import pandas as pd
import os

class SpotifyDataset:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = SpotifyDataset()
        return cls._instance

    def __init__(self):
        base_dir = os.path.join(os.path.dirname(__file__), 'data')
        # 支持多种常见的 CSV 文件名
        possible_names = ['dataset.csv', 'tracks.csv', 'spotify_tracks.csv', 'spotify_data.csv']
        
        self.csv_path = None
        for name in possible_names:
            p = os.path.join(base_dir, name)
            if os.path.exists(p):
                self.csv_path = p
                break
        
        self.df = None
        self.load_data()

    def load_data(self):
        if not self.csv_path:
            print(f"[WARN] 未找到数据集文件。推荐功能将无法使用。")
            return

        print(f"[INFO] 正在加载百万级数据集: {self.csv_path} (内存占用较大，请稍候)...")
        try:
            # 读取 CSV
            self.df = pd.read_csv(self.csv_path)
            
            # 1. 清理列名 (去除空格)
            self.df.columns = self.df.columns.str.strip()
            
            # 2. 统一 ID 列名
            if 'track_id' in self.df.columns:
                self.df.rename(columns={'track_id': 'id'}, inplace=True)
            
            # 3. 确保 ID 是字符串且不为空
            if 'id' in self.df.columns:
                self.df['id'] = self.df['id'].astype(str)
                self.df.drop_duplicates(subset=['id'], inplace=True)
                self.df.set_index('id', inplace=True, drop=False) # 保留 id 列以便后续使用
                print(f"[INFO] 数据集加载完成! 包含 {len(self.df)} 首歌曲。")
            else:
                print(f"[ERROR] CSV 中未找到 'id' 或 'track_id' 列，无法建立索引。")
                self.df = None
                
        except Exception as e:
            print(f"[ERROR] 加载数据集失败: {e}")
            self.df = None

    def get_track_features(self, track_id):
        """获取单曲特征 (用于前端展示)"""
        if self.df is None or track_id not in self.df.index:
            return None
        try:
            # loc[track_id] 可能会返回 DataFrame (如果有重复 ID)，我们需要 Series
            row = self.df.loc[track_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return self._row_to_dict(row)
        except:
            return None

    def get_track_features_by_name(self, track_name, artist_name):
        """通过歌名和歌手名查找特征 (备用方案)"""
        if self.df is None:
            return None
        
        try:
            # 1. 尝试精确匹配 (最快)
            # 注意：CSV 中的 artist_name 可能包含多个歌手，或者格式不同
            # 这里做一个简单的尝试
            matches = self.df[
                (self.df['track_name'] == track_name) & 
                (self.df['artist_name'] == artist_name)
            ]
            
            if matches.empty:
                # 2. 尝试稍微宽松的匹配 (忽略大小写) - 性能警告：这在百万级数据上可能较慢
                # 为了优化，我们可以先只匹配 track_name (假设重名歌曲远少于总数)
                # 然后在结果中筛选 artist
                
                # 筛选同名歌曲 (忽略大小写)
                # 注意：str.lower() 操作比较耗时，但如果只对 track_name 做，可能还行？
                # 不，还是先尝试精确匹配 track_name，然后模糊匹配 artist
                
                potential_matches = self.df[self.df['track_name'] == track_name]
                
                if potential_matches.empty:
                    # 如果连歌名都精确匹配不到，那可能真的没有，或者大小写差异
                    # 考虑到性能，这里不再做全表 lower() 扫描
                    return None
                
                # 在同名歌曲中查找 artist
                # 检查 artist_name 是否包含目标 artist (忽略大小写)
                target_artist = artist_name.lower()
                for _, row in potential_matches.iterrows():
                    db_artist = str(row['artist_name']).lower()
                    if target_artist in db_artist or db_artist in target_artist:
                        # 返回完整行字典，包含 'id' 以供上层匹配使用
                        return row.to_dict()

                return None
            else:
                # 返回完整行字典，包含 'id' 以供上层匹配使用
                return matches.iloc[0].to_dict()
                
        except Exception as e:
            print(f"[WARN] 按名称查找失败: {e}")
            return None

    def _row_to_dict(self, row):
        return {
            'danceability': row.get('danceability', '-'),
            'energy': row.get('energy', '-'),
            'valence': row.get('valence', '-'),
            'acousticness': row.get('acousticness', '-'),
            'instrumentalness': row.get('instrumentalness', '-'),
            'tempo': row.get('tempo', '-'),
            'genre': row.get('genre', 'Unknown')
        }

    def get_dataframe(self):
        """返回完整的 DataFrame 给推荐算法使用"""
        return self.df

    # --- 新增：用于歌曲列表/前端展示的便捷方法 ---
    def list_tracks(self, limit=50, offset=0, genre=None, year=None, search=None, sort_by='popularity'):
        """
        从离线 CSV 中分页返回歌曲列表。
        仅使用 DataFrame 切片，避免在接口层做重计算。
        """
        if self.df is None or self.df.empty:
            return [], 0

        df_view = self.df

        # 轻量筛选：genre / year / 关键词
        if genre:
            df_view = df_view[df_view['genre'].str.contains(str(genre), case=False, na=False)]
        if year:
            df_view = df_view[df_view['year'] == int(year)]
        if search:
            q = str(search).lower()
            df_view = df_view[
                df_view['track_name'].str.lower().str.contains(q, na=False) |
                df_view['artist_name'].str.lower().str.contains(q, na=False)
            ]

        total = len(df_view)
        start = max(int(offset), 0)
        end = start + int(limit)
        if sort_by not in {'popularity', 'name'}:
            sort_by = 'popularity'
        if sort_by == 'name':
            sliced = df_view.sort_values(by='track_name', ascending=True).iloc[start:end]
        else:
            sliced = df_view.sort_values(by='popularity', ascending=False).iloc[start:end]

        records = sliced[['id', 'track_name', 'artist_name', 'genre', 'year', 'popularity']].fillna('Unknown').to_dict('records')
        return records, total

    def get_track_record(self, track_id):
        """返回包含主要字段的单条歌曲记录，用于离线详情页。"""
        if self.df is None or track_id not in self.df.index:
            return None
        row = self.df.loc[track_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        # 提供尽量完整的信息以便前端展示
        return {
            'id': row.get('id'),
            'track_name': row.get('track_name', 'Unknown'),
            'artist_name': row.get('artist_name', 'Unknown'),
            'genre': row.get('genre', 'Unknown'),
            'year': row.get('year', 'Unknown'),
            'popularity': row.get('popularity', '-'),
            'danceability': row.get('danceability', '-'),
            'energy': row.get('energy', '-'),
            'valence': row.get('valence', '-'),
            'acousticness': row.get('acousticness', '-'),
            'instrumentalness': row.get('instrumentalness', '-'),
            'tempo': row.get('tempo', '-'),
        }
