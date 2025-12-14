import pandas as pd
import numpy as np
from dataset_service import SpotifyDataset
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import os
import pickle
import time
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(os.getenv('RECOMMENDER_LOG_LEVEL', 'INFO'))

# --- 1. 定义 MLP Autoencoder ---

class Autoencoder(nn.Module):
    """
    标准自动编码器 (MLP Autoencoder)
    结构简单，训练快，专门用于特征压缩和降维
    """
    def __init__(self, input_dim, latent_dim=32):
        super(Autoencoder, self).__init__()
        
        # Encoder (MLP 映射)
        # 将高维特征映射到低维潜在空间
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim) # 输出 32 维向量
        )
        
        # Decoder (重构)
        # 尝试从潜在向量还原原始特征
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid() # 输出范围 [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# --- 2. 推荐系统核心类 ---

class ContentBasedRecommender:
    def __init__(self, progress_callback=None, fallback_loose=None):
        self.progress_callback = progress_callback
        self.device = self._check_hardware()
        self._update_progress(5, "正在加载数据集...")
        
        self.dataset = SpotifyDataset.get_instance()
        self.df = self.dataset.get_dataframe()
        
        # 核心音频特征 (扩展特征集以提升精度)
        self.feature_cols = [
            'danceability', 'energy', 'valence', 
            'acousticness', 'instrumentalness', 'speechiness',
            'tempo', 'loudness',
            'liveness', 'mode', 'key', 'duration_ms', 'popularity'
        ]
        
        self.scaler = MinMaxScaler()
        self.scaled_features = None
        self.model = None
        self.embeddings = None
        
        # 模型缓存路径
        self.cache_dir = os.path.join(os.path.dirname(__file__), 'model_cache')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.scaler_path = os.path.join(self.cache_dir, 'scaler.pkl')
        self.model_weights_path = os.path.join(self.cache_dir, 'ae_model.pth')
        self.embeddings_path = os.path.join(self.cache_dir, 'embeddings.npy')
        
        if self.df is not None:
            self._preprocess_data()
            self._init_model()
        else:
            logger.warning("推荐引擎初始化失败: 数据集为空")
        # Fallback matching mode: strict by default (match artist exactly),
        # can be overridden by constructor arg or environment variable RECOMMENDER_FALLBACK_LOOSE=1
        env_loose = os.getenv('RECOMMENDER_FALLBACK_LOOSE', '').lower() in ('1', 'true', 'yes')
        if fallback_loose is None:
            self.fallback_loose = env_loose
        else:
            self.fallback_loose = bool(fallback_loose)
        logger.info(f"Fallback loose matching: {self.fallback_loose}")

    def _update_progress(self, percent, message):
        if self.progress_callback:
            self.progress_callback(percent, message)

    def _check_hardware(self):
        logger.info("正在检测硬件环境...")
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"发现 GPU 加速: {torch.cuda.get_device_name(0)}")
            self._update_progress(3, f"GPU 加速已开启 ({torch.cuda.get_device_name(0)})")
        else:
            device = torch.device("cpu")
            logger.info("未发现 GPU，将使用 CPU 进行训练")
            self._update_progress(3, "使用 CPU 进行训练 (未检测到 GPU)")
        return device

    def _preprocess_data(self):
        self._update_progress(10, "正在清洗和预处理数据...")
        # 数据清洗
        for col in self.feature_cols:
            if col not in self.df.columns:
                logger.warning(f"缺失列: {col}，尝试填充 0")
                self.df[col] = 0
                
        self.df = self.df.dropna(subset=self.feature_cols)
        for col in self.feature_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df = self.df.dropna(subset=self.feature_cols)
        
        # 优化: 处理离群值 (Outliers)
        logger.info("正在处理数据离群值...")
        for col in self.feature_cols:
            time.sleep(0.05) # 让出 CPU
            lower = self.df[col].quantile(0.01)
            upper = self.df[col].quantile(0.99)
            self.df[col] = self.df[col].clip(lower, upper)

        # 特征归一化 [0, 1]
        if os.path.exists(self.scaler_path):
            try:
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("[Step 1] 加载预训练的特征缩放器...")
                self.scaled_features = self.scaler.transform(self.df[self.feature_cols])
                self._update_progress(15, "特征缩放完成...")
                return
            except Exception as e:
                logger.warning(f"加载 Scaler 失败: {e}，将重新拟合。")

        logger.info("[Step 1] 数据预处理: 将音频特征归一化到 [0, 1] 区间...")
        self.scaled_features = self.scaler.fit_transform(self.df[self.feature_cols])
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        self._update_progress(15, "特征缩放完成...")

    def _init_model(self):
        self._update_progress(20, "初始化深度学习模型架构 (MLP Autoencoder)...")
        logger.info("[Step 2] 初始化 MLP Autoencoder...")
        
        input_dim = self.scaled_features.shape[1]
        self.model = Autoencoder(input_dim=input_dim).to(self.device)
        
        # 尝试加载预训练模型
        if os.path.exists(self.model_weights_path) and os.path.exists(self.embeddings_path):
            logger.info("发现预训练模型，正在加载...")
            try:
                state_dict = torch.load(self.model_weights_path, map_location=self.device, weights_only=True)

                # 检查维度
                saved_input_dim = state_dict['encoder.0.weight'].shape[1]
                if saved_input_dim != input_dim:
                    logger.warning(f"模型输入维度不匹配 (Saved: {saved_input_dim}, Current: {input_dim})，将重新训练...")
                    raise ValueError("Input dimension mismatch")

                self.model.load_state_dict(state_dict)
                self.model.eval()

                self.embeddings = np.load(self.embeddings_path)

                if len(self.embeddings) == len(self.df):
                    logger.info(f"[SUCCESS] 模型加载完成。已索引 {len(self.df)} 首歌曲。")
                    self._update_progress(100, "模型加载完成！")
                    return
                else:
                    logger.warning("数据集大小已变更，将重新训练...")
            except Exception as e:
                logger.warning(f"加载模型失败 ({e})，将重新训练...")

        logger.info("[Step 3] 开始训练 MLP Autoencoder...")
        self._update_progress(25, "准备训练数据...")
        
        train_data = torch.FloatTensor(self.scaled_features)
        
        # Batch Size 256
        batch_size = 256
        dataset = TensorDataset(train_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 使用 MSE Loss 和 Adam
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        epochs = 20 # MLP 收敛很快，20 epoch 足够
        
        for epoch in range(epochs):
            time.sleep(0.1) # 让出 CPU

            total_loss = 0
            for batch_idx, (data,) in enumerate(dataloader):
                data = data.to(self.device)
                optimizer.zero_grad()
                
                encoded, decoded = self.model(data)
                loss = criterion(decoded, data)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            if np.isnan(avg_loss):
                logger.error(f"训练出现异常: Loss 变为 NaN (Epoch {epoch+1})")
                break

            p = 30 + int((epoch + 1) / epochs * 50)
            self._update_progress(p, f"正在训练神经网络 (Epoch {epoch+1}/{epochs})... Loss: {avg_loss:.6f}")

        logger.info("保存模型权重...")
        self._update_progress(85, "保存模型权重...")
        torch.save(self.model.state_dict(), self.model_weights_path)

        logger.info("[Step 4] 生成全库音乐指纹 (Embeddings)...")
        self._update_progress(90, "生成全库音乐指纹...")
        
        self.model.eval()
        embeddings_list = []
        predict_loader = DataLoader(dataset, batch_size=4096, shuffle=False)
        
        with torch.no_grad():
            for (data,) in predict_loader:
                data = data.to(self.device)
                encoded, _ = self.model(data)
                embeddings_list.append(encoded.cpu().numpy())
        
        self.embeddings = np.concatenate(embeddings_list, axis=0)
        np.save(self.embeddings_path, self.embeddings)
        
        logger.info(f"[SUCCESS] 推荐系统就绪。已索引 {len(self.df)} 首歌曲。")
        self._update_progress(100, "初始化完成！")

    def recommend(self, seed_track_infos, limit=50):
        """
        基于 MLP Autoencoder 的推荐 (Max Similarity Strategy)
        支持两种输入格式：
        - 列表字符串 id：['id1','id2',...]
        - 列表字典：[{ 'id':..., 'name':..., 'artist':... }, ...]
        """
        logger.info("启动智能推荐流程 (MLP Autoencoder - Max Sim)")

        if self.df is None or self.embeddings is None or not seed_track_infos:
            return []

        # 1. Input
        # 兼容老的只传 id 的调用
        seed_ids = []
        # If input is list of strings
        if isinstance(seed_track_infos, (list, tuple)) and seed_track_infos and isinstance(seed_track_infos[0], str):
            seed_ids = [str(x) for x in seed_track_infos]
        else:
            # Expect list of dicts with possible keys 'id','name','artist'
            for item in seed_track_infos:
                if not item:
                    continue
                if isinstance(item, dict) and item.get('id'):
                    seed_ids.append(str(item.get('id')))
                else:
                    # fallback: ignore
                    continue

        # Normalize to strings
        seed_ids = [str(x) for x in seed_ids]

        # DEBUG: 记录输入信息以便排查匹配问题
        logger.debug(f"原始 seed_track_infos: {seed_track_infos}")
        logger.debug(f"解析后 seed_ids: {seed_ids}")

        # Check which seed ids exist in dataset
        seed_mask = self.df.index.isin(seed_ids)

        # If some provided ids are not found, attempt to fallback by name+artist when available
        missing_ids = [sid for sid in seed_ids if sid not in list(self.df.index)]
        if missing_ids:
            logger.debug(f"缺失的 ids: {missing_ids}")
        if missing_ids:
            # Build a map from provided infos if available
            id_map = {}
            if isinstance(seed_track_infos, (list, tuple)):
                for item in seed_track_infos:
                    if isinstance(item, dict):
                        _id = item.get('id')
                        name = item.get('name')
                        artist = item.get('artist')
                        if _id and str(_id) in missing_ids and name and artist:
                            id_map[str(_id)] = (name, artist)

            from dataset_service import SpotifyDataset
            ds = SpotifyDataset.get_instance()
            for mid in missing_ids:
                if mid in id_map:
                    name, artist = id_map[mid]
                    try:
                        row = ds.get_track_features_by_name(name, artist)
                        logger.debug(f"回退查找 for id={mid}, name={name}, artist={artist} -> row: {row}")
                        if row and row.get('id'):
                            # Enforce strict or loose artist matching depending on configuration
                            db_artist = str(row.get('artist_name', '')).strip().lower()
                            target_artist = str(artist).strip().lower() if artist else ''
                            matched = False
                            if self.fallback_loose:
                                # Loose: accept if either contains the other
                                if target_artist and (target_artist in db_artist or db_artist in target_artist):
                                    matched = True
                            else:
                                # Strict: require exact match (case-insensitive)
                                if db_artist and target_artist and db_artist == target_artist:
                                    matched = True

                            if matched:
                                found_id = str(row.get('id'))
                                seed_ids.append(found_id)
                                logger.debug(f"回退匹配成功: {mid} -> {found_id} (db_artist={db_artist}, target={target_artist})")
                            else:
                                logger.debug(f"回退匹配被拒绝(严格模式): {mid} (db_artist={db_artist}, target={target_artist})")
                    except Exception as e:
                        logger.debug(f"回退查找失败 for {mid}: {e}")

        # Recompute mask after possible fallbacks
        seed_mask = self.df.index.isin(seed_ids)
        if not np.any(seed_mask):
            logger.warning("歌单中的歌曲未在数据库中找到。")
            return self.df.sample(limit).to_dict('records')

        logger.info(f"[Step 1] 输入分析: 识别到 {np.sum(seed_mask)} 首有效种子歌曲。")
        # 2. Latent Mapping
        seed_vectors = self.embeddings[seed_mask]
        logger.info(f"[Step 2] 深度编码: 已将种子歌曲映射到 32维 潜在风格空间。")

        # 3. Similarity Search (Max Similarity Strategy)
        # 策略变更: 不再计算平均口味，而是为每首种子歌曲寻找相似歌曲，然后取最大值。
        # 这能更好地保留歌单的多样性 (例如同时包含古典和金属)。
        logger.info("[Step 3] 全库检索: 正在计算相似度 (Max Strategy)...")
        
        from sklearn.preprocessing import normalize
        
        # 归一化向量 (L2 Norm) 以便直接使用点积计算余弦相似度
        # shape: (N_db, 32)
        db_norm = normalize(self.embeddings, axis=1)
        # shape: (N_seeds, 32)
        seeds_norm = normalize(seed_vectors, axis=1)
        
        # 初始化最大相似度数组
        n_db = self.embeddings.shape[0]
        max_scores = np.full(n_db, -1.0)
        
        # 逐个种子计算相似度并更新最大值 (内存优化)
        # 相当于: 对于库里的每首歌，它与我歌单里最像的那首歌有多像？
        for i in range(seeds_norm.shape[0]):
            # dot product: (32,) @ (32, N_db) -> (N_db,)
            sim = np.dot(db_norm, seeds_norm[i])
            max_scores = np.maximum(max_scores, sim)
            
        scores = max_scores
        
        # 排除种子歌曲自身 (避免推荐已有的歌)
        scores[seed_mask] = -1
        
        # 获取 Top N
        top_indices = scores.argsort()[-limit:][::-1]
        recommendations = self.df.iloc[top_indices]
        
        top_score = scores[top_indices][0]
        logger.info(f"[SUCCESS] 推荐生成完毕! 最佳匹配度: {top_score:.4f}")
        logger.debug("="*50 + "\n")
        
        return recommendations.to_dict('records')
