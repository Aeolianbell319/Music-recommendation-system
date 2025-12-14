import kagglehub
import os
import shutil
import glob

def setup_dataset():
    print("="*50)
    print("正在通过 KaggleHub 下载 Spotify Tracks Dataset...")
    print("这可能需要几分钟，取决于您的网速 (文件约 100MB+)")
    print("="*50)
    
    try:
        # Download latest version
        # path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
        print("正在下载 1 Million Tracks Dataset (可能需要更长时间)...")
        path = kagglehub.dataset_download("amitanshjoshi/spotify-1million-tracks")
        print(f"\n[下载完成] 原始路径: {path}")

        # Define target directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(current_dir, 'data')
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            print(f"[创建目录] {target_dir}")

        # Find the CSV file in the downloaded folder
        csv_files = glob.glob(os.path.join(path, "*.csv"))

        if csv_files:
            source_file = csv_files[0]
            target_file = os.path.join(target_dir, 'dataset.csv')
            
            print(f"[处理中] 正在将文件移动到项目目录...")
            shutil.copy2(source_file, target_file)
            
            print(f"\n[成功] 数据集已就绪！")
            print(f"文件位置: {target_file}")
            print("现在您可以运行 'python app.py' 来体验完整的特征分析功能了。")
        else:
            print(f"\n[错误] 在下载路径中未找到 .csv 文件。")
            print(f"请手动检查: {path}")

    except Exception as e:
        print(f"\n[失败] 下载或移动过程中出错: {e}")
        print("如果自动下载失败，请尝试手动下载并放入 data 文件夹。")

if __name__ == "__main__":
    setup_dataset()
