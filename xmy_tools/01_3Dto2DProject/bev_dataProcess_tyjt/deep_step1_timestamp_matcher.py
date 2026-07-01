import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class TimestampMatcher:
    def __init__(self, data_root: str, output_root: str = "output"):
        self.data_root = Path(data_root).expanduser()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_root) / "01_TimestampMatching" / current_time
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_path = self.data_root / "datasets" / "G51102400001M00_20250728191726_2.0HZ"
        
        # 相机名称映射
        self.camera_mapping = {
            'SC_R1_Ae_CpcN_CAMR': 'SC_1A_CamR',
            'SC_R1_Be_CpcN_CAMR': 'SC_1B_CamR', 
            'SC_R1_Ce_CpcN_CAMR': 'SC_1C_CamR',
            'SC_R1_De_CpcN_CAMR': 'SC_1D_CamR'
        }
        
        self.camera_names = list(self.camera_mapping.keys())
        self.camera_folders = list(self.camera_mapping.values())
        
        print(f"数据根目录: {self.data_root}")
        print(f"输出目录: {self.output_dir}")
    
    def extract_timestamp_from_filename(self, filename: str) -> str:
        """从文件名中提取时间戳"""
        return Path(filename).stem
    
    def find_matching_files(self, timestamp: str) -> Dict:
        """根据时间戳查找匹配的文件"""
        files = {
            'lidar_pcd': None,
            'lidar_npy': None,
            'labels': None,
            'images': {}
        }
        
        # 查找点云文件
        lidar_dir = self.dataset_path / "lidar" / "pcd"
        if lidar_dir.exists():
            pcd_files = list(lidar_dir.glob(f"{timestamp}.pcd"))
            npy_files = list(lidar_dir.glob(f"{timestamp}.npy"))
            files['lidar_pcd'] = pcd_files[0] if pcd_files else None
            files['lidar_npy'] = npy_files[0] if npy_files else None
        
        # 查找标签文件
        label_dir = self.dataset_path / "lidar" / "label"
        if label_dir.exists():
            label_files = list(label_dir.glob(f"{timestamp}.json"))
            files['labels'] = label_files[0] if label_files else None
        
        # 查找图像文件
        for cam_name, cam_folder in self.camera_mapping.items():
            image_dir = self.dataset_path / cam_folder / "image_dc"
            if image_dir.exists():
                image_files = list(image_dir.glob(f"{timestamp}.jpg"))
                files['images'][cam_name] = image_files[0] if image_files else None
        
        return files
    
    def get_all_timestamps(self) -> List[str]:
        """获取所有可用的时间戳"""
        timestamps = set()
        
        # 从点云文件获取时间戳
        lidar_dir = self.dataset_path / "lidar" / "pcd"
        if lidar_dir.exists():
            for file in lidar_dir.glob("*.pcd"):
                timestamps.add(self.extract_timestamp_from_filename(file.name))
        
        # 从标签文件获取时间戳
        label_dir = self.dataset_path / "lidar" / "label"
        if label_dir.exists():
            for file in label_dir.glob("*.json"):
                timestamps.add(self.extract_timestamp_from_filename(file.name))
        
        return sorted(list(timestamps))
    
    def run_matching_analysis(self):
        """运行时间戳匹配分析"""
        print("开始时间戳匹配分析...")
        
        timestamps = self.get_all_timestamps()
        print(f"找到 {len(timestamps)} 个时间戳")
        
        # 验证每个时间戳的文件完整性
        results = []
        for i, ts in enumerate(timestamps):
            if i % 100 == 0:
                print(f"处理进度: {i}/{len(timestamps)}")
                
            files = self.find_matching_files(ts)
            
            result = {
                'timestamp': ts,
                'lidar_pcd': files['lidar_pcd'] is not None,
                'lidar_npy': files['lidar_npy'] is not None,
                'labels': files['labels'] is not None
            }
            
            for cam_name in self.camera_names:
                result[f'image_{cam_name}'] = files['images'].get(cam_name) is not None
            
            results.append(result)
        
        df = pd.DataFrame(results)
        
        # 保存统计结果
        stats = {
            'total_timestamps': len(df),
            'complete_samples': len(df[df.all(axis=1)]),
            'missing_lidar_pcd': len(df[~df['lidar_pcd']]),
            'missing_lidar_npy': len(df[~df['lidar_npy']]),
            'missing_labels': len(df[~df['labels']]),
            'camera_missing_stats': {
                cam: len(df[~df[f'image_{cam}']]) for cam in self.camera_names
            }
        }
        
        # 保存结果
        with open(self.output_dir / "timestamp_matching_stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        df.to_csv(self.output_dir / "timestamp_matching_results.csv", index=False)
        
        # 保存完整样本列表
        complete_samples = df[df.all(axis=1)]['timestamp'].tolist()
        with open(self.output_dir / "complete_samples.txt", 'w') as f:
            for ts in complete_samples:
                f.write(f"{ts}\n")
        
        print(f"时间戳匹配分析完成！")
        print(f"总时间戳: {stats['total_timestamps']}")
        print(f"完整样本: {stats['complete_samples']}")
        print(f"输出目录: {self.output_dir}")
        
        return df, stats

if __name__ == "__main__":
    data_path = "/home/tyjt/ws/02_docker/cv2-tests/dataset/2d3d4d_20250728_weiyuan"
    
    matcher = TimestampMatcher(data_path)
    df, stats = matcher.run_matching_analysis()