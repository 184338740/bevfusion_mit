import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional
import open3d as o3d

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.data_root = Path(config.data_root)
        self.dataset_path = self.data_root / "datasets" / config.dataset_name
        
    def load_calibration(self) -> Dict:
        calib_path = self.data_root / "calib" / self.config.calib_file
        if not calib_path.exists():
            raise FileNotFoundError(f"标定文件不存在: {calib_path}")
        
        with open(calib_path, 'r') as f:
            data = json.load(f)
            print(f"标定文件中的相机: {[k for k in data.keys() if k.startswith('SC_R1_')]}")
            return data
    
    def load_image(self, camera_folder: str, timestamp: str) -> Optional[np.ndarray]:
        img_path = self.dataset_path / camera_folder / self.config.image_folder / f"{timestamp}.jpg"
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                print(f"加载图像: {img_path.name} ({img.shape[1]}x{img.shape[0]})")
                return img
        print(f"未找到图像: {img_path}")
        return None
    
    def load_labels(self, timestamp: str) -> List[Dict]:
        label_path = self.dataset_path / "lidar" / "label" / f"{timestamp}.json"
        if not label_path.exists():
            print(f"未找到标签文件: {label_path}")
            return []
        
        try:
            with open(label_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    labels = data.get('objects', [])
                else:
                    labels = data
                
                print(f"加载标签: {len(labels)} 个对象")
                for label in labels[:3]:
                    if 'box3d' in label:
                        print(f"  标签: {label.get('type', 'unknown')}, box3d: {label['box3d']}")
                return labels
        except Exception as e:
            print(f"加载标签文件失败: {e}")
            return []
    
    def load_pointcloud(self, timestamp: str) -> Optional[np.ndarray]:
        pcd_path = self.dataset_path / "lidar" / "pcd" / f"{timestamp}.pcd"
        npy_path = self.dataset_path / "lidar" / "pcd" / f"{timestamp}.npy"
        
        if pcd_path.exists():
            try:
                pcd = o3d.io.read_point_cloud(str(pcd_path))
                points = np.asarray(pcd.points)
                print(f"加载点云: {len(points)} 点")
                if len(points) > 0:
                    print(f"点云范围: x[{points[:,0].min():.1f}, {points[:,0].max():.1f}], "
                          f"y[{points[:,1].min():.1f}, {points[:,1].max():.1f}], "
                          f"z[{points[:,2].min():.1f}, {points[:,2].max():.1f}]")
                return points
            except Exception as e:
                print(f"加载点云失败: {e}")
        elif npy_path.exists():
            points = np.load(str(npy_path))
            print(f"加载点云: {len(points)} 点")
            return points
        
        print("未找到点云文件")
        return None