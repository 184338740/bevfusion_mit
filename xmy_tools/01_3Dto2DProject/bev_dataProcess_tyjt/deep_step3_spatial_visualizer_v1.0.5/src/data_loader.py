"""
数据加载模块
版本: v1.0.5
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import open3d as o3d
from config import BEVConfig

class DataLoader:
    """统一数据加载器"""
    
    def __init__(self, config: BEVConfig):
        self.config = config
        self.data_root = Path(config.data_root)
        self.dataset_path = self.data_root / "datasets" / config.dataset_name
        
        # 验证路径存在性
        self._validate_paths()
    
    def _validate_paths(self):
        """验证必要路径是否存在"""
        required_paths = [
            self.data_root / "calib" / self.config.calib_file,
            self.dataset_path
        ]
        
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"路径不存在: {path}")
    
    def load_calibration(self) -> Dict:
        """加载标定文件"""
        calib_path = self.data_root / "calib" / self.config.calib_file
        print(f"加载标定文件: {calib_path}")
        
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
            cameras = [k for k in calib_data.keys() if k.startswith('SC_R1_')]
            print(f"标定文件中的相机: {cameras}")
            return calib_data
    
    def load_image(self, camera_folder: str, timestamp: str) -> Optional[np.ndarray]:
        """加载图像"""
        img_path = self.dataset_path / camera_folder / self.config.image_folder / f"{timestamp}.jpg"
        
        if not img_path.exists():
            print(f"未找到图像: {img_path}")
            return None
        
        img = cv2.imread(str(img_path))
        if img is not None:
            print(f"加载图像: {img_path.name} ({img.shape[1]}x{img.shape[0]})")
            return img
        
        return None
    
    def load_labels(self, timestamp: str) -> List[Dict]:
        """加载3D标签"""
        label_path = self.dataset_path / "lidar" / "label" / f"{timestamp}.json"
        
        if not label_path.exists():
            print(f"未找到标签文件: {label_path}")
            return []
        
        try:
            with open(label_path, 'r') as f:
                data = json.load(f)
                labels = data.get('objects', []) if isinstance(data, dict) else data
                
                print(f"加载标签: {len(labels)} 个对象")
                for label in labels[:3]:  # 只打印前3个标签信息
                    if 'box3d' in label:
                        print(f"  标签: {label.get('type', 'unknown')}, box3d: {label['box3d']}")
                return labels
        except Exception as e:
            print(f"加载标签文件失败: {e}")
            return []
    
    def load_pointcloud(self, timestamp: str) -> Optional[np.ndarray]:
        """加载点云"""
        # 尝试加载.pcd文件
        pcd_path = self.dataset_path / "lidar" / "pcd" / f"{timestamp}.pcd"
        if pcd_path.exists():
            try:
                pcd = o3d.io.read_point_cloud(str(pcd_path))
                points = np.asarray(pcd.points)
                self._log_pointcloud_info(points, "pcd")
                return points
            except Exception as e:
                print(f"加载点云失败: {e}")
        
        # 尝试加载.npy文件
        npy_path = self.dataset_path / "lidar" / "pcd" / f"{timestamp}.npy"
        if npy_path.exists():
            points = np.load(str(npy_path))
            self._log_pointcloud_info(points, "npy")
            return points
        
        print("未找到点云文件")
        return None
    
    def _log_pointcloud_info(self, points: np.ndarray, format_type: str):
        """记录点云信息"""
        if len(points) == 0:
            print(f"加载点云: 0 点 ({format_type})")
            return
        
        print(f"加载点云: {len(points)} 点 ({format_type})")
        if len(points) > 0:
            print(f"点云范围: x[{points[:,0].min():.1f}, {points[:,0].max():.1f}], "
                  f"y[{points[:,1].min():.1f}, {points[:,1].max():.1f}], "
                  f"z[{points[:,2].min():.1f}, {points[:,2].max():.1f}]")