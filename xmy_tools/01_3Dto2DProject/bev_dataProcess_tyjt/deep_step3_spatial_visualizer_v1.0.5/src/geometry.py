"""
几何计算工具模块
版本: v1.0.5
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

class GeometryUtils:
    """几何计算工具类"""
    
    @staticmethod
    def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
        """四元数转旋转矩阵"""
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
        ])
    
    @staticmethod
    def parse_transform(data: Dict) -> np.ndarray:
        """解析变换矩阵"""
        translation = np.array([data['tx'], data['ty'], data['tz']])
        rotation = GeometryUtils.quaternion_to_matrix(
            data['rx'], data['ry'], data['rz'], data['rw']
        )
        
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform
    
    @staticmethod
    def create_3d_box_corners(box3d: List[float], rotation: float) -> np.ndarray:
        """创建3D边界框的8个角点"""
        x, y, z, l, w, h = box3d
        
        # 创建边界框的8个角点 (局部坐标系)
        corners = np.array([
            [l/2, w/2, h/2], [l/2, w/2, -h/2], [l/2, -w/2, h/2], [l/2, -w/2, -h/2],
            [-l/2, w/2, h/2], [-l/2, w/2, -h/2], [-l/2, -w/2, h/2], [-l/2, -w/2, -h/2]
        ])
        
        # 应用旋转 (绕Z轴)
        rot_matrix = np.array([
            [np.cos(rotation), -np.sin(rotation), 0],
            [np.sin(rotation), np.cos(rotation), 0],
            [0, 0, 1]
        ])
        
        corners_rotated = corners @ rot_matrix.T
        corners_world = corners_rotated + np.array([x, y, z])
        
        return corners_world
    
    @staticmethod
    def project_points_to_image(points: np.ndarray, camera_params: Dict, 
                              image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """将3D点投影到图像平面"""
        if len(points) == 0:
            return np.array([]), np.array([])
        
        # 过滤相机后面的点
        front_mask = points[:, 2] > 0.1
        if not np.any(front_mask):
            return np.array([]), np.array([])
        
        points_camera = points[front_mask]
        depths = points_camera[:, 2]
        
        # 相机内参投影
        intrinsic = np.array([
            [camera_params['fx'], 0, camera_params['cx']],
            [0, camera_params['fy'], camera_params['cy']],
            [0, 0, 1]
        ])
        
        # 投影到图像平面
        points_2d_homo = (intrinsic @ points_camera.T).T
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
        
        # 过滤图像范围外的点
        h, w = image_size
        valid_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
                    (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
        
        valid_points = points_2d[valid_mask]
        valid_depths = depths[valid_mask]
        
        return valid_points, valid_depths

class CalibrationManager:
    """标定参数管理器"""
    
    def __init__(self, calib_data: Dict):
        self.calib_data = calib_data
    
    def get_camera_params(self, camera_name: str) -> Optional[Dict]:
        """获取相机参数，优先使用_new后缀"""
        new_name = f"{camera_name}_new"
        if new_name in self.calib_data:
            return self.calib_data[new_name]
        return self.calib_data.get(camera_name)
    
    def get_camera_transform(self, camera_name: str) -> Optional[np.ndarray]:
        """获取相机变换矩阵"""
        cam_params = self.get_camera_params(camera_name)
        return GeometryUtils.parse_transform(cam_params) if cam_params else None
    
    def get_group_transform(self) -> np.ndarray:
        """获取group变换矩阵"""
        if 'group2map' in self.calib_data:
            return GeometryUtils.parse_transform(self.calib_data['group2map'])
        return np.eye(4)
    
    def transform_points_to_camera(self, points: np.ndarray, camera_name: str) -> np.ndarray:
        """将点从group坐标系变换到相机坐标系"""
        camera_transform = self.get_camera_transform(camera_name)
        group_transform = self.get_group_transform()
        
        if camera_transform is None:
            return points
        
        # 转换为齐次坐标并变换
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        points_map = (group_transform @ points_homo.T).T
        camera_inv = np.linalg.inv(camera_transform)
        points_camera = (camera_inv @ points_map.T).T
        
        return points_camera[:, :3]