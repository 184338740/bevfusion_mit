import numpy as np
from typing import Dict, List, Optional, Tuple

class Transformer:
    def __init__(self, calib_data: Dict):
        self.calib_data = calib_data
    
    def get_camera_params(self, camera_name: str) -> Optional[Dict]:
        new_name = f"{camera_name}_new"
        if new_name in self.calib_data:
            print(f"使用_new后缀标定参数: {new_name}")
            return self.calib_data[new_name]
        if camera_name in self.calib_data:
            return self.calib_data[camera_name]
        print(f"警告: 未找到相机 {camera_name} 的标定参数")
        return None
    
    def quaternion_to_matrix(self, x: float, y: float, z: float, w: float) -> np.ndarray:
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
        ])
    
    def parse_transform(self, data: Dict) -> np.ndarray:
        translation = np.array([data['tx'], data['ty'], data['tz']])
        rotation = self.quaternion_to_matrix(data['rx'], data['ry'], data['rz'], data['rw'])
        
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform
    
    def transform_points_to_camera(self, points: np.ndarray, camera_name: str) -> np.ndarray:
        cam_params = self.get_camera_params(camera_name)
        if cam_params is None:
            return points
        
        camera_transform = self.parse_transform(cam_params)
        
        if 'group2map' in self.calib_data:
            group_transform = self.parse_transform(self.calib_data['group2map'])
        else:
            group_transform = np.eye(4)
        
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        points_map = (group_transform @ points_homo.T).T
        camera_inv = np.linalg.inv(camera_transform)
        points_camera = (camera_inv @ points_map.T).T
        
        return points_camera[:, :3]
    
    def create_3d_box_corners(self, box3d: List[float], rotation: float) -> np.ndarray:
        x, y, z, l, w, h = box3d
        
        corners = np.array([
            [l/2, w/2, h/2], [l/2, w/2, -h/2], [l/2, -w/2, h/2], [l/2, -w/2, -h/2],
            [-l/2, w/2, h/2], [-l/2, w/2, -h/2], [-l/2, -w/2, h/2], [-l/2, -w/2, -h/2]
        ])
        
        rot_matrix = np.array([
            [np.cos(rotation), -np.sin(rotation), 0],
            [np.sin(rotation), np.cos(rotation), 0],
            [0, 0, 1]
        ])
        
        corners_rotated = corners @ rot_matrix.T
        corners_world = corners_rotated + np.array([x, y, z])
        
        return corners_world
    
    def project_points_to_image(self, points: np.ndarray, camera_name: str, image_size: tuple) -> Tuple[np.ndarray, np.ndarray]:
        cam_params = self.get_camera_params(camera_name)
        if cam_params is None:
            return np.array([]), np.array([])
        
        print(f"  {camera_name}: 投影 {len(points)} 个点")
        points_camera = self.transform_points_to_camera(points, camera_name)
        
        front_mask = points_camera[:, 2] > 0.1
        if not np.any(front_mask):
            print(f"  {camera_name}: 无前方点 (所有点都在相机后面)")
            return np.array([]), np.array([])
        
        points_camera = points_camera[front_mask]
        depths = points_camera[:, 2]
        
        print(f"  {camera_name}: 前方点 {len(points_camera)}, 深度范围 [{depths.min():.1f}, {depths.max():.1f}]")
        
        intrinsic = np.array([
            [cam_params['fx'], 0, cam_params['cx']],
            [0, cam_params['fy'], cam_params['cy']],
            [0, 0, 1]
        ])
        
        points_2d_homo = (intrinsic @ points_camera.T).T
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
        
        h, w = image_size
        valid_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
                    (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
        
        valid_points = points_2d[valid_mask]
        valid_depths = depths[valid_mask]
        
        print(f"  {camera_name}: 图像内点 {len(valid_points)}")
        return valid_points, valid_depths