#!/usr/bin/env python3
"""
3D GT Label投影可视化工具 - 最终修复版本
修复A/C相机映射和输出路径
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

class GTLabelProjector:
    def __init__(self, data_root: str, calib_path: str):
        self.data_root = Path(data_root).expanduser()
        self.calib_path = Path(calib_path).expanduser()
        self.calib_data = self.load_calibration()
        
        # 正确的相机映射：标定名称 -> 实际文件夹名称
        self.camera_mapping = {
            'SC_R1_Aw_CpcS_CAMR': 'SC_1A_CamR',  # A相机：标定名称 -> A文件夹
            'SC_R1_Bn_CpcW_CAMR': 'SC_1B_CamR',  # B相机
            'SC_R1_Ce_CpcN_CAMR': 'SC_1C_CamR',  # C相机：标定名称 -> C文件夹  
            'SC_R1_Ds_CpcE_CAMR': 'SC_1D_CamR'   # D相机
        }
        
        # 创建标准化的输出路径
        today = datetime.now().strftime("%Y%m%d")
        self.output_dir = Path(f"output/3d_gt_projection/{today}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 颜色配置
        self.colors = {
            'car': (0, 255, 0),      # 绿色
            'truck': (255, 0, 0),    # 蓝色
            'cyclist': (0, 0, 255),  # 红色
            'pedestrian': (255, 255, 0),  # 青色
            'van': (255, 0, 255),    # 紫色
            'tricycle': (0, 255, 255),  # 黄色
            'default': (255, 255, 255)  # 白色
        }
        
        print("3D GT Label投影工具启动")
        print(f"数据路径: {self.data_root}")
        print(f"输出路径: {self.output_dir}")
    
    def load_calibration(self) -> Dict:
        with open(self.calib_path, 'r') as f:
            return json.load(f)
    
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
    
    def get_camera_calib(self, camera_name: str) -> Dict:
        if camera_name not in self.calib_data:
            return None
        
        cam_data = self.calib_data[camera_name]
        extrinsic = self.parse_transform(cam_data)
        
        intrinsic = np.array([
            [cam_data['fx'], 0, cam_data['cx']],
            [0, cam_data['fy'], cam_data['cy']],
            [0, 0, 1]
        ])
        
        return {
            'extrinsic': extrinsic,
            'intrinsic': intrinsic
        }
    
    def get_group_transform(self) -> np.ndarray:
        if 'group2map' not in self.calib_data:
            return None
        return self.parse_transform(self.calib_data['group2map'])
    
    def load_gt_labels(self, timestamp: str) -> List[Dict]:
        label_path = self.data_root / 'lidar' / 'label' / f"{timestamp}.json"
        if not label_path.exists():
            print(f"标签文件不存在: {label_path}")
            return []
        
        with open(label_path, 'r') as f:
            labels = json.load(f)
            print(f"加载3D标签: {len(labels)} 个对象")
            return labels
    
    def create_3d_box_corners(self, box3d: List[float], rotation: float) -> np.ndarray:
        x, y, z, l, w, h = box3d
        
        corners = np.array([
            [l/2, w/2, h/2],   [l/2, w/2, -h/2],
            [l/2, -w/2, h/2],  [l/2, -w/2, -h/2],
            [-l/2, w/2, h/2],  [-l/2, w/2, -h/2],
            [-l/2, -w/2, h/2], [-l/2, -w/2, -h/2]
        ])
        
        rot_matrix = np.array([
            [np.cos(rotation), -np.sin(rotation), 0],
            [np.sin(rotation), np.cos(rotation), 0],
            [0, 0, 1]
        ])
        corners = corners @ rot_matrix.T
        corners += np.array([x, y, z])
        
        return corners
    
    def create_orientation_arrow(self, box3d: List[float], rotation: float) -> np.ndarray:
        x, y, z, l, w, h = box3d
        
        start_point = np.array([l/2, 0, 0])
        end_point = np.array([l/2 + 2.0, 0, 0])
        
        rot_matrix = np.array([
            [np.cos(rotation), -np.sin(rotation), 0],
            [np.sin(rotation), np.cos(rotation), 0],
            [0, 0, 1]
        ])
        
        start_point = start_point @ rot_matrix.T
        end_point = end_point @ rot_matrix.T
        start_point += np.array([x, y, z])
        end_point += np.array([x, y, z])
        
        return np.array([start_point, end_point])
    
    def transform_points_to_camera(self, points: np.ndarray, camera_name: str) -> np.ndarray:
        camera_calib = self.get_camera_calib(camera_name)
        group_transform = self.get_group_transform()
        
        if camera_calib is None or group_transform is None:
            return points
        
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        points_map = (group_transform @ points_homo.T).T
        camera_inv = np.linalg.inv(camera_calib['extrinsic'])
        points_camera = (camera_inv @ points_map.T).T
        
        return points_camera[:, :3]
    
    def project_points_to_image(self, points: np.ndarray, camera_name: str, image_size: tuple) -> np.ndarray:
        camera_calib = self.get_camera_calib(camera_name)
        if camera_calib is None:
            return np.array([])
        
        points_cam = self.transform_points_to_camera(points, camera_name)
        front_mask = points_cam[:, 2] > 0
        if not np.any(front_mask):
            return np.array([])
        
        points_cam = points_cam[front_mask]
        intrinsic = camera_calib['intrinsic']
        points_2d_homo = (intrinsic @ points_cam.T).T
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
        
        h, w = image_size
        valid_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
                    (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
        
        return points_2d[valid_mask]
    
    def draw_3d_box_on_image(self, image: np.ndarray, box_corners_2d: np.ndarray, 
                           color: tuple, orientation_arrow_2d: np.ndarray = None) -> np.ndarray:
        vis_image = image.copy()
        
        if len(box_corners_2d) != 8:
            return vis_image
        
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 3),
            (4, 5), (4, 6), (5, 7), (6, 7),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        for i, j in edges:
            pt1 = (int(box_corners_2d[i, 0]), int(box_corners_2d[i, 1]))
            pt2 = (int(box_corners_2d[j, 0]), int(box_corners_2d[j, 1]))
            
            if (0 <= pt1[0] < vis_image.shape[1] and 0 <= pt1[1] < vis_image.shape[0] and
                0 <= pt2[0] < vis_image.shape[1] and 0 <= pt2[1] < vis_image.shape[0]):
                cv2.line(vis_image, pt1, pt2, color, 2)
        
        if orientation_arrow_2d is not None and len(orientation_arrow_2d) == 2:
            start_pt = (int(orientation_arrow_2d[0, 0]), int(orientation_arrow_2d[0, 1]))
            end_pt = (int(orientation_arrow_2d[1, 0]), int(orientation_arrow_2d[1, 1]))
            
            if (0 <= start_pt[0] < vis_image.shape[1] and 0 <= start_pt[1] < vis_image.shape[0] and
                0 <= end_pt[0] < vis_image.shape[1] and 0 <= end_pt[1] < vis_image.shape[0]):
                cv2.arrowedLine(vis_image, start_pt, end_pt, (255, 255, 0), 3, tipLength=0.3)
        
        return vis_image
    
    def project_gt_labels_to_camera(self, timestamp: str, camera_name: str) -> Dict:
        print(f"处理相机: {camera_name}")
        
        labels = self.load_gt_labels(timestamp)
        if not labels:
            return {}
        
        # 使用正确的映射关系：标定名称 -> 实际文件夹
        folder_name = self.camera_mapping[camera_name]
        img_path = self.data_root / folder_name / 'image_dc' / f"{timestamp}.jpg"
        if not img_path.exists():
            print(f"图像文件不存在: {img_path}")
            return {}
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"无法加载图像: {img_path}")
            return {}
        
        image_size = image.shape[:2]
        print(f"  图像尺寸: {image_size[1]}x{image_size[0]}")
        
        vis_image = image.copy()
        objects_projected = 0
        
        for label in labels:
            obj_type = label.get('type', 'unknown')
            box3d = label['box3d']
            rotation = label['rotation'][0]
            
            color = self.colors.get(obj_type, self.colors['default'])
            
            box_corners_3d = self.create_3d_box_corners(box3d, rotation)
            orientation_arrow_3d = self.create_orientation_arrow(box3d, rotation)
            
            box_corners_2d = self.project_points_to_image(box_corners_3d, camera_name, image_size)
            orientation_arrow_2d = self.project_points_to_image(orientation_arrow_3d, camera_name, image_size)
            
            if len(box_corners_2d) > 0:
                vis_image = self.draw_3d_box_on_image(vis_image, box_corners_2d, color, orientation_arrow_2d)
                objects_projected += 1
        
        # 使用标准化的输出文件名
        output_filename = f"3d_gt_projection_{camera_name}_{timestamp}.jpg"
        output_path = self.output_dir / output_filename
        cv2.imwrite(str(output_path), vis_image)
        
        print(f"  保存结果: {output_filename}")
        print(f"  投影统计: {objects_projected}/{len(labels)} 个对象")
        
        return {
            'camera': camera_name,
            'objects_projected': objects_projected,
            'objects_total': len(labels),
            'output_path': output_path
        }
    
    def run_projection(self, timestamp: str):
        print("=" * 50)
        print("3D GT Label投影 - 最终修复版本")
        print("=" * 50)
        print(f"时间戳: {timestamp}")
        print(f"输出目录: {self.output_dir}")
        
        label_path = self.data_root / 'lidar' / 'label' / f"{timestamp}.json"
        if not label_path.exists():
            print(f"标签文件不存在: {label_path}")
            return
        
        results = {}
        for camera_name in self.camera_mapping.keys():
            result = self.project_gt_labels_to_camera(timestamp, camera_name)
            results[camera_name] = result
        
        self.generate_summary_report(results, timestamp)
    
    def generate_summary_report(self, results: Dict, timestamp: str):
        print("\n" + "=" * 50)
        print("3D GT Label投影总结")
        print("=" * 50)
        
        total_objects = 0
        total_projected = 0
        
        print(f"时间戳: {timestamp}")
        print("投影结果:")
        
        for camera_name, result in results.items():
            if result:
                objects_total = result['objects_total']
                objects_projected = result['objects_projected']
                projection_rate = objects_projected / objects_total if objects_total > 0 else 0
                
                print(f"  {camera_name}: {objects_projected}/{objects_total} 对象 ({projection_rate:.1%})")
                
                total_objects += objects_total
                total_projected += objects_projected
        
        overall_rate = total_projected / total_objects if total_objects > 0 else 0
        print(f"\n总体统计: {total_projected}/{total_objects} 对象 ({overall_rate:.1%})")
        print(f"输出目录: {self.output_dir}")

def main():
    data_root = "~/ws/02_docker/cv2-tests/dataset/2d3d4d_20250728_weiyuan/datasets/G51102400001M00_20250728191726_2.0HZ"
    calib_path = "~/下载/2d3d4d_20250728_weiyuan/calib/sensor2map_calib.json"
    timestamp = "1753701446899737835"
    
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请安装: pip install opencv-python numpy")
        return
    
    projector = GTLabelProjector(data_root, calib_path)
    projector.run_projection(timestamp)

if __name__ == "__main__":
    main()