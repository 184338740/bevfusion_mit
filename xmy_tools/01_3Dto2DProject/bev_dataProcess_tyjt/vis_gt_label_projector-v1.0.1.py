import os
import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

class Label3DProjector:
    def __init__(self, 
                 data_root: str, 
                 calib_path: str, 
                 task_abbr: str = "label_proj"):
        self.data_root = Path(data_root).expanduser()
        self.calib_path = Path(calib_path).expanduser()
        self.task_abbr = task_abbr
        
        self.time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_root = Path(f"output/{self.task_abbr}/{self.time_str}")
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        self.calib_data = self._load_calibration()
        self.camera_mapping = {
            'SC_R1_Aw_CpcS_CAMR': 'SC_1A_CamR',
            'SC_R1_Bn_CpcW_CAMR': 'SC_1B_CamR',
            'SC_R1_Ce_CpcN_CAMR': 'SC_1C_CamR',
            'SC_R1_Ds_CpcE_CAMR': 'SC_1D_CamR'
        }
        self.valid_cameras = {
            cam_name: folder_name 
            for cam_name, folder_name in self.camera_mapping.items()
            if cam_name in self.calib_data
        }
        self.colors = {
            'car': (0, 255, 0), 'truck': (255, 0, 0), 'cyclist': (0, 0, 255),
            'pedestrian': (255, 255, 0), 'van': (255, 0, 255),
            'tricycle': (0, 255, 255), 'default': (255, 255, 255)
        }
        # 完整边集（恢复12条边）
        self.edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),  # 前端面
            (4, 5), (5, 7), (7, 6), (6, 4),  # 后端面
            (0, 4), (1, 5), (2, 6), (3, 7)   # 前后连接
        ]
        
        print(f"输出路径: {self.output_root}")
        print(f"有效相机: {list(self.valid_cameras.keys())}")

    def _load_calibration(self) -> Dict:
        if not self.calib_path.exists():
            raise FileNotFoundError(f"标定文件不存在: {self.calib_path}")
        with open(self.calib_path, "r") as f:
            return json.load(f)

    def _quat_to_rot(self, x: float, y: float, z: float, w: float) -> np.ndarray:
        return np.array([
            [1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]
        ])

    def _get_transform_matrix(self, sensor_name: str) -> np.ndarray:
        if sensor_name not in self.calib_data:
            raise KeyError(f"标定数据中无{sensor_name}的参数")
        
        data = self.calib_data[sensor_name]
        R = self._quat_to_rot(data['rx'], data['ry'], data['rz'], data['rw'])
        T = np.array([data['tx'], data['ty'], data['tz']])
        
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = T
        return transform

    def _3dbox_to_vertices(self, box3d: List[float], rotation: float) -> np.ndarray:
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
        return corners @ rot_matrix.T + np.array([x, y, z])

    def _project_single_label(self, label: Dict, cam_name: str) -> Tuple[np.ndarray, np.ndarray, tuple]:
        """返回所有顶点的2D坐标和Z值"""
        box3d = label["box3d"]
        rotation = label["rotation"][0]
        obj_type = label.get('type', 'default')
        color = self.colors.get(obj_type, self.colors['default'])
        
        # 生成3D顶点
        vertices_group = self._3dbox_to_vertices(box3d, rotation)
        group2map = self._get_transform_matrix("group2map")
        cam2map = self._get_transform_matrix(cam_name)
        map2cam = np.linalg.inv(cam2map)
        
        # 转换到相机坐标系
        vertices_group_homo = np.hstack([vertices_group, np.ones((8, 1))])
        vertices_map_homo = (group2map @ vertices_group_homo.T).T
        vertices_cam_homo = (map2cam @ vertices_map_homo.T).T
        vertex_z = vertices_cam_homo[:, 2]  # Z值（相机坐标系）
        
        # 透视投影
        cam_data = self.calib_data[cam_name]
        intrinsic = np.array([[cam_data['fx'], 0, cam_data['cx']],
                             [0, cam_data['fy'], cam_data['cy']],
                             [0, 0, 1]])
        points_2d_homo = (intrinsic @ vertices_cam_homo[:, :3].T).T
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
        
        return points_2d, vertex_z, color

    def _draw_all_labels_on_cam(self, timestamp: str, cam_name: str) -> str:
        print(f"----- 处理相机: {cam_name} -----")
        
        img_folder = self.valid_cameras[cam_name]
        img_path = self.data_root / img_folder / "image" / f"{timestamp}.jpg"
        if not img_path.exists():
            print(f"Error: 图像不存在: {img_path}")
            return ""
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Error: 无法加载图像: {img_path}")
            return ""
        h, w = image.shape[:2]
        vis_image = image.copy()
        
        # 获取相机参数（用于过滤）
        cam_data = self.calib_data[cam_name]
        img_width, img_height = cam_data.get('width', w), cam_data.get('height', h)
        
        # 加载标签
        label_paths = [
            self.data_root / "lidar" / "label" / f"{timestamp}.json",
            self.data_root / "label" / f"{timestamp}.json",
            self.data_root / f"{timestamp}.json"
        ]
        label_path = next((p for p in label_paths if p.exists()), None)
        if not label_path:
            print(f"Error: 标签文件不存在")
            return ""
        
        with open(label_path, "r") as f:
            labels = json.load(f)
        labels = labels if isinstance(labels, list) else labels.get("objects", [])
        print(f"  加载{len(labels)}个3D标签，批量投影中...")
        
        projected_count = 0
        
        for label in labels:
            try:
                points_2d, vertex_z, color = self._project_single_label(label, cam_name)
                box3d = label["box3d"]
                l, w_box = box3d[3], box3d[4]  # 3D框的长和宽（用于判断合理边长）
                max_reasonable_length = max(l, w_box) * 1.5  # 合理边长上限（3D空间）
                
                # 绘制3D框边
                for (i, j) in self.edges:
                    # 顶点坐标和深度
                    pt1 = (points_2d[i, 0], points_2d[i, 1])
                    pt2 = (points_2d[j, 0], points_2d[j, 1])
                    z1, z2 = vertex_z[i], vertex_z[j]

                    # 1. 过滤相机后方的边（至少一端Z>0）
                    if z1 <= 0 and z2 <= 0:
                        continue
                    
                    # 2. 计算2D线段长度（像素）
                    line_length = np.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1])
                    
                    # 3. 过滤异常长线（基于3D框尺寸的合理判断）
                    if line_length > max(img_width, img_height) * 0.5:  # 超过图像一半宽高的线视为异常
                        continue
                    
                    # 4. 判断顶点是否在图像内
                    in1 = (0 <= pt1[0] < img_width) and (0 <= pt1[1] < img_height)
                    in2 = (0 <= pt2[0] < img_width) and (0 <= pt2[1] < img_height)
                    
                    # 5. 分情况绘制
                    if in1 and in2:
                        # 两点都在图像内
                        cv2.line(vis_image, (int(pt1[0]), int(pt1[1])), 
                                 (int(pt2[0]), int(pt2[1])), color, 2)
                    elif in1 or in2:
                        # 一点在内一点在外（绘制到边界）
                        (x_in, y_in), (x_out, y_out) = (pt1, pt2) if in1 else (pt2, pt1)
                        # 计算与边界的交点
                        dx = x_out - x_in
                        dy = y_out - y_in
                        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                            continue  # 零长度线段
                        
                        # 计算与最近边界的交点
                        t_list = []
                        if dx != 0:
                            t_left = (0 - x_in) / dx
                            if 0 < t_left < 1:
                                t_list.append(t_left)
                            t_right = (img_width - 1 - x_in) / dx
                            if 0 < t_right < 1:
                                t_list.append(t_right)
                        if dy != 0:
                            t_top = (0 - y_in) / dy
                            if 0 < t_top < 1:
                                t_list.append(t_top)
                            t_bottom = (img_height - 1 - y_in) / dy
                            if 0 < t_bottom < 1:
                                t_list.append(t_bottom)
                        
                        if t_list:
                            t = min(t_list)  # 取最近的交点
                            x_inter = x_in + t * dx
                            y_inter = y_in + t * dy
                            cv2.line(vis_image, (int(x_in), int(y_in)), 
                                     (int(x_inter), int(y_inter)), color, 2)
                
                projected_count += 1
            except Exception as e:
                print(f"  [Error] 单个标签投影失败: {e}")
        
        # 添加统计信息
        cv2.putText(vis_image, f"Camera: {cam_name}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(vis_image, f"Total: {len(labels)}, Projected: {projected_count}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        # 保存结果
        save_name = f"all_labels_{cam_name}_{timestamp}.jpg"
        save_path = self.output_root / save_name
        cv2.imwrite(str(save_path), vis_image)
        print(f"  保存结果: {save_name}")
        return str(save_path)

    def process_single_timestamp(self, timestamp: str):
        print(f"\n===== 处理时间戳: {timestamp} =====")
        for cam_name in self.valid_cameras.keys():
            self._draw_all_labels_on_cam(timestamp, cam_name)

    def run(self, timestamp: str):
        print(f"开始3D标签批量投影任务: {self.task_abbr}")
        self.process_single_timestamp(timestamp)
        print(f"\n所有结果保存在: {self.output_root}")


if __name__ == "__main__":
    DATA_ROOT = "~/ws/02_docker/cv2-tests/dataset/2d3d4d_20250728_weiyuan/datasets/G51102400001M00_20250728191726_2.0HZ"
    CALIB_PATH = "~/下载/2d3d4d_20250728_weiyuan/calib/sensor2map_calib.json"
    TASK_ABBR = "3d_label_proj"
    TARGET_TIMESTAMP = "1753701446899737835"
    
    try:
        projector = Label3DProjector(
            data_root=DATA_ROOT,
            calib_path=CALIB_PATH,
            task_abbr=TASK_ABBR
        )
        projector.run(TARGET_TIMESTAMP)
    except Exception as e:
        print(f"Error: 运行失败: {e}")