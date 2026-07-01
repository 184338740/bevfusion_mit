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
        """初始化3D标签投影器，优化边界处投影效果"""
        self.data_root = Path(data_root).expanduser()
        self.calib_path = Path(calib_path).expanduser()
        self.task_abbr = task_abbr
        
        # 输出路径（精确到秒）
        self.time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_root = Path(f"output/{self.task_abbr}/{self.time_str}")
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 加载标定数据
        self.calib_data = self._load_calibration()
        
        # 相机映射（与标定文件匹配）
        self.camera_mapping = {
            'SC_R1_Aw_CpcS_CAMR': 'SC_1A_CamR',
            'SC_R1_Bn_CpcW_CAMR': 'SC_1B_CamR',
            'SC_R1_Ce_CpcN_CAMR': 'SC_1C_CamR',
            'SC_R1_Ds_CpcE_CAMR': 'SC_1D_CamR'
        }
        
        # 有效相机（含完整标定）
        self.valid_cameras = {
            cam_name: folder_name 
            for cam_name, folder_name in self.camera_mapping.items()
            if cam_name in self.calib_data
        }
        
        # 目标颜色映射
        self.colors = {
            'car': (0, 255, 0), 'truck': (255, 0, 0), 'cyclist': (0, 0, 255),
            'pedestrian': (255, 255, 0), 'van': (255, 0, 255),
            'tricycle': (0, 255, 255), 'default': (255, 255, 255)
        }
        
        print(f"📌 输出路径: {self.output_root}")
        print(f"✅ 有效相机: {list(self.valid_cameras.keys())}")

    def _load_calibration(self) -> Dict:
        """加载标定文件（内参+外参）"""
        if not self.calib_path.exists():
            raise FileNotFoundError(f"标定文件不存在: {self.calib_path}")
        with open(self.calib_path, "r") as f:
            return json.load(f)

    def _quat_to_rot(self, x: float, y: float, z: float, w: float) -> np.ndarray:
        """四元数转旋转矩阵（x,y,z,w顺序）"""
        return np.array([
            [1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]
        ])

    def _get_transform_matrix(self, sensor_name: str) -> np.ndarray:
        """生成传感器到map的4x4变换矩阵（R+T）"""
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
        """生成3D框8个顶点（group坐标系）"""
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

    def _project_single_label(self, label: Dict, cam_name: str, image_size: tuple) -> Tuple[np.ndarray, np.ndarray, tuple, np.ndarray]:
        """投影单个3D标签，返回：
        - 所有顶点的像素坐标（含无效点，用NaN标记）
        - 箭头坐标
        - 颜色
        - 原始顶点索引（用于边匹配）
        """
        box3d = label["box3d"]
        rotation = label["rotation"][0]
        obj_type = label.get('type', 'default')
        color = self.colors.get(obj_type, self.colors['default'])
        
        # 生成3D顶点（8个）
        vertices_group = self._3dbox_to_vertices(box3d, rotation)
        
        # 坐标转换（group→map→相机）
        group2map = self._get_transform_matrix("group2map")
        cam2map = self._get_transform_matrix(cam_name)
        map2cam = np.linalg.inv(cam2map)
        
        vertices_group_homo = np.hstack([vertices_group, np.ones((8, 1))])
        vertices_map_homo = (group2map @ vertices_group_homo.T).T
        vertices_cam_homo = (map2cam @ vertices_map_homo.T).T
        vertices_cam = vertices_cam_homo[:, :3]
        
        # 透视投影（保留所有8个顶点，无效点用NaN标记）
        cam_data = self.calib_data[cam_name]
        intrinsic = np.array([[cam_data['fx'], 0, cam_data['cx']],
                             [0, cam_data['fy'], cam_data['cy']],
                             [0, 0, 1]])
        points_2d_homo = (intrinsic @ vertices_cam.T).T
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]  # 可能含无效值（Z≤0时）
        
        # 标记无效点（相机后方或超出图像边界）
        h, w = image_size
        valid = (vertices_cam[:, 2] > 0) &  (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
        points_2d[~valid] = np.nan  # 无效点用NaN标记
        
        # 生成航向箭头
        x, y, z, l, w_box, h = box3d
        arrow_3d = np.array([
            [x + l/2 * np.cos(rotation), y + l/2 * np.sin(rotation), z],
            [x + (l/2 + 2.0) * np.cos(rotation), y + (l/2 + 2.0) * np.sin(rotation), z]
        ])
        arrow_cam = self._transform_points_to_camera(arrow_3d, cam_name)
        arrow_2d_homo = (intrinsic @ arrow_cam.T).T
        arrow_2d = arrow_2d_homo[:, :2] / arrow_2d_homo[:, 2:3]
        
        return points_2d.astype(np.float32), arrow_2d.astype(np.float32), color, np.arange(8)  # 保留原始索引

    def _transform_points_to_camera(self, points: np.ndarray, camera_name: str) -> np.ndarray:
        """点坐标转换到相机坐标系"""
        try:
            group2map = self._get_transform_matrix("group2map")
            cam2map = self._get_transform_matrix(camera_name)
            map2cam = np.linalg.inv(cam2map)
        except KeyError:
            return np.array([])
        
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        points_map = (group2map @ points_homo.T).T
        points_cam = (map2cam @ points_map.T).T
        return points_cam[:, :3]

    def _draw_all_labels_on_cam(self, timestamp: str, cam_name: str) -> str:
        """批量绘制所有标签，优化边界框线绘制"""
        print(f"----- 处理相机: {cam_name} -----")
        
        # 加载图像
        img_folder = self.valid_cameras[cam_name]
        img_path = self.data_root / img_folder / "image_dc" / f"{timestamp}.jpg"
        if not img_path.exists():
            print(f"❌ 图像不存在: {img_path}")
            return ""
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"❌ 无法加载图像: {img_path}")
            return ""
        h, w = image_size = image.shape[:2]
        vis_image = image.copy()
        
        # 加载标签
        label_paths = [
            self.data_root / "lidar" / "label" / f"{timestamp}.json",
            self.data_root / "label" / f"{timestamp}.json",
            self.data_root / f"{timestamp}.json"
        ]
        label_path = next((p for p in label_paths if p.exists()), None)
        if not label_path:
            print(f"❌ 标签文件不存在")
            return ""
        
        with open(label_path, "r") as f:
            labels = json.load(f)
        labels = labels if isinstance(labels, list) else labels.get("objects", [])
        print(f"  加载{len(labels)}个3D标签，批量投影中...")
        
        # 3D框边索引（8个顶点的连接关系）
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 3),  # 前端面
            (4, 5), (4, 6), (5, 7), (6, 7),  # 后端面
            (0, 4), (1, 5), (2, 6), (3, 7)   # 连接前后
        ]
        projected_count = 0
        
        for label in labels:
            try:
                # 获取所有顶点（含无效点，用NaN标记）
                points_2d, arrow_2d, color, indices = self._project_single_label(label, cam_name, image_size)
                if points_2d is None:
                    continue
                
                # 绘制3D框边（处理边界点逻辑）
                for i, j in edges:
                    # 跳过两端都无效的边
                    if np.isnan(points_2d[i, 0]) and np.isnan(points_2d[j, 0]):
                        continue
                    
                    # 一端有效、一端无效：计算与图像边界的交点并绘制到边界
                    if np.isnan(points_2d[i, 0]):
                        # i无效，j有效：从j向i方向延长到边界
                        pt_valid = (int(points_2d[j, 0]), int(points_2d[j, 1]))
                        pt_invalid = (points_2d[i, 0], points_2d[i, 1])
                        pt_boundary = self._get_boundary_intersection(pt_valid, pt_invalid, w, h)
                        if pt_boundary:
                            cv2.line(vis_image, pt_valid, pt_boundary, color, 2)
                    elif np.isnan(points_2d[j, 0]):
                        # j无效，i有效：从i向j方向延长到边界
                        pt_valid = (int(points_2d[i, 0]), int(points_2d[i, 1]))
                        pt_invalid = (points_2d[j, 0], points_2d[j, 1])
                        pt_boundary = self._get_boundary_intersection(pt_valid, pt_invalid, w, h)
                        if pt_boundary:
                            cv2.line(vis_image, pt_valid, pt_boundary, color, 2)
                    else:
                        # 两端都有效：直接绘制
                        pt1 = (int(points_2d[i, 0]), int(points_2d[i, 1]))
                        pt2 = (int(points_2d[j, 0]), int(points_2d[j, 1]))
                        cv2.line(vis_image, pt1, pt2, color, 2)
                
                # 绘制航向箭头（过滤无效箭头）
                if len(arrow_2d) == 2 and not np.isnan(arrow_2d).any():
                    start_pt = (int(arrow_2d[0, 0]), int(arrow_2d[0, 1]))
                    end_pt = (int(arrow_2d[1, 0]), int(arrow_2d[1, 1]))
                    if 0 <= start_pt[0] < w and 0 <= start_pt[1] < h and \
                       0 <= end_pt[0] < w and 0 <= end_pt[1] < h:
                        cv2.arrowedLine(vis_image, start_pt, end_pt, (255, 255, 0), 3, tipLength=0.3)
                
                projected_count += 1
            except Exception as e:
                print(f"  ⚠️ 单个标签投影失败: {e}")
        
        # 添加统计信息
        cv2.putText(vis_image, f"Camera: {cam_name}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Total: {len(labels)}, Projected: {projected_count}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 保存结果
        save_name = f"all_labels_{cam_name}_{timestamp}.jpg"
        save_path = self.output_root / save_name
        cv2.imwrite(str(save_path), vis_image)
        print(f"  ✅ 保存结果: {save_name}")
        return str(save_path)

    def _get_boundary_intersection(self, pt_valid: Tuple[int, int], pt_invalid: Tuple[float, float], w: int, h: int) -> Tuple[int, int]:
        """计算有效点到无效点的连线与图像边界的交点（解决边界框线断裂）"""
        x1, y1 = pt_valid
        x2, y2 = pt_invalid
        
        # 线段参数方程：x = x1 + t*(x2-x1), y = y1 + t*(y2-y1), t∈[0,1]
        # 求与图像边界（x=0, x=w-1, y=0, y=h-1）的交点
        t_list = []
        
        # 与x=0的交点
        if x2 != x1:
            t = (0 - x1) / (x2 - x1)
            if 0 <= t <= 1:
                y = y1 + t*(y2 - y1)
                if 0 <= y < h:
                    t_list.append((t, (0, int(y))))
        
        # 与x=w-1的交点
        if x2 != x1:
            t = (w-1 - x1) / (x2 - x1)
            if 0 <= t <= 1:
                y = y1 + t*(y2 - y1)
                if 0 <= y < h:
                    t_list.append((t, (w-1, int(y))))
        
        # 与y=0的交点
        if y2 != y1:
            t = (0 - y1) / (y2 - y1)
            if 0 <= t <= 1:
                x = x1 + t*(x2 - x1)
                if 0 <= x < w:
                    t_list.append((t, (int(x), 0)))
        
        # 与y=h-1的交点
        if y2 != y1:
            t = (h-1 - y1) / (y2 - y1)
            if 0 <= t <= 1:
                x = x1 + t*(x2 - x1)
                if 0 <= x < w:
                    t_list.append((t, (int(x), h-1)))
        
        # 取t最小的交点（最近的边界）
        if t_list:
            t_list.sort()
            return t_list[0][1]
        return None

    def process_single_timestamp(self, timestamp: str):
        """处理单个时间戳"""
        print(f"\n===== 处理时间戳: {timestamp} =====")
        for cam_name in self.valid_cameras.keys():
            self._draw_all_labels_on_cam(timestamp, cam_name)

    def run(self, timestamp: str):
        """运行主流程"""
        print(f"🚀 开始3D标签批量投影任务: {self.task_abbr}")
        self.process_single_timestamp(timestamp)
        print(f"\n🎉 所有结果保存在: {self.output_root}")


if __name__ == "__main__":
    # 配置参数
    DATA_ROOT = "~/下载/2d3d4d_20250728_weiyuan/datasets/G51102400001M00_20250728191726_2.0HZ"
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
        print(f"❌ 运行失败: {e}")