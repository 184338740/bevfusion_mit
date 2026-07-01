import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

class SpatialVisualizer:
    def __init__(self, config: Dict):
        self.config = config
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config['output_root']) / config['task_name'] / timestamp
        self.pointcloud_dir = self.output_dir / "pointcloud_projection"
        self.label_dir = self.output_dir / "3d_label_projection"
        self.pcd_3dbox_dir = self.output_dir / "pcd_3dbox_visualization"
        self.fov_dir = self.output_dir / "fov_visualization"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pointcloud_dir.mkdir(parents=True, exist_ok=True)
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.pcd_3dbox_dir.mkdir(parents=True, exist_ok=True)
        self.fov_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据路径
        self.data_root = Path(config['data_root'])
        self.dataset_path = self.data_root / "datasets" / "G51102400001M00_20250728191726_2.0HZ"
        
        # 标定文件
        self.calib_path = self.data_root / "calib" / config['calib_file']
        self.calib_data = self.load_calibration()
        
        # 相机配置
        self.camera_config = config['camera_config']
        
        print(f"初始化完成 - 输出目录: {self.output_dir}")
    
    def load_calibration(self) -> Dict:
        print(f"加载标定文件: {self.calib_path}")
        
        if not self.calib_path.exists():
            raise FileNotFoundError(f"标定文件不存在: {self.calib_path}")
        
        with open(self.calib_path, 'r') as f:
            data = json.load(f)
            print(f"标定文件中的相机: {[k for k in data.keys() if k.startswith('SC_R1_')]}")
            return data
    
    def get_camera_params(self, camera_name: str) -> Optional[Dict]:
        """获取相机参数，优先使用_new后缀的去畸变后参数"""
        # 优先尝试_new后缀名称
        new_name = f"{camera_name}_new"
        if new_name in self.calib_data:
            print(f"使用_new后缀标定参数: {new_name}")
            return self.calib_data[new_name]
        
        # 回退到原始名称
        if camera_name in self.calib_data:
            return self.calib_data[camera_name]
        
        print(f"警告: 未找到相机 {camera_name} 的标定参数")
        return None
    
    def quaternion_to_matrix(self, x: float, y: float, z: float, w: float) -> np.ndarray:
        """四元数转旋转矩阵"""
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
        ])
    
    def parse_transform(self, data: Dict) -> np.ndarray:
        """解析变换矩阵"""
        translation = np.array([data['tx'], data['ty'], data['tz']])
        rotation = self.quaternion_to_matrix(data['rx'], data['ry'], data['rz'], data['rw'])
        
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform
    
    def transform_points_to_camera(self, points: np.ndarray, camera_name: str) -> np.ndarray:
        """将点从group坐标系变换到相机坐标系"""
        cam_params = self.get_camera_params(camera_name)
        if cam_params is None:
            return points
        
        # 获取变换矩阵
        camera_transform = self.parse_transform(cam_params)
        
        # 如果有group2map，应用变换
        if 'group2map' in self.calib_data:
            group_transform = self.parse_transform(self.calib_data['group2map'])
        else:
            group_transform = np.eye(4)
        
        # 转换为齐次坐标
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        
        # Group -> Map -> Camera
        points_map = (group_transform @ points_homo.T).T
        camera_inv = np.linalg.inv(camera_transform)
        points_camera = (camera_inv @ points_map.T).T
        
        return points_camera[:, :3]
    
    def load_image(self, camera_folder: str, timestamp: str) -> Optional[np.ndarray]:
        img_path = self.dataset_path / camera_folder / self.config['image_folder'] / f"{timestamp}.jpg"
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
                for label in labels[:3]:  # 打印前3个标签信息
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
    
    def create_3d_box_corners(self, box3d: List[float], rotation: float) -> np.ndarray:
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
        
        # 旋转并平移
        corners_rotated = corners @ rot_matrix.T
        corners_world = corners_rotated + np.array([x, y, z])
        
        return corners_world
    
    def project_points_to_image(self, points: np.ndarray, camera_name: str, image_size: tuple) -> Tuple[np.ndarray, np.ndarray]:
        """完整的点云投影流程"""
        cam_params = self.get_camera_params(camera_name)
        if cam_params is None:
            return np.array([]), np.array([])
        
        print(f"  {camera_name}: 投影 {len(points)} 个点")
        
        # 1. 坐标变换到相机坐标系
        points_camera = self.transform_points_to_camera(points, camera_name)
        
        # 2. 过滤相机后面的点
        front_mask = points_camera[:, 2] > 0.1
        if not np.any(front_mask):
            print(f"  {camera_name}: 无前方点 (所有点都在相机后面)")
            return np.array([]), np.array([])
        
        points_camera = points_camera[front_mask]
        depths = points_camera[:, 2]
        
        print(f"  {camera_name}: 前方点 {len(points_camera)}, 深度范围 [{depths.min():.1f}, {depths.max():.1f}]")
        
        # 3. 相机内参投影
        intrinsic = np.array([
            [cam_params['fx'], 0, cam_params['cx']],
            [0, cam_params['fy'], cam_params['cy']],
            [0, 0, 1]
        ])
        
        # 投影到图像平面
        points_2d_homo = (intrinsic @ points_camera.T).T
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
        
        # 4. 过滤图像范围外的点
        h, w = image_size
        valid_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
                    (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
        
        valid_points = points_2d[valid_mask]
        valid_depths = depths[valid_mask]
        
        print(f"  {camera_name}: 图像内点 {len(valid_points)}")
        
        if len(valid_points) > 0:
            print(f"  {camera_name}: 投影点坐标范围 x[{valid_points[:,0].min():.1f}, {valid_points[:,0].max():.1f}], "
                  f"y[{valid_points[:,1].min():.1f}, {valid_points[:,1].max():.1f}]")
        
        return valid_points, valid_depths
    
    def draw_3d_boxes(self, image: np.ndarray, boxes_2d: List[np.ndarray], colors: List[tuple]) -> np.ndarray:
        vis_image = image.copy()
        edges = [(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(0,4),(1,5),(2,6),(3,7)]
        
        for box_idx, (box_corners, color) in enumerate(zip(boxes_2d, colors)):
            if len(box_corners) != 8:
                continue
                
            for i, j in edges:
                pt1 = tuple(box_corners[i].astype(int))
                pt2 = tuple(box_corners[j].astype(int))
                if (0 <= pt1[0] < vis_image.shape[1] and 0 <= pt1[1] < vis_image.shape[0] and
                    0 <= pt2[0] < vis_image.shape[1] and 0 <= pt2[1] < vis_image.shape[0]):
                    cv2.line(vis_image, pt1, pt2, color, 2)
            
            # 在第一个角点添加标签编号
            center = box_corners[0].astype(int)
            if 0 <= center[0] < vis_image.shape[1] and 0 <= center[1] < vis_image.shape[0]:
                cv2.putText(vis_image, str(box_idx), tuple(center), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_image
    
    def create_label_projection(self, timestamp: str):
        print(f"\n创建3D标签投影: {timestamp}")
        
        labels = self.load_labels(timestamp)
        if not labels:
            print("无标签数据，跳过")
            return
        
        # 收集相机图像
        camera_images = {}
        for calib_name, folder_name in self.camera_config.items():
            img = self.load_image(folder_name, timestamp)
            if img is not None:
                camera_images[calib_name] = img
        
        if not camera_images:
            print("无相机图像，跳过")
            return
        
        # 确定统一尺寸
        base_size = next(iter(camera_images.values())).shape[1], next(iter(camera_images.values())).shape[0]
        print(f"图像尺寸: {base_size}")
        
        # 创建2x2布局
        combined = np.zeros((base_size[1]*2, base_size[0]*2, 3), dtype=np.uint8)
        positions = {
            'SC_R1_Aw_CpcS_CAMR': (0, 0),
            'SC_R1_Bn_CpcW_CAMR': (base_size[0], 0), 
            'SC_R1_Ce_CpcN_CAMR': (0, base_size[1]),
            'SC_R1_Ds_CpcE_CAMR': (base_size[0], base_size[1])
        }
        
        label_colors = {
            'car': (0,255,0), 
            'truck': (255,0,0), 
            'cyclist': (0,0,255),
            'pedestrian': (255,255,0),
            'default': (255,255,255)
        }
        
        total_projected = 0
        
        for cam_name, (x, y) in positions.items():
            if cam_name in camera_images:
                img = cv2.resize(camera_images[cam_name], base_size)
                boxes_2d = []
                colors = []
                
                for label_idx, label in enumerate(labels):
                    if 'box3d' not in label or len(label['box3d']) != 6:
                        continue
                    
                    # 创建3D边界框
                    rotation = label.get('rotation', [0])[0]
                    corners_3d = self.create_3d_box_corners(label['box3d'], rotation)
                    
                    # 投影到图像
                    corners_2d, _ = self.project_points_to_image(corners_3d, cam_name, (base_size[1], base_size[0]))
                    
                    if len(corners_2d) == 8:
                        boxes_2d.append(corners_2d)
                        colors.append(label_colors.get(label.get('type', 'default'), label_colors['default']))
                        total_projected += 1
                        print(f"  {cam_name}: 标签 {label_idx} 投影成功")
                    else:
                        print(f"  {cam_name}: 标签 {label_idx} 投影失败 - 得到 {len(corners_2d)} 个点")
                
                # 绘制边界框
                img_with_boxes = self.draw_3d_boxes(img, boxes_2d, colors)
                combined[y:y+base_size[1], x:x+base_size[0]] = img_with_boxes
                
                # 添加文字
                text = f"{self.camera_config[cam_name]} - {len(boxes_2d)} objs"
                cv2.putText(combined, text, (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(f"  {cam_name}: 绘制 {len(boxes_2d)} 个边界框")
            else:
                # 如果没有图像，显示黑色背景
                combined[y:y+base_size[1], x:x+base_size[0]] = 0
                print(f"  {cam_name}: 无图像")
        
        output_path = self.label_dir / f"3d_label_projection_{timestamp}.jpg"
        cv2.imwrite(str(output_path), combined)
        print(f"保存3D标签投影: {output_path}")
        print(f"总计投影边界框: {total_projected}/{len(labels)}")
        
        # 保存配置文件
        config_path = self.label_dir / f"label_projection_config_{timestamp}.txt"
        with open(config_path, 'w') as f:
            f.write("3D Label Projection Configuration:\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total labels: {len(labels)}\n")
            f.write(f"Total projected boxes: {total_projected}\n")
            f.write(f"Image size: {base_size}\n")
            f.write("\nCamera mapping:\n")
            for calib_name, folder_name in self.camera_config.items():
                f.write(f"  {calib_name} -> {folder_name}\n")
            f.write("\nLabel colors:\n")
            for label_type, color in label_colors.items():
                f.write(f"  {label_type}: {color}\n")
        
        print(f"保存配置文件: {config_path}")

    def create_pointcloud_projection(self, timestamp: str):
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib不可用，跳过点云投影")
            return
            
        print(f"\n创建点云投影: {timestamp}")
        
        points = self.load_pointcloud(timestamp)
        if points is None:
            print("无点云数据，跳过")
            return
        
        # 收集相机图像
        camera_images = {}
        for calib_name, folder_name in self.camera_config.items():
            img = self.load_image(folder_name, timestamp)
            if img is not None:
                camera_images[calib_name] = img
        
        if not camera_images:
            print("无相机图像，跳过")
            return
        
        # 确定统一尺寸
        base_size = next(iter(camera_images.values())).shape[1], next(iter(camera_images.values())).shape[0]
        
        # 创建2x4布局
        fig, axes = plt.subplots(4, 2, figsize=(20, 32))
        camera_layout = {
            'SC_R1_Aw_CpcS_CAMR': 0,
            'SC_R1_Bn_CpcW_CAMR': 1, 
            'SC_R1_Ce_CpcN_CAMR': 2,
            'SC_R1_Ds_CpcE_CAMR': 3
        }
        
        total_points_projected = 0
        
        for cam_name, row_idx in camera_layout.items():
            folder_name = self.camera_config[cam_name]
            
            if cam_name in camera_images:
                img = cv2.resize(camera_images[cam_name], base_size)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 原图
                axes[row_idx, 0].imshow(img_rgb)
                axes[row_idx, 0].set_title(f'{folder_name} - Original')
                axes[row_idx, 0].axis('off')
                
                # 点云投影
                axes[row_idx, 1].imshow(img_rgb)
                points_2d, depths = self.project_points_to_image(points, cam_name, (base_size[1], base_size[0]))
                
                if len(points_2d) > 0:
                    # 采样
                    max_points = self.config['point_cloud_config']['max_points']
                    if max_points and len(points_2d) > max_points:
                        indices = np.random.choice(len(points_2d), max_points, replace=False)
                        points_2d, depths = points_2d[indices], depths[indices]
                    
                    # 使用hot色彩映射，确保深度值正确归一化
                    norm_depths = (depths - depths.min()) / (depths.max() - depths.min() + 1e-8)
                    
                    scatter = axes[row_idx, 1].scatter(
                        points_2d[:, 0], points_2d[:, 1], 
                        c=norm_depths, cmap='hot', vmin=0, vmax=1,
                        s=self.config['point_cloud_config']['point_size'],
                        alpha=self.config['point_cloud_config']['alpha']
                    )
                    
                    if row_idx == 0:
                        plt.colorbar(scatter, ax=axes[row_idx, 1], label='Depth (normalized)')
                    
                    total_points_projected += len(points_2d)
                    points_info = f" - {len(points_2d)} points, depth[{depths.min():.1f}-{depths.max():.1f}]m"
                else:
                    points_info = " - 0 points"
                
                axes[row_idx, 1].set_title(f'{folder_name} - Point Cloud{points_info}')
                axes[row_idx, 1].axis('off')
            else:
                # 黑色背景
                black_bg = np.zeros((base_size[1], base_size[0], 3))
                axes[row_idx, 0].imshow(black_bg)
                axes[row_idx, 0].set_title(f'{folder_name} - No Image')
                axes[row_idx, 0].axis('off')
                axes[row_idx, 1].imshow(black_bg)
                axes[row_idx, 1].set_title(f'{folder_name} - No Image')
                axes[row_idx, 1].axis('off')
        
        plt.tight_layout()
        output_path = self.pointcloud_dir / f"point_cloud_projection_{timestamp}.jpg"
        plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        
        # 保存配置
        config_path = self.pointcloud_dir / f"point_cloud_config_{timestamp}.txt"
        with open(config_path, 'w') as f:
            f.write("Point Cloud Configuration:\n")
            for key, value in self.config['point_cloud_config'].items():
                f.write(f"{key}: {value}\n")
            f.write(f"\nTotal points projected: {total_points_projected}\n")
            f.write(f"Color map: hot\n")
        
        print(f"保存点云投影: {output_path}")
        print(f"保存配置: {config_path}")
        print(f"总投影点数: {total_points_projected}")

    def create_pcd_3dbox_visualization(self, timestamp: str):
        """3.2 功能：在点云上绘制3D边界框"""
        print(f"\n创建点云3D边界框可视化: {timestamp}")
        
        points = self.load_pointcloud(timestamp)
        labels = self.load_labels(timestamp)
        
        if points is None or len(points) == 0:
            print("无点云数据，跳过")
            # 创建空的提示文件
            output_path = self.pcd_3dbox_dir / f"pcd_3dbox_{timestamp}.jpg"
            # 创建空白图像
            blank_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
            cv2.putText(blank_img, "No Point Cloud Data", (150, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imwrite(str(output_path), blank_img)
            return
        
        if not labels:
            print("无标签数据，跳过")
            # 创建空白图像
            output_path = self.pcd_3dbox_dir / f"pcd_3dbox_{timestamp}.jpg"
            blank_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
            cv2.putText(blank_img, "No Label Data", (200, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imwrite(str(output_path), blank_img)
            return
        
        # 使用matplotlib绘制3D边界框
        self.create_pcd_3dbox_matplotlib(points, labels, timestamp)

    def create_pcd_3dbox_matplotlib(self, points: np.ndarray, labels: List[Dict], timestamp: str):
        """使用matplotlib绘制3D边界框（BEV视角）"""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib也不可用，跳过3D边界框可视化")
            return
            
        print("使用matplotlib绘制3D边界框(BEV视角)")
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # 设置BEV视角范围
        bev_range = 75.0
        ax.set_xlim(-bev_range, bev_range)
        ax.set_ylim(-bev_range, bev_range)
        
        # 绘制坐标轴和网格
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        grid_step = 10
        for i in range(-int(bev_range/grid_step), int(bev_range/grid_step)+1):
            ax.axhline(y=i*grid_step, color='gray', linestyle=':', alpha=0.2)
            ax.axvline(x=i*grid_step, color='gray', linestyle=':', alpha=0.2)
        
        # 绘制点云（BEV视图，只显示在范围内的点）
        max_points = 10000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points_display = points[indices]
        else:
            points_display = points
        
        # 过滤在范围内的点
        in_range_mask = (points_display[:, 0] >= -bev_range) & (points_display[:, 0] <= bev_range) & \
                       (points_display[:, 1] >= -bev_range) & (points_display[:, 1] <= bev_range)
        points_in_range = points_display[in_range_mask]
        
        if len(points_in_range) > 0:
            # 点云颜色基于高度
            colors = points_in_range[:, 2]
            scatter = ax.scatter(points_in_range[:, 0], points_in_range[:, 1], 
                               c=colors, cmap='viridis', s=1, alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='Height (m)')
        
        # 绘制边界框
        label_colors = {
            'car': 'red',
            'truck': 'blue', 
            'cyclist': 'green',
            'pedestrian': 'orange',
            'default': 'black'
        }
        
        boxes_drawn = 0
        
        for label_idx, label in enumerate(labels):
            if 'box3d' not in label or len(label['box3d']) != 6:
                continue
                
            box3d = label['box3d']
            rotation = label.get('rotation', [0])[0]
            label_type = label.get('type', 'default')
            color = label_colors.get(label_type, label_colors['default'])
            
            # 检查边界框是否在显示范围内
            x, y, z, l, w, h = box3d
            if not (-bev_range <= x <= bev_range and -bev_range <= y <= bev_range):
                continue
            
            # 创建边界框角点 (BEV视图，只关注底面)
            corners_local = np.array([
                [l/2, w/2, 0],   # 前右
                [l/2, -w/2, 0],  # 前左
                [-l/2, -w/2, 0], # 后左
                [-l/2, w/2, 0]   # 后右
            ])
            
            # 应用旋转 (绕Z轴)
            rot_matrix = np.array([
                [np.cos(rotation), -np.sin(rotation), 0],
                [np.sin(rotation), np.cos(rotation), 0],
                [0, 0, 1]
            ])
            
            corners_rotated = corners_local @ rot_matrix.T
            corners_world = corners_rotated + np.array([x, y, 0])
            
            # 绘制边界框 (线条，不填充)
            corners_2d = corners_world[:, :2]
            
            # 连接角点形成矩形
            for i in range(4):
                start_point = corners_2d[i]
                end_point = corners_2d[(i+1) % 4]
                ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                       color=color, linewidth=2, alpha=0.8)
            
            # 绘制朝向箭头
            front_center = (corners_2d[0] + corners_2d[1]) / 2
            center = np.mean(corners_2d, axis=0)
            arrow_vec = front_center - center
            arrow_length = min(l, w) * 0.6
            arrow_end = center + arrow_vec / np.linalg.norm(arrow_vec) * arrow_length
            
            ax.arrow(center[0], center[1], 
                    arrow_end[0]-center[0], arrow_end[1]-center[1],
                    head_width=0.5, head_length=0.8, fc=color, ec=color, alpha=0.8)
            
            # 修改标签显示：更小的文字，无背景框
            ax.text(x, y + 1, f'{label_type}', fontsize=6, color=color, 
                   ha='center', va='bottom', weight='bold')
            
            boxes_drawn += 1
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'BEV 3D Bounding Box Visualization - {timestamp}\nRange: ±{bev_range}m')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 创建图例
        legend_elements = []
        for label_type, color in label_colors.items():
            if any(label.get('type', 'default') == label_type for label in labels):
                legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=label_type))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        output_path = self.pcd_3dbox_dir / f"pcd_3dbox_{timestamp}.jpg"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存配置信息
        config_path = self.pcd_3dbox_dir / f"pcd_3dbox_config_{timestamp}.txt"
        with open(config_path, 'w') as f:
            f.write("BEV 3D Box Visualization Configuration:\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"BEV Range: ±{bev_range}m\n")
            f.write(f"Total points: {len(points)}\n")
            f.write(f"Points in range: {len(points_in_range) if 'points_in_range' in locals() else 0}\n")
            f.write(f"Total boxes: {boxes_drawn}\n")
            f.write("\nLabel colors:\n")
            for label_type, color in label_colors.items():
                f.write(f"  {label_type}: {color}\n")
        
        print(f"使用matplotlib保存点云3D边界框可视化: {output_path}")
        print(f"绘制边界框: {boxes_drawn} 个")

    def create_fov_visualization(self, timestamp: str):
        """3.4 功能：相机FOV边界在BEV平面上的可视化"""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib不可用，跳过FOV可视化")
            return
            
        print(f"\n创建FOV边界可视化: {timestamp}")
        
        # 使用matplotlib绘制BEV视图
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # 设置BEV视角范围（画布大小）
        bev_range = 75.0  # 从中心到边缘的距离
        ax.set_xlim(-bev_range, bev_range)
        ax.set_ylim(-bev_range, bev_range)
        
        # 绘制坐标轴
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # 绘制网格
        grid_step = 10
        for i in range(-int(bev_range/grid_step), int(bev_range/grid_step)+1):
            ax.axhline(y=i*grid_step, color='gray', linestyle=':', alpha=0.2)
            ax.axvline(x=i*grid_step, color='gray', linestyle=':', alpha=0.2)
        
        # 绘制group点位置 (原点)
        group_pos = [0, 0]  # group坐标系原点
        ax.scatter(group_pos[0], group_pos[1], color='black', s=200, marker='o', 
                  label='Group Origin', zorder=10, edgecolors='white', linewidth=2)
        ax.text(group_pos[0]+1, group_pos[1]+1, 'Group', fontsize=12, weight='bold', color='black')
        
        # 相机颜色和配置
        camera_configs = {
            'SC_R1_Aw_CpcS_CAMR': {'color': 'red', 'label': 'Cam A'},
            'SC_R1_Bn_CpcW_CAMR': {'color': 'green', 'label': 'Cam B'},  
            'SC_R1_Ce_CpcN_CAMR': {'color': 'blue', 'label': 'Cam C'},
            'SC_R1_Ds_CpcE_CAMR': {'color': 'orange', 'label': 'Cam D'}
        }
        
        fov_info = []
        cameras_found = 0
        
        # 获取group2map变换矩阵
        if 'group2map' in self.calib_data:
            group2map_transform = self.parse_transform(self.calib_data['group2map'])
            map2group_transform = np.linalg.inv(group2map_transform)
            print(f"使用group2map变换矩阵")
        else:
            map2group_transform = np.eye(4)
            print(f"未找到group2map，使用单位矩阵")
        
        for cam_name, config in camera_configs.items():
            color = config['color']
            label = config['label']
            
            cam_params = self.get_camera_params(cam_name)
            if cam_params is None:
                print(f"  {cam_name}: 未找到标定参数，跳过")
                continue
            
            # 获取相机在map坐标系中的变换矩阵
            camera2map_transform = self.parse_transform(cam_params)
            
            # 计算相机在group坐标系中的位置: camera_group = map2group * camera_map
            camera_group_transform = map2group_transform @ camera2map_transform
            camera_pos_group = camera_group_transform[:3, 3]
            
            # 只取XY平面位置
            camera_pos_2d = [camera_pos_group[0], camera_pos_group[1]]
            
            # 获取相机在group坐标系中的旋转矩阵
            camera_rot_group = camera_group_transform[:3, :3]
            
            # 计算相机朝向 (假设相机前向是Z轴负方向或X轴，需要根据实际标定调整)
            # 尝试不同的前向向量
            forward_candidates = [
                [0, 0, 1],   # Z轴正方向
                [0, 0, -1],  # Z轴负方向  
                [1, 0, 0],   # X轴正方向
                [-1, 0, 0],  # X轴负方向
            ]
            
            # 选择最合理的前向向量（朝向外的方向）
            best_forward = None
            max_dot_product = -1
            
            for forward in forward_candidates:
                forward_world = camera_rot_group @ forward
                forward_2d = forward_world[:2]
                forward_2d_norm = forward_2d / (np.linalg.norm(forward_2d) + 1e-8)
                
                # 计算从group指向相机的向量
                camera_to_group = -np.array(camera_pos_2d)
                camera_to_group_norm = camera_to_group / (np.linalg.norm(camera_to_group) + 1e-8)
                
                # 如果前向向量与从相机指向group的向量点积为负，说明相机朝向group外部
                dot_product = np.dot(forward_2d_norm, camera_to_group_norm)
                if dot_product > max_dot_product:
                    max_dot_product = dot_product
                    best_forward = forward_world
            
            if best_forward is not None:
                forward_2d = best_forward[:2]
                camera_yaw = np.arctan2(forward_2d[1], forward_2d[0])
            else:
                # 默认朝向：从group指向相机的反方向（朝外）
                camera_to_group = np.array(camera_pos_2d)
                if np.linalg.norm(camera_to_group) > 0.1:
                    camera_yaw = np.arctan2(camera_to_group[1], camera_to_group[0]) + np.pi
                else:
                    camera_yaw = 0
            
            print(f"  {cam_name}: group坐标系位置 ({camera_pos_2d[0]:.2f}, {camera_pos_2d[1]:.2f}), "
                  f"朝向 {np.degrees(camera_yaw):.1f}°")
            
            # 创建FOV视锥 (BEV平面)
            fov_h = 60  # 水平FOV，度
            max_fov_range = 150.0  # 最大FOV可视范围扩展到150m
            
            # 计算FOV边界角度
            left_angle = camera_yaw - np.radians(fov_h/2)
            right_angle = camera_yaw + np.radians(fov_h/2)
            
            # 绘制多个距离的扇形弧线 (每30米一个，最大到150m)
            distance_intervals = [30, 60, 90, 120, 150]  # 距离间隔扩展到150m
            line_styles = ['-', '--', ':', '-.', '-']  # 不同距离的线型
            line_widths = [1.0, 0.8, 0.6, 0.5, 0.4]   # 不同距离的线宽
            alphas = [0.6, 0.5, 0.4, 0.3, 0.2]        # 不同距离的透明度
            
            for i, distance in enumerate(distance_intervals):
                # 绘制扇形弧线（即使超出画布范围也绘制，会被自动裁剪）
                theta = np.linspace(left_angle, right_angle, 30)
                fov_arc = [
                    camera_pos_2d[0] + distance * np.cos(theta),
                    camera_pos_2d[1] + distance * np.sin(theta)
                ]
                ax.plot(fov_arc[0], fov_arc[1], color=color, linewidth=line_widths[i], 
                       alpha=alphas[i], linestyle=line_styles[i])
                
                # 在弧线上标记距离值 (选择弧线中点位置)
                mid_angle = (left_angle + right_angle) / 2
                text_pos = [
                    camera_pos_2d[0] + distance * np.cos(mid_angle),
                    camera_pos_2d[1] + distance * np.sin(mid_angle)
                ]
                
                # 只显示在画布范围内的距离标记
                if (-bev_range <= text_pos[0] <= bev_range and 
                    -bev_range <= text_pos[1] <= bev_range):
                    
                    # 调整文本位置，避免重叠
                    text_offset = 3
                    text_angle_deg = np.degrees(mid_angle)
                    if text_angle_deg > 90 and text_angle_deg < 270:
                        text_pos[0] -= text_offset
                    else:
                        text_pos[0] += text_offset
                    
                    if text_angle_deg > 180:
                        text_pos[1] -= text_offset
                    else:
                        text_pos[1] += text_offset
                    
                    ax.text(text_pos[0], text_pos[1], f'{distance}m', fontsize=6, color=color,
                           ha='center', va='center', weight='bold',
                           bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.9, edgecolor=color))
            
            # 绘制FOV边界射线 - 使用更柔和的样式
            left_end = [
                camera_pos_2d[0] + max_fov_range * np.cos(left_angle),
                camera_pos_2d[1] + max_fov_range * np.sin(left_angle)
            ]
            right_end = [
                camera_pos_2d[0] + max_fov_range * np.cos(right_angle),
                camera_pos_2d[1] + max_fov_range * np.sin(right_angle)
            ]
            
            # 绘制边界射线 - 使用更细的线条和更低的透明度
            ax.plot([camera_pos_2d[0], left_end[0]], [camera_pos_2d[1], left_end[1]], 
                   color=color, linewidth=1.2, alpha=0.3, linestyle='-')
            ax.plot([camera_pos_2d[0], right_end[0]], [camera_pos_2d[1], right_end[1]], 
                   color=color, linewidth=1.2, alpha=0.3, linestyle='-')
            
            # 填充FOV区域 (半透明，只填充到画布边界)
            fill_range = min(max_fov_range, bev_range * 1.5)  # 填充范围适当扩展
            theta = np.linspace(left_angle, right_angle, 30)
            fov_arc_fill = [
                camera_pos_2d[0] + fill_range * np.cos(theta),
                camera_pos_2d[1] + fill_range * np.sin(theta)
            ]
            fov_fill_x = [camera_pos_2d[0]] + list(fov_arc_fill[0]) + [camera_pos_2d[0]]
            fov_fill_y = [camera_pos_2d[1]] + list(fov_arc_fill[1]) + [camera_pos_2d[1]]
            ax.fill(fov_fill_x, fov_fill_y, color=color, alpha=0.04)  # 进一步降低填充透明度
            
            # 绘制相机位置
            ax.scatter(camera_pos_2d[0], camera_pos_2d[1], color=color, s=120, marker='s', 
                      label=label, zorder=10, edgecolors='white', linewidth=2)
            
            # 绘制相机朝向箭头
            arrow_length = 2
            arrow_end = [
                camera_pos_2d[0] + arrow_length * np.cos(camera_yaw),
                camera_pos_2d[1] + arrow_length * np.sin(camera_yaw)
            ]
            ax.arrow(camera_pos_2d[0], camera_pos_2d[1], 
                    arrow_end[0]-camera_pos_2d[0], arrow_end[1]-camera_pos_2d[1],
                    head_width=1, head_length=1.5, fc=color, ec=color, alpha=0.8, linewidth=2)
            
            # 添加相机标签
            ax.text(camera_pos_2d[0]+1, camera_pos_2d[1]+1, label, 
                   fontsize=10, weight='bold', color=color, 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            
            fov_info.append(f"{cam_name}: 位置 ({camera_pos_2d[0]:.1f}, {camera_pos_2d[1]:.1f}), FOV {fov_h}度, 朝向 {np.degrees(camera_yaw):.1f}°")
            cameras_found += 1
        
        # 绘制点云数据（如果存在）
        points = self.load_pointcloud(timestamp)
        if points is not None and len(points) > 0:
            # 点云已经在group坐标系中，直接使用
            # 采样点云
            max_points = 3000
            if len(points) > max_points:
                indices = np.random.choice(len(points), max_points, replace=False)
                points_display = points[indices]
            else:
                points_display = points
            
            # 过滤在范围内的点
            in_range_mask = (points_display[:, 0] >= -bev_range) & (points_display[:, 0] <= bev_range) & \
                           (points_display[:, 1] >= -bev_range) & (points_display[:, 1] <= bev_range)
            points_in_range = points_display[in_range_mask]
            
            if len(points_in_range) > 0:
                ax.scatter(points_in_range[:, 0], points_in_range[:, 1], 
                          c='gray', s=2, alpha=0.4, label='Point Cloud', zorder=1)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'BEV Camera FOV Visualization - {timestamp}\nSensor Deployment Overview (FOV up to 150m)', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        output_path = self.fov_dir / f"fov_visualization_{timestamp}.jpg"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存配置信息
        config_path = self.fov_dir / f"fov_config_{timestamp}.txt"
        with open(config_path, 'w') as f:
            f.write("BEV FOV Visualization Configuration:\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"BEV Range: ±{bev_range}m (150m x 150m)\n")
            f.write(f"FOV Distance intervals: {distance_intervals}m\n")
            f.write(f"Max FOV Range: {max_fov_range}m\n")
            if points is not None:
                f.write(f"Total points: {len(points)}\n")
            f.write(f"Cameras found: {cameras_found}\n")
            f.write("\nCamera FOV information:\n")
            for info in fov_info:
                f.write(f"  {info}\n")
        
        print(f"保存FOV可视化: {output_path}")
        print(f"保存配置: {config_path}")
   
    def run(self, timestamps: List[str]):
        print(f"开始处理 {len(timestamps)} 个样本")
        
        for i, timestamp in enumerate(timestamps):
            print(f"\n[{i+1}/{len(timestamps)}] 处理: {timestamp}")
            self.create_label_projection(timestamp)
            self.create_pointcloud_projection(timestamp)
            self.create_pcd_3dbox_visualization(timestamp)
            self.create_fov_visualization(timestamp)
        
        print(f"\n完成! 结果保存在: {self.output_dir}")


def main():
    # 配置区域 - 在这里修改
    config = {
        'data_root': "/home/tyjt/ws/02_docker/cv2-tests/dataset/2d3d4d_20250728_weiyuan",
        'calib_file': "sensor2map_calib.json",  # 标定文件名
        'image_folder': "image_dc",  # 图像文件夹名
        
        'camera_config': {
            'SC_R1_Aw_CpcS_CAMR': 'SC_1A_CamR',
            'SC_R1_Bn_CpcW_CAMR': 'SC_1B_CamR',
            'SC_R1_Ce_CpcN_CAMR': 'SC_1C_CamR', 
            'SC_R1_Ds_CpcE_CAMR': 'SC_1D_CamR'
        },
        
        'point_cloud_config': {
            'max_points': 5000,
            'color_scheme': 'hot',
            'point_size': 3,
            'alpha': 0.8,
        },
        
        'sample_timestamps': ["1753701446899737835"],
        'output_root': "output",
        'task_name': "03_visualization",
    }
    
    # 检查路径
    data_root = Path(config['data_root'])
    calib_path = data_root / "calib" / config['calib_file']
    
    print(f"数据根目录: {data_root}")
    print(f"标定文件: {calib_path}")
    
    if not calib_path.exists():
        print(f"错误: 标定文件不存在")
        print(f"请检查配置: data_root 和 calib_file")
        return
    
    # 运行可视化
    visualizer = SpatialVisualizer(config)
    visualizer.run(config['sample_timestamps'])


if __name__ == "__main__":
    main()