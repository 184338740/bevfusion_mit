import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    import matplotlib.cm as cm
    from matplotlib.colors import LinearSegmentedColormap
    matplotlib.use('Agg')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class EnhancedSpatialVisualizer:
    def __init__(self, config: Dict):
        """
        初始化可视化器
        config: 包含所有配置的字典
        """
        self.config = config
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建输出目录
        self.output_base_dir = Path(config['output_root']) / config['task_name'] / current_time
        self.pointcloud_output_dir = self.output_base_dir / "pointcloud_projection"
        self.label_output_dir = self.output_base_dir / "3d_label_projection"
        
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.pointcloud_output_dir.mkdir(parents=True, exist_ok=True)
        self.label_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据路径
        self.data_root = Path(config['data_root']).expanduser()
        self.dataset_path = self.data_root / "datasets" / "G51102400001M00_20250728191726_2.0HZ"
        
        # 标定文件路径
        self.calib_path = self.data_root / "calib" / config['calib_file']
        
        # 相机配置
        self.camera_mapping = config['camera_mapping']
        self.camera_names = list(self.camera_mapping.keys())
        
        # 图像类型
        self.image_folder = config['image_folder']
        
        # 点云配置
        self.point_cloud_config = config['point_cloud_config']
        
        # 3D Label颜色配置
        self.label_colors = {
            'car': (0, 255, 0),      # 绿色
            'truck': (255, 0, 0),    # 蓝色
            'cyclist': (0, 0, 255),  # 红色
            'pedestrian': (255, 255, 0),  # 青色
            'van': (255, 0, 255),    # 紫色
            'tricycle': (0, 255, 255),  # 黄色
            'default': (255, 255, 255)  # 白色
        }
        
        # 初始化颜色映射
        self.setup_color_maps()
        
        # 加载标定文件
        self.calib_data = self.load_calibration()
        
        # 自动检测标定文件中的相机名称并更新映射
        self.actual_camera_names = self.detect_actual_camera_names()
        
        print(f"✅ 可视化器初始化完成")
        print(f"📁 数据路径: {self.data_root}")
        print(f"🎯 标定文件: {self.calib_path.name}")
        print(f"🖼️ 图像文件夹: {self.image_folder}")
        print(f"📊 输出目录: {self.output_base_dir}")
        print(f"📷 检测到的相机名称: {list(self.actual_camera_names.keys())}")
    
    def detect_actual_camera_names(self) -> Dict:
        """检测标定文件中实际的相机名称并创建映射"""
        actual_names = {}
        
        for expected_name in self.camera_names:
            # 检查标定文件中是否存在预期的相机名称
            if expected_name in self.calib_data:
                actual_names[expected_name] = expected_name
                print(f"  ✅ 找到相机: {expected_name}")
            else:
                # 尝试查找带_new后缀的名称
                new_name = f"{expected_name}_new"
                if new_name in self.calib_data:
                    actual_names[expected_name] = new_name
                    print(f"  🔄 找到_new后缀相机: {new_name} (映射到 {expected_name})")
                else:
                    # 如果都找不到，使用原始名称（可能会在后续报错）
                    actual_names[expected_name] = expected_name
                    print(f"  ⚠️ 未找到相机 {expected_name}，使用原始名称")
        
        return actual_names
    
    def get_actual_camera_name(self, camera_name: str) -> str:
        """获取标定文件中实际的相机名称"""
        return self.actual_camera_names.get(camera_name, camera_name)
    
    def setup_color_maps(self):
        """设置颜色映射"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        self.matplotlib_cmaps = {
            'rainbow': plt.cm.rainbow,
            'thermal': plt.cm.hot,
            'jet': plt.cm.jet,
            'hot': plt.cm.hot,
            'cool': plt.cm.cool,
            'viridis': plt.cm.viridis,
            'plasma': plt.cm.plasma,
            'inferno': plt.cm.inferno,
            'magma': plt.cm.magma,
        }
    
    def load_calibration(self) -> Dict:
        """加载标定文件"""
        print(f"🔍 正在加载标定文件: {self.calib_path}")
        
        if not self.calib_path.exists():
            raise FileNotFoundError(f"❌ 标定文件不存在: {self.calib_path}")
        
        with open(self.calib_path, 'r', encoding='utf-8') as f:
            calib_data = json.load(f)
        
        print("✅ 标定文件加载成功!")
        
        # 打印标定文件中的相机名称
        camera_keys = [key for key in calib_data.keys() if key.startswith('SC_R1_') and key != 'group2map']
        print(f"📷 标定文件中的相机: {camera_keys}")
        
        return calib_data
    
    def load_camera_image(self, camera_folder: str, timestamp: str) -> Optional[np.ndarray]:
        """加载相机图像"""
        img_path = self.dataset_path / camera_folder / self.image_folder / f"{timestamp}.jpg"
        
        if not img_path.exists():
            print(f"  ❌ 未找到图像: {img_path}")
            return None
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ❌ 无法加载图像: {img_path}")
            return None
        
        print(f"  ✅ 加载图像: {img_path.name} ({image.shape[1]}x{image.shape[0]})")
        return image
    
    def load_labels(self, file_path: Path) -> List[Dict]:
        """加载3D标签数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            
            if isinstance(labels, list):
                return labels
            elif isinstance(labels, dict):
                if 'objects' in labels:
                    return labels['objects']
                elif 'labels' in labels:
                    return labels['labels']
                else:
                    return []
            else:
                return []
                
        except Exception as e:
            print(f"❌ 加载标签文件 {file_path} 时出错: {e}")
            return []
    
    def create_3d_box_corners(self, box3d: List[float], rotation: float) -> np.ndarray:
        """创建3D边界框的8个角点"""
        x, y, z, l, w, h = box3d
        
        # 在物体局部坐标系下的角点
        corners = np.array([
            [l/2, w/2, h/2],   [l/2, w/2, -h/2],
            [l/2, -w/2, h/2],  [l/2, -w/2, -h/2],
            [-l/2, w/2, h/2],  [-l/2, w/2, -h/2],
            [-l/2, -w/2, h/2], [-l/2, -w/2, -h/2]
        ])
        
        # 应用旋转 (绕Z轴)
        rot_matrix = np.array([
            [np.cos(rotation), -np.sin(rotation), 0],
            [np.sin(rotation), np.cos(rotation), 0],
            [0, 0, 1]
        ])
        
        corners = corners @ rot_matrix.T
        corners += np.array([x, y, z])
        
        return corners
    
    def create_orientation_arrow(self, box3d: List[float], rotation: float) -> np.ndarray:
        """创建方向箭头"""
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
    
    def smart_point_sampling(self, points_2d: np.ndarray, depths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """智能点云采样"""
        max_points = self.point_cloud_config['max_points']
        
        if max_points is None or len(points_2d) <= max_points:
            return points_2d, depths
        
        strategy = self.point_cloud_config['sampling_strategy']
        print(f"  点云采样: {len(points_2d)} -> {max_points} (策略: {strategy})")
        
        if strategy == 'random':
            indices = np.random.choice(len(points_2d), max_points, replace=False)
            return points_2d[indices], depths[indices]
        
        elif strategy == 'uniform':
            step = len(points_2d) // max_points
            indices = np.arange(0, len(points_2d), step)[:max_points]
            return points_2d[indices], depths[indices]
        
        elif strategy == 'depth_aware':
            depth_bins = 5
            points_per_bin = max_points // depth_bins
            
            sampled_points = []
            sampled_depths = []
            
            depth_min, depth_max = np.min(depths), np.max(depths)
            bin_edges = np.linspace(depth_min, depth_max, depth_bins + 1)
            
            for i in range(depth_bins):
                mask = (depths >= bin_edges[i]) & (depths < bin_edges[i+1])
                bin_points = points_2d[mask]
                bin_depths = depths[mask]
                
                if len(bin_points) > points_per_bin:
                    indices = np.random.choice(len(bin_points), points_per_bin, replace=False)
                    sampled_points.append(bin_points[indices])
                    sampled_depths.append(bin_depths[indices])
                else:
                    sampled_points.append(bin_points)
                    sampled_depths.append(bin_depths)
            
            if sampled_points:
                all_sampled_points = np.vstack(sampled_points)
                all_sampled_depths = np.concatenate(sampled_depths)
            else:
                all_sampled_points = np.array([])
                all_sampled_depths = np.array([])
            
            if len(all_sampled_points) > max_points:
                indices = np.random.choice(len(all_sampled_points), max_points, replace=False)
                all_sampled_points = all_sampled_points[indices]
                all_sampled_depths = all_sampled_depths[indices]
            
            return all_sampled_points, all_sampled_depths
        
        else:
            indices = np.random.choice(len(points_2d), max_points, replace=False)
            return points_2d[indices], depths[indices]
    
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
    
    def get_camera_calib(self, camera_name: str) -> Dict:
        """获取相机标定参数"""
        actual_camera_name = self.get_actual_camera_name(camera_name)
        
        if actual_camera_name not in self.calib_data:
            print(f"❌ 错误: 标定文件中未找到相机 {actual_camera_name} (原始名称: {camera_name})")
            return None
        
        cam_data = self.calib_data[actual_camera_name]
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
        """获取group到map的变换矩阵"""
        if 'group2map' not in self.calib_data:
            print("❌ 错误: 标定文件中未找到group2map变换")
            return np.eye(4)
        return self.parse_transform(self.calib_data['group2map'])
    
    def transform_points_to_camera(self, points: np.ndarray, camera_name: str) -> np.ndarray:
        """将点从group坐标系变换到相机坐标系"""
        camera_calib = self.get_camera_calib(camera_name)
        group_transform = self.get_group_transform()
        
        if camera_calib is None or group_transform is None:
            print(f"❌ 无法获取 {camera_name} 的标定参数")
            return points
        
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        
        # 先变换到map坐标系
        points_map = (group_transform @ points_homo.T).T
        
        # 再变换到相机坐标系
        camera_inv = np.linalg.inv(camera_calib['extrinsic'])
        points_camera = (camera_inv @ points_map.T).T
        
        return points_camera[:, :3]
    
    def project_points_to_image(self, points: np.ndarray, camera_name: str, image_size: tuple) -> Tuple[np.ndarray, np.ndarray]:
        """将3D点投影到图像平面"""
        camera_calib = self.get_camera_calib(camera_name)
        if camera_calib is None:
            return np.array([]), np.array([])
        
        # 变换到相机坐标系
        points_cam = self.transform_points_to_camera(points, camera_name)
        
        # 过滤相机后面的点
        front_mask = points_cam[:, 2] > 0.1
        if not np.any(front_mask):
            return np.array([]), np.array([])
        
        points_cam = points_cam[front_mask]
        depths = points_cam[:, 2]
        intrinsic = camera_calib['intrinsic']
        
        # 投影到图像平面
        points_2d_homo = (intrinsic @ points_cam.T).T
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
        
        # 过滤图像范围外的点
        h, w = image_size
        valid_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
                    (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
        
        valid_points_2d = points_2d[valid_mask]
        valid_depths = depths[valid_mask]
        
        return valid_points_2d, valid_depths
    
    def draw_3d_box_on_image(self, image: np.ndarray, box_corners_2d: np.ndarray, 
                           color: tuple, orientation_arrow_2d: np.ndarray = None) -> np.ndarray:
        """在图像上绘制3D边界框"""
        vis_image = image.copy()
        
        if len(box_corners_2d) != 8:
            return vis_image
        
        # 边界框连线
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 3),  # 底面
            (4, 5), (4, 6), (5, 7), (6, 7),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7)   # 侧面
        ]
        
        for i, j in edges:
            pt1 = (int(box_corners_2d[i, 0]), int(box_corners_2d[i, 1]))
            pt2 = (int(box_corners_2d[j, 0]), int(box_corners_2d[j, 1]))
            
            # 检查点是否在图像范围内
            if (0 <= pt1[0] < vis_image.shape[1] and 0 <= pt1[1] < vis_image.shape[0] and
                0 <= pt2[0] < vis_image.shape[1] and 0 <= pt2[1] < vis_image.shape[0]):
                cv2.line(vis_image, pt1, pt2, color, 2)
        
        # 绘制方向箭头
        if orientation_arrow_2d is not None and len(orientation_arrow_2d) == 2:
            start_pt = (int(orientation_arrow_2d[0, 0]), int(orientation_arrow_2d[0, 1]))
            end_pt = (int(orientation_arrow_2d[1, 0]), int(orientation_arrow_2d[1, 1]))
            
            if (0 <= start_pt[0] < vis_image.shape[1] and 0 <= start_pt[1] < vis_image.shape[0] and
                0 <= end_pt[0] < vis_image.shape[1] and 0 <= end_pt[1] < vis_image.shape[0]):
                cv2.arrowedLine(vis_image, start_pt, end_pt, (255, 255, 0), 3, tipLength=0.3)
        
        return vis_image
    
    def get_color_map(self):
        """获取当前配置的颜色映射"""
        if not MATPLOTLIB_AVAILABLE:
            return 'viridis'
            
        scheme = self.point_cloud_config['color_scheme']
        return self.matplotlib_cmaps.get(scheme, plt.cm.viridis)
    
    def create_3d_label_projection(self, timestamp: str):
        """创建3D标签投影图像（ABCD四个相机拼在一起）"""
        print(f"\n📦 创建3D标签投影: {timestamp}")
        
        # 查找标签文件
        label_dir = self.dataset_path / "lidar" / "label"
        label_files = list(label_dir.glob(f"{timestamp}.json"))
        if not label_files:
            print(f"❌ 未找到标签文件: {timestamp}")
            return
        
        try:
            # 加载标签
            labels = self.load_labels(label_files[0])
            print(f"加载3D标签: {len(labels)} 个对象")
            
            if not labels:
                print("没有可用的标签数据")
                return
            
            # 收集所有相机的图像
            camera_images = {}
            image_sizes = {}
            
            for camera_name in self.camera_names:
                folder_name = self.camera_mapping[camera_name]
                image = self.load_camera_image(folder_name, timestamp)
                if image is not None:
                    camera_images[camera_name] = image
                    image_sizes[camera_name] = image.shape[:2]
            
            if not any(img is not None for img in camera_images.values()):
                print("❌ 未找到任何相机图像")
                return
            
            # 确定统一的图像尺寸
            base_size = None
            for img in camera_images.values():
                if img is not None:
                    base_size = (img.shape[1], img.shape[0])
                    break
            
            if base_size is None:
                print("❌ 无法确定图像尺寸")
                return
            
            # 创建2x2的合并图像
            combined_width = base_size[0] * 2
            combined_height = base_size[1] * 2
            combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            
            # 相机位置布局
            camera_positions = {
                'SC_R1_Aw_CpcS_CAMR_new': (0, 0),           # 左上
                'SC_R1_Bn_CpcW_CAMR_new': (base_size[0], 0), # 右上
                'SC_R1_Ce_CpcN_CAMR_new': (0, base_size[1]), # 左下
                'SC_R1_Ds_CpcE_CAMR_new': (base_size[0], base_size[1]) # 右下
            }
            
            # 处理每个相机
            total_projected = 0
            for camera_name, (x_offset, y_offset) in camera_positions.items():
                if camera_images[camera_name] is not None:
                    img = camera_images[camera_name]
                    if img.shape[:2] != (base_size[1], base_size[0]):
                        img = cv2.resize(img, base_size)
                    
                    vis_image = img.copy()
                    objects_projected = 0
                    
                    # 投影标签
                    for label in labels:
                        obj_type = label.get('type', 'unknown')
                        box3d = label.get('box3d', [])
                        rotation = label.get('rotation', [0])[0]
                        
                        if len(box3d) != 6:
                            continue
                        
                        color = self.label_colors.get(obj_type, self.label_colors['default'])
                        
                        # 创建3D边界框角点
                        box_corners_3d = self.create_3d_box_corners(box3d, rotation)
                        orientation_arrow_3d = self.create_orientation_arrow(box3d, rotation)
                        
                        # 投影到图像平面
                        box_corners_2d = self.project_points_to_image(box_corners_3d, camera_name, (base_size[1], base_size[0]))[0]
                        orientation_arrow_2d = self.project_points_to_image(orientation_arrow_3d, camera_name, (base_size[1], base_size[0]))[0]
                        
                        # 绘制边界框
                        if len(box_corners_2d) == 8:
                            vis_image = self.draw_3d_box_on_image(vis_image, box_corners_2d, color, orientation_arrow_2d)
                            objects_projected += 1
                    
                    total_projected += objects_projected
                    print(f"  ✅ {camera_name}: 投影 {objects_projected} 个对象")
                    
                else:
                    # 创建黑色背景
                    vis_image = np.zeros((base_size[1], base_size[0], 3), dtype=np.uint8)
                    print(f"  ❌ {camera_name}: 无图像，使用黑色背景")
                
                # 将图像放置到合并图像中
                combined_image[y_offset:y_offset+base_size[1], x_offset:x_offset+base_size[0]] = vis_image
                
                # 添加相机名称标签
                label_text = f"{self.camera_mapping[camera_name]}"
                if camera_images[camera_name] is not None:
                    label_text += f" - {objects_projected} objects"
                
                cv2.putText(combined_image, label_text, 
                           (x_offset + 10, y_offset + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 保存合并图像到标签投影目录
            output_filename = f"3d_label_projection_{timestamp}.jpg"
            output_path = self.label_output_dir / output_filename
            cv2.imwrite(str(output_path), combined_image)
            
            print(f"✅ 保存3D标签投影: {output_filename}")
            print(f"📊 总体投影统计: {total_projected}/{len(labels)} 个对象")
            
        except Exception as e:
            print(f"❌ 创建3D标签投影失败: {e}")
            import traceback
            traceback.print_exc()
    
    def create_point_cloud_projection(self, timestamp: str):
        """创建点云投影图像（ABCD四个相机拼在一起，包含原图对比）"""
        if not MATPLOTLIB_AVAILABLE:
            print("❌ matplotlib未安装，跳过点云投影可视化")
            return
            
        print(f"\n☁️ 创建点云投影: {timestamp}")
        
        # 查找点云文件
        lidar_dir = self.dataset_path / "lidar" / "pcd"
        pcd_files = list(lidar_dir.glob(f"{timestamp}.pcd"))
        npy_files = list(lidar_dir.glob(f"{timestamp}.npy"))
        
        if not pcd_files and not npy_files:
            print(f"❌ 未找到点云文件: {timestamp}")
            return
        
        try:
            import open3d as o3d
            
            # 加载点云
            if pcd_files:
                pcd = o3d.io.read_point_cloud(str(pcd_files[0]))
                points = np.asarray(pcd.points)
            elif npy_files:
                points = np.load(str(npy_files[0]))
            
            print(f"✅ 加载点云: {len(points)} 个点")
            print(f"⚙️ 点云配置: {self.point_cloud_config}")
            
            # 收集所有相机的图像
            camera_images = {}
            image_sizes = {}
            
            for camera_name in self.camera_names:
                folder_name = self.camera_mapping[camera_name]
                image = self.load_camera_image(folder_name, timestamp)
                if image is not None:
                    camera_images[camera_name] = image
                    image_sizes[camera_name] = image.shape[:2]
            
            if not any(img is not None for img in camera_images.values()):
                print("❌ 未找到任何相机图像")
                return
            
            # 确定统一的图像尺寸
            base_size = None
            for img in camera_images.values():
                if img is not None:
                    base_size = (img.shape[1], img.shape[0])
                    break
            
            if base_size is None:
                print("❌ 无法确定图像尺寸")
                return
            
            # 获取颜色映射
            color_map = self.get_color_map()
            print(f"🎨 使用的颜色映射: {self.point_cloud_config['color_scheme']}")
            
            # 创建2x4的合并图像
            fig, axes = plt.subplots(4, 2, figsize=(20, 32))
            
            # 相机位置布局
            camera_layout = {
                'SC_R1_Aw_CpcS_CAMR_new': 0,
                'SC_R1_Bn_CpcW_CAMR_new': 1,
                'SC_R1_Ce_CpcN_CAMR_new': 2,
                'SC_R1_Ds_CpcE_CAMR_new': 3
            }
            
            total_points_projected = 0
            
            for camera_name, row_idx in camera_layout.items():
                folder_name = self.camera_mapping[camera_name]
                
                if camera_images[camera_name] is not None:
                    img = camera_images[camera_name]
                    if img.shape[:2] != (base_size[1], base_size[0]):
                        img = cv2.resize(img, base_size)
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # 投影到图像平面
                    points_2d, depths = self.project_points_to_image(points, camera_name, (base_size[1], base_size[0]))
                    
                    # 应用点云采样
                    if self.point_cloud_config['enable_sampling'] and self.point_cloud_config['max_points'] is not None:
                        points_2d, depths = self.smart_point_sampling(points_2d, depths)
                    
                    # 左侧：原图
                    ax_original = axes[row_idx, 0]
                    ax_original.imshow(img_rgb)
                    ax_original.set_title(f'{folder_name} - Original', fontsize=12)
                    ax_original.axis('off')
                    
                    # 右侧：点云投影
                    ax_projection = axes[row_idx, 1]
                    ax_projection.imshow(img_rgb)
                    
                    # 绘制投影点
                    if len(points_2d) > 0:
                        scatter = ax_projection.scatter(
                            points_2d[:, 0], points_2d[:, 1], 
                            c=depths, 
                            cmap=color_map,
                            s=self.point_cloud_config['point_size'], 
                            alpha=self.point_cloud_config['alpha']
                        )
                        
                        if row_idx == 0 and self.point_cloud_config['show_colorbar']:
                            cbar = plt.colorbar(scatter, ax=ax_projection, label='Depth (m)')
                            cbar.set_label('Depth (m)', fontsize=10)
                        
                        total_points_projected += len(points_2d)
                        points_info = f" - {len(points_2d)} points"
                    else:
                        points_info = " - 0 points"
                    
                    config_info = f" (Max: {self.point_cloud_config['max_points']}, Color: {self.point_cloud_config['color_scheme']})"
                    ax_projection.set_title(f'{folder_name} - Point Cloud{points_info}{config_info}', fontsize=12)
                    ax_projection.axis('off')
                    
                    print(f"  ✅ {camera_name}: 投影 {len(points_2d)} 个点")
                    
                else:
                    black_bg = np.zeros((base_size[1], base_size[0], 3), dtype=np.uint8)
                    
                    ax_original = axes[row_idx, 0]
                    ax_original.imshow(black_bg)
                    ax_original.set_title(f'{folder_name} - No Image', fontsize=12)
                    ax_original.axis('off')
                    
                    ax_projection = axes[row_idx, 1]
                    ax_projection.imshow(black_bg)
                    ax_projection.set_title(f'{folder_name} - No Image', fontsize=12)
                    ax_projection.axis('off')
                    
                    print(f"  ❌ {camera_name}: 无图像")
            
            plt.tight_layout()
            
            # 保存合并图像到点云投影目录
            output_filename = f"point_cloud_projection_{timestamp}.jpg"
            output_path = self.pointcloud_output_dir / output_filename
            plt.savefig(output_path, bbox_inches='tight', 
                       dpi=self.point_cloud_config['dpi'], 
                       facecolor='white')
            plt.close()
            
            print(f"✅ 保存点云投影: {output_filename}")
            print(f"📊 总体点云投影统计: {total_points_projected} 个点")
            
        except Exception as e:
            print(f"❌ 创建点云投影失败: {e}")
            import traceback
            traceback.print_exc()
    
    def run_visualization(self, timestamps: List[str]):
        """运行可视化流程"""
        print("=" * 50)
        print("🚀 开始空间可视化")
        print("=" * 50)
        
        for i, timestamp in enumerate(timestamps):
            print(f"\n[{i+1}/{len(timestamps)}] 处理: {timestamp}")
            self.create_3d_label_projection(timestamp)
            self.create_point_cloud_projection(timestamp)
        
        print(f"\n🎉 可视化完成!")
        print(f"📁 点云投影: {self.pointcloud_output_dir}")
        print(f"📁 3D标签投影: {self.label_output_dir}")


def main():
    """主函数 - 在这里统一配置所有参数"""
    
    # ==================== 【在这里修改配置】 ====================
    config = {
        # 【必改】数据路径
        'data_root': "/home/tyjt/ws/02_docker/cv2-tests/dataset/2d3d4d_20250728_weiyuan",
        
        # 【选改】标定配置 - 直接填写标定文件名称
        'calib_file': "sensor2map_calib.json",  # 填写实际存在的文件名
        
        # 【选改】图像配置 - 直接填写图像文件夹名称
        'image_folder': "image_dc",  # 填写实际存在的文件夹名
        
        # 【选改】点云显示配置
        'point_cloud_config': {
            'max_points': None,      # 最大点数，None=显示所有
            'color_scheme': 'hot',   # 颜色方案
            'point_size': 3,         # 点大小
            'enable_sampling': True,
            'sampling_strategy': 'depth_aware',
            'alpha': 0.8,
            'show_colorbar': True,
            'dpi': 200,
        },
        
        # 【选改】样本配置
        'sample_timestamps': ["1753701446899737835"],
        
        # 【通常不改】输出配置
        'output_root': "output",
        'task_name': "03_visualization",
        
        # 【通常不改】相机映射
        'camera_mapping': {
            'SC_R1_Aw_CpcS_CAMR_new': 'SC_1A_CamR',
            'SC_R1_Bn_CpcW_CAMR_new': 'SC_1B_CamR', 
            'SC_R1_Ce_CpcN_CAMR_new': 'SC_1C_CamR',
            'SC_R1_Ds_CpcE_CAMR_new': 'SC_1D_CamR'
        }
    }
    # ==================== 【配置结束】 ====================
    
    # 打印当前配置的完整路径
    data_root = Path(config['data_root']).expanduser()
    calib_full_path = data_root / "calib" / config['calib_file']
    image_full_path = data_root / "datasets" / "G51102400001M00_20250728191726_2.0HZ" / "SC_1A_CamR" / config['image_folder']
    
    print(f"📁 数据根目录: {data_root}")
    print(f"🎯 标定文件路径: {calib_full_path}")
    print(f"🖼️ 图像文件夹示例: {image_full_path}")
    
    # 检查标定文件是否存在
    if not calib_full_path.exists():
        print(f"❌ 标定文件不存在!")
        print(f"💡 请检查并修改 main() 函数中的 'calib_file' 配置")
        print(f"   当前配置: 'calib_file': '{config['calib_file']}'")
        return
    
    # 检查图像文件夹是否存在
    if not image_full_path.parent.exists():
        print(f"❌ 图像文件夹不存在!")
        print(f"💡 请检查并修改 main() 函数中的 'data_root' 配置")
        print(f"   当前配置: 'data_root': '{config['data_root']}'")
        return
    
    # 创建可视化器并运行
    visualizer = EnhancedSpatialVisualizer(config)
    visualizer.run_visualization(config['sample_timestamps'])


if __name__ == "__main__":
    main()