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
    matplotlib.use('Agg')  # 使用非交互式后端
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib未安装，点云投影可视化将不可用")

class SpatialVisualizer:
    def __init__(self, data_root: str, output_root: str = "output"):
        self.data_root = Path(data_root).expanduser()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建不同的输出目录
        self.output_dir = Path(output_root) / "SpatialVisualization" / current_time
        self.label_output_dir = self.output_dir / "3d_label_projection"
        self.pointcloud_output_dir = self.output_dir / "point_cloud_projection"
        
        # 创建目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label_output_dir.mkdir(parents=True, exist_ok=True)
        self.pointcloud_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 标定文件路径
        self.calib_path = self.data_root / "calib" / "sensor2map_calib.json"
        self.dataset_path = self.data_root / "datasets" / "G51102400001M00_20250728191726_2.0HZ"
        
        # 正确的相机映射（基于成功案例）
        self.camera_mapping = {
            'SC_R1_Aw_CpcS_CAMR': 'SC_1A_CamR',  # A相机
            'SC_R1_Bn_CpcW_CAMR': 'SC_1B_CamR',  # B相机
            'SC_R1_Ce_CpcN_CAMR': 'SC_1C_CamR',  # C相机  
            'SC_R1_Ds_CpcE_CAMR': 'SC_1D_CamR'   # D相机
        }
        
        self.camera_names = list(self.camera_mapping.keys())
        
        # 加载标定文件
        self.calib_data = self.load_calibration()
        
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
        
        print(f"数据根目录: {self.data_root}")
        print(f"输出主目录: {self.output_dir}")
        print(f"3D标签投影目录: {self.label_output_dir}")
        print(f"点云投影目录: {self.pointcloud_output_dir}")
        print(f"相机映射: {self.camera_mapping}")
        if not MATPLOTLIB_AVAILABLE:
            print("警告: matplotlib未安装，点云投影可视化将不可用")
    
    def load_calibration(self) -> Dict:
        """加载标定文件"""
        print(f"加载标定文件: {self.calib_path}")
        
        if not self.calib_path.exists():
            alt_paths = [
                self.data_root.parent / "calib" / "sensor2map_calib.json",
                self.data_root / "calib" / "G51102400001M00" / "sensor2map_calib.json"
            ]
            for path in alt_paths:
                if path.exists():
                    self.calib_path = path
                    print(f"使用备用标定文件: {path}")
                    break
            else:
                raise FileNotFoundError(f"未找到标定文件")
        
        with open(self.calib_path, 'r', encoding='utf-8') as f:
            calib_data = json.load(f)
        
        print("标定文件加载成功!")
        return calib_data
    
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
        if camera_name not in self.calib_data:
            print(f"警告: 标定文件中未找到相机 {camera_name}")
            return None
        
        cam_data = self.calib_data[camera_name]
        extrinsic = self.parse_transform(cam_data)
        
        # 使用实际的相机内参
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
            print("警告: 标定文件中未找到group2map变换")
            return np.eye(4)
        return self.parse_transform(self.calib_data['group2map'])
    
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
            print(f"加载标签文件 {file_path} 时出错: {e}")
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
    
    def transform_points_to_camera(self, points: np.ndarray, camera_name: str) -> np.ndarray:
        """将点从group坐标系变换到相机坐标系"""
        camera_calib = self.get_camera_calib(camera_name)
        group_transform = self.get_group_transform()
        
        if camera_calib is None or group_transform is None:
            print(f"无法获取 {camera_name} 的标定参数")
            return points
        
        # 变换流程: P_camera = T_camera^{-1} * T_group * P_object
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        
        # 先变换到map坐标系
        points_map = (group_transform @ points_homo.T).T
        
        # 再变换到相机坐标系
        camera_inv = np.linalg.inv(camera_calib['extrinsic'])
        points_camera = (camera_inv @ points_map.T).T
        
        return points_camera[:, :3]
    
    def project_points_to_image(self, points: np.ndarray, camera_name: str, image_size: tuple) -> np.ndarray:
        """将3D点投影到图像平面"""
        camera_calib = self.get_camera_calib(camera_name)
        if camera_calib is None:
            return np.array([])
        
        # 变换到相机坐标系
        points_cam = self.transform_points_to_camera(points, camera_name)
        
        # 过滤相机后面的点
        front_mask = points_cam[:, 2] > 0.1
        if not np.any(front_mask):
            return np.array([])
        
        points_cam = points_cam[front_mask]
        intrinsic = camera_calib['intrinsic']
        
        # 投影到图像平面
        points_2d_homo = (intrinsic @ points_cam.T).T
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
        
        # 过滤图像范围外的点
        h, w = image_size
        valid_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
                    (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
        
        return points_2d[valid_mask]
    
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
    
    def create_combined_3d_label_projection(self, timestamp: str):
        """创建合并的3D标签投影图像（ABCD四个相机拼在一起）"""
        print(f"\n创建合并的3D标签投影: {timestamp}")
        
        # 查找标签文件
        label_dir = self.dataset_path / "lidar" / "label"
        label_files = list(label_dir.glob(f"{timestamp}.json"))
        if not label_files:
            print(f"未找到标签文件: {timestamp}")
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
                image_dir = self.dataset_path / folder_name / "image_dc"
                image_files = list(image_dir.glob(f"{timestamp}.jpg"))
                
                if image_files:
                    image = cv2.imread(str(image_files[0]))
                    if image is not None:
                        camera_images[camera_name] = image
                        image_sizes[camera_name] = image.shape[:2]
                        print(f"找到 {camera_name} 图像: {image.shape[1]}x{image.shape[0]}")
                    else:
                        print(f"无法加载 {camera_name} 图像")
                        camera_images[camera_name] = None
                else:
                    print(f"未找到 {camera_name} 图像文件")
                    camera_images[camera_name] = None
            
            # 如果没有找到任何图像，返回
            if not any(img is not None for img in camera_images.values()):
                print("未找到任何相机图像")
                return
            
            # 确定统一的图像尺寸（使用第一个有效图像的尺寸）
            base_size = None
            for img in camera_images.values():
                if img is not None:
                    base_size = (img.shape[1], img.shape[0])  # (width, height)
                    break
            
            if base_size is None:
                print("无法确定图像尺寸")
                return
            
            # 创建2x2的合并图像
            combined_width = base_size[0] * 2
            combined_height = base_size[1] * 2
            combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            
            # 相机位置布局
            camera_positions = {
                'SC_R1_Aw_CpcS_CAMR': (0, 0),           # 左上
                'SC_R1_Bn_CpcW_CAMR': (base_size[0], 0), # 右上
                'SC_R1_Ce_CpcN_CAMR': (0, base_size[1]), # 左下
                'SC_R1_Ds_CpcE_CAMR': (base_size[0], base_size[1]) # 右下
            }
            
            # 处理每个相机
            total_projected = 0
            for camera_name, (x_offset, y_offset) in camera_positions.items():
                if camera_images[camera_name] is not None:
                    # 使用实际图像
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
                        
                        color = self.colors.get(obj_type, self.colors['default'])
                        
                        # 创建3D边界框角点
                        box_corners_3d = self.create_3d_box_corners(box3d, rotation)
                        orientation_arrow_3d = self.create_orientation_arrow(box3d, rotation)
                        
                        # 投影到图像平面
                        box_corners_2d = self.project_points_to_image(box_corners_3d, camera_name, (base_size[1], base_size[0]))
                        orientation_arrow_2d = self.project_points_to_image(orientation_arrow_3d, camera_name, (base_size[1], base_size[0]))
                        
                        # 绘制边界框
                        if len(box_corners_2d) == 8:
                            vis_image = self.draw_3d_box_on_image(vis_image, box_corners_2d, color, orientation_arrow_2d)
                            objects_projected += 1
                    
                    total_projected += objects_projected
                    print(f"  {camera_name}: 投影 {objects_projected} 个对象")
                    
                else:
                    # 创建黑色背景
                    vis_image = np.zeros((base_size[1], base_size[0], 3), dtype=np.uint8)
                    print(f"  {camera_name}: 无图像，使用黑色背景")
                
                # 将图像放置到合并图像中
                combined_image[y_offset:y_offset+base_size[1], x_offset:x_offset+base_size[0]] = vis_image
                
                # 添加相机名称标签
                label_text = f"{self.camera_mapping[camera_name]} ({camera_name})"
                if camera_images[camera_name] is not None:
                    label_text += f" - {objects_projected} objects"
                
                cv2.putText(combined_image, label_text, 
                           (x_offset + 10, y_offset + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 保存合并图像到标签投影目录
            output_filename = f"combined_3d_label_projection_{timestamp}.jpg"
            output_path = self.label_output_dir / output_filename
            cv2.imwrite(str(output_path), combined_image)
            
            print(f"保存合并图像: {output_filename}")
            print(f"总体投影统计: {total_projected}/{len(labels)} 个对象")
            
        except Exception as e:
            print(f"创建合并3D标签投影失败: {e}")
            import traceback
            traceback.print_exc()
    
    def create_combined_point_cloud_projection(self, timestamp: str):
        """创建合并的点云投影图像（ABCD四个相机拼在一起，包含原图对比）"""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib未安装，跳过点云投影可视化")
            return
            
        print(f"\n创建合并的点云投影: {timestamp}")
        
        # 查找点云文件
        lidar_dir = self.dataset_path / "lidar" / "pcd"
        pcd_files = list(lidar_dir.glob(f"{timestamp}.pcd"))
        npy_files = list(lidar_dir.glob(f"{timestamp}.npy"))
        
        if not pcd_files and not npy_files:
            print(f"未找到点云文件: {timestamp}")
            return
        
        try:
            import open3d as o3d
            
            # 加载点云
            if pcd_files:
                pcd = o3d.io.read_point_cloud(str(pcd_files[0]))
                points = np.asarray(pcd.points)
            elif npy_files:
                points = np.load(str(npy_files[0]))
            
            print(f"加载点云: {len(points)} 个点")
            
            # 收集所有相机的图像
            camera_images = {}
            image_sizes = {}
            
            for camera_name in self.camera_names:
                folder_name = self.camera_mapping[camera_name]
                image_dir = self.dataset_path / folder_name / "image_dc"
                image_files = list(image_dir.glob(f"{timestamp}.jpg"))
                
                if image_files:
                    image = cv2.imread(str(image_files[0]))
                    if image is not None:
                        camera_images[camera_name] = image
                        image_sizes[camera_name] = image.shape[:2]
                        print(f"找到 {camera_name} 图像: {image.shape[1]}x{image.shape[0]}")
                    else:
                        print(f"无法加载 {camera_name} 图像")
                        camera_images[camera_name] = None
                else:
                    print(f"未找到 {camera_name} 图像文件")
                    camera_images[camera_name] = None
            
            # 如果没有找到任何图像，返回
            if not any(img is not None for img in camera_images.values()):
                print("未找到任何相机图像")
                return
            
            # 确定统一的图像尺寸
            base_size = None
            for img in camera_images.values():
                if img is not None:
                    base_size = (img.shape[1], img.shape[0])  # (width, height)
                    break
            
            if base_size is None:
                print("无法确定图像尺寸")
                return
            
            # 创建2x4的合并图像：左侧4个原图，右侧4个点云投影
            fig, axes = plt.subplots(4, 2, figsize=(20, 32))
            
            # 相机位置布局
            camera_layout = {
                'SC_R1_Aw_CpcS_CAMR': 0,  # 第一行
                'SC_R1_Bn_CpcW_CAMR': 1,  # 第二行
                'SC_R1_Ce_CpcN_CAMR': 2,  # 第三行
                'SC_R1_Ds_CpcE_CAMR': 3   # 第四行
            }
            
            # 处理每个相机
            total_points_projected = 0
            
            for camera_name, row_idx in camera_layout.items():
                folder_name = self.camera_mapping[camera_name]
                
                if camera_images[camera_name] is not None:
                    # 使用实际图像
                    img = camera_images[camera_name]
                    if img.shape[:2] != (base_size[1], base_size[0]):
                        img = cv2.resize(img, base_size)
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # 变换点云到相机坐标系
                    points_camera = self.transform_points_to_camera(points, camera_name)
                    
                    # 投影到图像平面
                    points_2d = self.project_points_to_image(points, camera_name, (base_size[1], base_size[0]))
                    
                    # 左侧：原图
                    ax_original = axes[row_idx, 0]
                    ax_original.imshow(img_rgb)
                    ax_original.set_title(f'{folder_name} ({camera_name}) - Original', fontsize=12)
                    ax_original.axis('off')
                    
                    # 右侧：点云投影
                    ax_projection = axes[row_idx, 1]
                    ax_projection.imshow(img_rgb)
                    
                    # 绘制投影点
                    if len(points_2d) > 0:
                        # 获取有效的深度值
                        valid_depth_mask = points_camera[:, 2] > 0.1
                        if np.any(valid_depth_mask) and len(points_2d) == np.sum(valid_depth_mask):
                            depths = points_camera[valid_depth_mask, 2]
                            scatter = ax_projection.scatter(points_2d[:, 0], points_2d[:, 1], 
                                                          c=depths, cmap='viridis', s=1, alpha=0.6)
                            # 为第一行的点云投影添加colorbar
                            if row_idx == 0:
                                plt.colorbar(scatter, ax=ax_projection, label='Depth (m)')
                        else:
                            # 如果不匹配，只绘制点不显示深度
                            ax_projection.scatter(points_2d[:, 0], points_2d[:, 1], 
                                                c='red', s=1, alpha=0.6)
                        
                        total_points_projected += len(points_2d)
                        points_info = f" - {len(points_2d)} points"
                    else:
                        points_info = " - 0 points"
                    
                    ax_projection.set_title(f'{folder_name} ({camera_name}) - Point Cloud Projection{points_info}', fontsize=12)
                    ax_projection.axis('off')
                    
                    print(f"  {camera_name}: 投影 {len(points_2d)} 个点")
                    
                else:
                    # 创建黑色背景
                    black_bg = np.zeros((base_size[1], base_size[0], 3), dtype=np.uint8)
                    
                    # 左侧：原图（黑色）
                    ax_original = axes[row_idx, 0]
                    ax_original.imshow(black_bg)
                    ax_original.set_title(f'{folder_name} ({camera_name}) - No Image', fontsize=12)
                    ax_original.axis('off')
                    
                    # 右侧：点云投影（黑色）
                    ax_projection = axes[row_idx, 1]
                    ax_projection.imshow(black_bg)
                    ax_projection.set_title(f'{folder_name} ({camera_name}) - No Image', fontsize=12)
                    ax_projection.axis('off')
                    
                    print(f"  {camera_name}: 无图像")
            
            plt.tight_layout()
            
            # 保存合并图像到点云投影目录
            output_filename = f"combined_point_cloud_projection_{timestamp}.jpg"
            output_path = self.pointcloud_output_dir / output_filename
            plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
            plt.close()
            
            print(f"保存合并点云投影: {output_filename}")
            print(f"总体点云投影统计: {total_points_projected} 个点")
            
        except Exception as e:
            print(f"创建合并点云投影失败: {e}")
            import traceback
            traceback.print_exc()
    
    def run_visualization(self, timestamps: List[str]):
        """运行可视化流程"""
        print("=" * 60)
        print("3D空间转换和多模态可视化")
        print("=" * 60)
        print(f"处理 {len(timestamps)} 个样本")
        print(f"输出主目录: {self.output_dir}")
        print(f"3D标签投影目录: {self.label_output_dir}")
        print(f"点云投影目录: {self.pointcloud_output_dir}")
        
        for i, timestamp in enumerate(timestamps):
            print(f"\n{'='*40}")
            print(f"处理样本 {i+1}/{len(timestamps)}: {timestamp}")
            print(f"{'='*40}")
            
            # 合并的3D标签投影可视化
            self.create_combined_3d_label_projection(timestamp)
            
            # 合并的点云投影可视化
            self.create_combined_point_cloud_projection(timestamp)
        
        print(f"\n可视化流程完成！")
        print(f"3D标签投影结果: {self.label_output_dir}")
        print(f"点云投影结果: {self.pointcloud_output_dir}")

if __name__ == "__main__":
    data_path = "/home/tyjt/ws/02_docker/cv2-tests/dataset/2d3d4d_20250728_weiyuan"
    
    # 测试样本
    sample_timestamps = ["1753701446899737835"]
    
    visualizer = SpatialVisualizer(data_path)
    visualizer.run_visualization(sample_timestamps)