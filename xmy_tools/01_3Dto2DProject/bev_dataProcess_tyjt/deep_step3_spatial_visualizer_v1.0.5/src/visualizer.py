"""
可视化模块
版本: v1.0.5
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from config import BEVConfig, LABEL_COLORS, BEV_LABEL_COLORS
from data_loader import DataLoader
from geometry import GeometryUtils, CalibrationManager

class BaseVisualizer:
    """可视化器基类"""
    
    def __init__(self, data_loader: DataLoader, calib_manager: CalibrationManager, output_dir: Path):
        self.data_loader = data_loader
        self.calib_manager = calib_manager
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, timestamp: str, config_data: Dict):
        """保存配置文件"""
        config_path = self.output_dir / f"{self.__class__.__name__.lower()}_config_{timestamp}.txt"
        with open(config_path, 'w') as f:
            for key, value in config_data.items():
                f.write(f"{key}: {value}\n")
        print(f"保存配置: {config_path}")

class PointCloudVisualizer(BaseVisualizer):
    """点云投影可视化器"""
    
    def create_visualization(self, timestamp: str):
        """创建点云投影可视化"""
        print(f"\n创建点云投影: {timestamp}")
        
        points = self.data_loader.load_pointcloud(timestamp)
        if points is None:
            print("无点云数据，跳过")
            return
        
        camera_images = self._load_camera_images(timestamp)
        if not camera_images:
            print("无相机图像，跳过")
            return
        
        base_size = self._get_base_image_size(camera_images)
        fig, axes = plt.subplots(4, 2, figsize=(20, 32))
        
        total_points = self._plot_pointcloud_projections(points, camera_images, base_size, axes)
        
        # 保存结果和配置
        output_path = self.output_dir / f"point_cloud_projection_{timestamp}.jpg"
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        
        self._save_pointcloud_config(timestamp, total_points, base_size)
        print(f"保存点云投影: {output_path}, 总投影点数: {total_points}")
    
    def _load_camera_images(self, timestamp: str) -> Dict[str, np.ndarray]:
        """加载所有相机图像"""
        camera_images = {}
        for calib_name, folder_name in self.data_loader.config.camera_config.items():
            img = self.data_loader.load_image(folder_name, timestamp)
            if img is not None:
                camera_images[calib_name] = img
        return camera_images
    
    def _get_base_image_size(self, camera_images: Dict[str, np.ndarray]) -> Tuple[int, int]:
        """获取基础图像尺寸"""
        first_img = next(iter(camera_images.values()))
        return first_img.shape[1], first_img.shape[0]
    
    def _plot_pointcloud_projections(self, points: np.ndarray, camera_images: Dict[str, np.ndarray],
                                   base_size: Tuple[int, int], axes) -> int:
        """绘制点云投影"""
        camera_layout = {
            'SC_R1_Aw_CpcS_CAMR': 0, 'SC_R1_Bn_CpcW_CAMR': 1, 
            'SC_R1_Ce_CpcN_CAMR': 2, 'SC_R1_Ds_CpcE_CAMR': 3
        }
        
        total_points = 0
        
        for cam_name, row_idx in camera_layout.items():
            folder_name = self.data_loader.config.camera_config[cam_name]
            total_points += self._plot_single_camera_projection(points, cam_name, folder_name, 
                                                              camera_images, base_size, axes, row_idx)
        
        return total_points
    
    def _plot_single_camera_projection(self, points: np.ndarray, cam_name: str, folder_name: str,
                                     camera_images: Dict[str, np.ndarray], base_size: Tuple[int, int],
                                     axes, row_idx: int) -> int:
        """绘制单个相机的点云投影"""
        if cam_name in camera_images:
            img = cv2.resize(camera_images[cam_name], base_size)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 显示原图
            axes[row_idx, 0].imshow(img_rgb)
            axes[row_idx, 0].set_title(f'{folder_name} - Original')
            axes[row_idx, 0].axis('off')
            
            # 显示点云投影
            axes[row_idx, 1].imshow(img_rgb)
            points_2d, depths = self._project_points_for_camera(points, cam_name, base_size)
            
            points_count = 0
            if len(points_2d) > 0:
                points_2d, depths = self._apply_point_sampling(points_2d, depths)
                self._plot_points(axes[row_idx, 1], points_2d, depths, row_idx)
                points_count = len(points_2d)
                points_info = f" - {points_count} points"
            else:
                points_info = " - 0 points"
            
            axes[row_idx, 1].set_title(f'{folder_name} - Point Cloud{points_info}')
            axes[row_idx, 1].axis('off')
            return points_count
        else:
            self._plot_empty_camera(axes, row_idx, folder_name, base_size)
            return 0
    
    def _project_points_for_camera(self, points: np.ndarray, cam_name: str, 
                                 base_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """为指定相机投影点云"""
        cam_params = self.calib_manager.get_camera_params(cam_name)
        if cam_params is None:
            return np.array([]), np.array([])
        
        points_camera = self.calib_manager.transform_points_to_camera(points, cam_name)
        return GeometryUtils.project_points_to_image(points_camera, cam_params, 
                                                   (base_size[1], base_size[0]))
    
    def _apply_point_sampling(self, points_2d: np.ndarray, depths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用点云采样"""
        max_points = self.data_loader.config.point_cloud_max_points
        if max_points and len(points_2d) > max_points:
            indices = np.random.choice(len(points_2d), max_points, replace=False)
            return points_2d[indices], depths[indices]
        return points_2d, depths
    
    def _plot_points(self, ax, points_2d: np.ndarray, depths: np.ndarray, row_idx: int):
        """绘制点云"""
        norm_depths = (depths - depths.min()) / (depths.max() - depths.min() + 1e-8)
        scatter = ax.scatter(
            points_2d[:, 0], points_2d[:, 1], 
            c=norm_depths, cmap='hot', vmin=0, vmax=1,
            s=self.data_loader.config.point_size,
            alpha=self.data_loader.config.point_alpha
        )
        
        if row_idx == 0:
            plt.colorbar(scatter, ax=ax, label='Depth (normalized)')
    
    def _plot_empty_camera(self, axes, row_idx: int, folder_name: str, base_size: Tuple[int, int]):
        """绘制空相机占位"""
        black_bg = np.zeros((base_size[1], base_size[0], 3))
        axes[row_idx, 0].imshow(black_bg)
        axes[row_idx, 0].set_title(f'{folder_name} - No Image')
        axes[row_idx, 0].axis('off')
        axes[row_idx, 1].imshow(black_bg)
        axes[row_idx, 1].set_title(f'{folder_name} - No Image')
        axes[row_idx, 1].axis('off')
    
    def _save_pointcloud_config(self, timestamp: str, total_points: int, base_size: Tuple[int, int]):
        """保存点云配置"""
        config_data = {
            "Point Cloud Projection Configuration": "",
            "Timestamp": timestamp,
            "Total points projected": total_points,
            "Color map": "hot",
            "Max points": self.data_loader.config.point_cloud_max_points,
            "Point size": self.data_loader.config.point_size,
            "Alpha": self.data_loader.config.point_alpha,
            "Image size": base_size
        }
        self.save_config(timestamp, config_data)

class LabelVisualizer(BaseVisualizer):
    """3D标签投影可视化器"""
    
    def create_visualization(self, timestamp: str):
        """创建3D标签投影可视化"""
        print(f"\n创建3D标签投影: {timestamp}")
        
        labels = self.data_loader.load_labels(timestamp)
        if not labels:
            print("无标签数据，跳过")
            return
        
        camera_images = self._load_camera_images(timestamp)
        if not camera_images:
            print("无相机图像，跳过")
            return
        
        base_size = self._get_base_image_size(camera_images)
        combined_image, total_projected = self._create_combined_image(labels, camera_images, base_size)
        
        # 保存结果
        output_path = self.output_dir / f"3d_label_projection_{timestamp}.jpg"
        cv2.imwrite(str(output_path), combined_image)
        self._save_label_config(timestamp, len(labels), total_projected, base_size)
        print(f"保存3D标签投影: {output_path}, 投影边界框: {total_projected}/{len(labels)}")
    
    def _load_camera_images(self, timestamp: str) -> Dict[str, np.ndarray]:
        """加载所有相机图像"""
        camera_images = {}
        for calib_name, folder_name in self.data_loader.config.camera_config.items():
            img = self.data_loader.load_image(folder_name, timestamp)
            if img is not None:
                camera_images[calib_name] = img
        return camera_images
    
    def _get_base_image_size(self, camera_images: Dict[str, np.ndarray]) -> Tuple[int, int]:
        """获取基础图像尺寸"""
        first_img = next(iter(camera_images.values()))
        return first_img.shape[1], first_img.shape[0]
    
    def _create_combined_image(self, labels: List[Dict], camera_images: Dict[str, np.ndarray],
                             base_size: Tuple[int, int]) -> Tuple[np.ndarray, int]:
        """创建合并图像"""
        positions = {
            'SC_R1_Aw_CpcS_CAMR': (0, 0),
            'SC_R1_Bn_CpcW_CAMR': (base_size[0], 0), 
            'SC_R1_Ce_CpcN_CAMR': (0, base_size[1]),
            'SC_R1_Ds_CpcE_CAMR': (base_size[0], base_size[1])
        }
        
        combined = np.zeros((base_size[1]*2, base_size[0]*2, 3), dtype=np.uint8)
        total_projected = 0
        
        for cam_name, (x, y) in positions.items():
            if cam_name in camera_images:
                img = cv2.resize(camera_images[cam_name], base_size)
                boxes_2d, colors, projected = self._project_labels_for_camera(labels, cam_name, base_size)
                
                img_with_boxes = self._draw_3d_boxes(img, boxes_2d, colors)
                combined[y:y+base_size[1], x:x+base_size[0]] = img_with_boxes
                
                # 添加文字说明
                folder_name = self.data_loader.config.camera_config[cam_name]
                text = f"{folder_name} - {len(boxes_2d)} objs"
                cv2.putText(combined, text, (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                
                total_projected += projected
            else:
                combined[y:y+base_size[1], x:x+base_size[0]] = 0
        
        return combined, total_projected
    
    def _project_labels_for_camera(self, labels: List[Dict], cam_name: str, 
                                 base_size: Tuple[int, int]) -> Tuple[List[np.ndarray], List[tuple], int]:
        """为指定相机投影标签"""
        boxes_2d, colors = [], []
        projected_count = 0
        
        for label in labels:
            if 'box3d' not in label or len(label['box3d']) != 6:
                continue
            
            rotation = label.get('rotation', [0])[0]
            corners_3d = GeometryUtils.create_3d_box_corners(label['box3d'], rotation)
            
            cam_params = self.calib_manager.get_camera_params(cam_name)
            if cam_params:
                corners_camera = self.calib_manager.transform_points_to_camera(corners_3d, cam_name)
                corners_2d, _ = GeometryUtils.project_points_to_image(
                    corners_camera, cam_params, (base_size[1], base_size[0])
                )
                
                if len(corners_2d) == 8:
                    boxes_2d.append(corners_2d)
                    label_type = label.get('type', 'default')
                    colors.append(LABEL_COLORS.get(label_type, LABEL_COLORS['default']))
                    projected_count += 1
        
        return boxes_2d, colors, projected_count
    
    def _draw_3d_boxes(self, image: np.ndarray, boxes_2d: List[np.ndarray], colors: List[tuple]) -> np.ndarray:
        """在图像上绘制3D边界框"""
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
            
            # 添加标签编号
            center = box_corners[0].astype(int)
            if 0 <= center[0] < vis_image.shape[1] and 0 <= center[1] < vis_image.shape[0]:
                cv2.putText(vis_image, str(box_idx), tuple(center), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_image
    
    def _save_label_config(self, timestamp: str, total_labels: int, total_projected: int, base_size: Tuple[int, int]):
        """保存标签配置"""
        config_data = {
            "3D Label Projection Configuration": "",
            "Timestamp": timestamp,
            "Total labels": total_labels,
            "Total projected boxes": total_projected,
            "Image size": base_size
        }
        self.save_config(timestamp, config_data)

class BEVVisualizer(BaseVisualizer):
    """BEV融合可视化器"""
    
    def create_visualization(self, timestamp: str):
        """创建BEV融合可视化"""
        print(f"\n创建BEV融合可视化: {timestamp}")
        
        points = self.data_loader.load_pointcloud(timestamp)
        labels = self.data_loader.load_labels(timestamp)
        
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # 设置BEV视图
        self._setup_bev_axes(ax)
        
        # 绘制各个元素
        points_count = self._draw_pointcloud(ax, points)
        boxes_drawn, labels_count, obstacle_types = self._draw_3d_boxes(ax, labels)
        cameras_found = self._draw_camera_fov(ax)
        
        # 添加图例和标题
        self._add_legends_and_title(ax, timestamp, points_count, labels_count, cameras_found, obstacle_types)
        
        # 保存结果
        output_path = self.output_dir / f"bev_fusion_{timestamp}.jpg"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self._save_bev_config(timestamp, points_count, labels_count, boxes_drawn, cameras_found, obstacle_types)
        print(f"保存BEV融合可视化: {output_path}")
    
    def _setup_bev_axes(self, ax):
        """设置BEV坐标轴"""
        bev_range = self.data_loader.config.bev_range
        ax.set_xlim(-bev_range, bev_range)
        ax.set_ylim(-bev_range, bev_range)
        
        # 绘制坐标轴和网格
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        grid_step = 10
        for i in range(-int(bev_range/grid_step), int(bev_range/grid_step)+1):
            ax.axhline(y=i*grid_step, color='gray', linestyle=':', alpha=0.2)
            ax.axvline(x=i*grid_step, color='gray', linestyle=':', alpha=0.2)
        
        # 绘制group点位置
        group_pos = [0, 0]
        ax.scatter(group_pos[0], group_pos[1], color='black', s=200, marker='o', 
                  label='Group Origin', zorder=20, edgecolors='white', linewidth=2)
        ax.text(group_pos[0]+1, group_pos[1]+1, 'Group', fontsize=12, weight='bold', color='black')
    
    def _draw_pointcloud(self, ax, points: Optional[np.ndarray]) -> int:
        """绘制点云"""
        if points is None or len(points) == 0:
            return 0
        
        points_count = len(points)
        bev_range = self.data_loader.config.bev_range
        
        # 采样点云
        max_points = 5000
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
                               c=colors, cmap='viridis', s=2, alpha=0.5, label='Point Cloud', zorder=1)
            plt.colorbar(scatter, ax=ax, label='Height (m)')
        
        return points_count
    
    def _draw_3d_boxes(self, ax, labels: List[Dict]) -> Tuple[int, int, set]:
        """绘制3D边界框"""
        if not labels:
            return 0, 0, set()
        
        boxes_drawn = 0
        labels_count = len(labels)
        obstacle_types = set()
        bev_range = self.data_loader.config.bev_range
        
        for label_idx, label in enumerate(labels):
            if 'box3d' not in label or len(label['box3d']) != 6:
                continue
                
            box3d = label['box3d']
            rotation = label.get('rotation', [0])[0]
            label_type = label.get('type', 'default')
            color = BEV_LABEL_COLORS.get(label_type, BEV_LABEL_COLORS['default'])
            
            # 记录障碍物类型
            obstacle_types.add(label_type)
            
            x, y, z, l, w, h = box3d
            if not (-bev_range <= x <= bev_range and -bev_range <= y <= bev_range):
                continue
            
            # 创建并绘制边界框
            corners_2d = self._create_box_corners_2d(box3d, rotation)
            self._draw_single_box(ax, corners_2d, color)
            
            # 绘制朝向箭头
            self._draw_box_orientation(ax, corners_2d, l, w, color)
            
            # 添加ID和类型标签
            self._add_box_labels(ax, x, y, label_idx, label_type, color)
            
            boxes_drawn += 1
        
        return boxes_drawn, labels_count, obstacle_types
    
    def _create_box_corners_2d(self, box3d: List[float], rotation: float) -> np.ndarray:
        """创建2D边界框角点"""
        x, y, z, l, w, h = box3d
        
        corners_local = np.array([
            [l/2, w/2, 0], [l/2, -w/2, 0], [-l/2, -w/2, 0], [-l/2, w/2, 0]
        ])
        
        rot_matrix = np.array([
            [np.cos(rotation), -np.sin(rotation), 0],
            [np.sin(rotation), np.cos(rotation), 0],
            [0, 0, 1]
        ])
        
        corners_rotated = corners_local @ rot_matrix.T
        corners_world = corners_rotated + np.array([x, y, 0])
        
        return corners_world[:, :2]
    
    def _draw_single_box(self, ax, corners_2d: np.ndarray, color: str):
        """绘制单个边界框"""
        for i in range(4):
            start_point = corners_2d[i]
            end_point = corners_2d[(i+1) % 4]
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                   color=color, linewidth=2, alpha=0.8, zorder=10)
    
    def _draw_box_orientation(self, ax, corners_2d: np.ndarray, l: float, w: float, color: str):
        """绘制边界框朝向箭头"""
        front_center = (corners_2d[0] + corners_2d[1]) / 2
        center = np.mean(corners_2d, axis=0)
        arrow_vec = front_center - center
        arrow_length = min(l, w) * 0.6
        arrow_end = center + arrow_vec / np.linalg.norm(arrow_vec) * arrow_length
        
        ax.arrow(center[0], center[1], 
                arrow_end[0]-center[0], arrow_end[1]-center[1],
                head_width=0.5, head_length=0.8, fc=color, ec=color, alpha=0.8, zorder=10)
    
    def _add_box_labels(self, ax, x: float, y: float, label_idx: int, label_type: str, color: str):
        """添加边界框标签"""
        # 显示ID
        ax.text(x, y + 1, f'ID:{label_idx}', fontsize=8, color=color, 
               ha='center', va='bottom', weight='bold', zorder=11,
               bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8, edgecolor=color))
        
        # 显示类型
        ax.text(x, y - 1, f'{label_type}', fontsize=7, color=color,
               ha='center', va='top', weight='bold', zorder=11)
    
    def _draw_camera_fov(self, ax) -> int:
        """绘制相机FOV"""
        cameras_found = 0
        
        # 获取group变换矩阵
        group2map_transform = self.calib_manager.get_group_transform()
        map2group_transform = np.linalg.inv(group2map_transform)
        
        for cam_name in self.data_loader.config.camera_config.keys():
            camera_transform = self.calib_manager.get_camera_transform(cam_name)
            if camera_transform is None:
                continue
            
            # 计算相机在group坐标系中的位置和朝向
            camera_pos_2d, camera_yaw = self._calculate_camera_pose(camera_transform, map2group_transform)
            if camera_pos_2d is None:
                continue
            
            # 绘制相机FOV
            self._draw_single_camera_fov(ax, cam_name, camera_pos_2d, camera_yaw)
            cameras_found += 1
        
        return cameras_found
    
    def _calculate_camera_pose(self, camera_transform: np.ndarray, map2group_transform: np.ndarray) -> Tuple[Optional[Tuple[float, float]], float]:
        """计算相机位置和朝向"""
        camera_group_transform = map2group_transform @ camera_transform
        camera_pos_group = camera_group_transform[:3, 3]
        camera_pos_2d = (camera_pos_group[0], camera_pos_group[1])
        
        camera_rot_group = camera_group_transform[:3, :3]
        
        # 计算相机朝向
        forward_candidates = [[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0]]
        best_forward = None
        max_dot_product = -1
        
        for forward in forward_candidates:
            forward_world = camera_rot_group @ forward
            forward_2d = forward_world[:2]
            if np.linalg.norm(forward_2d) < 1e-8:
                continue
                
            forward_2d_norm = forward_2d / np.linalg.norm(forward_2d)
            camera_to_group = -np.array(camera_pos_2d)
            
            if np.linalg.norm(camera_to_group) > 1e-8:
                camera_to_group_norm = camera_to_group / np.linalg.norm(camera_to_group)
                dot_product = np.dot(forward_2d_norm, camera_to_group_norm)
                if dot_product > max_dot_product:
                    max_dot_product = dot_product
                    best_forward = forward_world
        
        if best_forward is not None:
            forward_2d = best_forward[:2]
            camera_yaw = np.arctan2(forward_2d[1], forward_2d[0])
        else:
            camera_to_group = np.array(camera_pos_2d)
            if np.linalg.norm(camera_to_group) > 0.1:
                camera_yaw = np.arctan2(camera_to_group[1], camera_to_group[0]) + np.pi
            else:
                camera_yaw = 0
        
        return camera_pos_2d, camera_yaw
    
    def _draw_single_camera_fov(self, ax, cam_name: str, camera_pos_2d: Tuple[float, float], camera_yaw: float):
        """绘制单个相机的FOV"""
        color = self.data_loader.config.camera_colors.get(cam_name, 'red')
        label = self.data_loader.config.camera_labels.get(cam_name, 'Cam')
        
        fov_h = self.data_loader.config.fov_angle
        max_fov_range = self.data_loader.config.fov_range
        
        left_angle = camera_yaw - np.radians(fov_h/2)
        right_angle = camera_yaw + np.radians(fov_h/2)
        
        # 绘制FOV弧线
        distance_intervals = [30, 60, 90, 120, 150]
        for i, distance in enumerate(distance_intervals):
            theta = np.linspace(left_angle, right_angle, 30)
            fov_arc_x = camera_pos_2d[0] + distance * np.cos(theta)
            fov_arc_y = camera_pos_2d[1] + distance * np.sin(theta)
            
            ax.plot(fov_arc_x, fov_arc_y, color=color, linewidth=1.0-i*0.1, 
                   alpha=0.6-i*0.1, linestyle=['-', '--', ':', '-.', '-'][i], zorder=5)
        
        # 绘制FOV边界射线
        left_end = (
            camera_pos_2d[0] + max_fov_range * np.cos(left_angle),
            camera_pos_2d[1] + max_fov_range * np.sin(left_angle)
        )
        right_end = (
            camera_pos_2d[0] + max_fov_range * np.cos(right_angle),
            camera_pos_2d[1] + max_fov_range * np.sin(right_angle)
        )
        
        ax.plot([camera_pos_2d[0], left_end[0]], [camera_pos_2d[1], left_end[1]], 
               color=color, linewidth=1.2, alpha=0.5, zorder=5)
        ax.plot([camera_pos_2d[0], right_end[0]], [camera_pos_2d[1], right_end[1]], 
               color=color, linewidth=1.2, alpha=0.5, zorder=5)
        
        # 绘制相机位置
        ax.scatter(camera_pos_2d[0], camera_pos_2d[1], color=color, s=100, marker='s', 
                  label=f'{label} Camera', zorder=15, edgecolors='white', linewidth=2)
        
        # 绘制相机朝向箭头
        arrow_length = 2
        arrow_end = (
            camera_pos_2d[0] + arrow_length * np.cos(camera_yaw),
            camera_pos_2d[1] + arrow_length * np.sin(camera_yaw)
        )
        ax.arrow(camera_pos_2d[0], camera_pos_2d[1], 
                arrow_end[0]-camera_pos_2d[0], arrow_end[1]-camera_pos_2d[1],
                head_width=1, head_length=1.5, fc=color, ec=color, alpha=0.8, linewidth=2, zorder=15)
    
    def _add_legends_and_title(self, ax, timestamp: str, points_count: int, labels_count: int, 
                              cameras_found: int, obstacle_types: set):
        """添加图例和标题"""
        # 添加障碍物类型图例
        if obstacle_types:
            obstacle_elements = []
            for obs_type in sorted(obstacle_types):
                color = BEV_LABEL_COLORS.get(obs_type, BEV_LABEL_COLORS['default'])
                obstacle_elements.append(plt.Line2D([0], [0], color=color, lw=4, label=obs_type))
            
            obstacle_legend = ax.legend(handles=obstacle_elements, 
                                      loc='upper left', 
                                      title='Obstacle Types',
                                      bbox_to_anchor=(0, 1),
                                      framealpha=0.9)
            ax.add_artist(obstacle_legend)
        
        # 添加其他图例
        ax.legend(loc='upper right', 
                 bbox_to_anchor=(1, 1),
                 framealpha=0.9)
        
        # 设置标题和标签
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        title_parts = []
        if points_count > 0:
            title_parts.append(f"Points: {points_count}")
        if labels_count > 0:
            title_parts.append(f"Labels: {labels_count}")
        if cameras_found > 0:
            title_parts.append(f"Cameras: {cameras_found}")
        
        title = f'BEV Fusion Visualization - {timestamp}'
        if title_parts:
            title += f'\n({", ".join(title_parts)})'
            
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _save_bev_config(self, timestamp: str, points_count: int, labels_count: int, 
                        boxes_drawn: int, cameras_found: int, obstacle_types: set):
        """保存BEV配置"""
        config_data = {
            "BEV Fusion Visualization Configuration": "",
            "Timestamp": timestamp,
            "BEV Range": f"±{self.data_loader.config.bev_range}m",
            "FOV Range": f"up to {self.data_loader.config.fov_range}m",
            "Total points": points_count,
            "Total labels": labels_count,
            "Boxes drawn": boxes_drawn,
            "Cameras found": cameras_found,
            "Obstacle types present": sorted(obstacle_types),
            "FOV angle": f"{self.data_loader.config.fov_angle} degrees"
        }
        self.save_config(timestamp, config_data)