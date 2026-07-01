import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import config
from data_loader import DataLoader
from transformer import Transformer

class ProjectionVisualizer:
    def __init__(self, data_loader: DataLoader, transformer: Transformer, output_dir: Path):
        self.data_loader = data_loader
        self.transformer = transformer
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
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
            
            center = box_corners[0].astype(int)
            if 0 <= center[0] < vis_image.shape[1] and 0 <= center[1] < vis_image.shape[0]:
                cv2.putText(vis_image, str(box_idx), tuple(center), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_image
    
    def create_pointcloud_projection(self, timestamp: str):
        print(f"\n创建点云投影: {timestamp}")
        
        points = self.data_loader.load_pointcloud(timestamp)
        if points is None:
            print("无点云数据，跳过")
            return
        
        camera_images = {}
        for calib_name, folder_name in self.data_loader.config.camera_config.items():
            img = self.data_loader.load_image(folder_name, timestamp)
            if img is not None:
                camera_images[calib_name] = img
        
        if not camera_images:
            print("无相机图像，跳过")
            return
        
        base_size = next(iter(camera_images.values())).shape[1], next(iter(camera_images.values())).shape[0]
        
        fig, axes = plt.subplots(4, 2, figsize=(20, 32))
        camera_layout = {
            'SC_R1_Aw_CpcS_CAMR': 0,
            'SC_R1_Bn_CpcW_CAMR': 1, 
            'SC_R1_Ce_CpcN_CAMR': 2,
            'SC_R1_Ds_CpcE_CAMR': 3
        }
        
        total_points_projected = 0
        
        for cam_name, row_idx in camera_layout.items():
            folder_name = self.data_loader.config.camera_config[cam_name]
            
            if cam_name in camera_images:
                img = cv2.resize(camera_images[cam_name], base_size)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                axes[row_idx, 0].imshow(img_rgb)
                axes[row_idx, 0].set_title(f'{folder_name} - Original')
                axes[row_idx, 0].axis('off')
                
                axes[row_idx, 1].imshow(img_rgb)
                points_2d, depths = self.transformer.project_points_to_image(points, cam_name, (base_size[1], base_size[0]))
                
                if len(points_2d) > 0:
                    # 确保max_points配置正确使用
                    max_points = self.data_loader.config.point_cloud_config.get('max_points', 50000)
                    print(f"  {cam_name}: 原始点数 {len(points_2d)}, 最大点数限制: {max_points}")
                    
                    if max_points and len(points_2d) > max_points:
                        indices = np.random.choice(len(points_2d), max_points, replace=False)
                        points_2d = points_2d[indices]
                        depths = depths[indices]
                        print(f"  {cam_name}: 采样后点数 {len(points_2d)}")
                    
                    norm_depths = (depths - depths.min()) / (depths.max() - depths.min() + 1e-8)
                    
                    scatter = axes[row_idx, 1].scatter(
                        points_2d[:, 0], points_2d[:, 1], 
                        c=norm_depths, cmap='hot', vmin=0, vmax=1,
                        s=self.data_loader.config.point_cloud_config.get('point_size', 3),
                        alpha=self.data_loader.config.point_cloud_config.get('alpha', 0.8)
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
                black_bg = np.zeros((base_size[1], base_size[0], 3))
                axes[row_idx, 0].imshow(black_bg)
                axes[row_idx, 0].set_title(f'{folder_name} - No Image')
                axes[row_idx, 0].axis('off')
                axes[row_idx, 1].imshow(black_bg)
                axes[row_idx, 1].set_title(f'{folder_name} - No Image')
                axes[row_idx, 1].axis('off')
        
        plt.tight_layout()
        output_path = self.output_dir / f"point_cloud_projection_{timestamp}.jpg"
        plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        
        # 保存点云投影配置文件
        config_path = self.output_dir / f"point_cloud_config_{timestamp}.txt"
        with open(config_path, 'w') as f:
            f.write("Point Cloud Projection Configuration:\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total points projected: {total_points_projected}\n")
            f.write(f"Color map: hot\n")
            f.write("\nPoint Cloud Settings:\n")
            for key, value in self.data_loader.config.point_cloud_config.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nImage size: {base_size}\n")
            f.write("\nCamera mapping:\n")
            for calib_name, folder_name in self.data_loader.config.camera_config.items():
                f.write(f"  {calib_name} -> {folder_name}\n")
        
        print(f"保存点云投影: {output_path}")
        print(f"保存配置: {config_path}")
        print(f"总投影点数: {total_points_projected}")
    
    def create_label_projection(self, timestamp: str):
        print(f"\n创建3D标签投影: {timestamp}")
        
        labels = self.data_loader.load_labels(timestamp)
        if not labels:
            print("无标签数据，跳过")
            return
        
        camera_images = {}
        for calib_name, folder_name in self.data_loader.config.camera_config.items():
            img = self.data_loader.load_image(folder_name, timestamp)
            if img is not None:
                camera_images[calib_name] = img
        
        if not camera_images:
            print("无相机图像，跳过")
            return
        
        base_size = next(iter(camera_images.values())).shape[1], next(iter(camera_images.values())).shape[0]
        
        combined = np.zeros((base_size[1]*2, base_size[0]*2, 3), dtype=np.uint8)
        positions = {
            'SC_R1_Aw_CpcS_CAMR': (0, 0),
            'SC_R1_Bn_CpcW_CAMR': (base_size[0], 0), 
            'SC_R1_Ce_CpcN_CAMR': (0, base_size[1]),
            'SC_R1_Ds_CpcE_CAMR': (base_size[0], base_size[1])
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
                    
                    rotation = label.get('rotation', [0])[0]
                    corners_3d = self.transformer.create_3d_box_corners(label['box3d'], rotation)
                    corners_2d, _ = self.transformer.project_points_to_image(corners_3d, cam_name, (base_size[1], base_size[0]))
                    
                    if len(corners_2d) == 8:
                        boxes_2d.append(corners_2d)
                        colors.append(config.LABEL_COLORS.get(label.get('type', 'default'), config.LABEL_COLORS['default']))
                        total_projected += 1
                        print(f"  {cam_name}: 标签 {label_idx} 投影成功")
                    else:
                        print(f"  {cam_name}: 标签 {label_idx} 投影失败 - 得到 {len(corners_2d)} 个点")
                
                img_with_boxes = self.draw_3d_boxes(img, boxes_2d, colors)
                combined[y:y+base_size[1], x:x+base_size[0]] = img_with_boxes
                
                text = f"{self.data_loader.config.camera_config[cam_name]} - {len(boxes_2d)} objs"
                cv2.putText(combined, text, (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                print(f"  {cam_name}: 绘制 {len(boxes_2d)} 个边界框")
            else:
                combined[y:y+base_size[1], x:x+base_size[0]] = 0
                print(f"  {cam_name}: 无图像")
        
        output_path = self.output_dir / f"3d_label_projection_{timestamp}.jpg"
        cv2.imwrite(str(output_path), combined)
        
        # 保存3D标签投影配置文件
        config_path = self.output_dir / f"label_projection_config_{timestamp}.txt"
        with open(config_path, 'w') as f:
            f.write("3D Label Projection Configuration:\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total labels: {len(labels)}\n")
            f.write(f"Total projected boxes: {total_projected}\n")
            f.write(f"Image size: {base_size}\n")
            f.write("\nCamera mapping:\n")
            for calib_name, folder_name in self.data_loader.config.camera_config.items():
                f.write(f"  {calib_name} -> {folder_name}\n")
            f.write("\nLabel colors (BGR format):\n")
            for label_type, color in config.LABEL_COLORS.items():
                f.write(f"  {label_type}: {color}\n")
        
        print(f"保存3D标签投影: {output_path}")
        print(f"保存配置文件: {config_path}")
        print(f"总计投影边界框: {total_projected}/{len(labels)}")