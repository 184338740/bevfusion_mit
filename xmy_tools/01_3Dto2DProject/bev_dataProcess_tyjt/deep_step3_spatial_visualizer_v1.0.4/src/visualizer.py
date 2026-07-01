import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import config
from data_loader import DataLoader
from transformer import Transformer

class BEVVisualizer:
    def __init__(self, data_loader: DataLoader, transformer: Transformer, output_dir: Path):
        self.data_loader = data_loader
        self.transformer = transformer
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_bev_visualization(self, timestamp: str):
        print(f"\n创建BEV融合可视化: {timestamp}")
        
        points = self.data_loader.load_pointcloud(timestamp)
        labels = self.data_loader.load_labels(timestamp)
        
        fig, ax = plt.subplots(figsize=(15, 15))
        
        bev_range = 75.0
        ax.set_xlim(-bev_range, bev_range)
        ax.set_ylim(-bev_range, bev_range)
        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        grid_step = 10
        for i in range(-int(bev_range/grid_step), int(bev_range/grid_step)+1):
            ax.axhline(y=i*grid_step, color='gray', linestyle=':', alpha=0.2)
            ax.axvline(x=i*grid_step, color='gray', linestyle=':', alpha=0.2)
        
        group_pos = [0, 0]
        ax.scatter(group_pos[0], group_pos[1], color='black', s=200, marker='o', 
                  label='Group Origin', zorder=20, edgecolors='white', linewidth=2)
        ax.text(group_pos[0]+1, group_pos[1]+1, 'Group', fontsize=12, weight='bold', color='black')
        
        points_count = 0
        if points is not None and len(points) > 0:
            points_count = len(points)
            max_points = 5000
            if len(points) > max_points:
                indices = np.random.choice(len(points), max_points, replace=False)
                points_display = points[indices]
            else:
                points_display = points
            
            in_range_mask = (points_display[:, 0] >= -bev_range) & (points_display[:, 0] <= bev_range) & \
                           (points_display[:, 1] >= -bev_range) & (points_display[:, 1] <= bev_range)
            points_in_range = points_display[in_range_mask]
            
            if len(points_in_range) > 0:
                colors = points_in_range[:, 2]
                scatter = ax.scatter(points_in_range[:, 0], points_in_range[:, 1], 
                                   c=colors, cmap='viridis', s=2, alpha=0.5, label='Point Cloud', zorder=1)
                plt.colorbar(scatter, ax=ax, label='Height (m)')
        
        boxes_drawn = 0
        labels_count = 0
        
        # 收集所有出现的障碍物类型，用于图例
        obstacle_types_present = set()
        
        if labels and len(labels) > 0:
            labels_count = len(labels)
            for label_idx, label in enumerate(labels):
                if 'box3d' not in label or len(label['box3d']) != 6:
                    continue
                    
                box3d = label['box3d']
                rotation = label.get('rotation', [0])[0]
                label_type = label.get('type', 'default')
                color = config.BEV_LABEL_COLORS.get(label_type, config.BEV_LABEL_COLORS['default'])
                
                # 记录出现的障碍物类型
                obstacle_types_present.add(label_type)
                
                x, y, z, l, w, h = box3d
                if not (-bev_range <= x <= bev_range and -bev_range <= y <= bev_range):
                    continue
                
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
                
                corners_2d = corners_world[:, :2]
                for i in range(4):
                    start_point = corners_2d[i]
                    end_point = corners_2d[(i+1) % 4]
                    ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                           color=color, linewidth=2, alpha=0.8, zorder=10)
                
                front_center = (corners_2d[0] + corners_2d[1]) / 2
                center = np.mean(corners_2d, axis=0)
                arrow_vec = front_center - center
                arrow_length = min(l, w) * 0.6
                arrow_end = center + arrow_vec / np.linalg.norm(arrow_vec) * arrow_length
                
                ax.arrow(center[0], center[1], 
                        arrow_end[0]-center[0], arrow_end[1]-center[1],
                        head_width=0.5, head_length=0.8, fc=color, ec=color, alpha=0.8, zorder=10)
                
                # 功能1: 在障碍物上显示ID
                ax.text(x, y + 1, f'ID:{label_idx}', fontsize=8, color=color, 
                       ha='center', va='bottom', weight='bold', zorder=11,
                       bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8, edgecolor=color))
                
                # 显示障碍物类型
                ax.text(x, y - 1, f'{label_type}', fontsize=7, color=color,
                       ha='center', va='top', weight='bold', zorder=11)
                
                boxes_drawn += 1
        
        camera_configs = {
            'SC_R1_Aw_CpcS_CAMR': {'color': 'red', 'label': 'Cam A'},
            'SC_R1_Bn_CpcW_CAMR': {'color': 'green', 'label': 'Cam B'},  
            'SC_R1_Ce_CpcN_CAMR': {'color': 'blue', 'label': 'Cam C'},
            'SC_R1_Ds_CpcE_CAMR': {'color': 'orange', 'label': 'Cam D'}
        }
        
        cameras_found = 0
        
        if 'group2map' in self.transformer.calib_data:
            group2map_transform = self.transformer.parse_transform(self.transformer.calib_data['group2map'])
            map2group_transform = np.linalg.inv(group2map_transform)
        else:
            map2group_transform = np.eye(4)
        
        for cam_name, config_cam in camera_configs.items():
            color = config_cam['color']
            label = config_cam['label']
            
            cam_params = self.transformer.get_camera_params(cam_name)
            if cam_params is None:
                continue
            
            camera2map_transform = self.transformer.parse_transform(cam_params)
            camera_group_transform = map2group_transform @ camera2map_transform
            camera_pos_group = camera_group_transform[:3, 3]
            camera_pos_2d = [camera_pos_group[0], camera_pos_group[1]]
            
            camera_rot_group = camera_group_transform[:3, :3]
            
            forward_candidates = [
                [0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0],
            ]
            
            best_forward = None
            max_dot_product = -1
            
            for forward in forward_candidates:
                forward_world = camera_rot_group @ forward
                forward_2d = forward_world[:2]
                forward_2d_norm = forward_2d / (np.linalg.norm(forward_2d) + 1e-8)
                
                camera_to_group = -np.array(camera_pos_2d)
                camera_to_group_norm = camera_to_group / (np.linalg.norm(camera_to_group) + 1e-8)
                
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
            
            fov_h = 60
            max_fov_range = 150.0
            
            left_angle = camera_yaw - np.radians(fov_h/2)
            right_angle = camera_yaw + np.radians(fov_h/2)
            
            distance_intervals = [30, 60, 90, 120, 150]
            line_styles = ['-', '--', ':', '-.', '-']
            line_widths = [1.0, 0.8, 0.6, 0.5, 0.4]
            alphas = [0.6, 0.5, 0.4, 0.3, 0.2]
            
            for i, distance in enumerate(distance_intervals):
                theta = np.linspace(left_angle, right_angle, 30)
                fov_arc = [
                    camera_pos_2d[0] + distance * np.cos(theta),
                    camera_pos_2d[1] + distance * np.sin(theta)
                ]
                ax.plot(fov_arc[0], fov_arc[1], color=color, linewidth=line_widths[i], 
                       alpha=alphas[i], linestyle=line_styles[i], zorder=5)
                
                mid_angle = (left_angle + right_angle) / 2
                text_pos = [
                    camera_pos_2d[0] + distance * np.cos(mid_angle),
                    camera_pos_2d[1] + distance * np.sin(mid_angle)
                ]
                
                if (-bev_range <= text_pos[0] <= bev_range and 
                    -bev_range <= text_pos[1] <= bev_range):
                    
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
                           ha='center', va='center', weight='bold', zorder=6,
                           bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.9, edgecolor=color))
            
            left_end = [
                camera_pos_2d[0] + max_fov_range * np.cos(left_angle),
                camera_pos_2d[1] + max_fov_range * np.sin(left_angle)
            ]
            right_end = [
                camera_pos_2d[0] + max_fov_range * np.cos(right_angle),
                camera_pos_2d[1] + max_fov_range * np.sin(right_angle)
            ]
            
            ax.plot([camera_pos_2d[0], left_end[0]], [camera_pos_2d[1], left_end[1]], 
                   color=color, linewidth=1.2, alpha=0.5, linestyle='-', zorder=5)
            ax.plot([camera_pos_2d[0], right_end[0]], [camera_pos_2d[1], right_end[1]], 
                   color=color, linewidth=1.2, alpha=0.5, linestyle='-', zorder=5)
            
            fill_range = min(max_fov_range, bev_range * 1.5)
            theta = np.linspace(left_angle, right_angle, 30)
            fov_arc_fill = [
                camera_pos_2d[0] + fill_range * np.cos(theta),
                camera_pos_2d[1] + fill_range * np.sin(theta)
            ]
            fov_fill_x = [camera_pos_2d[0]] + list(fov_arc_fill[0]) + [camera_pos_2d[0]]
            fov_fill_y = [camera_pos_2d[1]] + list(fov_arc_fill[1]) + [camera_pos_2d[1]]
            ax.fill(fov_fill_x, fov_fill_y, color=color, alpha=0.04, zorder=2)
            
            ax.scatter(camera_pos_2d[0], camera_pos_2d[1], color=color, s=100, marker='s', 
                      label=f'{label} Camera', zorder=15, edgecolors='white', linewidth=2)
            
            arrow_length = 2
            arrow_end = [
                camera_pos_2d[0] + arrow_length * np.cos(camera_yaw),
                camera_pos_2d[1] + arrow_length * np.sin(camera_yaw)
            ]
            ax.arrow(camera_pos_2d[0], camera_pos_2d[1], 
                    arrow_end[0]-camera_pos_2d[0], arrow_end[1]-camera_pos_2d[1],
                    head_width=1, head_length=1.5, fc=color, ec=color, alpha=0.8, linewidth=2, zorder=15)
            
            cameras_found += 1
        
        # 功能2: 在左上角添加障碍物类型图例（不覆盖相机图例）
        if obstacle_types_present:
            obstacle_legend_elements = []
            for obs_type in sorted(obstacle_types_present):
                color = config.BEV_LABEL_COLORS.get(obs_type, config.BEV_LABEL_COLORS['default'])
                obstacle_legend_elements.append(plt.Line2D([0], [0], color=color, lw=4, label=obs_type))
            
            # 创建第一个图例（障碍物类型）- 放在左上角
            obstacle_legend = ax.legend(handles=obstacle_legend_elements, 
                                      loc='upper left', 
                                      title='Obstacle Types',
                                      bbox_to_anchor=(0, 1),
                                      framealpha=0.9, 
                                      fancybox=True, 
                                      shadow=True)
            
            # 将障碍物类型图例添加到图表中
            ax.add_artist(obstacle_legend)
            
            # 创建第二个图例（相机和其他）- 放在右上角
            # matplotlib会自动收集所有label，我们不需要手动创建
            ax.legend(loc='upper right', 
                     bbox_to_anchor=(1, 1),
                     framealpha=0.9,
                     fancybox=True,
                     shadow=True)
        
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
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"bev_fusion_{timestamp}.jpg"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存BEV融合可视化配置文件
        config_path = self.output_dir / f"bev_fusion_config_{timestamp}.txt"
        with open(config_path, 'w') as f:
            f.write("BEV Fusion Visualization Configuration:\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"BEV Range: ±{bev_range}m\n")
            f.write(f"FOV Range: up to 150m\n")
            f.write(f"Total points: {points_count}\n")
            f.write(f"Total labels: {labels_count}\n")
            f.write(f"Boxes drawn: {boxes_drawn}\n")
            f.write(f"Cameras found: {cameras_found}\n")
            f.write(f"Obstacle types present: {sorted(obstacle_types_present)}\n")
            f.write("\nCamera FOV settings:\n")
            f.write(f"  Horizontal FOV: {fov_h} degrees\n")
            f.write(f"  Max FOV range: {max_fov_range}m\n")
            f.write(f"  Distance intervals: {distance_intervals}\n")
            f.write("\nLabel colors:\n")
            for label_type, color in config.BEV_LABEL_COLORS.items():
                f.write(f"  {label_type}: {color}\n")
        
        print(f"保存BEV融合可视化: {output_path}")
        print(f"保存配置: {config_path}")
        print(f"点云点数: {points_count}, 标签数: {labels_count}, 相机数: {cameras_found}")
        print(f"出现的障碍物类型: {sorted(obstacle_types_present)}")