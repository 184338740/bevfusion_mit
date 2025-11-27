#!/usr/bin/env python3
"""
Step2-3: nuscenes数据可视化验证工具 - 修复输出路径和类别映射
版本: step2_3_visOfficeRender-v6.0-图像使用绝对路径.py
功能: 修复文字拥挤、NoneType解包错误，添加单相机可视化选项
"""

import os
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import cv2
import json
import gc
import matplotlib.patches as patches
from matplotlib import colors as mcolors



# ==================== 配置参数 ====================
Mode = "Local"
if Mode == "Local":
    NUSC_ROOT = "./output/step1/nuscenes_tyjt"
    OUTPUT_DIR = "./output/step2_3_vis"
    NUSC_VERSION = "v1.0-tyjt"
    # # 指定需要处理的数据包名称列表
    # DATA_PACKAGES_TO_PROCESS = [
    #     "2d3d4d_20250728_weiyuan", 
    #     "2d3d4d_20250913_weiyuan"
    # ]
elif Mode == "A100":
    # NUSC_ROOT = "/cephfsdata/users/mingyuan/ws/01/00_RawData"
    NUSC_ROOT = "./output/step1-v6/nuscenes_tyjt"
    OUTPUT_DIR = "./output/step2_3_vis-v6"
    NUSC_VERSION = "v1.0-tyjt"
    # DATA_PACKAGES_TO_PROCESS = [
    #     "2d3d_20250114",
    #     "2d3d4d_20241218",
    #     "2d3d4d_20250618_weiyuan",
    #     "2d3d4d_20250728_weiyuan",
    #     "2d3d_20250218",
    #     "2d3d4d_20250117",
    #     "2d3d4d_20250218",
    #     "2d3d4d_20250403",
    #     "2d3d4d_20250708_weiyuan",
    #     "2d3d_20250221",
    #     "2d3d4d_20250213",
    #     "2d3d4d_20241122_wuxi",
    #     "2d3d4d_20250408",
    # ]
else:
    print(f"❌【Error】: TYJT_ROOT & DATA_PACKAGES_TO_PROCESS: None")




# 控制处理的样本数量 - 修改为采样策略
SAMPLING_STRATEGY = "per_scene"  # 可选: "per_scene"（每个场景MAX_SAMPLES）, "uniform"（全量等距采样）
MAX_SAMPLES_PER_SCENE = 5        # 每个场景最多处理的样本数量

# 类别映射回标准nuscenes类别用于可视化
CATEGORY_MAPPING_VIS = {
    # ============ 带tyjt前缀版本
    "tyjt.car": "vehicle.car",
    "tyjt.truck": "vehicle.truck", 
    "tyjt.construction_truck": "vehicle.construction",
    "tyjt.van": "vehicle.car",
    "tyjt.bus": "vehicle.bus",
    "tyjt.robot": "vehicle.construction",
    "tyjt.pedestrian": "human.pedestrian.adult", 
    "tyjt.cyclist": "vehicle.motorcycle",
    "tyjt.bicycle": "vehicle.bicycle",
    "tyjt.tricycle": "vehicle.motorcycle",
    "tyjt.tricyclist": "human.pedestrian.adult",
    "tyjt.trolley": "vehicle.trailer",
    "tyjt.cone": "movable_object.trafficcone",
    "tyjt.barrier": "movable_object.barrier",
    "tyjt.other": "movable_object.debris",
    # =================== 不带前缀版本
    "car": "vehicle.car",
    "truck": "vehicle.truck", 
    "construction_truck": "vehicle.construction",
    "van": "vehicle.car",
    "bus": "vehicle.bus",
    "robot": "vehicle.construction",
    "pedestrian": "human.pedestrian.adult", 
    "cyclist": "vehicle.motorcycle",
    "bicycle": "vehicle.bicycle",
    "tricycle": "vehicle.motorcycle",
    "tricyclist": "human.pedestrian.adult",
    "trolley": "vehicle.trailer",
    "cone": "movable_object.trafficcone",
    "barrier": "movable_object.barrier",
    "other": "movable_object.debris"
}

# 类别颜色映射 - 解决文字拥挤问题
CATEGORY_COLORS = {
    "vehicle.car": "red",
    "vehicle.truck": "blue",
    "vehicle.construction": "green",
    "vehicle.bus": "purple",
    "vehicle.motorcycle": "orange",
    "vehicle.bicycle": "cyan",
    "vehicle.trailer": "brown",
    "human.pedestrian.adult": "magenta",
    "movable_object.trafficcone": "yellow",
    "movable_object.barrier": "pink",
    "movable_object.debris": "gray"
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'render_sample_official': False,        # 暂时关闭官方渲染（有类别问题）
    'render_sample_data_official': False,   # 暂时关闭官方传感器数据渲染
    'render_pointcloud_projection': True,   # 点云投影
    'render_bev_views': True,               # BEV视图
    'debug_calibration_info': True,         # 标定信息调试
    'test_projection_accuracy': True,       # 投影准确性测试
    'test_annotation_projection': True,     # 标注投影测试
    'custom_visualization': True,           # 自定义可视化（修复类别问题）
    'save_individual_camera_views': False,   # 新增：保存单个相机视图
}

# ==================== 工具函数 ====================
def quaternion_to_rotation_matrix(q):
    """四元数转旋转矩阵 [w, x, y, z]"""
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])

def create_output_folders():
    """创建输出文件夹结构"""
    print("\n📁 创建输出文件夹结构")
    print("=" * 50)
    
    folders = {
        'custom_samples': '自定义样本渲染',
        'pointcloud_projections': '点云投影',
        'bev_views': 'BEV视图',
        'calibration_debug': '标定调试信息',
        'annotation_projection': '标注投影测试',
        'summary_reports': '汇总报告'
    }
    
    for folder_name, description in folders.items():
        folder_path = Path(OUTPUT_DIR) / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 创建: {folder_name} - {description}")
    
    camera_channels = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK']
    for cam in camera_channels:
        (Path(OUTPUT_DIR) / "pointcloud_projections" / cam).mkdir(parents=True, exist_ok=True)
        (Path(OUTPUT_DIR) / "annotation_projection" / cam).mkdir(parents=True, exist_ok=True)
        (Path(OUTPUT_DIR) / "calibration_debug" / cam).mkdir(parents=True, exist_ok=True)
        (Path(OUTPUT_DIR) / "custom_samples" / cam).mkdir(parents=True, exist_ok=True)
    
    print("✅ 所有文件夹创建完成")
    return folders

def transform_points_to_camera(nusc, points_3d, camera_token):
    """将点从global坐标系转换到camera坐标系"""
    try:
        points = points_3d.copy()
        if points.shape[0] == 3:
            points = np.vstack([points, np.ones(points.shape[1])])
        
        cam_data = nusc.get('sample_data', camera_token)
        cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
        
        ego_rotation = quaternion_to_rotation_matrix(ego_pose['rotation'])
        ego_translation = np.array(ego_pose['translation'])
        
        global_from_ego = np.eye(4)
        global_from_ego[:3, :3] = ego_rotation
        global_from_ego[:3, 3] = ego_translation
        
        cam_rotation = quaternion_to_rotation_matrix(cam_calib['rotation'])
        cam_translation = np.array(cam_calib['translation'])
        
        ego_from_cam = np.eye(4)
        ego_from_cam[:3, :3] = cam_rotation
        ego_from_cam[:3, 3] = cam_translation
        
        global_from_cam = global_from_ego @ ego_from_cam
        cam_from_global = np.linalg.inv(global_from_cam)
        
        points_camera = cam_from_global @ points
        return points_camera[:3, :]
        
    except Exception as e:
        print(f"坐标转换失败: {e}")
        return points_3d

def get_yaw_from_quaternion(q):
    """从四元数提取yaw角"""
    w, x, y, z = q
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw

# ==================== 标注框投影函数 ====================
def get_3dbox_corners(ann):
    """获取3D框的8个角点"""
    w, l, h = ann['size']
    
    # 3D框的8个角点（在物体坐标系）
    corners = np.array([
        [l/2, w/2, h/2],   # 前右上
        [l/2, w/2, -h/2],  # 前右下  
        [l/2, -w/2, h/2],  # 前左上
        [l/2, -w/2, -h/2], # 前左下
        [-l/2, w/2, h/2],  # 后右上
        [-l/2, w/2, -h/2], # 后右下
        [-l/2, -w/2, h/2], # 后左上
        [-l/2, -w/2, -h/2] # 后左下
    ]).T
    
    # 应用旋转
    rotation = quaternion_to_rotation_matrix(ann['rotation'])
    corners = rotation @ corners
    
    # 应用平移
    corners = corners + np.array(ann['translation']).reshape(3, 1)
    
    return corners

def project_3d_to_2d(nusc, points_3d, camera_token):
    """将3D点投影到2D图像"""
    try:
        cam_data = nusc.get('sample_data', camera_token)
        cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        
        if 'camera_intrinsic' not in cam_calib:
            return None
            
        intrinsic = np.array(cam_calib['camera_intrinsic'])
        
        # 坐标系转换: global -> camera
        points_camera = transform_points_to_camera(nusc, points_3d, camera_token)
        
        # 投影到图像平面
        points_2d_homo = intrinsic @ points_camera
        points_2d = points_2d_homo[:2, :] / points_2d_homo[2, :]
        
        return points_2d
        
    except Exception as e:
        print(f"3D投影失败: {e}")
        return None

def is_bbox_in_image(corners_2d, image_width=1920, image_height=1080):
    """检查2D边界框是否在图像内 - 带边界检查"""
    x_coords = corners_2d[0, :]
    y_coords = corners_2d[1, :]
    
    # 检查是否有至少一个角点在图像内（带边界容差）
    margin = 50  # 边界容差
    in_image = np.any((x_coords >= -margin) & (x_coords < image_width + margin) & 
                     (y_coords >= -margin) & (y_coords < image_height + margin))
    return in_image

def get_category_color(category_name):
    """根据类别名称获取颜色"""
    mapped_category = CATEGORY_MAPPING_VIS.get(category_name, category_name)
    return CATEGORY_COLORS.get(mapped_category, "white")

def draw_2d_bbox(ax, corners_2d, ann):
    """在图像上绘制2D边界框 - 优化文字显示"""
    try:
        # 计算2D边界框的包围盒
        x_min, x_max = np.min(corners_2d[0, :]), np.max(corners_2d[0, :])
        y_min, y_max = np.min(corners_2d[1, :]), np.max(corners_2d[1, :])
        
        width = x_max - x_min
        height = y_max - y_min
        
        # 跳过太小的框
        if width < 10 or height < 10:
            return
        
        # 获取类别颜色
        category = ann['category_name']
        display_category = CATEGORY_MAPPING_VIS.get(category, category)
        bbox_color = get_category_color(category)
        
        # 创建矩形框
        rect = patches.Rectangle((x_min, y_min), width, height, 
                               linewidth=1.5, edgecolor=bbox_color, facecolor='none')
        ax.add_patch(rect)
        
        # 优化文字标签 - 只在框足够大时显示
        if width > 50 and height > 30:
            # 简化类别名称显示
            short_name = display_category.split('.')[-1]
            label_text = short_name
            
            # 添加文字背景
            ax.text(x_min, y_min - 5, label_text, 
                   fontsize=8, color=bbox_color, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
        
    except Exception as e:
        print(f"绘制2D框失败: {e}")


def render_annotations_on_image(nusc, sample_token, camera_channel, ax):
    """在图像上绘制3D标注的2D投影 - 使用英文"""
    try:
        sample = nusc.get('sample', sample_token)
        camera_token = sample['data'][camera_channel]
        
        annotation_count = 0
        problematic_annotations = 0
        
        for ann_token in sample['anns']:
            try:
                ann = nusc.get('sample_annotation', ann_token)
                
                # 获取3D框的8个角点
                corners_3d = get_3dbox_corners(ann)
                
                # 投影到2D图像
                corners_2d = project_3d_to_2d(nusc, corners_3d, camera_token)
                
                if corners_2d is not None:
                    # 严格的坐标有效性检查
                    valid_coords = True
                    
                    # 检查坐标是否有限且合理
                    if not np.isfinite(corners_2d).all():
                        valid_coords = False
                    
                    # 检查坐标范围是否合理（考虑图像边界）
                    elif np.any(corners_2d < -1000) or np.any(corners_2d > 3000):
                        valid_coords = False
                    
                    # 检查是否在图像内
                    elif not is_bbox_in_image(corners_2d):
                        valid_coords = False
                    
                    if valid_coords:
                        # 绘制边界框
                        draw_2d_bbox(ax, corners_2d, ann)
                        annotation_count += 1
                    else:
                        problematic_annotations += 1
                        
            except Exception as e:
                problematic_annotations += 1
                continue
        
        if problematic_annotations > 0:
            print(f"      Warning {camera_channel}: Skipped {problematic_annotations} problematic annotations")
            
        return annotation_count
                
    except Exception as e:
        print(f"    Annotation rendering failed: {e}")
        return 0



# ==================== 可视化函数 ====================
def render_single_camera_view(nusc, sample_token, camera_channel, output_path):
    """渲染单个相机视图 - 使用英文"""
    try:
        sample = nusc.get('sample', sample_token)
        camera_token = sample['data'][camera_channel]
        
        img_path = nusc.get_sample_data_path(camera_token)
        img = cv2.imread(img_path)
        if img is None:
            print(f"   Cannot load image: {img_path}")
            return False
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.imshow(img)
        
        # 绘制标注框
        annotation_count = render_annotations_on_image(nusc, sample_token, camera_channel, ax)
        
        ax.set_title(f'{camera_channel} - {annotation_count} Annotations', fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   Saved single camera view: {output_path}")
        return True
        
    except Exception as e:
        print(f"   Single camera view failed: {e}")
        plt.close()
        return False

def render_custom_sample_visualization(nusc, sample_token, output_path):
    """修复版自定义样本可视化 - 解决文字显示问题"""
    try:
        sample = nusc.get('sample', sample_token)
        
        # 使用简单的2x3布局，增加整体图形尺寸
        fig, axes = plt.subplots(2, 3, figsize=(22, 14))  # 增加图形尺寸
        axes = axes.flatten()
        
        camera_channels = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK']
        
        # 调试信息：检查样本数据
        print(f"    Checking sample data:")
        print(f"      Sample token: {sample_token}")
        print(f"      Annotation count: {len(sample['anns'])}")
        print(f"      Sensor data: {list(sample['data'].keys())}")
        
        for i, cam_channel in enumerate(camera_channels):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            if cam_channel in sample['data']:
                cam_token = sample['data'][cam_channel]
                
                try:
                    # 获取图像路径
                    img_path = nusc.get_sample_data_path(cam_token)
                    print(f"      Camera {cam_channel}: {img_path}")
                    
                    # 加载图像
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # 显示图像
                        ax.imshow(img)
                        
                        # 绘制标注框 - 使用修复后的函数
                        annotation_count = render_annotations_on_image_fixed(nusc, sample_token, cam_channel, ax, img.shape[1], img.shape[0])
                        
                        ax.set_title(f'{cam_channel}\n({annotation_count} bboxes)', fontsize=10)
                        ax.axis('off')
                        
                        print(f"      OK {cam_channel}: Image loaded, {annotation_count} annotations")
                        
                    else:
                        print(f"      FAIL {cam_channel}: Image load failed")
                        ax.text(0.5, 0.5, f'Load Failed\n{cam_channel}', 
                               transform=ax.transAxes, ha='center', va='center')
                        ax.set_title(f'{cam_channel} - Failed', fontsize=10)
                        
                except Exception as e:
                    print(f"      ERROR {cam_channel}: Processing failed - {e}")
                    ax.text(0.5, 0.5, f'Error\n{str(e)[:30]}', 
                           transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'{cam_channel} - Error', fontsize=10)
            else:
                ax.text(0.5, 0.5, f'No Data\n{cam_channel}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{cam_channel} - No Data', fontsize=10)
        
        # BEV视图
        if len(axes) > 4:
            try:
                render_bev_custom(nusc, sample_token, axes[4])
                axes[4].set_title('BEV View - 3D Boxes', fontsize=12)
                print(f"      OK BEV View: Rendered successfully")
            except Exception as e:
                print(f"      ERROR BEV View: Render failed - {e}")
                axes[4].text(0.5, 0.5, f'BEV Render Failed', 
                           transform=axes[4].transAxes, ha='center', va='center')
                axes[4].set_title('BEV View - Error', fontsize=12)
        
        # 样本信息 - 完全修复省略号问题
        if len(axes) > 5:
            info_ax = axes[5]
            info_ax.axis('off')
            
            scene = nusc.get('scene', sample['scene_token'])
            
            # 获取时间戳信息
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = nusc.get('sample_data', lidar_token)
            filename = Path(lidar_data['filename']).name
            
            # 优化信息显示 - 避免省略号
            # 1. 使用更紧凑的格式
            # 2. 分行显示关键信息
            # 3. 使用更小的字体但增加行间距
            
            sample_info_lines = [
                "=== SAMPLE INFO ===",
                f"Token: {sample_token[:16]}",
                f"Scene: {scene['name']}",
                f"File: {filename}",
                f"Time: {sample['timestamp']}",
                f"Annotations: {len(sample['anns'])}",
                "==================="
            ]
            
            # 动态调整字体大小基于信息长度
            max_line_length = max(len(line) for line in sample_info_lines)
            if max_line_length > 40:
                font_size = 8
            elif max_line_length > 30:
                font_size = 9
            else:
                font_size = 10
            
            # 使用多行文本，每行单独处理
            y_position = 0.95
            line_height = 0.07  # 增加行间距
            
            for i, line in enumerate(sample_info_lines):
                if i == 0 or i == len(sample_info_lines) - 1:  # 首行和末行
                    weight = 'bold'
                    color = 'darkblue'
                else:
                    weight = 'normal'
                    color = 'black'
                
                info_ax.text(0.05, y_position, line, 
                           transform=info_ax.transAxes,
                           fontsize=font_size, 
                           fontfamily='monospace',
                           verticalalignment='top',
                           weight=weight,
                           color=color)
                y_position -= line_height
            
            # 添加背景框使信息更清晰
            from matplotlib.patches import FancyBboxPatch
            bbox = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                boxstyle="round,pad=0.02",
                                facecolor="lightblue", alpha=0.2,
                                edgecolor="gray", linewidth=1)
            info_ax.add_patch(bbox)
            
            # 设置信息区域的标题
            info_ax.set_title('Sample Information', fontsize=12, pad=10, weight='bold')
        
        # 调整布局，确保所有元素可见
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, 
                          wspace=0.2, hspace=0.3)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   Saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"   Sample visualization failed: {e}")
        import traceback
        traceback.print_exc()
        plt.close()
        return False




def render_annotations_on_image_fixed(nusc, sample_token, camera_channel, ax, img_width, img_height):
    """修复版标注渲染 - 解决文字在图像外的问题"""
    try:
        sample = nusc.get('sample', sample_token)
        camera_token = sample['data'][camera_channel]
        
        annotation_count = 0
        problematic_annotations = 0
        
        for ann_token in sample['anns']:
            try:
                ann = nusc.get('sample_annotation', ann_token)
                
                # 获取3D框的8个角点
                corners_3d = get_3dbox_corners(ann)
                
                # 投影到2D图像
                corners_2d = project_3d_to_2d(nusc, corners_3d, camera_token)
                
                if corners_2d is not None:
                    # 严格的坐标有效性检查
                    valid_coords = True
                    
                    # 检查坐标是否有限且合理
                    if not np.isfinite(corners_2d).all():
                        valid_coords = False
                    
                    # 检查坐标范围是否合理（考虑图像边界）
                    elif np.any(corners_2d < -1000) or np.any(corners_2d > 3000):
                        valid_coords = False
                    
                    # 检查是否在图像内
                    elif not is_bbox_in_image(corners_2d, img_width, img_height):
                        valid_coords = False
                    
                    if valid_coords:
                        # 绘制边界框 - 使用修复后的函数
                        draw_2d_bbox_fixed(ax, corners_2d, ann, img_width, img_height)
                        annotation_count += 1
                    else:
                        problematic_annotations += 1
                        
            except Exception as e:
                problematic_annotations += 1
                continue
        
        if problematic_annotations > 0:
            print(f"      Warning {camera_channel}: Skipped {problematic_annotations} problematic annotations")
            
        return annotation_count
                
    except Exception as e:
        print(f"    Annotation rendering failed: {e}")
        return 0

def draw_2d_bbox_fixed(ax, corners_2d, ann, img_width, img_height):
    """修复版2D边界框绘制 - 解决文字在图像外的问题"""
    try:
        # 计算2D边界框的包围盒
        x_min, x_max = np.min(corners_2d[0, :]), np.max(corners_2d[0, :])
        y_min, y_max = np.min(corners_2d[1, :]), np.max(corners_2d[1, :])
        
        width = x_max - x_min
        height = y_max - y_min
        
        # 跳过太小的框
        if width < 10 or height < 10:
            return
        
        # 获取类别颜色
        category = ann['category_name']
        display_category = CATEGORY_MAPPING_VIS.get(category, category)
        bbox_color = get_category_color(category)
        
        # 创建矩形框
        rect = patches.Rectangle((x_min, y_min), width, height, 
                               linewidth=1.5, edgecolor=bbox_color, facecolor='none')
        ax.add_patch(rect)
        
        # 优化文字标签 - 智能位置计算
        if width > 50 and height > 30:
            # 简化类别名称显示
            short_name = display_category.split('.')[-1]
            label_text = short_name
            
            # 计算文字位置 - 确保在图像内
            text_x = x_min
            text_y = y_min - 5  # 在框上方
            
            # 如果文字可能超出图像上边界，放在框内
            if text_y < 10:
                text_y = y_min + 15
            
            # 如果文字可能超出图像左边界
            if text_x < 5:
                text_x = 5
            
            # 如果文字可能超出图像右边界
            max_text_width = len(label_text) * 8  # 估算文字宽度
            if text_x + max_text_width > img_width - 5:
                text_x = img_width - max_text_width - 5
            
            # 添加文字背景
            ax.text(text_x, text_y, label_text, 
                   fontsize=8, color=bbox_color, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
        
    except Exception as e:
        print(f"绘制2D框失败: {e}")


def render_bev_custom(nusc, sample_token, ax=None, output_path=None):
    """自定义BEV视图渲染 - 支持独立保存和嵌入模式"""
    try:
        sample = nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        
        # 如果没有传入ax，创建新的figure
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            standalone_fig = True
        else:
            standalone_fig = False
        
        # 加载点云数据
        pc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_token))
        points = pc.points[:3, :]
        
        # 绘制点云
        if points.shape[1] > 0:
            scatter = ax.scatter(points[0, :], points[1, :], s=1, c=points[2, :], 
                              cmap='viridis', alpha=0.6)
            if standalone_fig:
                plt.colorbar(scatter, ax=ax, label='Height (m)')
        
        # 绘制3D标注框
        annotation_count = 0
        for ann_token in sample['anns']:
            try:
                ann = nusc.get('sample_annotation', ann_token)
                draw_3d_bbox_bev(ax, ann)
                annotation_count += 1
            except Exception as e:
                continue
        
        # 设置坐标轴和标题
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 设置显示范围
        if points.shape[1] > 0:
            x_center, y_center = np.mean(points[0, :]), np.mean(points[1, :])
            range_val = 50
            ax.set_xlim(x_center - range_val, x_center + range_val)
            ax.set_ylim(y_center - range_val, y_center + range_val)
        
        # 添加标题
        title = f'BEV View - {annotation_count} Annotations'
        if standalone_fig:
            ax.set_title(title, fontsize=14)
        
        # 如果是独立图表且指定了输出路径，保存图像
        if standalone_fig and output_path:
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            return True
        elif standalone_fig:
            # 如果没有输出路径但需要显示，返回figure
            plt.tight_layout()
            return fig
        else:
            # 如果是嵌入模式，直接返回True
            return True
        
    except Exception as e:
        print(f"    ❌ BEV渲染失败: {e}")
        if ax is None:
            plt.close()
        return False



def draw_3d_bbox_bev(ax, ann):
    """在BEV视图中绘制3D边界框"""
    try:
        center = np.array(ann['translation'][:2])  # 只取x,y
        w, l, _ = ann['size']
        yaw = get_yaw_from_quaternion(ann['rotation'])
        
        # 计算BEV矩形的四个角点
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        
        corners = np.array([
            [l/2 * cos_yaw - w/2 * sin_yaw, l/2 * sin_yaw + w/2 * cos_yaw],
            [l/2 * cos_yaw + w/2 * sin_yaw, l/2 * sin_yaw - w/2 * cos_yaw],
            [-l/2 * cos_yaw + w/2 * sin_yaw, -l/2 * sin_yaw - w/2 * cos_yaw],
            [-l/2 * cos_yaw - w/2 * sin_yaw, -l/2 * sin_yaw + w/2 * cos_yaw]
        ])
        
        corners += center
        
        # 获取类别颜色
        category = ann['category_name']
        display_category = CATEGORY_MAPPING_VIS.get(category, category)
        bbox_color = get_category_color(category)
        
        # 绘制矩形
        from matplotlib.patches import Polygon
        polygon = Polygon(corners, closed=True, 
                         edgecolor=bbox_color, facecolor='none', linewidth=2)
        ax.add_patch(polygon)
        
        # 绘制方向箭头
        arrow_length = l / 2
        arrow_end = center + arrow_length * np.array([cos_yaw, sin_yaw])
        ax.arrow(center[0], center[1], 
                arrow_end[0] - center[0], arrow_end[1] - center[1],
                head_width=0.5, head_length=0.5, fc=bbox_color, ec=bbox_color)
        
    except Exception as e:
        pass

def render_pointcloud_projection_custom(nusc, sample_token, camera_channel, output_path):
    """自定义点云投影渲染"""
    try:
        sample = nusc.get('sample', sample_token)
        camera_token = sample['data'][camera_channel]
        
        img_path = nusc.get_sample_data_path(camera_token)
        img = cv2.imread(img_path)
        if img is None:
            print(f"   ❌ 无法加载图像: {img_path}")
            return False
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        lidar_token = sample['data']['LIDAR_TOP']
        pc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_token))
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.imshow(img)
        
        points_camera = transform_points_to_camera(nusc, pc.points[:3, :], camera_token)
        
        cam_calib = nusc.get('calibrated_sensor', 
                            nusc.get('sample_data', camera_token)['calibrated_sensor_token'])
        
        if 'camera_intrinsic' in cam_calib and cam_calib['camera_intrinsic']:
            intrinsic = np.array(cam_calib['camera_intrinsic'])
            
            valid_indices = points_camera[2, :] > 0.1
            points_cam_valid = points_camera[:3, valid_indices]
            
            if points_cam_valid.shape[1] > 0:
                points_2d_homo = intrinsic @ points_cam_valid
                points_2d = points_2d_homo[:2, :] / points_2d_homo[2, :]
                
                in_image = (points_2d[0, :] >= 0) & (points_2d[0, :] < img.shape[1]) & \
                          (points_2d[1, :] >= 0) & (points_2d[1, :] < img.shape[0])
                
                points_2d_in_image = points_2d[:, in_image]
                depths = points_cam_valid[2, in_image]
                
                if points_2d_in_image.shape[1] > 0:
                    scatter = ax.scatter(points_2d_in_image[0, :], points_2d_in_image[1, :], 
                                       c=depths, s=2, cmap='viridis', alpha=0.7)
                    plt.colorbar(scatter, ax=ax, label='Depth (m)')
        
        ax.set_title(f'Point Cloud Projection - {camera_channel}', fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   💾 保存点云投影: {output_path}")
        return True
        
    except Exception as e:
        print(f"   ❌ 点云投影失败: {e}")
        plt.close()
        return False

# ==================== 修复原有的调试和测试函数 ====================
def debug_calibration_info_comprehensive(nusc, sample_token, sample_index):
    """调试标定信息"""
    print(f"\n🔧 调试标定信息 - 样本 {sample_token[:8]}")
    print("=" * 40)
    
    sample = nusc.get('sample', sample_token)
    calib_info = {}
    
    # 获取时间戳信息用于文件名
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    timestamp = Path(lidar_data['filename']).stem
    
    for sensor_type, sensor_token in sample['data'].items():
        try:
            sensor_data = nusc.get('sample_data', sensor_token)
            calib_token = sensor_data['calibrated_sensor_token']
            calib = nusc.get('calibrated_sensor', calib_token)
            ego_pose = nusc.get('ego_pose', sensor_data['ego_pose_token'])
            
            # 计算传感器在global坐标系中的位置
            ego_rotation = quaternion_to_rotation_matrix(ego_pose['rotation'])
            sensor_rotation_ego = quaternion_to_rotation_matrix(calib['rotation'])
            
            # sensor在global中的旋转
            sensor_rotation_global = ego_rotation @ sensor_rotation_ego
            # sensor在global中的位置
            sensor_position_global = ego_rotation @ np.array(calib['translation']) + np.array(ego_pose['translation'])
            
            calib_info[sensor_type] = {
                'calibrated_sensor_token': calib_token,
                'sensor_to_ego_translation': calib['translation'],
                'sensor_to_ego_rotation': calib['rotation'],
                'sensor_in_global_position': sensor_position_global.tolist(),
                'ego_to_global_translation': ego_pose['translation'],
                'ego_to_global_rotation': ego_pose['rotation'],
                'camera_intrinsic': calib.get('camera_intrinsic', []) if 'camera_intrinsic' in calib else 'N/A'
            }
            
            print(f"  📷 {sensor_type}:")
            print(f"    传感器->ego平移: [{calib['translation'][0]:.2f}, {calib['translation'][1]:.2f}, {calib['translation'][2]:.2f}]")
            print(f"    传感器->ego旋转: [{calib['rotation'][0]:.3f}, {calib['rotation'][1]:.3f}, {calib['rotation'][2]:.3f}, {calib['rotation'][3]:.3f}]")
            print(f"    传感器在global位置: [{sensor_position_global[0]:.1f}, {sensor_position_global[1]:.1f}, {sensor_position_global[2]:.1f}]")
            
            if 'camera_intrinsic' in calib and calib['camera_intrinsic']:
                intrinsic = np.array(calib['camera_intrinsic'])
                print(f"    相机内参:")
                print(f"      fx={intrinsic[0,0]:.1f}, fy={intrinsic[1,1]:.1f}")
                print(f"      cx={intrinsic[0,2]:.1f}, cy={intrinsic[1,2]:.1f}")
            
            # 为每个传感器单独保存标定信息
            if sensor_type.startswith('CAM'):
                sensor_calib_path = Path(OUTPUT_DIR) / "calibration_debug" / sensor_type / f"sample_{timestamp}_calibration.json"
                with open(sensor_calib_path, 'w') as f:
                    json.dump(calib_info[sensor_type], f, indent=2, ensure_ascii=False)
                print(f"    💾 标定信息保存至: {sensor_calib_path}")
            
        except Exception as e:
            print(f"  ❌ 加载{sensor_type}标定失败: {e}")
    
    # 保存完整的标定信息
    calib_path = Path(OUTPUT_DIR) / "calibration_debug" / f"sample_{timestamp}_all_calibration.json"
    with open(calib_path, 'w') as f:
        json.dump(calib_info, f, indent=2, ensure_ascii=False)
    
    return calib_info

def test_projection_accuracy(nusc, sample_token, camera_channel):
    """测试投影准确性 - 修复返回类型问题"""
    print(f"\n🎯 测试投影准确性 - {camera_channel}")
    print("=" * 40)
    
    try:
        sample = nusc.get('sample', sample_token)
        camera_token = sample['data'][camera_channel]
        
        # 获取点云
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        pc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_token))
        
        # 获取相机参数
        cam_calib = nusc.get('calibrated_sensor', 
                            nusc.get('sample_data', camera_token)['calibrated_sensor_token'])
        ego_pose = nusc.get('ego_pose', 
                           nusc.get('sample_data', camera_token)['ego_pose_token'])
        
        # 转换点云到相机坐标系
        points_ego = pc.points[:3, :]  # 取xyz
        
        # ego -> global 变换
        ego_rotation = quaternion_to_rotation_matrix(ego_pose['rotation'])
        ego_translation = np.array(ego_pose['translation'])
        points_global = ego_rotation @ points_ego + ego_translation.reshape(3, 1)
        
        # global -> camera 变换
        cam_rotation_ego = quaternion_to_rotation_matrix(cam_calib['rotation'])
        cam_translation_ego = np.array(cam_calib['translation'])
        
        # camera -> global 变换
        cam_rotation_global = ego_rotation @ cam_rotation_ego
        cam_translation_global = ego_rotation @ cam_translation_ego + ego_translation
        
        # global -> camera 变换（逆变换）
        global_rotation_cam = cam_rotation_global.T
        global_translation_cam = -global_rotation_cam @ cam_translation_global
        
        # 将global坐标系的点转换到camera坐标系
        points_camera = global_rotation_cam @ points_global + global_translation_cam.reshape(3, 1)
        
        # 投影到图像平面
        if 'camera_intrinsic' in cam_calib and cam_calib['camera_intrinsic']:
            intrinsic = np.array(cam_calib['camera_intrinsic'])
            
            # 过滤在相机前方的点 (z > 0)
            depths = points_camera[2, :]
            valid_indices = depths > 0.1
            points_cam_valid = points_camera[:3, valid_indices]
            
            if points_cam_valid.shape[1] == 0:
                print(f"  ⚠️  没有有效的投影点（所有点都在相机后方）")
                return False, {}
            
            # 投影
            points_2d_homo = intrinsic @ points_cam_valid
            points_2d = points_2d_homo[:2, :] / points_2d_homo[2, :]
            
            # 统计投影结果
            image_width = 1920
            image_height = 1080
            in_image = (points_2d[0, :] >= 0) & (points_2d[0, :] < image_width) & \
                      (points_2d[1, :] >= 0) & (points_2d[1, :] < image_height)
            
            total_points = points_cam_valid.shape[1]
            points_in_image = np.sum(in_image)
            projection_ratio = points_in_image / total_points * 100
            
            print(f"  📊 投影统计:")
            print(f"    总点数: {total_points}")
            print(f"    在图像内点数: {points_in_image}")
            print(f"    投影比例: {projection_ratio:.1f}%")
            print(f"    有效点深度范围: [{depths[valid_indices].min():.1f}, {depths[valid_indices].max():.1f}]m")
            
            # 修复：确保所有数值都是Python原生类型
            return True, {
                'total_points': int(total_points),
                'points_in_image': int(points_in_image),
                'projection_ratio': float(projection_ratio),
                'min_depth': float(depths[valid_indices].min()),
                'max_depth': float(depths[valid_indices].max())
            }
        else:
            print(f"  ❌ 无相机内参数据")
            return False, {}
            
    except Exception as e:
        print(f"  ❌ 投影测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def test_annotation_projection(nusc, sample_token, camera_channel, sample_index):
    """测试3D标注投影"""
    print(f"\n🎯 测试3D标注投影 - {camera_channel}")
    print("=" * 40)
    
    try:
        sample = nusc.get('sample', sample_token)
        camera_token = sample['data'][camera_channel]
        
        # 获取时间戳信息用于文件名
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        timestamp = Path(lidar_data['filename']).stem
        
        # 获取相机参数
        cam_calib = nusc.get('calibrated_sensor', 
                            nusc.get('sample_data', camera_token)['calibrated_sensor_token'])
        ego_pose = nusc.get('ego_pose', 
                           nusc.get('sample_data', camera_token)['ego_pose_token'])
        
        # 获取该样本的所有标注
        annotations = []
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            instance = nusc.get('instance', ann['instance_token'])
            category = nusc.get('category', instance['category_token'])
            
            annotations.append({
                'token': ann_token,
                'category': category['name'],
                'translation': ann['translation'],
                'size': ann['size'],
                'rotation': ann['rotation']
            })
        
        print(f"  标注数量: {len(annotations)}")
        
        annotation_results = {
            'camera_channel': camera_channel,
            'timestamp': timestamp,
            'total_annotations': len(annotations),
            'projection_details': []
        }
        
        # 检查每个标注是否在相机视野内
        in_view_count = 0
        for i, ann in enumerate(annotations):
            # 将标注中心点从ego坐标系转换到相机坐标系
            point_ego = np.array(ann['translation'])
            
            # ego -> global
            ego_rotation = quaternion_to_rotation_matrix(ego_pose['rotation'])
            ego_translation = np.array(ego_pose['translation'])
            point_global = ego_rotation @ point_ego + ego_translation
            
            # global -> camera
            cam_rotation_ego = quaternion_to_rotation_matrix(cam_calib['rotation'])
            cam_translation_ego = np.array(cam_calib['translation'])
            
            cam_rotation_global = ego_rotation @ cam_rotation_ego
            cam_translation_global = ego_rotation @ cam_translation_ego + ego_translation
            
            global_rotation_cam = cam_rotation_global.T
            global_translation_cam = -global_rotation_cam @ cam_translation_global
            
            point_camera = global_rotation_cam @ point_global + global_translation_cam
            
            # 检查是否在相机前方
            in_front = bool(point_camera[2] > 0)
            
            projection_info = {
                'annotation_index': i,
                'category': ann['category'],
                'ego_position': ann['translation'],
                'camera_position': [float(x) for x in point_camera],
                'in_front_of_camera': in_front,
                'in_image': False,
                'image_coordinates': None
            }
            
            if 'camera_intrinsic' in cam_calib and cam_calib['camera_intrinsic'] and in_front:
                intrinsic = np.array(cam_calib['camera_intrinsic'])
                point_2d_homo = intrinsic @ point_camera
                point_2d = point_2d_homo[:2] / point_2d_homo[2]
                
                # 检查是否在图像范围内
                image_width = 1920
                image_height = 1080
                in_image = bool((0 <= point_2d[0] < image_width) and (0 <= point_2d[1] < image_height))
                
                projection_info['in_image'] = in_image
                projection_info['image_coordinates'] = [float(x) for x in point_2d]
                
                if in_image:
                    in_view_count += 1
                    status = "✅ 在视野内"
                else:
                    status = "⚠️  在相机前方但不在图像内"
            else:
                if not in_front:
                    status = "❌ 在相机后方"
                else:
                    status = "❌ 无内参数据"
            
            annotation_results['projection_details'].append(projection_info)
            
            print(f"  标注 {i}: {ann['category']} - {status}")
            if in_front and projection_info['image_coordinates']:
                print(f"    图像坐标: [{projection_info['image_coordinates'][0]:.1f}, {projection_info['image_coordinates'][1]:.1f}]")
            print(f"    3D位置: [{ann['translation'][0]:.1f}, {ann['translation'][1]:.1f}, {ann['translation'][2]:.1f}]")
            print(f"    相机坐标系: [{point_camera[0]:.1f}, {point_camera[1]:.1f}, {point_camera[2]:.1f}]")
        
        annotation_results['in_view_count'] = in_view_count
        annotation_results['in_view_ratio'] = float(in_view_count / len(annotations) * 100) if annotations else 0.0
        
        print(f"  总结: {in_view_count}/{len(annotations)} 个标注在相机视野内")
        
        # 保存标注投影结果到对应传感器文件夹
        annotation_path = Path(OUTPUT_DIR) / "annotation_projection" / camera_channel / f"sample_{timestamp}_projection.json"
        with open(annotation_path, 'w') as f:
            json.dump(annotation_results, f, indent=2, ensure_ascii=False)
        print(f"  💾 标注投影结果保存至: {annotation_path}")
        
        return True, annotation_results
        
    except Exception as e:
        print(f"  ❌ 标注投影测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def get_sampled_samples(nusc, sampling_strategy, max_samples_per_scene):
    """根据采样策略获取要处理的样本列表"""
    print(f"\n🎯 采样策略: {sampling_strategy}")
    print(f"   每个场景最大样本数: {max_samples_per_scene}")
    
    sampled_samples = []
    
    if sampling_strategy == "per_scene":
        # 每个场景采样MAX_SAMPLES_PER_SCENE个样本
        for scene in nusc.scene:
            scene_token = scene['token']
            scene_name = scene['name']
            
            # 获取该场景的所有样本
            scene_samples = []
            first_sample_token = scene['first_sample_token']
            current_sample = nusc.get('sample', first_sample_token)
            
            while current_sample is not None:
                scene_samples.append(current_sample)
                if current_sample['next']:
                    current_sample = nusc.get('sample', current_sample['next'])
                else:
                    break
            
            print(f"   📊 场景 {scene_name}: {len(scene_samples)} 个样本")
            
            # 采样策略：如果样本数量超过限制，等距采样
            if len(scene_samples) > max_samples_per_scene:
                step = len(scene_samples) // max_samples_per_scene
                sampled_scene_samples = scene_samples[::step][:max_samples_per_scene]
                print(f"      🔄 等距采样: {len(scene_samples)} -> {len(sampled_scene_samples)} 个样本")
            else:
                sampled_scene_samples = scene_samples
            
            sampled_samples.extend(sampled_scene_samples)
    
    elif sampling_strategy == "uniform":
        # 全量等距采样
        total_samples = len(nusc.sample)
        if total_samples > max_samples_per_scene:
            step = total_samples // max_samples_per_scene
            sampled_samples = nusc.sample[::step][:max_samples_per_scene]
            print(f"   🔄 全量等距采样: {total_samples} -> {len(sampled_samples)} 个样本")
        else:
            sampled_samples = nusc.sample
            print(f"   📊 使用全量样本: {len(sampled_samples)} 个样本")
    
    else:
        # 默认使用全量样本
        sampled_samples = nusc.sample
        print(f"   📊 使用全量样本: {len(sampled_samples)} 个样本")
    
    print(f"   ✅ 最终处理样本数: {len(sampled_samples)}")
    return sampled_samples


def run_custom_visualization(nusc, sampling_strategy="per_scene", max_samples_per_scene=5):
    """运行自定义可视化流程 - 修复版"""
    print(f"\n🎨 运行自定义可视化")
    print("=" * 50)
    
    # 获取采样后的样本列表
    sampled_samples = get_sampled_samples(nusc, sampling_strategy, max_samples_per_scene)
    
    # 系统信息调试 - 修复版本获取方式
    print(f"\n🔧 系统调试信息:")
    print(f"   Python版本: {sys.version}")
    print(f"   OpenCV版本: {cv2.__version__}")
    
    # 正确获取matplotlib版本
    import matplotlib
    print(f"   Matplotlib版本: {matplotlib.__version__}")
    print(f"   NumPy版本: {np.__version__}")
    
    success_stats = {
        'custom_sample_vis': 0,
        'individual_camera_vis': 0,
        'pointcloud_projection': 0,
        'bev_views': 0,
        'projection_test': 0,
        'annotation_projection_test': 0
    }
    
    for i, sample in enumerate(sampled_samples):
        sample_token = sample['token']
        
        # 获取时间戳信息
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        timestamp = Path(lidar_data['filename']).stem
        
        # 获取场景信息
        scene = nusc.get('scene', sample['scene_token'])
        scene_name = scene['name']
        
        print(f"\n📊 样本 {i+1}/{len(sampled_samples)}: {sample_token[:8]}... (场景: {scene_name}, 时间戳: {timestamp})")
        
        # 自定义样本可视化
        if VISUALIZATION_CONFIG['custom_visualization']:
            sample_path = Path(OUTPUT_DIR) / "custom_samples" / f"scene_{scene_name}_sample_{timestamp}_custom.png"
            if render_custom_sample_visualization(nusc, sample_token, str(sample_path)):
                success_stats['custom_sample_vis'] += 1
        
        # 单个相机视图
        if VISUALIZATION_CONFIG['save_individual_camera_views']:
            for camera_channel in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK']:
                if camera_channel in sample['data']:
                    single_cam_path = Path(OUTPUT_DIR) / "custom_samples" / camera_channel / f"scene_{scene_name}_sample_{timestamp}_{camera_channel}.png"
                    if render_single_camera_view(nusc, sample_token, camera_channel, str(single_cam_path)):
                        success_stats['individual_camera_vis'] += 1
        
        # 点云投影
        if VISUALIZATION_CONFIG['render_pointcloud_projection']:
            for camera_channel in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK']:
                if camera_channel in sample['data']:
                    pc_path = Path(OUTPUT_DIR) / "pointcloud_projections" / camera_channel / f"scene_{scene_name}_sample_{timestamp}.png"
                    if render_pointcloud_projection_custom(nusc, sample_token, camera_channel, str(pc_path)):
                        success_stats['pointcloud_projection'] += 1
        
        # BEV视图
        if VISUALIZATION_CONFIG['render_bev_views']:
            try:
                bev_path = Path(OUTPUT_DIR) / "bev_views" / f"scene_{scene_name}_sample_{timestamp}_bev.png"
                if render_bev_custom(nusc, sample_token, output_path=str(bev_path)):
                    success_stats['bev_views'] += 1
                    print(f"   💾 保存BEV视图: {bev_path}")
                else:
                    print(f"   ❌ BEV视图保存失败")
            except Exception as e:
                print(f"   ❌ BEV视图处理异常: {e}")
        
        # 调试信息
        if VISUALIZATION_CONFIG['debug_calibration_info']:
            debug_calibration_info_comprehensive(nusc, sample_token, i+1)
        
        # 投影测试
        if VISUALIZATION_CONFIG['test_projection_accuracy']:
            for camera_channel in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK']:
                if camera_channel in sample['data']:
                    try:
                        result = test_projection_accuracy(nusc, sample_token, camera_channel)
                        if isinstance(result, tuple) and len(result) == 2:
                            success, test_result = result
                            if success:
                                success_stats['projection_test'] += 1
                        else:
                            print(f"  ⚠️  投影测试返回异常结果: {result}")
                    except Exception as e:
                        print(f"  ❌ 投影测试异常: {e}")
        
        # 标注投影测试
        if VISUALIZATION_CONFIG['test_annotation_projection']:
            for camera_channel in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK']:
                if camera_channel in sample['data']:
                    try:
                        result = test_annotation_projection(nusc, sample_token, camera_channel, i+1)
                        if isinstance(result, tuple) and len(result) == 2:
                            success, test_result = result
                            if success:
                                success_stats['annotation_projection_test'] += 1
                        else:
                            print(f"  ⚠️  标注投影测试返回异常结果: {result}")
                    except Exception as e:
                        print(f"  ❌ 标注投影测试异常: {e}")
        
        # 清理内存
        if i % 5 == 0:
            gc.collect()
            print(f"  🧹 内存清理完成")
    
    print(f"\n📊 自定义可视化统计:")
    total_processed = len(sampled_samples)
    print(f"   总处理样本数: {total_processed}")
    print(f"   自定义样本可视化: {success_stats['custom_sample_vis']}/{total_processed}")
    print(f"   单相机视图: {success_stats['individual_camera_vis']}/{total_processed * 4}")
    print(f"   点云投影: {success_stats['pointcloud_projection']}/{total_processed * 4}")
    print(f"   BEV视图: {success_stats['bev_views']}/{total_processed}")
    print(f"   投影测试: {success_stats['projection_test']}/{total_processed * 4}")
    print(f"   标注投影测试: {success_stats['annotation_projection_test']}/{total_processed * 4}")
    
    return success_stats

def main():
    """主函数"""
    print("🚀 Step2-3: nuscenes数据可视化验证工具 v4.12-fixed-issues")
    print(f"数据路径: {NUSC_ROOT}")
    print(f"输出路径: {OUTPUT_DIR}")
    print(f"版本: {NUSC_VERSION}")
    print(f"采样策略: {SAMPLING_STRATEGY}")
    print(f"每个场景最大样本数: {MAX_SAMPLES_PER_SCENE}")
    print(f"单相机视图保存: {'开启' if VISUALIZATION_CONFIG['save_individual_camera_views'] else '关闭'}")
    
    # 加载nuscenes数据
    try:
        nusc = NuScenes(version=NUSC_VERSION, dataroot=NUSC_ROOT, verbose=True)
        print(f"✅ NuScenes加载成功")
        print(f"   样本数量: {len(nusc.sample)}")
        print(f"   场景数量: {len(nusc.scene)}")
        
        # 显示场景信息
        print(f"\n📋 场景列表:")
        for scene in nusc.scene:
            sample_count = 0
            current_sample = nusc.get('sample', scene['first_sample_token'])
            while current_sample is not None:
                sample_count += 1
                if current_sample['next']:
                    current_sample = nusc.get('sample', current_sample['next'])
                else:
                    break
            print(f"   🎬 {scene['name']}: {sample_count} 个样本")
            
    except Exception as e:
        print(f"❌ NuScenes加载失败: {e}")
        return
    
    # 创建输出目录
    create_output_folders()
    
    # 运行自定义可视化
    success_stats = run_custom_visualization(nusc, SAMPLING_STRATEGY, MAX_SAMPLES_PER_SCENE)
    
    print(f"\n🎉 Step2-3 完成!")
    print("✅ 自定义可视化完成")
    print("📊 所有结果已保存到相应文件夹")
    
    # 显示输出目录结构
    output_path = Path(OUTPUT_DIR)
    print(f"\n📁 输出目录结构:")
    for folder in output_path.iterdir():
        if folder.is_dir():
            file_count = len(list(folder.rglob('*.png'))) + len(list(folder.rglob('*.json')))
            print(f"   📁 {folder.name}/ - {file_count} 个文件")


if __name__ == "__main__":
    main()