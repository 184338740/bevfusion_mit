#!/usr/bin/env python3
"""
Step2-3: nuscenes数据可视化验证工具 - 修复输出路径
版本: v4.9-visualization-fixed
功能: 纯可视化验证，修复输出路径和数据保存问题
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

# ==================== 配置参数 ====================
NUSC_ROOT = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt"
OUTPUT_DIR = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step2_3_vis"  # 固定路径
NUSC_VERSION = "v1.0-tyjt"

# 新增: MAX_SAMPLES控制参数
MAX_SAMPLES = None  # None: 处理所有样本, 正整数: 每个场景最多处理指定数量样本

# 新增: 可视化功能配置开关
VISUALIZATION_CONFIG = {
    'render_sample_official': True,        # 官方样本渲染
    'render_sample_data_official': True,   # 官方传感器数据渲染  
    'render_pointcloud_projection': True,  # 点云投影
    'render_bev_views': True,              # BEV视图
    'debug_calibration_info': True,        # 标定信息调试
    'test_projection_accuracy': True,      # 投影准确性测试
    'test_annotation_projection': True,    # 标注投影测试
}

# ==================== 工具函数 ====================
def quaternion_to_rotation_matrix(q):
    """四元数转旋转矩阵 [w, x, y, z] - NuScenes标准顺序"""
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])

def create_output_folders():
    """创建所有输出文件夹 - 修复版本"""
    print("\n📁 创建输出文件夹结构")
    print("=" * 50)
    
    folders = {
        'official_samples': '官方样本渲染',
        'official_sensor_data': '官方传感器数据渲染', 
        'pointcloud_projections': '点云投影',
        'bev_views': 'BEV视图',
        'calibration_debug': '标定调试信息',
        'annotation_projection': '标注投影测试',
        'summary_reports': '汇总报告'
    }
    
    # 创建主文件夹
    for folder_name, description in folders.items():
        folder_path = Path(OUTPUT_DIR) / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 创建: {folder_name} - {description}")
    
    # 创建子文件夹 - 修复：为每个传感器创建子文件夹
    camera_channels = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK']
    
    for cam in camera_channels:
        # pointcloud_projections 子文件夹
        (Path(OUTPUT_DIR) / "pointcloud_projections" / cam).mkdir(parents=True, exist_ok=True)
        # official_sensor_data 子文件夹  
        (Path(OUTPUT_DIR) / "official_sensor_data" / cam).mkdir(parents=True, exist_ok=True)
        # calibration_debug 子文件夹
        (Path(OUTPUT_DIR) / "calibration_debug" / cam).mkdir(parents=True, exist_ok=True)
        # annotation_projection 子文件夹
        (Path(OUTPUT_DIR) / "annotation_projection" / cam).mkdir(parents=True, exist_ok=True)
    
    print("✅ 所有子文件夹创建完成")
    return folders

# ==================== 可视化功能实现 ====================
def render_sample_official(nusc, sample_token, output_path):
    """使用官方render_sample渲染样本"""
    try:
        plt.figure(figsize=(20, 12))
        nusc.render_sample(sample_token)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   💾 保存官方样本渲染: {output_path}")
        return True
    except Exception as e:
        print(f"   ❌ 官方样本渲染失败: {e}")
        plt.close()
        return False

def render_sample_data_official(nusc, sample_data_token, output_path, with_anns=True):
    """使用官方render_sample_data渲染传感器数据"""
    try:
        plt.figure(figsize=(15, 10))
        nusc.render_sample_data(sample_data_token, with_anns=with_anns)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   💾 保存传感器数据渲染: {output_path}")
        return True
    except Exception as e:
        print(f"   ❌ 传感器数据渲染失败: {e}")
        plt.close()
        return False

def render_pointcloud_projection_fixed(nusc, sample_token, camera_channel, output_path):
    """修复点云投影渲染 - 使用官方API"""
    try:
        plt.figure(figsize=(15, 10))
        
        # 使用官方点云投影函数
        nusc.render_pointcloud_in_image(
            sample_token, 
            pointsensor_channel='LIDAR_TOP',
            camera_channel=camera_channel,
            render_intensity=True
        )
        
        # 优化点云显示
        ax = plt.gca()
        for collection in ax.collections:
            if hasattr(collection, '_sizes'):
                collection.set_sizes([2])
            if hasattr(collection, '_facecolors'):
                collection.set_alpha(0.6)
        
        plt.title(f"Point Cloud Projection - {camera_channel}", fontsize=14)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   💾 保存点云投影: {output_path}")
        return True
    except Exception as e:
        print(f"   ❌ 点云投影失败: {e}")
        plt.close()
        return False

def render_bev_comprehensive(nusc, sample_token, output_path):
    """综合BEV视图渲染 - 使用官方API"""
    try:
        sample = nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        
        plt.figure(figsize=(16, 12))
        nusc.render_sample_data(lidar_token, with_anns=True)
        
        # 添加标题和信息
        plt.title(f"BEV Visualization\nSample: {sample_token[:8]}\nAnnotations: {len(sample['anns'])}", 
                 fontsize=16, pad=20)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   💾 保存BEV视图: {output_path}")
        return True
    except Exception as e:
        print(f"   ❌ BEV渲染失败: {e}")
        plt.close()
        return False


def debug_calibration_info_comprehensive(nusc, sample_token, sample_index):
    """综合调试标定信息 - 使用时间戳作为文件名"""
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
            
            # 修复：为每个传感器单独保存标定信息 - 使用时间戳作为文件名
            if sensor_type.startswith('CAM'):
                sensor_calib_path = Path(OUTPUT_DIR) / "calibration_debug" / sensor_type / f"sample_{timestamp}_calibration.json"
                with open(sensor_calib_path, 'w') as f:
                    json.dump(calib_info[sensor_type], f, indent=2, ensure_ascii=False)
                print(f"    💾 标定信息保存至: {sensor_calib_path}")
            
        except Exception as e:
            print(f"  ❌ 加载{sensor_type}标定失败: {e}")
    
    # 保存完整的标定信息 - 使用时间戳作为文件名
    calib_path = Path(OUTPUT_DIR) / "calibration_debug" / f"sample_{timestamp}_all_calibration.json"
    with open(calib_path, 'w') as f:
        json.dump(calib_info, f, indent=2, ensure_ascii=False)
    
    return calib_info


def test_projection_accuracy(nusc, sample_token, camera_channel):
    """测试投影准确性 - 修复JSON序列化问题"""
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
                'total_points': int(total_points),  # 转换为Python int
                'points_in_image': int(points_in_image),  # 转换为Python int
                'projection_ratio': float(projection_ratio),  # 转换为Python float
                'min_depth': float(depths[valid_indices].min()),  # 转换为Python float
                'max_depth': float(depths[valid_indices].max())   # 转换为Python float
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
    """测试3D标注在图像中的投影 - 使用时间戳作为文件名"""
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
            
            # 检查是否在相机前方 - 修复：转换为Python原生bool
            in_front = bool(point_camera[2] > 0)  # 修复：添加bool()转换
            
            projection_info = {
                'annotation_index': i,
                'category': ann['category'],
                'ego_position': ann['translation'],
                'camera_position': [float(x) for x in point_camera],  # 修复：转换为Python float
                'in_front_of_camera': in_front,  # 现在已经是Python bool
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
                projection_info['image_coordinates'] = [float(x) for x in point_2d]  # 修复：转换为Python float
                
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
        
        # 修复：保存标注投影结果到对应传感器文件夹 - 使用时间戳作为文件名
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



# ==================== 主流程函数 ====================
def test_official_api_functions(nusc, num_samples=None):
    """测试官方API功能 - 使用原始文件名保存"""
    print(f"\n🎯 测试官方API功能")
    print("=" * 50)
    
    # 确定要处理的样本数量
    if num_samples is None:
        if MAX_SAMPLES is not None and MAX_SAMPLES > 0:
            num_samples = min(MAX_SAMPLES, len(nusc.sample))
        else:
            num_samples = len(nusc.sample)
    
    print(f"📊 处理样本数量: {num_samples} (MAX_SAMPLES={MAX_SAMPLES})")
    
    success_stats = {
        'render_sample': 0,
        'render_sample_data': 0,
        'render_pointcloud': 0,
        'render_bev': 0,
        'projection_test': 0,
        'annotation_projection_test': 0
    }
    
    projection_results = {}
    annotation_results = {}
    
    for i in range(num_samples):
        sample = nusc.sample[i]
        sample_token = sample['token']
        
        # 获取时间戳信息用于文件名
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        timestamp = Path(lidar_data['filename']).stem
        
        print(f"\n📊 样本 {i+1}: {sample_token[:8]}... (时间戳: {timestamp})")
        
        # 调试标定信息
        if VISUALIZATION_CONFIG['debug_calibration_info']:
            calib_info = debug_calibration_info_comprehensive(nusc, sample_token, i+1)
        
        # 测试投影准确性
        if VISUALIZATION_CONFIG['test_projection_accuracy']:
            projection_results[i+1] = {}
            for camera_channel in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK']:
                if camera_channel in sample['data']:
                    success, result = test_projection_accuracy(nusc, sample_token, camera_channel)
                    if success:
                        success_stats['projection_test'] += 1
                        projection_results[i+1][camera_channel] = result
        
        # 测试3D标注投影
        if VISUALIZATION_CONFIG['test_annotation_projection']:
            annotation_results[i+1] = {}
            for camera_channel in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK']:
                if camera_channel in sample['data']:
                    success, result = test_annotation_projection(nusc, sample_token, camera_channel, i+1)
                    if success:
                        success_stats['annotation_projection_test'] += 1
                        annotation_results[i+1][camera_channel] = result
        
        # 测试render_sample - 使用时间戳作为文件名
        if VISUALIZATION_CONFIG['render_sample_official']:
            sample_path = Path(OUTPUT_DIR) / "official_samples" / f"sample_{timestamp}_official.png"
            if render_sample_official(nusc, sample_token, str(sample_path)):
                success_stats['render_sample'] += 1
        
        # 测试render_sample_data (所有传感器) - 使用时间戳作为文件名
        if VISUALIZATION_CONFIG['render_sample_data_official']:
            for sensor_type, sensor_token in sample['data'].items():
                if sensor_type.startswith('CAM'):
                    sensor_path = Path(OUTPUT_DIR) / "official_sensor_data" / sensor_type / f"sample_{timestamp}_{sensor_type}.png"
                    if render_sample_data_official(nusc, sensor_token, str(sensor_path), with_anns=True):
                        success_stats['render_sample_data'] += 1
                
                elif sensor_type == 'LIDAR_TOP':
                    lidar_path = Path(OUTPUT_DIR) / "official_sensor_data" / f"sample_{timestamp}_LIDAR.png"
                    if render_sample_data_official(nusc, sensor_token, str(lidar_path), with_anns=True):
                        success_stats['render_sample_data'] += 1
        
        # 测试点云投影 (所有相机) - 使用时间戳作为文件名
        if VISUALIZATION_CONFIG['render_pointcloud_projection']:
            for camera_channel in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK']:
                if camera_channel in sample['data']:
                    pc_path = Path(OUTPUT_DIR) / "pointcloud_projections" / camera_channel / f"sample_{timestamp}.png"
                    if render_pointcloud_projection_fixed(nusc, sample_token, camera_channel, str(pc_path)):
                        success_stats['render_pointcloud'] += 1
        
        # 测试BEV渲染 - 使用时间戳作为文件名
        if VISUALIZATION_CONFIG['render_bev_views']:
            bev_path = Path(OUTPUT_DIR) / "bev_views" / f"sample_{timestamp}_bev.png"
            if render_bev_comprehensive(nusc, sample_token, str(bev_path)):
                success_stats['render_bev'] += 1
        
        # 清理内存
        gc.collect()
    
    # 保存测试结果
    if VISUALIZATION_CONFIG['test_projection_accuracy']:
        projection_path = Path(OUTPUT_DIR) / "calibration_debug" / "projection_results.json"
        with open(projection_path, 'w') as f:
            json.dump(projection_results, f, indent=2, ensure_ascii=False)
        print(f"💾 投影测试结果保存至: {projection_path}")
    
    if VISUALIZATION_CONFIG['test_annotation_projection']:
        annotation_path = Path(OUTPUT_DIR) / "annotation_projection" / "annotation_results.json"
        with open(annotation_path, 'w') as f:
            json.dump(annotation_results, f, indent=2, ensure_ascii=False)
        print(f"💾 标注投影结果保存至: {annotation_path}")
    
    print(f"\n📊 官方API测试统计:")
    for func, count in success_stats.items():
        if func == 'render_sample':
            total_attempts = num_samples
        elif func == 'render_sample_data':
            total_attempts = num_samples * 5  # 大约5个传感器
        elif func == 'render_pointcloud':
            total_attempts = num_samples * 4  # 4个相机
        elif func == 'render_bev':
            total_attempts = num_samples
        elif func == 'projection_test':
            total_attempts = num_samples * 4  # 4个相机
        elif func == 'annotation_projection_test':
            total_attempts = num_samples * 4  # 4个相机
        
        success_rate = count / total_attempts * 100 if total_attempts > 0 else 0
        print(f"   {func}: {count}/{total_attempts} ({success_rate:.1f}%)")
    
    return projection_results, annotation_results



def main():
    """主函数"""
    print("🚀 Step2-3: nuscenes数据可视化验证工具 v4.9-fixed")
    print(f"数据路径: {NUSC_ROOT}")
    print(f"输出路径: {OUTPUT_DIR}")
    print(f"版本: {NUSC_VERSION}")
    print(f"MAX_SAMPLES: {MAX_SAMPLES}")
    print(f"可视化配置: {VISUALIZATION_CONFIG}")
    
    # 加载nuscenes数据
    try:
        nusc = NuScenes(version=NUSC_VERSION, dataroot=NUSC_ROOT, verbose=True)
        print(f"✅ NuScenes加载成功")
        print(f"   样本数量: {len(nusc.sample)}")
        print(f"   场景数量: {len(nusc.scene)}")
    except Exception as e:
        print(f"❌ NuScenes加载失败: {e}")
        return
    
    # 创建输出目录
    create_output_folders()
    
    # 测试官方API功能
    projection_results, annotation_results = test_official_api_functions(nusc)
    
    print(f"\n🎉 Step2-3 完成!")
    print("✅ 可视化验证完成")
    print("📊 可视化结果已按类型分文件夹保存")
    
    # 显示启用的功能
    enabled_functions = [k for k, v in VISUALIZATION_CONFIG.items() if v]
    print(f"🔧 启用的可视化功能: {len(enabled_functions)} 个")
    for func in enabled_functions:
        print(f"   ✅ {func}")

if __name__ == "__main__":
    main()