# step2_3_check_vis-v1.3-fixed.py
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

# 配置字体避免警告
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

def parse_args():
    parser = argparse.ArgumentParser(description="修复版可视化工具")
    parser.add_argument('--dataroot', type=str, default="./output/step1/nuscenes_tyjt_fixed/v1.0-tyjt")
    parser.add_argument('--version', type=str, default="v1.0-tyjt")
    parser.add_argument('--output_dir', type=str, default="./output/step2_fixed")
    parser.add_argument('--bev_range', type=int, default=75)
    parser.add_argument('--max_frames', type=int, default=3, help="最大处理帧数（测试用）")
    parser.add_argument('--calib_file', type=str, default="./output/step1/nuscenes_tyjt_fixed/v1.0-tyjt/v1.0-tyjt/calibrated_sensor.json")
    return parser.parse_args()

def load_nuscenes_calibration(calib_file):
    """加载nuscenes格式的标定数据"""
    if not os.path.exists(calib_file):
        print(f"❌ 标定文件不存在: {calib_file}")
        return None
    
    with open(calib_file, 'r') as f:
        calib_data = json.load(f)
    
    print(f"✅ 加载nuscenes标定文件: {calib_file}")
    print(f"   标定记录数量: {len(calib_data)}")
    
    calib_index = {}
    for calib in calib_data:
        token = calib['token']
        calib_index[token] = calib
    
    return calib_index

def get_camera_calibration_from_nusc(nusc, sample, cam_channel, calib_index):
    """从nuscenes数据中获取相机标定"""
    try:
        cam_sd_token = sample['data'][cam_channel]
        cam_sd = nusc.get('sample_data', cam_sd_token)
        
        calib_token = cam_sd['calibrated_sensor_token']
        calib = calib_index.get(calib_token)
        
        if not calib:
            print(f"❌ 找不到相机 {cam_channel} 的标定数据，token: {calib_token}")
            return None
        
        translation = np.array(calib['translation'])
        rotation_quat = Quaternion(calib['rotation'])
        intrinsics = np.array(calib['camera_intrinsic'])
        
        calibration = {
            'translation': translation,
            'rotation': rotation_quat,
            'intrinsics': intrinsics,
            'sensor_token': calib_token,
            'sensor_data': cam_sd
        }
        
        print(f"📷 相机 {cam_channel} 标定 (token: {calib_token[:8]}):")
        print(f"   位置: [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}]")
        print(f"   旋转: {rotation_quat}")
        
        return calibration
        
    except Exception as e:
        print(f"❌ 获取相机 {cam_channel} 标定失败: {e}")
        return None

def transform_pointcloud_to_camera_fixed(points, cam_calib):
    """修复版：点云坐标转换"""
    try:
        cam_translation = cam_calib['translation']
        cam_rotation = cam_calib['rotation']
        
        points_centered = points - cam_translation.reshape(3, 1)
        points_cam = np.dot(cam_rotation.inverse.rotation_matrix, points_centered)
        
        front_points = np.sum(points_cam[2] > 0.1)
        print(f"   前方点数: {front_points}/{points_cam.shape[1]}")
        
        return points_cam
        
    except Exception as e:
        print(f"❌ 坐标转换失败: {e}")
        return points

def draw_3d_boxes_fixed(nusc, sample, cam_calib, ax, cam_intrinsic, cam_channel):
    """修复版：绘制3D标注框"""
    boxes_drawn = 0
    try:
        cam_sd_token = sample['data'][cam_channel]
        cam_sd = nusc.get('sample_data', cam_sd_token)
        ego_pose = nusc.get('ego_pose', cam_sd['ego_pose_token'])
        
        ego_trans = np.array(ego_pose['translation'])
        ego_rot = Quaternion(ego_pose['rotation'])
        
        print(f"🔍 {cam_channel} 标注框投影调试:")
        print(f"  ego位置: {ego_trans}")
        print(f"  相机位置: {cam_calib['translation']}")
        
        for i, ann_token in enumerate(sample['anns']):
            try:
                ann = nusc.get('sample_annotation', ann_token)
                box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
                
                # 坐标转换
                box_ego = box.copy()
                box_ego.translate(-ego_trans)
                box_ego.rotate(ego_rot.inverse)
                
                box_cam = box_ego.copy()
                box_cam.translate(-np.array(cam_calib['translation']))
                box_cam.rotate(cam_calib['rotation'].inverse)
                
                corners_cam = box_cam.corners()
                corners_img = view_points(corners_cam, cam_intrinsic, normalize=True)
                
                front_corners = np.sum(corners_cam[2] > 0.1)
                
                if front_corners >= 4:
                    edges = [
                        (0,1), (1,2), (2,3), (3,0),
                        (4,5), (5,6), (6,7), (7,4),
                        (0,4), (1,5), (2,6), (3,7)
                    ]
                    
                    for edge_i, edge_j in edges:
                        if corners_cam[2, edge_i] > 0.1 and corners_cam[2, edge_j] > 0.1:
                            u1, v1 = corners_img[0, edge_i], corners_img[1, edge_i]
                            u2, v2 = corners_img[0, edge_j], corners_img[1, edge_j]
                            
                            if (0 <= u1 < 1920 or 0 <= u2 < 1920) and (0 <= v1 < 1080 or 0 <= v2 < 1080):
                                ax.plot([u1, u2], [v1, v2], 'lime', linewidth=2, alpha=0.8)
                    
                    boxes_drawn += 1
                    
            except Exception as e:
                print(f"    ❌ 标注{i+1}处理失败: {e}")
                continue
                
    except Exception as e:
        print(f"❌ 标注框投影调试失败: {e}")
    
    print(f"  ✅ 成功绘制 {boxes_drawn} 个标注框")
    return boxes_drawn

def visualize_combined_fixed(nusc, sample, cam_channel, output_dir, calib_index):
    """修复版：点云和3D框组合显示"""
    try:
        cam_calib = get_camera_calibration_from_nusc(nusc, sample, cam_channel, calib_index)
        if not cam_calib:
            return
        
        cam_sd_token = sample['data'][cam_channel]
        cam_sd = nusc.get('sample_data', cam_sd_token)
        img_path = os.path.join(nusc.dataroot, cam_sd['filename'])
        
        if not os.path.exists(img_path):
            print(f"❌ 图像文件不存在: {img_path}")
            return
            
        img = plt.imread(img_path)
        img_height, img_width = img.shape[:2]
        
        # 加载激光雷达点云
        lidar_sd_token = sample['data']['LIDAR_TOP']
        lidar_sd = nusc.get('sample_data', lidar_sd_token)
        lidar_path = os.path.join(nusc.dataroot, lidar_sd['filename'])
        
        if not os.path.exists(lidar_path):
            print(f"❌ 点云文件不存在: {lidar_path}")
            return
            
        points = np.load(lidar_path)
        if points.shape[1] >= 3:
            points = points[:, :3].T
        else:
            print(f"❌ 点云数据格式错误: {points.shape}")
            return
        
        # 坐标转换和投影
        points_cam = transform_pointcloud_to_camera_fixed(points, cam_calib)
        cam_intrinsic = cam_calib['intrinsics']
        points_img = view_points(points_cam, cam_intrinsic, normalize=True)
        
        # 过滤点云
        front_mask = points_cam[2] > 0.1
        points_img_front = points_img[:, front_mask]
        points_cam_front = points_cam[:, front_mask]
        
        img_mask = np.all(points_img_front[:2] > 0, axis=0)
        img_mask = np.logical_and(img_mask, points_img_front[0] < img_width)
        img_mask = np.logical_and(img_mask, points_img_front[1] < img_height)
        
        points_img_final = points_img_front[:, img_mask]
        points_cam_final = points_cam_front[:, img_mask]
        
        # 绘制组合图像
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(img)
        
        # 绘制点云 - 使用优化的参数
        if points_img_final.shape[1] > 0:
            depths = points_cam_final[2]
            scatter = ax.scatter(points_img_final[0], points_img_final[1], 
                               c=depths, s=6, alpha=0.4, cmap='plasma',
                               vmin=0, vmax=50)
            plt.colorbar(scatter, ax=ax, label='Depth (m)')
        
        # 绘制3D框
        boxes_drawn = draw_3d_boxes_fixed(nusc, sample, cam_calib, ax, cam_intrinsic, cam_channel)
        
        ax.set_title(f"Combined - {cam_channel} | Points: {points_img_final.shape[1]} | Boxes: {boxes_drawn}", 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        save_path = os.path.join(output_dir, f"{sample['token'][:8]}_{cam_channel}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        
        print(f"💾 保存组合图像: {os.path.basename(save_path)}")
        
    except Exception as e:
        print(f"❌ 组合可视化失败 {cam_channel}: {e}")
        plt.close('all')

def visualize_all_frames_fixed(nusc, output_dir, bev_range, max_frames, calib_index):
    """修复版：可视化所有帧"""
    pointcloud_output = os.path.join(output_dir, "pointcloud_projection")
    bbox_output = os.path.join(output_dir, "bbox_projection")
    combined_output = os.path.join(output_dir, "combined")
    
    os.makedirs(pointcloud_output, exist_ok=True)
    os.makedirs(bbox_output, exist_ok=True)
    os.makedirs(combined_output, exist_ok=True)
    
    frame_count = 0
    for scene in nusc.scene:
        if frame_count >= max_frames:
            break
            
        print(f"\n===== 处理场景：{scene['name']} =====")
        
        sample_tokens = []
        current_sample_token = scene['first_sample_token']
        while current_sample_token:
            sample_tokens.append(current_sample_token)
            current_sample = nusc.get('sample', current_sample_token)
            current_sample_token = current_sample.get('next', '')
        
        for idx, sample_token in enumerate(sample_tokens):
            if frame_count >= max_frames:
                break
                
            sample = nusc.get('sample', sample_token)
            print(f"\n处理帧 {idx+1}/{len(sample_tokens)}（token: {sample_token[:8]}）")
            
            # 4个相机视图
            for cam_channel in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_FRONT_LEFT']:
                visualize_combined_fixed(nusc, sample, cam_channel, combined_output, calib_index)
            
            frame_count += 1

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        calib_index = load_nuscenes_calibration(args.calib_file)
        if not calib_index:
            print("❌ 无法加载标定数据，终止程序")
            return
        
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
        print(f"✅ 已加载数据集（官方接口）：{args.dataroot}")
        
        visualize_all_frames_fixed(nusc, args.output_dir, args.bev_range, args.max_frames, calib_index)
        print(f"\n🎉 所有可视化结果已保存至：{args.output_dir}")
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()