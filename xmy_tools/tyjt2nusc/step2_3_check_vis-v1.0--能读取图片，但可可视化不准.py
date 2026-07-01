import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

# 中文显示配置
plt.rcParams["font.family"] = ["Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

def parse_args():
    parser = argparse.ArgumentParser(description="修复所有可视化问题的工具")
    parser.add_argument('--dataroot', type=str, default="./output/step1/nuscenes_tyjt/v1.0-tyjt")
    parser.add_argument('--version', type=str, default="v1.0-tyjt")
    parser.add_argument('--output_dir', type=str, default="./nusc_complete_fixed")
    parser.add_argument('--bev_range', type=int, default=75)
    return parser.parse_args()

def verify_data_completeness(nusc):
    """验证数据完备性"""
    print("===== 【官方工具】数据完备性验证 =====")
    valid = True
    required_tables = [
        'scene', 'sample', 'sample_data', 'calibrated_sensor',
        'sensor', 'ego_pose', 'sample_annotation', 'instance',
        'category', 'attribute', 'visibility', 'log'
    ]
    for table in required_tables:
        if not hasattr(nusc, table) or len(getattr(nusc, table)) == 0:
            print(f"❌ 元数据表 {table} 缺失或为空")
            valid = False
        else:
            print(f"✅ 元数据表 {table} 验证通过（{len(getattr(nusc, table))} 条记录）")
    print(f"===== 数据完备性验证 {'通过' if valid else '失败'} =====")
    return valid

def get_sample_tokens(nusc, scene_token):
    """获取场景所有帧的token"""
    sample_tokens = []
    current_sample_token = nusc.get('scene', scene_token)['first_sample_token']
    while current_sample_token:
        sample_tokens.append(current_sample_token)
        current_sample = nusc.get('sample', current_sample_token)
        current_sample_token = current_sample['next']
    return sample_tokens

def batch_rotate(points, quaternion):
    """批量旋转点云（[N,3] → [N,3]）"""
    rot_matrix = quaternion.rotation_matrix
    return np.dot(points, rot_matrix.T)

def visualize_all_frames(nusc, output_dir, bev_range):
    """可视化所有帧（修复所有问题）"""
    img_output = os.path.join(output_dir, "images_with_pc")  # 图像+点云+2D框
    bev_output = os.path.join(output_dir, "bev_views")       # BEV视图+3D框
    os.makedirs(img_output, exist_ok=True)
    os.makedirs(bev_output, exist_ok=True)
    
    for scene in nusc.scene:
        print(f"\n===== 处理场景：{scene['name']} =====")
        sample_tokens = get_sample_tokens(nusc, scene['token'])
        
        for idx, sample_token in enumerate(sample_tokens):
            sample = nusc.get('sample', sample_token)
            print(f"处理帧 {idx+1}/{len(sample_tokens)}（token: {sample_token[:8]}）")
            
            # 处理4个相机视图（修复坐标范围1e8问题）
            for cam_channel in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_FRONT_LEFT']:
                visualize_camera_view(nusc, sample, cam_channel, img_output, bev_range)
            
            # 处理BEV视图（修复Box框显示问题）
            visualize_bev_view(nusc, sample, bev_output, bev_range)

def visualize_camera_view(nusc, sample, cam_channel, output_dir, bev_range):
    """相机视图：图像+点云+3D框2D投影（修复坐标范围）"""
    # 1. 加载相机图像
    cam_sd_token = sample['data'][cam_channel]
    cam_sd = nusc.get('sample_data', cam_sd_token)
    img_path = os.path.join(nusc.dataroot, cam_sd['filename'])
    img = plt.imread(img_path)
    img_h, img_w = img.shape[:2]  # 获取图像尺寸用于过滤
    
    # 2. 加载激光雷达点云（世界坐标系）
    lidar_sd_token = sample['data']['LIDAR_TOP']
    lidar_sd = nusc.get('sample_data', lidar_sd_token)
    lidar_path = os.path.join(nusc.dataroot, lidar_sd['filename'])
    lidar_points = np.load(lidar_path)[:, :3]  # [N,3]
    
    # 3. 自车姿态（世界坐标系）
    ego_pose = nusc.get('ego_pose', cam_sd['ego_pose_token'])
    ego_trans = np.array(ego_pose['translation'])
    ego_rot_quat = Quaternion(ego_pose['rotation'])
    
    # 4. 点云→自车坐标系（核心：解决坐标范围过大）
    lidar_points_ego = lidar_points - ego_trans  # 平移
    lidar_points_ego = batch_rotate(lidar_points_ego, ego_rot_quat.inverse)  # 旋转
    
    # 5. 点云→相机坐标系
    cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    cs_trans = np.array(cs['translation'])
    cs_rot_quat = Quaternion(cs['rotation'])
    lidar_points_cam = lidar_points_ego - cs_trans  # 平移
    lidar_points_cam = batch_rotate(lidar_points_cam, cs_rot_quat.inverse)  # 旋转
    
    # 6. 过滤异常点（双重过滤确保范围正常）
    # 过滤超出BEV范围的点
    in_bev_range = (
        (lidar_points_ego[:, 0] > -bev_range) & (lidar_points_ego[:, 0] < bev_range) &
        (lidar_points_ego[:, 1] > -bev_range) & (lidar_points_ego[:, 1] < bev_range)
    )
    # 过滤超出图像范围的点（避免投影到1e8）
    points_img = view_points(lidar_points_cam.T, np.array(cs['camera_intrinsic']), normalize=True).T
    in_img_range = (
        (points_img[:, 0] > 0) & (points_img[:, 0] < img_w) &
        (points_img[:, 1] > 0) & (points_img[:, 1] < img_h) &
        (lidar_points_cam[:, 2] > 0)  # 相机前方
    )
    # 合并过滤条件
    valid = in_bev_range & in_img_range
    lidar_points_cam = lidar_points_cam[valid]
    points_img = points_img[valid].astype(int)
    
    # 7. 绘制图像+点云+3D框2D投影
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    # 绘制点云
    ax.scatter(points_img[:, 0], points_img[:, 1], c='r', s=1, alpha=0.5, label='激光雷达点云')
    # 绘制3D框的2D投影
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        # 世界坐标系3D框
        box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
        # 转换到自车坐标系
        box.translate(-ego_trans)
        box.rotate(ego_rot_quat.inverse)
        # 过滤超出BEV范围的框
        if (abs(box.center[0]) > bev_range) or (abs(box.center[1]) > bev_range):
            continue
        # 转换到相机坐标系
        box_cam = box.copy()
        box_cam.translate(-cs_trans)
        box_cam.rotate(cs_rot_quat.inverse)
        # 投影到图像并绘制
        corners = box_cam.corners()  # [3,8]
        corners_img = view_points(corners, np.array(cs['camera_intrinsic']), normalize=True)  # [2,8]
        # 只绘制在图像内的框边
        edges = [(0,1), (1,3), (3,2), (2,0), (4,5), (5,7), (7,6), (6,4), (0,4), (1,5), (2,6), (3,7)]
        for i, j in edges:
            # 检查两个端点是否都在图像内
            if (0 < corners_img[0,i] < img_w) and (0 < corners_img[1,i] < img_h) and \
               (0 < corners_img[0,j] < img_w) and (0 < corners_img[1,j] < img_h):
                ax.plot([corners_img[0,i], corners_img[0,j]], 
                        [corners_img[1,i], corners_img[1,j]], 'g-', linewidth=2)
    
    ax.set_title(f"相机视图：{cam_channel}（帧token: {sample['token'][:8]}）")
    ax.legend()
    save_path = os.path.join(output_dir, f"{sample['token'][:8]}_{cam_channel}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def visualize_bev_view(nusc, sample, output_dir, bev_range):
    """BEV视图：3D框鸟瞰投影（修复Box框显示）"""
    # 1. 自车姿态
    lidar_sd_token = sample['data']['LIDAR_TOP']
    lidar_sd = nusc.get('sample_data', lidar_sd_token)
    ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    ego_trans = np.array(ego_pose['translation'])
    ego_rot_quat = Quaternion(ego_pose['rotation'])
    
    # 2. 创建BEV图（范围[-75,75]）
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f"BEV视图（范围[-{bev_range},{bev_range}]，帧token: {sample['token'][:8]}）")
    
    # 3. 绘制所有3D框的BEV投影（修复框显示）
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        # 世界坐标系3D框
        box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
        # 转换到自车坐标系
        box.translate(-ego_trans)
        box.rotate(ego_rot_quat.inverse)
        # 过滤范围外的框
        if (abs(box.center[0]) > bev_range) or (abs(box.center[1]) > bev_range):
            continue
        
        # 核心：正确绘制3D框的BEV投影（底面+顶面边框）
        corners = box.corners()  # [3,8]，8个角点
        # 底面4个点（索引0-3）和顶面4个点（索引4-7）
        bottom_corners = corners[:2, [0,1,3,2,0]]  # xy平面，闭合四边形
        top_corners = corners[:2, [4,5,7,6,4]]
        # 绘制底面（实线）和顶面（虚线）
        ax.plot(bottom_corners[0, :], bottom_corners[1, :], 'b-', linewidth=2)
        ax.plot(top_corners[0, :], top_corners[1, :], 'b--', linewidth=1)
        # 绘制高度线（连接底面和顶面）
        for i in range(4):
            ax.plot([bottom_corners[0,i], top_corners[0,i]], 
                    [bottom_corners[1,i], top_corners[1,i]], 'b-', linewidth=1)
        # 标注类别
        ax.text(box.center[0], box.center[1], ann['category_name'], fontsize=8)
    
    # 4. 标记自车位置
    ax.plot(0, 0, 'ro', markersize=8, label='自车位置')
    ax.legend()
    save_path = os.path.join(output_dir, f"{sample['token'][:8]}_bev.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print(f"✅ 已加载数据集（1.0.9官方接口）：{args.dataroot}")
    
    if not verify_data_completeness(nusc):
        print("❌ 数据不完备，终止可视化")
        return
    
    visualize_all_frames(nusc, args.output_dir, args.bev_range)
    print(f"\n===== 所有可视化结果已保存至：{args.output_dir} =====")

if __name__ == '__main__':
    main()