# step2_3_check_vis-v1.7-nusc-api.py
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

def parse_args():
    parser = argparse.ArgumentParser(description="使用nuscenes API的可视化验证工具")
    parser.add_argument('--dataroot', type=str, default="./output/step1/nuscenes_tyjt/v1.0-tyjt")
    parser.add_argument('--version', type=str, default="v1.0-tyjt")
    parser.add_argument('--output_dir', type=str, default="./output/step2_nusc_api")
    parser.add_argument('--max_samples', type=int, default=10, help="最大处理样本数")
    return parser.parse_args()

def find_problem_samples_with_nusc(nusc):
    """使用nuscenes API查找问题样本"""
    print("🔍 使用nuscenes API查找问题样本...")
    
    problem_samples = []
    
    # 查找标注数量少的样本
    for sample in nusc.sample:
        ann_count = len(sample['anns'])
        if ann_count <= 3:
            problem_samples.append(sample['token'])
    
    print(f"找到 {len(problem_samples)} 个标注数量≤3的样本")
    
    # 显示前几个问题样本的详情
    for i, sample_token in enumerate(problem_samples[:5]):
        sample = nusc.get('sample', sample_token)
        print(f"\n样本 {sample_token[:8]}: {len(sample['anns'])} 个标注")
        
        for j, ann_token in enumerate(sample['anns']):
            ann = nusc.get('sample_annotation', ann_token)
            category = nusc.get('category', ann['category_token'])
            print(f"  标注{j+1}: {category['name']}")
            print(f"    位置: {ann['translation']}")
            print(f"    尺寸: {ann['size']}")
    
    return problem_samples

def visualize_sample_with_nusc(nusc, sample_token, output_dir):
    """使用nuscenes API可视化样本"""
    sample = nusc.get('sample', sample_token)
    
    print(f"\n🎯 可视化样本: {sample_token[:8]}")
    print(f"  标注数量: {len(sample['anns'])}")
    
    # 为每个相机通道可视化
    for cam_channel in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_FRONT_LEFT']:
        if cam_channel not in sample['data']:
            print(f"⚠️ 跳过 {cam_channel}: 无数据")
            continue
        
        try:
            # 获取相机数据
            cam_sd_token = sample['data'][cam_channel]
            cam_sd = nusc.get('sample_data', cam_sd_token)
            img_path = os.path.join(nusc.dataroot, cam_sd['filename'])
            
            if not os.path.exists(img_path):
                print(f"❌ 图像文件不存在: {img_path}")
                continue
            
            # 加载图像
            img = plt.imread(img_path)
            img_height, img_width = img.shape[:2]
            
            # 获取标定数据
            ego_pose = nusc.get('ego_pose', cam_sd['ego_pose_token'])
            calib_token = cam_sd['calibrated_sensor_token']
            cam_calib = nusc.get('calibrated_sensor', calib_token)
            cam_intrinsic = np.array(cam_calib['camera_intrinsic'])
            
            print(f"  {cam_channel}: 图像 {img_width}x{img_height}, 相机位置: {cam_calib['translation']}")
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.imshow(img)
            
            # 绘制3D框
            boxes_drawn = 0
            for ann_token in sample['anns']:
                try:
                    ann = nusc.get('sample_annotation', ann_token)
                    category = nusc.get('category', ann['category_token'])
                    
                    print(f"    处理 {category['name']}: 位置{ann['translation']}, 尺寸{ann['size']}")
                    
                    # 创建3D框
                    box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
                    
                    # 坐标转换
                    box_ego = box.copy()
                    ego_trans = np.array(ego_pose['translation'])
                    ego_rot = Quaternion(ego_pose['rotation'])
                    box_ego.translate(-ego_trans)
                    box_ego.rotate(ego_rot.inverse)
                    
                    box_cam = box_ego.copy()
                    cam_trans = np.array(cam_calib['translation'])
                    cam_rot = Quaternion(cam_calib['rotation'])
                    box_cam.translate(-cam_trans)
                    box_cam.rotate(cam_rot.inverse)
                    
                    corners_cam = box_cam.corners()
                    corners_img = view_points(corners_cam, cam_intrinsic, normalize=True)
                    
                    # 深度检查
                    depths = corners_cam[2]
                    front_corners = np.sum(depths > 0.1)
                    
                    # 检查是否在图像内
                    in_image_mask = (corners_img[0] >= 0) & (corners_img[0] < img_width) & \
                                   (corners_img[1] >= 0) & (corners_img[1] < img_height)
                    in_image_count = np.sum(in_image_mask)
                    
                    print(f"      深度范围: [{depths.min():.1f}, {depths.max():.1f}], "
                          f"前方角点: {front_corners}/8, 图像内: {in_image_count}/8")
                    
                    if front_corners >= 4 and in_image_count >= 4:
                        # 绘制3D框
                        edges = [
                            (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4),
                            (0,4), (1,5), (2,6), (3,7)
                        ]
                        
                        visible_edges = 0
                        for edge_i, edge_j in edges:
                            if depths[edge_i] > 0.1 and depths[edge_j] > 0.1:
                                u1, v1 = corners_img[0, edge_i], corners_img[1, edge_i]
                                u2, v2 = corners_img[0, edge_j], corners_img[1, edge_j]
                                
                                if (0 <= u1 < img_width and 0 <= v1 < img_height and
                                    0 <= u2 < img_width and 0 <= v2 < img_height):
                                    
                                    ax.plot([u1, u2], [v1, v2], 'lime', linewidth=2, alpha=0.8)
                                    visible_edges += 1
                        
                        if visible_edges >= 6:
                            boxes_drawn += 1
                            print(f"      ✅ 成功绘制 {category['name']} ({visible_edges}条边)")
                        else:
                            print(f"      ⚠️ {category['name']} 可见边不足: {visible_edges}")
                    else:
                        print(f"      ❌ {category['name']} 不在视野内")
                        
                except Exception as e:
                    print(f"    ❌ 标注处理失败: {e}")
                    continue
            
            # 设置标题
            ax.set_title(f"3D Boxes - {cam_channel} | Sample: {sample_token[:8]}\n"
                        f"Boxes: {boxes_drawn}/{len(sample['anns'])} | Image: {img_width}x{img_height}", 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # 保存图像
            save_path = os.path.join(output_dir, f"{sample_token[:8]}_{cam_channel}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
            plt.close(fig)
            
            print(f"💾 保存图像: {os.path.basename(save_path)}")
            
        except Exception as e:
            print(f"❌ {cam_channel} 可视化失败: {e}")
            import traceback
            traceback.print_exc()
            plt.close('all')

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        print(f"🚀 启动nuscenes API可视化工具...")
        print(f"数据路径: {args.dataroot}")
        print(f"版本: {args.version}")
        
        # 使用nuscenes API加载数据
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
        print(f"✅ 已加载数据集")
        
        # 查找问题样本
        problem_samples = find_problem_samples_with_nusc(nusc)
        
        if not problem_samples:
            print("❌ 未找到问题样本，使用前几个样本")
            problem_samples = [sample['token'] for sample in nusc.sample[:args.max_samples]]
        
        # 可视化问题样本
        print(f"\n🎯 开始可视化 {len(problem_samples[:args.max_samples])} 个样本...")
        for i, sample_token in enumerate(problem_samples[:args.max_samples]):
            print(f"\n[{i+1}/{min(len(problem_samples), args.max_samples)}]")
            visualize_sample_with_nusc(nusc, sample_token, args.output_dir)
        
        # 检查输出
        output_files = os.listdir(args.output_dir)
        if output_files:
            print(f"\n🎉 可视化完成! 生成的图像文件:")
            for file in sorted(output_files):
                if file.endswith('.png'):
                    print(f"  ✅ {file}")
        else:
            print(f"\n❌ 输出目录为空: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()