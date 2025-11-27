import mmcv
import numpy as np
import os

def check_filter_conditions():
    """检查验证时的过滤条件"""
    dataset_root = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt"
    val_info_path = os.path.join(dataset_root, "tyjt_infos_val.pkl")
    
    val_data = mmcv.load(val_info_path)
    val_infos = val_data['infos']
    
    print("=== GT标注框分析 ===")
    
    # 检查可能被过滤的条件
    for i, info in enumerate(val_infos[:5]):  # 检查前5个样本
        gt_boxes = info['gt_boxes']
        gt_names = info['gt_names']
        num_lidar_pts = info.get('num_lidar_pts', [])
        valid_flag = info.get('valid_flag', [])
        
        print(f"样本 {i} ({info['token'][:8]}):")
        print(f"  GT框数量: {len(gt_boxes)}")
        print(f"  类别: {np.unique(gt_names)}")
        
        if 'num_lidar_pts' in info:
            print(f"  点云点数: {num_lidar_pts[:5]}...")  # 前5个
            zero_points = sum(1 for n in num_lidar_pts if n == 0)
            print(f"  零点数框: {zero_points}/{len(num_lidar_pts)}")
        
        if 'valid_flag' in info:
            print(f"  有效标志: {valid_flag[:5]}...")  # 前5个
            invalid_boxes = sum(1 for v in valid_flag if not v)
            print(f"  无效框: {invalid_boxes}/{len(valid_flag)}")
        
        # 检查框的坐标范围
        if len(gt_boxes) > 0:
            centers = gt_boxes[:, :3]
            print(f"  框中心范围: X[{centers[:,0].min():.1f}, {centers[:,0].max():.1f}], "
                  f"Y[{centers[:,1].min():.1f}, {centers[:,1].max():.1f}], "
                  f"Z[{centers[:,2].min():.1f}, {centers[:,2].max():.1f}]")

if __name__ == "__main__":
    check_filter_conditions()
