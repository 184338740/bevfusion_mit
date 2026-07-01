import pickle
import torch
from mmdet3d.datasets import build_dataset
from mmcv import Config

def test_data_loading():
    print("=== 测试数据加载 ===")
    
    # 加载配置
    cfg = Config.fromfile('/mnt/bevfusion_mit_xmy/configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/tyjt_convfuser.yaml')
    
    # 构建数据集
    try:
        dataset = build_dataset(cfg.data.train)
        print(f"数据集构建成功! 长度: {len(dataset)}")
        
        # 测试加载第一个样本
        sample = dataset[0]
        print(f"样本加载成功!")
        print(f"样本键: {list(sample.keys())}")
        
        # 检查关键字段
        for key in ['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d']:
            if key in sample:
                value = sample[key]
                if hasattr(value, 'shape'):
                    print(f"{key}: {value.shape}")
                else:
                    print(f"{key}: {type(value)}")
            else:
                print(f"{key}: 不存在")
                
    except Exception as e:
        print(f"数据加载失败: {e}")
        import traceback
        traceback.print_exc()

def check_pkl_content():
    print("\n=== 检查PKL文件内容 ===")
    
    # 检查训练数据
    with open('/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_train_fixed.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    print(f"训练数据metadata: {train_data.get('metadata', '无metadata')}")
    print(f"训练样本数量: {len(train_data['infos'])}")
    
    # 检查第一个样本的详细结构
    sample = train_data['infos'][0]
    print(f"\n第一个样本结构:")
    print(f"token: {sample['token']}")
    print(f"timestamp: {sample['timestamp']}")
    print(f"相机数量: {len(sample['cams'])}")
    print(f"激光雷达路径: {sample['lidar_path']}")
    print(f"标注框形状: {sample['gt_boxes'].shape}")
    print(f"标注名称: {sample['gt_names']}")
    
    # 检查标注框格式
    print(f"\n标注框格式检查:")
    print(f"gt_boxes维度: {sample['gt_boxes'].shape[1]}")
    if sample['gt_boxes'].shape[1] == 7:
        print("标注框格式: [x, y, z, dx, dy, dz, yaw]")
    elif sample['gt_boxes'].shape[1] == 9:
        print("标注框格式: [x, y, z, dx, dy, dz, yaw, vx, vy]")
    else:
        print(f"未知标注框格式: {sample['gt_boxes'].shape[1]}维")

if __name__ == '__main__':
    check_pkl_content()
    test_data_loading()