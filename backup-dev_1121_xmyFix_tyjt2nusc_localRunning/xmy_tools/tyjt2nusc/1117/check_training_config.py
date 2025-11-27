import os
import yaml
import mmcv

def check_training_config():
    """检查训练配置文件"""
    config_path = "configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/tyjt_convfuser_tmp.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    print("=== 训练配置检查 ===")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 检查数据配置
    if 'data' in config:
        print("数据配置:")
        for split in ['train', 'val']:
            if split in config['data']:
                split_config = config['data'][split]
                print(f"\n{split.upper()}配置:")
                print(f"  type: {split_config.get('type', 'N/A')}")
                print(f"  ann_file: {split_config.get('ann_file', 'N/A')}")
                print(f"  dataset_root: {split_config.get('dataset_root', 'N/A')}")
                
                # 检查路径是否存在
                ann_file = split_config.get('ann_file', '')
                dataset_root = split_config.get('dataset_root', '')
                
                if ann_file:
                    print(f"  ann_file存在: {os.path.exists(ann_file)}")
                if dataset_root:
                    print(f"  dataset_root存在: {os.path.exists(dataset_root)}")
    
    # 检查验证配置
    if 'evaluation' in config:
        print(f"\n验证配置: {config['evaluation']}")
    
    # 检查模型配置
    if 'model' in config:
        print(f"\n模型类型: {config['model'].get('type', 'N/A')}")

def check_dataset_loading():
    """检查数据集加载过程"""
    print("\n=== 数据集加载检查 ===")
    
    # 检查验证集pkl文件内容
    val_info_path = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/tyjt_infos_val.pkl"
    if os.path.exists(val_info_path):
        data = mmcv.load(val_info_path)
        print(f"验证集pkl文件结构: {type(data)}")
        if isinstance(data, dict):
            print(f"数据键值: {data.keys()}")
            if 'infos' in data:
                infos = data['infos']
                print(f"样本数量: {len(infos)}")
                
                # 检查第一个样本的完整信息
                sample = infos[0]
                print(f"\n第一个样本的关键信息:")
                print(f"  token: {sample['token']}")
                print(f"  gt_names: {sample['gt_names']}")
                print(f"  gt_boxes形状: {sample['gt_boxes'].shape}")
                print(f"  时间戳: {sample['timestamp']}")
                
                # 检查是否有相机数据
                if 'cams' in sample:
                    print(f"  相机数量: {len(sample['cams'])}")
                    for cam_name, cam_info in sample['cams'].items():
                        print(f"    {cam_name}: {cam_info['data_path']}")

def check_validation_process():
    """检查验证过程可能的问题"""
    print("\n=== 验证过程检查 ===")
    
    # 检查nuscenes数据集版本
    dataset_root = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt"
    
    # 检查必要的nuscenes文件
    required_files = [
        "tyjt_infos_val.pkl",
        "tyjt_infos_train.pkl", 
        "samples/LIDAR_TOP",
        "v1.0-tyjt"
    ]
    
    print("必要文件检查:")
    for file in required_files:
        full_path = os.path.join(dataset_root, file)
        exists = os.path.exists(full_path)
        print(f"  {file}: {'✅' if exists else '❌'}")

if __name__ == "__main__":
    check_training_config()
    check_dataset_loading() 
    check_validation_process()
