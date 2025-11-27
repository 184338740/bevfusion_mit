import os
import json
import mmcv

def analyze_existing_versions():
    """分析现有版本文件"""
    dataset_root = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt"
    
    print("=== 分析现有版本文件 ===")
    
    # 检查所有版本文件
    version_files = [f for f in os.listdir(dataset_root) if f.startswith('v1.0') and f.endswith('.json')]
    print(f"找到版本文件: {version_files}")
    
    # 加载验证集信息
    val_info_path = os.path.join(dataset_root, "tyjt_infos_val.pkl")
    val_infos = mmcv.load(val_info_path)['infos']
    val_tokens = set(info['token'] for info in val_infos)
    print(f"验证集token数量: {len(val_tokens)}")
    
    # 分析每个版本
    for version_file in version_files:
        version_path = os.path.join(dataset_root, version_file)
        with open(version_path, 'r') as f:
            version_data = json.load(f)
        
        version_name = version_file.replace('.json', '')
        samples = version_data.get('samples', [])
        sample_tokens = set(sample['token'] for sample in samples)
        
        print(f"\n版本: {version_name}")
        print(f"  样本数量: {len(samples)}")
        print(f"  与验证集交集: {len(val_tokens & sample_tokens)}")
        
        # 检查是否有足够的验证集样本
        if len(val_tokens & sample_tokens) == len(val_tokens):
            print(f"  ✅ 包含所有验证集样本")
        else:
            print(f"  ❌ 缺失 {len(val_tokens) - len(val_tokens & sample_tokens)} 个验证集样本")
    
    # 检查训练配置使用的版本
    print(f"\n=== 训练配置检查 ===")
    config_path = "/mnt/bevfusion_mit_xmy/configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/tyjt_convfuser_tmp.yaml"
    
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        for split in ['train', 'val']:
            if split in config.get('data', {}):
                split_config = config['data'][split]
                version = split_config.get('version', '未指定')
                print(f"{split} 使用的版本: {version}")

if __name__ == "__main__":
    analyze_existing_versions()
