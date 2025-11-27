import os

def check_official_config():
    """检查官方配置"""
    config_path = "/mnt/bevfusion_mit_xmy/configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
    
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("=== 官方配置分析 ===")
        
        # 检查数据配置
        if 'data' in config:
            for split in ['train', 'val']:
                if split in config['data']:
                    split_config = config['data'][split]
                    version = split_config.get('version', '❌ 未指定')
                    ann_file = split_config.get('ann_file', '未指定')
                    dataset_root = split_config.get('dataset_root', '未指定')
                    print(f"{split}:")
                    print(f"  version: {version}")
                    print(f"  ann_file: {ann_file}")
                    print(f"  dataset_root: {dataset_root}")
    else:
        print(f"官方配置文件不存在: {config_path}")

if __name__ == "__main__":
    check_official_config()
