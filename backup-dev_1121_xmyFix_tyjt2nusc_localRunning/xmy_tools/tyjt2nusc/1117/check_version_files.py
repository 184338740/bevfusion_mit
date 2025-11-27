import os

def check_version_files():
    """检查nuscenes版本文件"""
    dataset_root = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt"
    
    print("=== 检查版本文件 ===")
    
    # 列出所有文件
    all_files = os.listdir(dataset_root)
    print(f"数据集根目录文件: {all_files}")
    
    # 检查版本文件
    version_files = [f for f in all_files if f.startswith('v1.0') and f.endswith('.json')]
    print(f"版本文件: {version_files}")
    
    # 检查配置中指定的版本
    config_path = "/mnt/bevfusion_mit_xmy/configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/tyjt_convfuser_tmp.yaml"
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        train_version = config['data']['train'].get('version', '未指定')
        val_version = config['data']['val'].get('version', '未指定')
        print(f"训练集版本: {train_version}")
        print(f"验证集版本: {val_version}")
        
        # 检查版本文件是否存在
        for version_name in [train_version, val_version]:
            if version_name != '未指定':
                version_file = f"{version_name}.json"
                version_path = os.path.join(dataset_root, version_file)
                exists = os.path.exists(version_path)
                print(f"  {version_file}: {'✅ 存在' if exists else '❌ 不存在'}")

if __name__ == "__main__":
    check_version_files()
