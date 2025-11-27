import os
import yaml
import mmcv

def analyze_config():
    """分析配置文件内容"""
    config_path = "/mnt/bevfusion_mit_xmy/configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/tyjt_convfuser_tmp.yaml"
    
    print("=== 配置文件分析 ===")
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    # 读取配置文件
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("配置文件内容:")
    print(yaml.dump(config, default_flow_style=False))
    
    # 检查关键配置
    print("\n=== 关键配置检查 ===")
    
    # 数据配置
    if 'data' in config:
        for split in ['train', 'val']:
            if split in config['data']:
                print(f"\n{split.upper()}配置:")
                split_config = config['data'][split]
                for key, value in split_config.items():
                    print(f"  {key}: {value}")
                
                # 检查路径是否存在
                if 'ann_file' in split_config:
                    ann_file = split_config['ann_file']
                    if ann_file.startswith('data_root'):
                        # 需要解析data_root
                        data_root = config.get('data_root', '')
                        ann_file = ann_file.replace('data_root', data_root).replace('+', '').replace("'", "").replace('"', '').strip()
                    exists = os.path.exists(ann_file)
                    print(f"  ann_file存在: {exists} ({ann_file})")
                
                if 'data_root' in split_config:
                    data_root = split_config['data_root']
                    exists = os.path.exists(data_root)
                    print(f"  data_root存在: {exists} ({data_root})")
    
    # 验证配置
    if 'evaluation' in config:
        print(f"\n验证配置: {config['evaluation']}")
    else:
        print(f"\n❌ 没有验证配置")
    
    # 模型配置
    if 'model' in config:
        print(f"\n模型类型: {config['model'].get('type', 'N/A')}")
    else:
        print(f"\n❌ 没有模型配置")

def check_data_paths_in_config():
    """检查配置中的数据路径"""
    print("\n=== 数据路径检查 ===")
    
    config_path = "/mnt/bevfusion_mit_xmy/configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/tyjt_convfuser_tmp.yaml"
    
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    print("配置文件中的路径引用:")
    print(config_content)

def validate_dataset_paths():
    """验证数据集路径"""
    print("\n=== 数据集路径验证 ===")
    
    dataset_root = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt"
    
    # 检查必要文件
    required_files = [
        "tyjt_infos_train.pkl",
        "tyjt_infos_val.pkl", 
        "samples/LIDAR_TOP",
        "v1.0-tyjt"
    ]
    
    for file in required_files:
        full_path = os.path.join(dataset_root, file)
        exists = os.path.exists(full_path)
        print(f"{'✅' if exists else '❌'} {file}: {full_path}")
        
        if exists and file.endswith('.pkl'):
            try:
                data = mmcv.load(full_path)
                if 'infos' in data:
                    print(f"   样本数量: {len(data['infos'])}")
            except Exception as e:
                print(f"   加载失败: {e}")

if __name__ == "__main__":
    analyze_config()
    check_data_paths_in_config()
    validate_dataset_paths()
