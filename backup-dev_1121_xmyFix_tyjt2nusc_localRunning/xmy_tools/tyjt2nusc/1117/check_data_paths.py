import os
import mmcv
from pathlib import Path

def check_data_paths():
    """详细检查数据路径问题"""
    val_info_path = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/tyjt_infos_val.pkl"
    val_infos = mmcv.load(val_info_path)['infos']
    
    print("=== 详细路径检查 ===")
    
    # 检查前5个样本
    for i, info in enumerate(val_infos[:5]):
        lidar_path = info['lidar_path']
        print(f"\n样本 {i}:")
        print(f"  Token: {info['token'][:8]}...")
        print(f"  Lidar路径: {lidar_path}")
        
        # 检查不同路径解析方式
        paths_to_check = [
            lidar_path,  # 原始路径
            f"/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/{lidar_path}",
            lidar_path.replace("/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/", ""),
        ]
        
        for path in paths_to_check:
            exists = os.path.exists(path)
            print(f"    路径: {path}")
            print(f"    存在: {exists}")
            if exists:
                file_size = os.path.getsize(path)
                print(f"    文件大小: {file_size} bytes")
                break

def check_dataset_root():
    """检查数据集根目录配置"""
    print("\n=== 数据集根目录检查 ===")
    
    # 检查可能的根目录
    possible_roots = [
        "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt",
        "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1",
        "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output",
    ]
    
    for root in possible_roots:
        samples_dir = os.path.join(root, "samples", "LIDAR_TOP")
        exists = os.path.exists(samples_dir)
        bin_files = list(Path(samples_dir).glob("*.bin")) if exists else []
        print(f"根目录: {root}")
        print(f"  samples/LIDAR_TOP存在: {exists}")
        print(f"  .bin文件数量: {len(bin_files)}")
        
        if bin_files:
            print(f"  示例文件: {bin_files[0].name}")

def check_config_paths():
    """检查训练配置文件中的路径设置"""
    print("\n=== 训练配置检查 ===")
    
    config_path = "configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/tyjt_convfuser_tmp.yaml"
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'data' in config and 'val' in config['data']:
            val_config = config['data']['val']
            print(f"验证集配置:")
            print(f"  ann_file: {val_config.get('ann_file', 'N/A')}")
            print(f"  dataset_root: {val_config.get('dataset_root', 'N/A')}")
            
            # 检查这些路径是否存在
            ann_file = val_config.get('ann_file', '')
            dataset_root = val_config.get('dataset_root', '')
            
            if ann_file:
                print(f"  ann_file存在: {os.path.exists(ann_file)}")
            if dataset_root:
                print(f"  dataset_root存在: {os.path.exists(dataset_root)}")

if __name__ == "__main__":
    check_data_paths()
    check_dataset_root()
    check_config_paths()
EOFcat > check_data_paths.py << 'EOF'
import os
import mmcv
from pathlib import Path

def check_data_paths():
    """详细检查数据路径问题"""
    val_info_path = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/tyjt_infos_val.pkl"
    val_infos = mmcv.load(val_info_path)['infos']
    
    print("=== 详细路径检查 ===")
    
    # 检查前5个样本
    for i, info in enumerate(val_infos[:5]):
        lidar_path = info['lidar_path']
        print(f"\n样本 {i}:")
        print(f"  Token: {info['token'][:8]}...")
        print(f"  Lidar路径: {lidar_path}")
        
        # 检查不同路径解析方式
        paths_to_check = [
            lidar_path,  # 原始路径
            f"/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/{lidar_path}",
            lidar_path.replace("/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/", ""),
        ]
        
        for path in paths_to_check:
            exists = os.path.exists(path)
            print(f"    路径: {path}")
            print(f"    存在: {exists}")
            if exists:
                file_size = os.path.getsize(path)
                print(f"    文件大小: {file_size} bytes")
                break

def check_dataset_root():
    """检查数据集根目录配置"""
    print("\n=== 数据集根目录检查 ===")
    
    # 检查可能的根目录
    possible_roots = [
        "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt",
        "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1",
        "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output",
    ]
    
    for root in possible_roots:
        samples_dir = os.path.join(root, "samples", "LIDAR_TOP")
        exists = os.path.exists(samples_dir)
        bin_files = list(Path(samples_dir).glob("*.bin")) if exists else []
        print(f"根目录: {root}")
        print(f"  samples/LIDAR_TOP存在: {exists}")
        print(f"  .bin文件数量: {len(bin_files)}")
        
        if bin_files:
            print(f"  示例文件: {bin_files[0].name}")

def check_config_paths():
    """检查训练配置文件中的路径设置"""
    print("\n=== 训练配置检查 ===")
    
    config_path = "configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/tyjt_convfuser_tmp.yaml"
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'data' in config and 'val' in config['data']:
            val_config = config['data']['val']
            print(f"验证集配置:")
            print(f"  ann_file: {val_config.get('ann_file', 'N/A')}")
            print(f"  dataset_root: {val_config.get('dataset_root', 'N/A')}")
            
            # 检查这些路径是否存在
            ann_file = val_config.get('ann_file', '')
            dataset_root = val_config.get('dataset_root', '')
            
            if ann_file:
                print(f"  ann_file存在: {os.path.exists(ann_file)}")
            if dataset_root:
                print(f"  dataset_root存在: {os.path.exists(dataset_root)}")

if __name__ == "__main__":
    check_data_paths()
    check_dataset_root()
    check_config_paths()
