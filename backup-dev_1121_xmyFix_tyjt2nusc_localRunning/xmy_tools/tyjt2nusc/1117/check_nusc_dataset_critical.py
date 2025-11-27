import os

def check_nusc_dataset_critical():
    """检查nuscenes_dataset.py中的关键配置"""
    nusc_file = "/mnt/bevfusion_mit_xmy/mmdet3d/datasets/nuscenes_dataset.py"
    
    print("=== 关键代码检查 ===")
    
    with open(nusc_file, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    # 查找_evaluate_single方法
    for i, line in enumerate(lines):
        if 'def _evaluate_single' in line:
            print(f"🔍 找到_evaluate_single方法 (第{i+1}行)")
            # 查找NuScenes初始化
            for j in range(i, min(i+50, len(lines))):
                if 'nusc = NuScenes' in lines[j]:
                    print(f"  第{j+1}行: {lines[j]}")
                    # 打印上下文
                    for k in range(max(0, j-3), min(j+8, len(lines))):
                        arrow = ">>>" if k == j else "   "
                        print(f"  {arrow} {k+1}: {lines[k]}")
                    break
            break
    
    print("\n=== 配置检查 ===")
    # 检查配置文件中版本设置
    config_path = "/mnt/bevfusion_mit_xmy/configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/tyjt_convfuser_tmp.yaml"
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("数据配置:")
        if 'data' in config:
            for split in ['train', 'val']:
                if split in config['data']:
                    split_config = config['data'][split]
                    version = split_config.get('version', '❌ 未指定')
                    print(f"  {split}: version = {version}")

if __name__ == "__main__":
    check_nusc_dataset_critical()
