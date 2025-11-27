import json,os
import mmcv

def check_nusc_version_samples():
    """检查nuscenes版本文件中的样本"""
    dataset_root = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt"
    
    # 检查版本文件

    version_files = [f for f in os.listdir(dataset_root) if f.startswith('v1.0') and f.endswith('.json')]
    print(f"版本文件: {version_files}")
    
    # 加载验证集

    val_info_path = os.path.join(dataset_root, "tyjt_infos_val.pkl")
    val_data = mmcv.load(val_info_path)
    val_infos = val_data['infos']
    val_tokens = [info['token'] for info in val_infos]
    
    print(f"验证集token数量: {len(val_tokens)}")
    
    # 检查每个版本文件

    for version_file in version_files:
        version_path = os.path.join(dataset_root, version_file)
        with open(version_path, 'r') as f:
            version_data = json.load(f)
        
        version_samples = version_data['samples']
        version_tokens = [sample['token'] for sample in version_samples]
        
        print(f"\n版本: {version_file}")
        print(f"  样本数量: {len(version_tokens)}")
        
        # 检查匹配

        common_tokens = set(val_tokens) & set(version_tokens)
        print(f"  与验证集共同样本: {len(common_tokens)}")
        
        if len(common_tokens) < len(val_tokens):
            missing = set(val_tokens) - set(version_tokens)
            print(f"  ❌ 缺失样本: {len(missing)}")
            print(f"    例如: {list(missing)[:3]}")

if __name__ == "__main__":
    check_nusc_version_samples()
