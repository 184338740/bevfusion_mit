import os
import json
from nuscenes import NuScenes

def fix_nuscenes_dataset():
    """修复nuscenes数据集版本问题"""
    dataset_root = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt"
    version = "v1.0-tyjt"
    
    print("=== 修复NuScenes数据集版本 ===")
    
    # 检查nuscenes元数据文件
    nuscenes_json = os.path.join(dataset_root, f"{version}.json")
    if not os.path.exists(nuscenes_json):
        print(f"❌ NuScenes元数据文件不存在: {nuscenes_json}")
        return False
    
    # 加载nuscenes元数据
    with open(nuscenes_json, 'r') as f:
        nuscenes_data = json.load(f)
    
    print(f"数据集版本: {nuscenes_data.get('version', 'N/A')}")
    print(f"样本数量: {len(nuscenes_data.get('samples', []))}")
    print(f"场景数量: {len(nuscenes_data.get('scenes', []))}")
    
    # 检查验证集样本token是否在nuscenes元数据中
    val_info_path = os.path.join(dataset_root, "tyjt_infos_val.pkl")
    import mmcv
    val_infos = mmcv.load(val_info_path)['infos']
    
    val_tokens = set(info['token'] for info in val_infos)
    nuscenes_tokens = set(sample['token'] for sample in nuscenes_data.get('samples', []))
    
    print(f"验证集token数量: {len(val_tokens)}")
    print(f"NuScenes元数据token数量: {len(nuscenes_tokens)}")
    print(f"交集数量: {len(val_tokens & nuscenes_tokens)}")
    
    # 如果token不匹配，需要修复
    if len(val_tokens & nuscenes_tokens) != len(val_tokens):
        print("❌ Token不匹配，需要修复数据集版本")
        return False
    else:
        print("✅ Token匹配正确")
        return True

def create_correct_version():
    """创建正确的数据集版本"""
    dataset_root = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt"
    
    # 检查现有的版本文件
    version_files = [f for f in os.listdir(dataset_root) if f.startswith('v1.0')]
    print(f"现有的版本文件: {version_files}")
    
    # 建议创建mini版本用于测试
    val_info_path = os.path.join(dataset_root, "tyjt_infos_val.pkl")
    import mmcv
    val_infos = mmcv.load(val_info_path)['infos']
    
    # 创建mini版本
    mini_data = {
        "version": "v1.0-mini",
        "samples": [],
        "scenes": [{"token": "test_scene", "name": "test", "description": "test"}]
    }
    
    # 添加验证集样本到mini版本
    for i, info in enumerate(val_infos[:10]):  # 只取前10个样本
        sample = {
            "token": info['token'],
            "timestamp": info['timestamp'],
            "scene_token": "test_scene",
            "prev": info.get('prev_token', ""),
            "next": "",
            "data": {
                "LIDAR_TOP": info['lidar_path'].replace(dataset_root + '/', ''),
                "CAM_FRONT": info['cams']['CAM_FRONT']['data_path'].replace(dataset_root + '/', ''),
                "CAM_FRONT_LEFT": info['cams']['CAM_FRONT_LEFT']['data_path'].replace(dataset_root + '/', ''),
                "CAM_FRONT_RIGHT": info['cams']['CAM_FRONT_RIGHT']['data_path'].replace(dataset_root + '/', ''),
                "CAM_BACK": info['cams']['CAM_BACK']['data_path'].replace(dataset_root + '/', '')
            }
        }
        mini_data["samples"].append(sample)
    
    # 保存mini版本
    mini_json_path = os.path.join(dataset_root, "v1.0-mini.json")
    with open(mini_json_path, 'w') as f:
        json.dump(mini_data, f, indent=2)
    
    print(f"✅ 创建mini版本: {mini_json_path}")
    print(f"包含 {len(mini_data['samples'])} 个样本")

if __name__ == "__main__":
    if not fix_nuscenes_dataset():
        print("\n尝试创建正确的版本...")
        create_correct_version()
