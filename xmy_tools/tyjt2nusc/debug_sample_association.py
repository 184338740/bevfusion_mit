# debug_sample_association.py
import json
from pathlib import Path

def debug_sample_association():
    """调试样本与传感器数据的关联关系"""
    data_root = Path("/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/output/step1/nuscenes_tyjt")
    
    # 加载数据
    with open(data_root / "v1.0-tyjt" / "sample.json", 'r') as f:
        samples = json.load(f)
    
    with open(data_root / "v1.0-tyjt" / "sample_data.json", 'r') as f:
        sample_data = json.load(f)
    
    print("=== 样本与传感器数据关联调试 ===")
    
    # 检查前3个样本
    for i, sample in enumerate(samples[:3]):
        sample_token = sample['token']
        print(f"\n🔍 样本 {i+1}: {sample_token}")
        print(f"   时间戳: {sample.get('timestamp', 'N/A')}")
        
        # 查找该样本的所有传感器数据
        sample_sensor_data = [sd for sd in sample_data if sd.get('sample_token') == sample_token]
        
        print(f"   关联的传感器数据数量: {len(sample_sensor_data)}")
        
        for sd in sample_sensor_data:
            filename = sd.get('filename', 'N/A')
            sensor_modality = "相机" if "CAM" in filename else "激光雷达" if "LIDAR" in filename else "未知"
            file_exists = (data_root / filename).exists()
            status = "✅" if file_exists else "❌"
            print(f"   {status} {sensor_modality}: {filename} (存在: {file_exists})")
    
    # 检查是否有样本没有关联任何传感器数据
    samples_with_data = set(sd['sample_token'] for sd in sample_data if 'sample_token' in sd)
    samples_without_data = [s for s in samples if s['token'] not in samples_with_data]
    
    print(f"\n📊 统计信息:")
    print(f"   总样本数: {len(samples)}")
    print(f"   有关联数据的样本数: {len(samples_with_data)}")
    print(f"   无关联数据的样本数: {len(samples_without_data)}")
    
    if samples_without_data:
        print(f"   前3个无数据样本: {[s['token'] for s in samples_without_data[:3]]}")

if __name__ == "__main__":
    debug_sample_association()