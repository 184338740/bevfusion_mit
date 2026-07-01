# debug_visualization_lookup.py
import json
from pathlib import Path

def debug_visualization_lookup():
    """调试可视化工具的数据查找逻辑"""
    data_root = Path("/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/output/step1/nuscenes_tyjt")
    
    # 加载数据
    with open(data_root / "v1.0-tyjt" / "sample.json", 'r') as f:
        samples = json.load(f)
    
    with open(data_root / "v1.0-tyjt" / "sample_data.json", 'r') as f:
        sample_data = json.load(f)
    
    print("=== 可视化工具数据查找调试 ===")
    
    # 模拟可视化工具查找传感器数据的过程
    sample_token = "eaceae02-b27d-b019-8472-68e0948cbcd2"  # 第一个样本
    
    print(f"🔍 查找样本: {sample_token}")
    
    # 方法1: 通过sample_data查找
    camera_data = []
    lidar_data = []
    
    for sd in sample_data:
        if sd.get('sample_token') == sample_token:
            filename = sd.get('filename', '')
            if 'CAM' in filename:
                camera_data.append(sd)
            elif 'LIDAR' in filename:
                lidar_data.append(sd)
    
    print(f"   找到相机数据: {len(camera_data)} 个")
    print(f"   找到激光雷达数据: {len(lidar_data)} 个")
    
    # 检查具体的传感器通道
    sensor_channels = {}
    for sd in sample_data:
        if sd.get('sample_token') == sample_token:
            filename = sd.get('filename', '')
            channel = filename.split('/')[1] if '/' in filename else 'unknown'
            sensor_channels[channel] = sd
    
    print(f"   传感器通道: {list(sensor_channels.keys())}")
    
    # 检查可视化工具可能使用的查找方法
    print(f"\n🔍 检查可视化工具可能的问题:")
    
    # 检查sample_data中的关键字段
    sample_data_record = next((sd for sd in sample_data if sd.get('sample_token') == sample_token), None)
    if sample_data_record:
        print(f"   sample_data关键字段:")
        print(f"     token: {sample_data_record.get('token')}")
        print(f"     sample_token: {sample_data_record.get('sample_token')}")
        print(f"     calibrated_sensor_token: {sample_data_record.get('calibrated_sensor_token')}")
        print(f"     filename: {sample_data_record.get('filename')}")
        print(f"     is_key_frame: {sample_data_record.get('is_key_frame')}")

if __name__ == "__main__":
    debug_visualization_lookup()