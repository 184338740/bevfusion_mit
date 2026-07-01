# check_step1_nusc.py
import os
import json
from nuscenes import NuScenes

def check_step1_output():
    """检查step1转换后的nuscenes数据集"""
    nusc_root = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1121/output/step1-v6/nuscenes_tyjt"
    
    print("=== 检查step1输出的nuscenes数据集 ===")
    
    # 加载nuscenes数据集
    nusc = NuScenes(version='v1.0-tyjt', dataroot=nusc_root, verbose=True)
    
    # 检查第一个样本
    sample = nusc.sample[0]
    print(f"样本token: {sample['token']}")
    print(f"样本中的传感器: {list(sample['data'].keys())}")
    
    # 检查相机数据
    for cam_name in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK']:
        if cam_name in sample['data']:
            cam_token = sample['data'][cam_name]
            cam_data = nusc.get('sample_data', cam_token)
            calibrated_sensor = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            
            print(f"\n{cam_name} 相机标定信息:")
            print(f"  标定token: {cam_data['calibrated_sensor_token']}")
            print(f"  标定字段: {list(calibrated_sensor.keys())}")
            print(f"  是否有rotation: {'rotation' in calibrated_sensor}")
            print(f"  是否有translation: {'translation' in calibrated_sensor}")

if __name__ == "__main__":
    check_step1_output()