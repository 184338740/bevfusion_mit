# step2_4_diagnose_data.py
import os
import json
import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion

def diagnose_nuscenes_data():
    """诊断nuscenes数据问题"""
    dataroot = "./output/step1/nuscenes_tyjt/v1.0-tyjt"
    version = "v1.0-tyjt"
    
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    
    print("🔍 ===== 数据诊断报告 =====")
    
    # 1. 检查ego_pose数据
    print("\n1. EGO_POSE 数据检查:")
    zero_ego_count = 0
    for i, ego_pose in enumerate(nusc.ego_pose[:10]):  # 检查前10个
        translation = np.array(ego_pose['translation'])
        if np.allclose(translation, [0, 0, 0]):
            zero_ego_count += 1
        print(f"   EgoPose {i}: 位置={translation}, 时间戳={ego_pose['timestamp']}")
    
    print(f"   前10个ego_pose中位置为0的数量: {zero_ego_count}/10")
    
    # 2. 检查样本标注分布
    print("\n2. 样本标注分布检查:")
    sample_ann_counts = []
    for sample in nusc.sample[:20]:  # 检查前20个样本
        ann_count = len(sample['anns'])
        sample_ann_counts.append(ann_count)
        print(f"   样本 {sample['token'][:8]}: {ann_count}个标注")
    
    print(f"   前20个样本平均标注数: {np.mean(sample_ann_counts):.1f}")
    
    # 3. 检查问题样本
    problem_samples = ['4cbc03f8', '9fe62709', '9b26b94f']
    print(f"\n3. 问题样本详细检查:")
    
    for sample_token in nusc.sample:
        token_short = sample_token['token'][:8]
        if token_short in problem_samples:
            sample = nusc.get('sample', sample_token['token'])
            print(f"\n   🔍 样本 {token_short}:")
            print(f"     标注数量: {len(sample['anns'])}")
            
            # 检查每个标注的详细信息
            for i, ann_token in enumerate(sample['anns']):
                ann = nusc.get('sample_annotation', ann_token)
                print(f"     标注{i+1}: {ann['category_name']}")
                print(f"       位置: {ann['translation']}")
                print(f"       尺寸: {ann['size']}")
                print(f"       旋转: {ann['rotation']}")
    
    # 4. 检查标定数据
    print(f"\n4. 标定数据检查:")
    calib_file = "./output/step1/nuscenes_tyjt/v1.0-tyjt/v1.0-tyjt/calibrated_sensor.json"
    with open(calib_file, 'r') as f:
        calib_data = json.load(f)
    
    print(f"   标定记录总数: {len(calib_data)}")
    
    # 统计各传感器的标定数量
    sensor_calib_count = {}
    for calib in calib_data[:50]:  # 检查前50个
        sensor_token = calib['sensor_token']
        if sensor_token not in sensor_calib_count:
            sensor_calib_count[sensor_token] = 0
        sensor_calib_count[sensor_token] += 1
    
    print(f"   各传感器标定数量: {sensor_calib_count}")
    
    # 5. 检查转换脚本可能的问题
    print(f"\n5. 转换脚本问题排查:")
    
    # 检查step1生成的JSON文件
    json_files = {
        'sample_annotation': 'sample_annotation.json',
        'ego_pose': 'ego_pose.json', 
        'calibrated_sensor': 'calibrated_sensor.json'
    }
    
    for key, filename in json_files.items():
        filepath = os.path.join(dataroot, version, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
            print(f"   {filename}: {len(data)}条记录")
            
            if key == 'ego_pose':
                # 检查ego_pose的translation
                zero_count = sum(1 for item in data[:10] if np.allclose(item['translation'], [0,0,0]))
                print(f"     前10条中translation为0的数量: {zero_count}/10")
            
            if key == 'sample_annotation':
                # 检查标注的rotation
                rotations = [item['rotation'] for item in data[:5]]
                print(f"     前5条rotation示例: {rotations[:2]}...")

if __name__ == '__main__':
    diagnose_nuscenes_data()