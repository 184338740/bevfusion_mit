#!/usr/bin/env python3
"""
诊断calibrated_sensor表的translation字段问题
"""

import json
from pathlib import Path

def diagnose_calibrated_sensor(data_dir):
    """诊断calibrated_sensor表的translation字段"""
    version_dir = Path(data_dir) / "v1.0-tyjt"
    
    print("🔍 诊断calibrated_sensor表")
    print("=" * 60)
    
    with open(version_dir / "calibrated_sensor.json", 'r') as f:
        sensors = json.load(f)
    
    problematic_sensors = []
    
    for i, sensor in enumerate(sensors):
        translation = sensor.get('translation', [])
        rotation = sensor.get('rotation', [])
        
        # 检查translation
        trans_problem = False
        if not isinstance(translation, list):
            trans_problem = True
        elif len(translation) != 3:
            trans_problem = True
        elif not all(isinstance(x, (int, float)) for x in translation):
            trans_problem = True
        
        # 检查rotation
        rot_problem = False
        if not isinstance(rotation, list):
            rot_problem = True
        elif len(rotation) != 4:
            rot_problem = True
        elif not all(isinstance(x, (int, float)) for x in rotation):
            rot_problem = True
        
        if trans_problem or rot_problem:
            problematic_sensors.append({
                'index': i,
                'token': sensor.get('token', 'unknown'),
                'sensor_token': sensor.get('sensor_token', 'unknown'),
                'translation': translation,
                'translation_issue': trans_problem,
                'rotation': rotation,
                'rotation_issue': rot_problem
            })
    
    print(f"总传感器标定记录数: {len(sensors)}")
    print(f"问题记录数: {len(problematic_sensors)}")
    
    if problematic_sensors:
        print("\n❌ 发现的问题记录:")
        for problem in problematic_sensors[:10]:  # 只显示前10个
            print(f"\n记录 {problem['index']}:")
            print(f"  token: {problem['token']}")
            print(f"  sensor_token: {problem['sensor_token']}")
            if problem['translation_issue']:
                print(f"  ❌ translation问题: {problem['translation']} (长度: {len(problem['translation'])})")
            else:
                print(f"  ✅ translation正常: {problem['translation']}")
            
            if problem['rotation_issue']:
                print(f"  ❌ rotation问题: {problem['rotation']} (长度: {len(problem['rotation'])})")
            else:
                print(f"  ✅ rotation正常: {problem['rotation']}")
    
    # 统计传感器类型
    print(f"\n📊 传感器类型统计:")
    with open(version_dir / "sensor.json", 'r') as f:
        sensor_types = json.load(f)
    
    sensor_map = {s['token']: s['channel'] for s in sensor_types}
    
    for problem in problematic_sensors[:5]:
        sensor_channel = sensor_map.get(problem['sensor_token'], 'unknown')
        print(f"  问题记录 {problem['index']}: {sensor_channel}")

if __name__ == "__main__":
    data_dir = "./output-1226-v1.9/step1/nuscenes_tyjt"
    diagnose_calibrated_sensor(data_dir)