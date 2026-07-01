# check_step3_process.py
import os
import pickle
import json

def check_pkl_generation_process():
    """检查pkl生成过程中的数据状态"""
    
    print("=== 对比1117和1121版本的数据源 ===")
    
    # 检查两个版本的nuscenes数据集基础信息
    nusc_1117 = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1117/output/step1/nuscenes_tyjt"
    nusc_1121 = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1121/output/step1-v6/nuscenes_tyjt"
    
    for version, path in [("1117", nusc_1117), ("1121", nusc_1121)]:
        print(f"\n=== {version}版本nuscenes数据集 ===")
        
        # 检查calibrated_sensor.json
        calib_file = os.path.join(path, "v1.0-tyjt", "calibrated_sensor.json")
        if os.path.exists(calib_file):
            with open(calib_file, 'r') as f:
                calib_data = json.load(f)
                if len(calib_data) > 0:
                    first_calib = calib_data[0]
                    print(f"  第一个标定数据键: {list(first_calib.keys())}")
                    print(f"  是否有rotation: {'rotation' in first_calib}")
                    print(f"  是否有translation: {'translation' in first_calib}")
        else:
            print(f"  ❌ 标定文件不存在: {calib_file}")

if __name__ == "__main__":
    check_pkl_generation_process()