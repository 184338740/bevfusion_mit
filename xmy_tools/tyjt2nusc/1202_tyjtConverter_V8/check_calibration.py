# check_calibration_fixed.py
import os
import pickle

def check_calibration_data():
    # nusc_root = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1121/output/step1-v6/nuscenes_tyjt"
    nusc_root = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1117/output/step1/nuscenes_tyjt"

    
    # 检查pkl文件中的标定数据
    with open(os.path.join(nusc_root, "tyjt_infos_train.pkl"), 'rb') as f:
        train_data = pickle.load(f)
    
    print("=== 检查pkl文件结构 ===")
    print("数据类型:", type(train_data))
    print("字典键:", list(train_data.keys()))
    
    # 正确的数据结构：{'infos': list, 'metadata': dict}
    if 'infos' in train_data and isinstance(train_data['infos'], list):
        infos = train_data['infos']
        print("样本数量:", len(infos))
        
        if len(infos) > 0:
            sample = infos[0]
            print("样本键:", list(sample.keys()))
            
            if 'cams' in sample:
                cam_name = list(sample['cams'].keys())[0]
                cam_info = sample['cams'][cam_name]
                print(f"相机 {cam_name} 的键:", list(cam_info.keys()))
                
                # 检查缺少的字段
                required_keys = ['sensor2lidar_rotation', 'sensor2lidar_translation', 'lidar2sensor_rotation', 'lidar2sensor_translation']
                for key in required_keys:
                    if key in cam_info:
                        print(f"✅ {key}: 存在")
                    else:
                        print(f"❌ {key}: 缺失")
            else:
                print("❌ 样本中没有 'cams' 字段")
    else:
        print("❌ 无法找到样本数据")

if __name__ == "__main__":
    check_calibration_data()