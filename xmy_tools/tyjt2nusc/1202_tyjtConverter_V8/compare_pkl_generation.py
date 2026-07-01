# compare_pkl_generation.py
import os
import pickle

def compare_pkl_files():
    """对比1117和1121版本生成的pkl文件差异"""
    
    # 1117版本（成功的）
    pkl_1117 = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1117/output/step1/nuscenes_tyjt/tyjt_infos_train.pkl"
    # 1121版本（失败的）  
    pkl_1121 = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1121/output/step1-v6/nuscenes_tyjt/tyjt_infos_train.pkl"
    
    if os.path.exists(pkl_1117) and os.path.exists(pkl_1121):
        print("=== 对比两个版本的pkl文件 ===")
        
        with open(pkl_1117, 'rb') as f:
            data_1117 = pickle.load(f)
        with open(pkl_1121, 'rb') as f:
            data_1121 = pickle.load(f)
        
        # 检查结构
        print(f"1117版本结构: {type(data_1117)}, 键: {list(data_1117.keys())}")
        print(f"1121版本结构: {type(data_1121)}, 键: {list(data_1121.keys())}")
        
        # 检查第一个样本的相机信息
        if 'infos' in data_1117 and 'infos' in data_1121:
            sample_1117 = data_1117['infos'][0]
            sample_1121 = data_1121['infos'][0]
            
            print(f"\n=== 第一个样本对比 ===")
            print(f"1117样本键: {list(sample_1117.keys())}")
            print(f"1121样本键: {list(sample_1121.keys())}")
            
            if 'cams' in sample_1117 and 'cams' in sample_1121:
                cam_1117 = list(sample_1117['cams'].values())[0]
                cam_1121 = list(sample_1121['cams'].values())[0]
                
                print(f"\n=== 相机信息对比 ===")
                print(f"1117相机键: {list(cam_1117.keys())}")
                print(f"1121相机键: {list(cam_1121.keys())}")
                
                # 检查关键标定字段
                key_fields = ['sensor2lidar_rotation', 'sensor2lidar_translation']
                for field in key_fields:
                    exists_1117 = field in cam_1117
                    exists_1121 = field in cam_1121
                    print(f"{field}: 1117={exists_1117}, 1121={exists_1121}")
                    
                    if exists_1117 and exists_1121:
                        # 检查值是否相同
                        val_1117 = cam_1117[field]
                        val_1121 = cam_1121[field]
                        print(f"  1117值形状: {np.array(val_1117).shape if hasattr(val_1117, '__len__') else 'scalar'}")
                        print(f"  1121值形状: {np.array(val_1121).shape if hasattr(val_1121, '__len__') else 'scalar'}")
            else:
                print("❌ 样本中没有cams字段")
        else:
            print("❌ pkl文件中没有infos字段")
    else:
        print("❌ 找不到pkl文件")
        print(f"1117版本存在: {os.path.exists(pkl_1117)}")
        print(f"1121版本存在: {os.path.exists(pkl_1121)}")

if __name__ == "__main__":
    compare_pkl_files()