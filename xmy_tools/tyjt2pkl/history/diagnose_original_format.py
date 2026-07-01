# diagnose_original_format.py
import pickle
import numpy as np
from pathlib import Path

def diagnose_original_format():
    """诊断原始BEVFusion PKL文件的实际格式"""
    original_pkl_path = "/mnt/bevfusion_mit_xmy/data/nuscenes/nuscenes_infos_train.pkl"
    
    print(f"正在诊断原始BEVFusion PKL文件: {original_pkl_path}")
    
    try:
        with open(original_pkl_path, 'rb') as f:
            original_data = pickle.load(f)
        
        print("✅ 原始PKL文件加载成功")
        
        # 检查第一个样本
        if 'infos' in original_data and original_data['infos']:
            first_sample = original_data['infos'][0]
            
            print("\n原始BEVFusion PKL格式诊断:")
            print("=" * 60)
            
            # 检查相机字段的实际类型
            if 'cams' in first_sample and first_sample['cams']:
                print("相机字段数据类型:")
                first_cam = list(first_sample['cams'].values())[0]
                
                camera_fields = ['cam_intrinsic', 'sensor2lidar_rotation', 'sensor2lidar_translation',
                               'sensor2ego_rotation', 'sensor2ego_translation',
                               'ego2global_rotation', 'ego2global_translation']
                
                for field in camera_fields:
                    if field in first_cam:
                        value = first_cam[field]
                        value_type = type(value).__name__
                        shape_info = f", shape: {value.shape}" if hasattr(value, 'shape') else ""
                        print(f"  {field}: {value_type}{shape_info}")
                        
                        # 如果是numpy数组，显示dtype
                        if isinstance(value, np.ndarray):
                            print(f"    dtype: {value.dtype}")
            
            # 检查标注字段
            print("\n标注字段数据类型:")
            annotation_fields = ['gt_boxes', 'gt_names', 'gt_velocity', 'num_lidar_pts', 'num_radar_pts', 'valid_flag']
            
            for field in annotation_fields:
                if field in first_sample:
                    value = first_sample[field]
                    value_type = type(value).__name__
                    shape_info = f", shape: {value.shape}" if hasattr(value, 'shape') else ""
                    print(f"  {field}: {value_type}{shape_info}")
                    
                    # 如果是numpy数组，显示dtype
                    if isinstance(value, np.ndarray):
                        print(f"    dtype: {value.dtype}")
            
            # 检查坐标系字段
            print("\n坐标系字段数据类型:")
            coord_fields = ['lidar2ego_translation', 'lidar2ego_rotation', 
                           'ego2global_translation', 'ego2global_rotation',
                           'lidar2global', 'ego2global']
            
            for field in coord_fields:
                if field in first_sample:
                    value = first_sample[field]
                    value_type = type(value).__name__
                    shape_info = f", shape: {value.shape}" if hasattr(value, 'shape') else ""
                    print(f"  {field}: {value_type}{shape_info}")
                    
        else:
            print("❌ 原始PKL文件中没有找到infos字段或infos为空")
            
    except FileNotFoundError:
        print(f"❌ 找不到原始PKL文件: {original_pkl_path}")
        print("请检查路径是否正确")
    except Exception as e:
        print(f"❌ 诊断过程中出错: {e}")

if __name__ == "__main__":
    diagnose_original_format()