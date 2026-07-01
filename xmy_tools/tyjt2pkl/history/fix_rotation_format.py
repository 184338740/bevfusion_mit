# fix_all_rotations.py
import pickle
import numpy as np
from scipy.spatial.transform import Rotation

def check_and_fix_rotations():
    # 检查训练文件
    with open('/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_train_correct.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    print("=== 检查训练文件旋转字段 ===")
    for i, info in enumerate(train_data['infos'][:5]):  # 只检查前5个样本
        print(f"样本 {i}:")
        for key, value in info.items():
            if 'rotation' in key:
                print(f"  {key}: {value} (长度: {len(value) if hasattr(value, '__len__') else 'N/A'})")
    
    # 修复所有旋转字段
    print("\n=== 修复所有旋转字段 ===")
    fixed_count = 0
    for info in train_data['infos']:
        for key in list(info.keys()):
            if 'rotation' in key:
                value = info[key]
                if hasattr(value, '__len__') and len(value) == 3:
                    print(f"修复 {key}: {value} -> 四元数")
                    # 欧拉角转四元数
                    rotation = Rotation.from_euler('xyz', value, degrees=False)
                    quaternion = rotation.as_quat()  # [x, y, z, w]
                    info[key] = quaternion.tolist()
                    fixed_count += 1
    
    print(f"训练文件修复了 {fixed_count} 个旋转字段")
    
    with open('/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_train_correct.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    
    # 同样修复验证文件
    with open('/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_val_correct.pkl', 'rb') as f:
        val_data = pickle.load(f)
    
    fixed_count = 0
    for info in val_data['infos']:
        for key in list(info.keys()):
            if 'rotation' in key:
                value = info[key]
                if hasattr(value, '__len__') and len(value) == 3:
                    rotation = Rotation.from_euler('xyz', value, degrees=False)
                    quaternion = rotation.as_quat()
                    info[key] = quaternion.tolist()
                    fixed_count += 1
    
    print(f"验证文件修复了 {fixed_count} 个旋转字段")
    
    with open('/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_val_correct.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    
    print("修复完成！")

if __name__ == '__main__':
    check_and_fix_rotations()