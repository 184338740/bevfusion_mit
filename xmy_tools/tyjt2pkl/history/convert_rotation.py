import pickle
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path

def rotation_matrix_to_quaternion(rot_matrix):
    """将3x3旋转矩阵转换为四元数 [w, x, y, z]"""
    try:
        # 转换为numpy数组并验证形状
        rot_matrix = np.array(rot_matrix, dtype=np.float64)
        if rot_matrix.shape != (3, 3):
            raise ValueError(f"旋转矩阵形状错误: {rot_matrix.shape}（需3x3）")
        
        # 转换为四元数（scipy默认返回[x, y, z, w]，调整为[w, x, y, z]）
        r = R.from_matrix(rot_matrix)
        quat = r.as_quat()  # [x, y, z, w]
        return [float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])]  # [w, x, y, z]
    except Exception as e:
        print(f"转换失败: {e}，使用单位四元数替代")
        return [1.0, 0.0, 0.0, 0.0]  # 单位四元数（无旋转）

def convert_pkl_rotations(input_pkl_path, output_pkl_path):
    """转换PKL文件中所有旋转矩阵为四元数"""
    print(f"开始处理: {input_pkl_path}")
    
    # 加载原始PKL
    with open(input_pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # 处理infos中的每帧数据
    if "infos" in data:
        for info in data["infos"]:
            # 1. 转换顶层lidar2ego_rotation
            if "lidar2ego_rotation" in info:
                info["lidar2ego_rotation"] = rotation_matrix_to_quaternion(
                    info["lidar2ego_rotation"]
                )
            
            # 2. 转换顶层ego2global_rotation
            if "ego2global_rotation" in info:
                info["ego2global_rotation"] = rotation_matrix_to_quaternion(
                    info["ego2global_rotation"]
                )
            
            # 3. 转换相机的sensor2ego_rotation和ego2global_rotation
            if "cams" in info:
                for cam_name, cam_info in info["cams"].items():
                    # 相机到自车的旋转
                    if "sensor2ego_rotation" in cam_info:
                        cam_info["sensor2ego_rotation"] = rotation_matrix_to_quaternion(
                            cam_info["sensor2ego_rotation"]
                        )
                    # 相机的自车到全局旋转
                    if "ego2global_rotation" in cam_info:
                        cam_info["ego2global_rotation"] = rotation_matrix_to_quaternion(
                            cam_info["ego2global_rotation"]
                        )
    
    # 保存转换后的PKL
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"转换完成，保存至: {output_pkl_path}\n")

def batch_convert(input_dir, output_dir):
    """批量转换目录下的PKL文件"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 需要转换的文件列表（根据你的实际文件调整）
    pkl_files = [
        "tyjt_infos_train_fixed.pkl",
        "tyjt_infos_val_fixed.pkl",
        "tyjt_dbinfos_train_fixed.pkl"
    ]
    
    for filename in pkl_files:
        input_path = os.path.join(input_dir, filename)
        if not os.path.exists(input_path):
            print(f"跳过不存在的文件: {input_path}\n")
            continue
        
        # 输出文件添加后缀"_quat"区分
        output_filename = f"{os.path.splitext(filename)[0]}_quat.pkl"
        output_path = os.path.join(output_dir, output_filename)
        
        convert_pkl_rotations(input_path, output_path)

if __name__ == "__main__":
    # 输入目录：原始PKL文件所在路径（根据你的实际路径调整）
    input_dir = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1"
    # 输出目录：转换后的PKL文件保存路径
    output_dir = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1"
    
    batch_convert(input_dir, output_dir)