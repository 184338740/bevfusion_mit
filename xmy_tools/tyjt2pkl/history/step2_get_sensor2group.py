# step2_generate_calibration.py
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """四元数转旋转矩阵"""
    return R.from_quat([qx, qy, qz, qw]).as_matrix()

def rotation_matrix_to_quaternion(rotation_matrix):
    """旋转矩阵转四元数 [x, y, z, w]"""
    return R.from_matrix(rotation_matrix).as_quat()

def create_transform_matrix(tx, ty, tz, rx, ry, rz, rw):
    """创建4x4齐次变换矩阵"""
    rotation = quaternion_to_rotation_matrix(rx, ry, rz, rw)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [tx, ty, tz]
    return transform

def invert_transform(transform):
    """求变换矩阵的逆"""
    inv_transform = np.eye(4)
    inv_rotation = transform[:3, :3].T
    inv_transform[:3, :3] = inv_rotation
    inv_transform[:3, 3] = -inv_rotation @ transform[:3, 3]
    return inv_transform

def generate_sensor2group_calib(sensor2map_file, output_file=None):
    """
    生成sensor到group坐标系的标定文件
    
    Args:
        sensor2map_file: 原始sensor2map标定文件路径
        output_file: 输出文件路径
    
    Returns:
        sensor2group_calib: 传感器到group坐标系的标定字典
    """
    
    print(f"加载标定文件: {sensor2map_file}")
    with open(sensor2map_file, 'r') as f:
        calib_data = json.load(f)
    
    # 获取group2map变换矩阵
    group2map = calib_data['group2map']
    group2map_transform = create_transform_matrix(
        group2map['tx'], group2map['ty'], group2map['tz'],
        group2map['rx'], group2map['ry'], group2map['rz'], group2map['rw']
    )
    
    # 计算map2group变换矩阵（group2map的逆）
    map2group_transform = invert_transform(group2map_transform)
    
    sensor2group_calib = {
        'metadata': {
            'description': 'TYJT传感器到group坐标系的标定文件',
            'original_file': str(sensor2map_file),
            'coordinate_system': 'group坐标系（对应BEVFusion的global坐标系）',
            'conversion_date': np.datetime64('now').astype(str)
        },
        'coordinate_systems': {
            'group': '路口中心坐标系，所有传感器的统一参考系',
            'map': '全局地图坐标系',
            'relationship': 'sensor2group = map2group × sensor2map'
        },
        'camera_mapping': {
            'SC_R1_Aw_CpcS_CAMR_new': 'SC_1A_CamR -> CAM_FRONT_RIGHT',
            'SC_R1_Bn_CpcW_CAMR_new': 'SC_1B_CamR -> CAM_FRONT', 
            'SC_R1_Ce_CpcN_CAMR_new': 'SC_1C_CamR -> CAM_BACK',
            'SC_R1_Ds_CpcE_CAMR_new': 'SC_1D_CamR -> CAM_FRONT_LEFT'
        },
        'transform_matrices': {
            'group2map': {
                'matrix': group2map_transform.tolist(),
                'translation': [group2map['tx'], group2map['ty'], group2map['tz']],
                'rotation_quaternion': [group2map['rx'], group2map['ry'], group2map['rz'], group2map['rw']]
            },
            'map2group': {
                'matrix': map2group_transform.tolist(),
                'translation': map2group_transform[:3, 3].tolist(),
                'rotation_quaternion': rotation_matrix_to_quaternion(map2group_transform[:3, :3]).tolist()
            }
        },
        'cameras': {}
    }
    
    # 相机传感器列表
    camera_sensors = [
        'SC_R1_Aw_CpcS_CAMR_new',  # SC_1A_CamR -> CAM_FRONT_RIGHT
        'SC_R1_Bn_CpcW_CAMR_new',  # SC_1B_CamR -> CAM_FRONT
        'SC_R1_Ce_CpcN_CAMR_new',  # SC_1C_CamR -> CAM_BACK
        'SC_R1_Ds_CpcE_CAMR_new',  # SC_1D_CamR -> CAM_FRONT_LEFT
    ]
    
    print("处理相机传感器标定...")
    for sensor_name in camera_sensors:
        if sensor_name not in calib_data:
            print(f"  警告: 传感器 {sensor_name} 不在标定文件中")
            continue
            
        sensor_calib = calib_data[sensor_name]
        
        # 传感器到map的变换矩阵
        sensor2map_transform = create_transform_matrix(
            sensor_calib['tx'], sensor_calib['ty'], sensor_calib['tz'],
            sensor_calib['rx'], sensor_calib['ry'], sensor_calib['rz'], sensor_calib['rw']
        )
        
        # 计算传感器到group的变换矩阵: sensor2group = map2group × sensor2map
        sensor2group_transform = map2group_transform @ sensor2map_transform
        
        # 提取变换分量
        rotation_matrix = sensor2group_transform[:3, :3]
        translation = sensor2group_transform[:3, 3]
        quaternion = rotation_matrix_to_quaternion(rotation_matrix)
        
        # 保存传感器到group的标定
        sensor2group_calib['cameras'][sensor_name] = {
            'sensor2group_transform': {
                'matrix': sensor2group_transform.tolist(),
                'translation': translation.tolist(),
                'rotation_quaternion': quaternion.tolist(),
                'rotation_matrix': rotation_matrix.tolist()
            },
            'sensor2map_original': {
                'matrix': sensor2map_transform.tolist(),
                'translation': [sensor_calib['tx'], sensor_calib['ty'], sensor_calib['tz']],
                'rotation_quaternion': [sensor_calib['rx'], sensor_calib['ry'], sensor_calib['rz'], sensor_calib['rw']]
            },
            'intrinsics': {
                'fx': sensor_calib['fx'],
                'fy': sensor_calib['fy'],
                'cx': sensor_calib['cx'],
                'cy': sensor_calib['cy'],
                'matrix': [
                    [sensor_calib['fx'], 0, sensor_calib['cx']],
                    [0, sensor_calib['fy'], sensor_calib['cy']],
                    [0, 0, 1]
                ]
            },
            'distortion': {
                'radial_distortion': sensor_calib.get('radial_distortion', []),
                'note': '图像已经过畸变校正'
            },
            'position_in_group': {
                'x': float(translation[0]),
                'y': float(translation[1]),
                'z': float(translation[2]),
                'distance_from_origin': float(np.linalg.norm(translation))
            }
        }
        print(f"  ✅ {sensor_name} 标定转换完成")
    
    if output_file:
        # 确保输出目录存在
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sensor2group_calib, f, indent=2, ensure_ascii=False)
        print(f"传感器到group标定文件已保存: {output_file}")
    
    return sensor2group_calib

def verify_coordinate_transform(sensor2group_calib):
    """验证坐标系转换的正确性"""
    
    print("\n=== 坐标系转换验证 ===")
    
    # 验证group2map和map2group互为逆矩阵
    group2map = np.array(sensor2group_calib['transform_matrices']['group2map']['matrix'])
    map2group = np.array(sensor2group_calib['transform_matrices']['map2group']['matrix'])
    
    identity_check = group2map @ map2group
    identity_error = np.abs(identity_check - np.eye(4)).max()
    print(f"逆矩阵验证误差: {identity_error:.6e} (应接近0)")
    
    if identity_error > 1e-10:
        print("⚠️  逆矩阵误差较大，请检查标定数据")
    
    # 验证传感器转换
    camera_sensors = [
        'SC_R1_Aw_CpcS_CAMR_new',
        'SC_R1_Bn_CpcW_CAMR_new', 
        'SC_R1_Ce_CpcN_CAMR_new',
        'SC_R1_Ds_CpcE_CAMR_new',
    ]
    
    all_errors = []
    for sensor_name in camera_sensors:
        if sensor_name in sensor2group_calib['cameras']:
            sensor_info = sensor2group_calib['cameras'][sensor_name]
            sensor2group = np.array(sensor_info['sensor2group_transform']['matrix'])
            sensor2map_orig = np.array(sensor_info['sensor2map_original']['matrix'])
            
            # 验证: sensor2group ≈ map2group × sensor2map
            reconstructed = map2group @ sensor2map_orig
            reconstruction_error = np.abs(sensor2group - reconstructed).max()
            all_errors.append(reconstruction_error)
            
            status = "✅" if reconstruction_error < 1e-10 else "⚠️ "
            print(f"{status} {sensor_name}: 重建误差 = {reconstruction_error:.6e}")
    
    if all_errors:
        max_error = max(all_errors)
        if max_error < 1e-10:
            print("🎉 所有相机变换验证通过！")
        else:
            print(f"⚠️  最大重建误差: {max_error:.6e}")

def generate_bevfusion_components(sensor2group_calib, output_file):
    """生成BEVFusion格式的变换分量"""
    
    bevfusion_components = {
        'metadata': {
            'description': 'BEVFusion格式的相机变换分量',
            'coordinate_system': 'group坐标系（对应global坐标系）',
            'note': '这些分量可以直接用于构建BEVFusion的相机数据'
        },
        'cameras': {}
    }
    
    camera_mapping = {
        'SC_R1_Aw_CpcS_CAMR_new': 'CAM_FRONT_RIGHT',
        'SC_R1_Bn_CpcW_CAMR_new': 'CAM_FRONT',
        'SC_R1_Ce_CpcN_CAMR_new': 'CAM_BACK',
        'SC_R1_Ds_CpcE_CAMR_new': 'CAM_FRONT_LEFT'
    }
    
    for sensor_name, nusc_cam in camera_mapping.items():
        if sensor_name in sensor2group_calib['cameras']:
            cam_calib = sensor2group_calib['cameras'][sensor_name]
            transform = cam_calib['sensor2group_transform']
            
            bevfusion_components['cameras'][nusc_cam] = {
                'tyjt_original_name': sensor_name,
                'sensor2lidar_rotation': transform['rotation_quaternion'],
                'sensor2lidar_translation': transform['translation'],
                'sensor2ego_rotation': transform['rotation_quaternion'],  # 简化假设
                'sensor2ego_translation': transform['translation'],
                'ego2global_rotation': [0.0, 0.0, 0.0, 1.0],  # 单位四元数
                'ego2global_translation': [0.0, 0.0, 0.0],
                'cam_intrinsic': cam_calib['intrinsics']['matrix'],
                'camera_position': cam_calib['position_in_group']
            }
    
    # 保存文件
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(bevfusion_components, f, indent=2, ensure_ascii=False)
    
    print(f"BEVFusion变换分量已保存: {output_file}")
    return bevfusion_components

def generate_summary_report(sensor2group_calib, output_dir):
    """生成标定摘要报告"""
    
    report_lines = []
    report_lines.append("TYJT到BEVFusion标定转换摘要报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    report_lines.append("1. 相机名称映射")
    report_lines.append("-" * 40)
    for sensor_name, mapping in sensor2group_calib['camera_mapping'].items():
        report_lines.append(f"  {sensor_name} -> {mapping}")
    
    report_lines.append("")
    report_lines.append("2. 相机位置信息（group坐标系）")
    report_lines.append("-" * 40)
    for sensor_name, cam_calib in sensor2group_calib['cameras'].items():
        position = cam_calib['position_in_group']
        mapping = sensor2group_calib['camera_mapping'][sensor_name]
        report_lines.append(f"  {mapping}:")
        report_lines.append(f"    位置: ({position['x']:.2f}, {position['y']:.2f}, {position['z']:.2f})")
        report_lines.append(f"    距离原点: {position['distance_from_origin']:.2f}m")
    
    report_lines.append("")
    report_lines.append("3. 相机内参")
    report_lines.append("-" * 40)
    for sensor_name, cam_calib in sensor2group_calib['cameras'].items():
        mapping = sensor2group_calib['camera_mapping'][sensor_name]
        intrinsics = cam_calib['intrinsics']
        report_lines.append(f"  {mapping}:")
        report_lines.append(f"    焦距: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")
        report_lines.append(f"    主点: cx={intrinsics['cx']:.1f}, cy={intrinsics['cy']:.1f}")
    
    # 保存报告
    report_file = Path(output_dir) / "calibration_summary.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    print(f"标定摘要报告已保存: {report_file}")

def main():
    """主函数"""
    
    # 配置路径
    dataset_root = "/mnt/dataset/tyjt_RawData"
    subscene = "2d3d4d_20250728_weiyuan"
    output_dir = "./step2_output"
    
    # 输入文件路径
    sensor2map_file = Path(dataset_root) / subscene / "calib" / "sensor2map_calib.json"
    
    # 输出文件路径
    sensor2group_output = Path(output_dir) / "sensor2group_calib.json"
    bevfusion_output = Path(output_dir) / "bevfusion_transform_components.json"
    
    print("🚀 开始生成标定文件...")
    print(f"输入文件: {sensor2map_file}")
    print(f"输出目录: {output_dir}")
    print()
    
    if not sensor2map_file.exists():
        print(f"❌ 错误: 标定文件不存在 {sensor2map_file}")
        return
    
    try:
        # 生成传感器到group的标定
        sensor2group_calib = generate_sensor2group_calib(sensor2map_file, sensor2group_output)
        
        # 验证坐标系转换
        verify_coordinate_transform(sensor2group_calib)
        
        # 生成BEVFusion格式的变换分量
        bevfusion_components = generate_bevfusion_components(sensor2group_calib, bevfusion_output)
        
        # 生成摘要报告
        generate_summary_report(sensor2group_calib, output_dir)
        
        print("\n🎉 标定文件生成完成！")
        print(f"📁 输出文件:")
        print(f"  - {sensor2group_output}")
        print(f"  - {bevfusion_output}")
        print(f"  - {Path(output_dir)/'calibration_summary.txt'}")
        
    except Exception as e:
        print(f"❌ 标定文件生成失败: {e}")
        raise

if __name__ == "__main__":
    main()