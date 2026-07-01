import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """四元数转旋转矩阵"""
    return R.from_quat([qx, qy, qz, qw]).as_matrix()

def rotation_matrix_to_quaternion(rotation_matrix):
    """旋转矩阵转四元数"""
    return R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]

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
    生成sensor到group坐标系的标定文件，并保存group2map标定参数
    
    Args:
        sensor2map_file: 原始sensor2map标定文件路径
        output_file: 输出文件路径，如不指定则返回字典
    
    Returns:
        sensor2group_calib: 传感器到group坐标系的标定字典
    """
    
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
    
    sensor2group_calib = {}
    
    # 保存group2map标定参数到sensor2group坐标系
    sensor2group_calib['group2map_original'] = {
        'transform_matrix': group2map_transform.tolist(),
        'translation': {
            'tx': group2map['tx'],
            'ty': group2map['ty'],
            'tz': group2map['tz']
        },
        'rotation': {
            'rx': group2map['rx'],
            'ry': group2map['ry'],
            'rz': group2map['rz'],
            'rw': group2map['rw']
        },
        'description': 'group坐标系到map坐标系的原始变换参数'
    }
    
    # 保存map2group标定参数
    map2group_quat = rotation_matrix_to_quaternion(map2group_transform[:3, :3])
    sensor2group_calib['map2group'] = {
        'transform_matrix': map2group_transform.tolist(),
        'translation': {
            'tx': float(map2group_transform[0, 3]),
            'ty': float(map2group_transform[1, 3]),
            'tz': float(map2group_transform[2, 3])
        },
        'rotation': {
            'rx': float(map2group_quat[0]),
            'ry': float(map2group_quat[1]),
            'rz': float(map2group_quat[2]),
            'rw': float(map2group_quat[3])
        },
        'description': 'map坐标系到group坐标系的变换参数（group2map的逆）'
    }
    
    # 相机传感器列表
    camera_sensors = [
        'SC_R1_Aw_CpcS_CAMR_new',  # SC_1A_CamR
        'SC_R1_Bn_CpcW_CAMR_new',  # SC_1B_CamR  
        'SC_R1_Ce_CpcN_CAMR_new',  # SC_1C_CamR
        'SC_R1_Ds_CpcE_CAMR_new',  # SC_1D_CamR
    ]
    
    # 处理每个相机传感器
    for sensor_name in camera_sensors:
        if sensor_name not in calib_data:
            print(f"警告: 传感器 {sensor_name} 不在标定文件中")
            continue
            
        sensor_calib = calib_data[sensor_name]
        
        # 传感器到map的变换矩阵
        sensor2map_transform = create_transform_matrix(
            sensor_calib['tx'], sensor_calib['ty'], sensor_calib['tz'],
            sensor_calib['rx'], sensor_calib['ry'], sensor_calib['rz'], sensor_calib['rw']
        )
        
        # 计算传感器到group的变换矩阵: sensor2group = map2group × sensor2map
        sensor2group_transform = map2group_transform @ sensor2map_transform
        
        # 提取变换参数
        rotation_matrix = sensor2group_transform[:3, :3]
        translation = sensor2group_transform[:3, 3]
        
        # 旋转矩阵转四元数
        quaternion = rotation_matrix_to_quaternion(rotation_matrix)
        
        # 保存传感器到group的标定
        sensor2group_calib[sensor_name] = {
            'sensor2group_transform': sensor2group_transform.tolist(),
            'sensor2map_original': {
                'transform_matrix': sensor2map_transform.tolist(),
                'translation': {
                    'tx': sensor_calib['tx'],
                    'ty': sensor_calib['ty'],
                    'tz': sensor_calib['tz']
                },
                'rotation': {
                    'rx': sensor_calib['rx'],
                    'ry': sensor_calib['ry'],
                    'rz': sensor_calib['rz'],
                    'rw': sensor_calib['rw']
                }
            },
            'translation': {
                'tx': float(translation[0]),
                'ty': float(translation[1]), 
                'tz': float(translation[2])
            },
            'rotation': {
                'rx': float(quaternion[0]),
                'ry': float(quaternion[1]),
                'rz': float(quaternion[2]),
                'rw': float(quaternion[3])
            },
            'intrinsics': {
                'fx': sensor_calib['fx'],
                'fy': sensor_calib['fy'], 
                'cx': sensor_calib['cx'],
                'cy': sensor_calib['cy']
            },
            'radial_distortion': sensor_calib.get('radial_distortion', []),
            'description': f'{sensor_name}传感器到group坐标系的变换参数'
        }
    
    # 添加group坐标系信息（单位矩阵）
    sensor2group_calib['group_coordinate_system'] = {
        'transform_matrix': np.eye(4).tolist(),
        'translation': {'tx': 0.0, 'ty': 0.0, 'tz': 0.0},
        'rotation': {'rx': 0.0, 'ry': 0.0, 'rz': 0.0, 'rw': 1.0},
        'description': 'group坐标系定义（原点）'
    }
    
    # 添加坐标系关系说明
    sensor2group_calib['coordinate_system_relationships'] = {
        'group': '路口的中心坐标系，所有传感器数据的统一参考系',
        'map': '全局地图坐标系',
        'sensor2group': 'sensor2group = map2group × sensor2map',
        'map2group': 'map2group = inverse(group2map)',
        'data_usage': '所有传感器数据应使用sensor2group变换到group坐标系'
    }
    
    if output_file:
        # 确保输出目录存在
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(sensor2group_calib, f, indent=2, ensure_ascii=False)
        print(f"传感器到group标定文件已保存: {output_file}")
    
    return sensor2group_calib

def verify_coordinate_transform(sensor2group_calib):
    """验证坐标系转换的正确性"""
    
    print("\n=== 坐标系转换验证 ===")
    
    # 验证group2map和map2group互为逆矩阵
    group2map = np.array(sensor2group_calib['group2map_original']['transform_matrix'])
    map2group = np.array(sensor2group_calib['map2group']['transform_matrix'])
    
    identity_check = group2map @ map2group
    identity_error = np.abs(identity_check - np.eye(4)).max()
    print(f"逆矩阵验证误差: {identity_error:.6f} (应接近0)")
    
    # 验证传感器转换
    camera_sensors = [
        'SC_R1_Aw_CpcS_CAMR_new',
        'SC_R1_Bn_CpcW_CAMR_new', 
        'SC_R1_Ce_CpcN_CAMR_new',
        'SC_R1_Ds_CpcE_CAMR_new',
    ]
    
    for sensor_name in camera_sensors:
        if sensor_name in sensor2group_calib:
            sensor_info = sensor2group_calib[sensor_name]
            sensor2group = np.array(sensor_info['sensor2group_transform'])
            sensor2map_orig = np.array(sensor_info['sensor2map_original']['transform_matrix'])
            
            # 验证: sensor2group ≈ map2group × sensor2map
            reconstructed = map2group @ sensor2map_orig
            reconstruction_error = np.abs(sensor2group - reconstructed).max()
            
            print(f"{sensor_name}: 重建误差 = {reconstruction_error:.6f}")
            
            # 显示传感器位置信息
            tx, ty, tz = sensor_info['translation']['tx'], sensor_info['translation']['ty'], sensor_info['translation']['tz']
            print(f"  group系位置: ({tx:.2f}, {ty:.2f}, {tz:.2f})")

def test_coordinate_transform():
    """测试坐标系转换"""
    
    # 示例使用
    dataset_root = "/mnt/dataset/tyjt_RawData"
    subscene = "2d3d4d_20250728_weiyuan"
    
    sensor2map_file = Path(dataset_root) / subscene / "calib" / "sensor2map_calib.json"
    output_file = Path(dataset_root) / subscene / "calib" / "sensor2group_calib.json"
    
    if sensor2map_file.exists():
        print("开始坐标系转换...")
        sensor2group_calib = generate_sensor2group_calib(sensor2map_file, output_file)
        
        # 验证转换结果
        verify_coordinate_transform(sensor2group_calib)
        
        # 显示关键信息
        print("\n=== 关键标定参数 ===")
        group2map = sensor2group_calib['group2map_original']['translation']
        print(f"group2map平移: ({group2map['tx']:.2f}, {group2map['ty']:.2f}, {group2map['tz']:.2f})")
        
        for sensor_name in ['SC_R1_Aw_CpcS_CAMR_new', 'SC_R1_Bn_CpcW_CAMR_new', 
                           'SC_R1_Ce_CpcN_CAMR_new', 'SC_R1_Ds_CpcE_CAMR_new']:
            if sensor_name in sensor2group_calib:
                calib = sensor2group_calib[sensor_name]
                tx, ty, tz = calib['translation']['tx'], calib['translation']['ty'], calib['translation']['tz']
                fx, fy = calib['intrinsics']['fx'], calib['intrinsics']['fy']
                print(f"\n{sensor_name}:")
                print(f"  group系位置: ({tx:.2f}, {ty:.2f}, {tz:.2f})")
                print(f"  相机内参: fx={fx:.1f}, fy={fy:.1f}")
                
    else:
        print(f"标定文件不存在: {sensor2map_file}")

if __name__ == "__main__":
    test_coordinate_transform()