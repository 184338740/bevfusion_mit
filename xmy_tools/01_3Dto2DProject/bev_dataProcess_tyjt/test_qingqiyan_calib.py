import json
import numpy as np

def load_calibration_data():
    with open('sensor2map_calib.json', 'r') as f:
        return json.load(f)

def load_target_calib():
    with open('calib_qingqiyan.json', 'r') as f:
        return json.load(f)

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """四元数转旋转矩阵"""
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),     2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),     1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),     2*(qy*qz+qx*qw),     1-2*(qx*qx+qy*qy)]
    ])

def analyze_transform_relationship():
    """分析map坐标系到group坐标系的变换关系"""
    
    print("🔍 分析map到group坐标系的变换关系")
    print("=" * 60)
    
    calib_data = load_calibration_data()
    target_calib = load_target_calib()
    
    camera_mapping = {
        'SC_R1_Aw_CpcS_CAMR': 'SC_R1_Aw_CpcS_CAMR_new',
        'SC_R1_Bn_CpcW_CAMR': 'SC_R1_Bn_CpcW_CAMR_new',
        'SC_R1_Ce_CpcN_CAMR': 'SC_R1_Ce_CpcN_CAMR_new',
        'SC_R1_Ds_CpcE_CAMR': 'SC_R1_Ds_CpcE_CAMR_new'
    }
    
    # 收集对应点
    map_points = []
    group_points = []
    
    for target_cam, source_cam in camera_mapping.items():
        if source_cam in calib_data and target_cam in target_calib:
            source_data = calib_data[source_cam]
            target_data = target_calib[target_cam]
            
            map_points.append([source_data['tx'], source_data['ty'], source_data['tz']])
            group_points.append([
                target_data['extrinsic'][3], 
                target_data['extrinsic'][7], 
                target_data['extrinsic'][11]
            ])
    
    map_points = np.array(map_points)
    group_points = np.array(group_points)
    
    print("📊 对应点数据:")
    for i, (map_pt, group_pt) in enumerate(zip(map_points, group_points)):
        print(f"  点{i+1}: map{map_pt} → group{group_pt}")
    
    # 计算刚性变换：group = R * map + t
    # 使用Umeyama算法计算最优刚性变换
    
    # 中心化
    map_center = np.mean(map_points, axis=0)
    group_center = np.mean(group_points, axis=0)
    
    map_centered = map_points - map_center
    group_centered = group_points - group_center
    
    # 计算协方差矩阵
    H = map_centered.T @ group_centered
    
    # SVD分解
    U, S, Vt = np.linalg.svd(H)
    
    # 计算旋转矩阵
    R = Vt.T @ U.T
    
    # 确保右手坐标系
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 计算平移向量
    t = group_center - R @ map_center
    
    print(f"\n🎯 计算出的变换参数:")
    print(f"  旋转矩阵 R:")
    print(f"    {R[0]}")
    print(f"    {R[1]}")
    print(f"    {R[2]}")
    print(f"  平移向量 t: {t}")
    
    # 验证变换
    print(f"\n✅ 变换验证:")
    max_error = 0
    for i, (map_pt, expected_group) in enumerate(zip(map_points, group_points)):
        transformed = R @ map_pt + t
        error = np.linalg.norm(transformed - expected_group)
        max_error = max(max_error, error)
        print(f"  点{i+1}: 误差 = {error:.6f}")
    
    print(f"  最大误差: {max_error:.6f}")
    
    return R, t

def generate_with_calculated_transform():
    """使用计算出的变换参数生成标定文件"""
    
    print("\n🎯 使用计算出的变换生成标定文件")
    print("=" * 60)
    
    calib_data = load_calibration_data()
    
    # 计算最优变换参数
    R_map2group, t_map2group = analyze_transform_relationship()
    
    camera_mapping = {
        'SC_R1_Aw_CpcS_CAMR': 'SC_R1_Aw_CpcS_CAMR_new',
        'SC_R1_Bn_CpcW_CAMR': 'SC_R1_Bn_CpcW_CAMR_new',
        'SC_R1_Ce_CpcN_CAMR': 'SC_R1_Ce_CpcN_CAMR_new',
        'SC_R1_Ds_CpcE_CAMR': 'SC_R1_Ds_CpcE_CAMR_new'
    }
    
    qingqiyan_calib = {"formated": "true"}
    
    print("\n📷 生成各相机标定:")
    
    for target_cam, source_cam in camera_mapping.items():
        if source_cam in calib_data:
            source_data = calib_data[source_cam]
            
            # 传感器在map坐标系中的位姿
            R_sensor_map = quaternion_to_rotation_matrix(
                source_data['rx'], source_data['ry'], 
                source_data['rz'], source_data['rw']
            )
            t_sensor_map = np.array([source_data['tx'], source_data['ty'], source_data['tz']])
            
            # 构建传感器到map的变换矩阵
            T_sensor_map = np.eye(4)
            T_sensor_map[:3, :3] = R_sensor_map
            T_sensor_map[:3, 3] = t_sensor_map
            
            # 构建map到group的变换矩阵
            T_map2group = np.eye(4)
            T_map2group[:3, :3] = R_map2group
            T_map2group[:3, 3] = t_map2group
            
            # 应用变换：T_sensor_group = T_map2group × T_sensor_map
            T_sensor_group = T_map2group @ T_sensor_map
            
            # 提取group坐标系中的旋转和平移
            R_sensor_group = T_sensor_group[:3, :3]
            t_sensor_group = T_sensor_group[:3, 3]
            
            print(f"\n✅ {target_cam}:")
            print(f"  map位置: [{t_sensor_map[0]:.3f}, {t_sensor_map[1]:.3f}, {t_sensor_map[2]:.3f}]")
            print(f"  group位置: [{t_sensor_group[0]:.6f}, {t_sensor_group[1]:.6f}, {t_sensor_group[2]:.6f}]")
            
            # 构建外参矩阵
            extrinsic = [
                R_sensor_group[0,0], R_sensor_group[0,1], R_sensor_group[0,2], t_sensor_group[0],
                R_sensor_group[1,0], R_sensor_group[1,1], R_sensor_group[1,2], t_sensor_group[1],
                R_sensor_group[2,0], R_sensor_group[2,1], R_sensor_group[2,2], t_sensor_group[2],
                0.0, 0.0, 0.0, 1.0
            ]
            
            # 内参矩阵
            intrinsic = [
                source_data['fx'], 0.0, source_data['cx'], 0.0,
                0.0, source_data['fy'], source_data['cy'], 0.0,
                0.0, 0.0, 1.0, 0.0
            ]
            
            qingqiyan_calib[target_cam] = {
                "intrinsic": intrinsic,
                "extrinsic": extrinsic
            }
    
    # 保存文件
    output_file = 'calib_qingqiyan_calculated_transform.json'
    with open(output_file, 'w') as f:
        json.dump(qingqiyan_calib, f, indent=4)
    
    print(f"\n💾 生成完成: {output_file}")
    
    # 验证与目标文件的一致性
    validate_consistency(qingqiyan_calib)
    
    return qingqiyan_calib

def validate_consistency(generated_calib):
    """验证生成文件与目标文件的一致性"""
    
    print("\n" + "=" * 60)
    print("🔍 验证生成结果一致性")
    print("=" * 60)
    
    target_calib = load_target_calib()
    
    camera_names = ['SC_R1_Aw_CpcS_CAMR', 'SC_R1_Bn_CpcW_CAMR', 
                   'SC_R1_Ce_CpcN_CAMR', 'SC_R1_Ds_CpcE_CAMR']
    
    all_matched = True
    tolerance = 1e-10
    
    for cam_name in camera_names:
        if cam_name in generated_calib and cam_name in target_calib:
            gen_cam = generated_calib[cam_name]
            target_cam = target_calib[cam_name]
            
            # 比较外参矩阵
            gen_extrinsic = np.array(gen_cam['extrinsic'])
            target_extrinsic = np.array(target_cam['extrinsic'])
            
            diff = np.max(np.abs(gen_extrinsic - target_extrinsic))
            matched = diff < tolerance
            
            status = "✅" if matched else "❌"
            print(f"{status} {cam_name}: 最大差异 = {diff:.10f}")
            
            if not matched:
                all_matched = False
                # 显示具体差异
                gen_pos = [gen_extrinsic[3], gen_extrinsic[7], gen_extrinsic[11]]
                target_pos = [target_extrinsic[3], target_extrinsic[7], target_extrinsic[11]]
                print(f"    生成位置: {gen_pos}")
                print(f"    目标位置: {target_pos}")
    
    print(f"\n🎯 最终结果: {'✅ 完全一致' if all_matched else '❌ 存在差异'}")
    
    if all_matched:
        print("🎉 成功找到正确的坐标系变换关系！")
    else:
        print("💡 可能需要更复杂的变换模型（如仿射变换）")

def main():
    """主函数"""
    
    print("🚀 寻找正确的map到group坐标系变换")
    print("=" * 60)
    
    # 使用计算出的最优变换生成标定文件
    result = generate_with_calculated_transform()
    
    print("\n📋 总结:")
    print("  通过对应点分析计算出了map到group的最优刚性变换")
    print("  这个变换应该能正确生成与目标文件一致的标定")

if __name__ == "__main__":
    main()