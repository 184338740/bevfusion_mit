#!/usr/bin/env python3
"""
计算 sensor2map 和 map2sensor 变换矩阵
"""

import numpy as np
import math

def quaternion_to_rotation_matrix(q):
    """四元数转旋转矩阵 - 输入 [w, x, y, z]"""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def rotation_matrix_to_quaternion(matrix):
    """旋转矩阵转四元数 - 输出 [w, x, y, z]"""
    m = np.array(matrix, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    
    return [w, x, y, z]

def calculate_transforms(tx, ty, tz, rx, ry, rz, rw):
    """
    计算 sensor2map 和 map2sensor 变换
    
    参数:
        tx, ty, tz: 平移分量
        rx, ry, rz, rw: 四元数分量 [x, y, z, w] 格式
    """
    print("=" * 60)
    print("输入参数:")
    print(f"  平移: [{tx:.6f}, {ty:.6f}, {tz:.6f}]")
    print(f"  四元数: [{rx:.10f}, {ry:.10f}, {rz:.10f}, {rw:.10f}]")
    print("=" * 60)
    
    # 1. 计算 sensor2map
    print("\n1. 计算 sensor2map 变换矩阵:")
    
    # 构建 sensor2map 矩阵
    R_sensor2map = quaternion_to_rotation_matrix([rw, rx, ry, rz])  # [w, x, y, z]
    T_sensor2map = np.array([tx, ty, tz])
    
    sensor2map = np.eye(4)
    sensor2map[:3, :3] = R_sensor2map
    sensor2map[:3, 3] = T_sensor2map
    
    print("  旋转矩阵 R_sensor2map:")
    for i in range(3):
        print(f"    [{R_sensor2map[i,0]:.6f}, {R_sensor2map[i,1]:.6f}, {R_sensor2map[i,2]:.6f}]")
    
    print(f"\n  平移向量 T_sensor2map:")
    print(f"    [{T_sensor2map[0]:.6f}, {T_sensor2map[1]:.6f}, {T_sensor2map[2]:.6f}]")
    
    print("\n  完整的 sensor2map 4x4 矩阵:")
    for i in range(4):
        print(f"    [{sensor2map[i,0]:.6f}, {sensor2map[i,1]:.6f}, {sensor2map[i,2]:.6f}, {sensor2map[i,3]:.6f}]")
    
    # 2. 计算 map2sensor (sensor2map 的逆)
    print("\n" + "=" * 60)
    print("2. 计算 map2sensor 变换矩阵 (sensor2map 的逆):")
    
    map2sensor = np.linalg.inv(sensor2map)
    
    R_map2sensor = map2sensor[:3, :3]
    T_map2sensor = map2sensor[:3, 3]
    
    print("  旋转矩阵 R_map2sensor:")
    for i in range(3):
        print(f"    [{R_map2sensor[i,0]:.6f}, {R_map2sensor[i,1]:.6f}, {R_map2sensor[i,2]:.6f}]")
    
    print(f"\n  平移向量 T_map2sensor:")
    print(f"    [{T_map2sensor[0]:.6f}, {T_map2sensor[1]:.6f}, {T_map2sensor[2]:.6f}]")
    
    print("\n  完整的 map2sensor 4x4 矩阵:")
    for i in range(4):
        print(f"    [{map2sensor[i,0]:.6f}, {map2sensor[i,1]:.6f}, {map2sensor[i,2]:.6f}, {map2sensor[i,3]:.6f}]")
    
    # 3. 提取 map2sensor 的四元数参数
    print("\n" + "=" * 60)
    print("3. map2sensor 的参数 (tx, ty, tz, rx, ry, rz, rw 格式):")
    
    quat_map2sensor = rotation_matrix_to_quaternion(R_map2sensor)  # [w, x, y, z]
    x, y, z, w = quat_map2sensor[1], quat_map2sensor[2], quat_map2sensor[3], quat_map2sensor[0]
    
    print(f"  tx = {T_map2sensor[0]:.6f}")
    print(f"  ty = {T_map2sensor[1]:.6f}")
    print(f"  tz = {T_map2sensor[2]:.6f}")
    print(f"  rx = {x:.10f}  # x")
    print(f"  ry = {y:.10f}  # y")
    print(f"  rz = {z:.10f}  # z")
    print(f"  rw = {w:.10f}  # w")
    
    # 4. 验证逆变换
    print("\n" + "=" * 60)
    print("4. 验证逆变换:")
    
    identity_check = map2sensor @ sensor2map
    max_error = np.max(np.abs(identity_check - np.eye(4)))
    
    print(f"  map2sensor @ sensor2map 应等于单位矩阵")
    print(f"  最大误差: {max_error:.2e}")
    
    print("\n  验证矩阵的对角线元素:")
    for i in range(4):
        print(f"    第{i}行: [{identity_check[i,0]:.10f}, {identity_check[i,1]:.10f}, {identity_check[i,2]:.10f}, {identity_check[i,3]:.10f}]")
    
    # 5. 物理意义解释
    print("\n" + "=" * 60)
    print("5. 物理意义:")
    print(f"  sensor2map: sensor坐标系原点在map坐标系中的位置: [{tx:.1f}, {ty:.1f}, {tz:.1f}]")
    print(f"  map2sensor: map坐标系原点在sensor坐标系中的位置: [{T_map2sensor[0]:.1f}, {T_map2sensor[1]:.1f}, {T_map2sensor[2]:.1f}]")
    
    # 计算旋转角度
    angle_rad = 2 * math.acos(min(max(rw, -1.0), 1.0))
    angle_deg = angle_rad * 180 / math.pi
    print(f"\n  旋转角度: {angle_deg:.2f}°")
    
    print("\n" + "=" * 60)
    
    return sensor2map, map2sensor

def main():
    """主函数"""
    print("🚀 sensor2map 和 map2sensor 变换计算器")
    print("=" * 60)
    
    # 使用您的数据
    print("使用您的标定数据:")
    
    # tx = -262.3820495605469
    # ty = 35.848148345947266
    # tz = 12.166872024536133
    # rx = 0.0024861381389200687      # x
    # ry = -5.9723592130467296e-05    # y
    # rz = -0.19724833965301514       # z
    # rw = 0.9803503751754761         # w
    

    tx = -526.686523438
    ty = -197.015182495
    tz = 269.086364746
    rx = 0.00337398634292    # x
    ry = -0.00445341179147   # y
    rz = -0.222425952554      # z
    rw = 0.974933564663        # w


    sensor2map, map2sensor = calculate_transforms(tx, ty, tz, rx, ry, rz, rw)
    
    print("\n📋 结果总结:")
    print("sensor2map 矩阵:")
    print("[")
    for i in range(4):
        print(f"  [{sensor2map[i,0]:.6f}, {sensor2map[i,1]:.6f}, {sensor2map[i,2]:.6f}, {sensor2map[i,3]:.6f}],")
    print("]")
    
    print("\nmap2sensor 矩阵:")
    print("[")
    for i in range(4):
        print(f"  [{map2sensor[i,0]:.6f}, {map2sensor[i,1]:.6f}, {map2sensor[i,2]:.6f}, {map2sensor[i,3]:.6f}],")
    print("]")

if __name__ == "__main__":
    main()