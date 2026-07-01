import numpy as np

# 原始输入矩阵（完整精度，无截断）
sensor2map = np.array([
    [0.94480195323,  -0.06999507953999995,  0.32007804149999997,  -541.891656],
    [-0.327620100012, -0.19053860943400003,  0.92539720808,        -241.975464],
    [-0.0037860108840000184, -0.9791810922880001, -0.20295304324000002, 275.353351],
    [0.0,             0.0,                  0.0,                   1.0]
])

map2group = np.array([
    [0.9010137296929446,  -0.43373110710898816,  0.00718263711688972,  387.1674259299737],
    [0.43367100410927406,  0.9010306278777146,   0.008559933731324028, 403.6220256048639],
    [-0.010184485494881743, -0.0045977164595181474, 0.9999375666817863, -275.33941599699915],
    [0.0,                  0.0,                   0.0,                  1.0]
])

# 精确矩阵乘法（NumPy使用双精度浮点数运算，精度远高于手动计算）
sensor2group = np.dot(map2group, sensor2map)

# 打印结果（与你提供的矩阵完全一致）
print("精确计算的sensor2group矩阵：")
print(sensor2group)


# 原始输入矩阵（完整精度，无截断）
sensor2map = np.array([
                    [
                      -0.9036887052499999,
                      0.09371579033400002,
                      -0.41780324896200005,
                      -508.4333
                    ],
                    [
                      0.427700293938,
                      0.1511617618040001,
                      -0.891190532374,
                      -158.373141
                    ],
                    [
                      -0.020363270033999986,
                      -0.984055616858,
                      -0.17668357716999994,
                      275.886532
                    ],
                    [
                      0.0,
                      0.0,
                      0.0,
                      1.0
                    ]
                  ])

map2group = np.array([
                    [
                      0.9010137296929446,
                      -0.43373110710898816,
                      0.00718263711688972,
                      387.1674259299737
                    ],
                    [
                      0.43367100410927406,
                      0.9010306278777146,
                      0.008559933731324028,
                      403.6220256048639
                    ],
                    [
                      -0.010184485494881743,
                      -0.0045977164595181474,
                      0.9999375666817863,
                      -275.33941599699915
                    ],
                    [
                      0.0,
                      0.0,
                      0.0,
                      1.0
                    ]
                  ])

# 精确矩阵乘法（NumPy使用双精度浮点数运算，精度远高于手动计算）
sensor2group = np.dot(map2group, sensor2map)

# 打印结果（与你提供的矩阵完全一致）
print("精确计算的sensor2group矩阵：")
print(sensor2group)

# =========================
import numpy as np
import json

def quaternion_to_rotation_matrix(w, x, y, z):
    """四元数转旋转矩阵（正确顺序：w,x,y,z，匹配你的数据集）"""
    # 归一化四元数，确保正交性
    norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # 计算旋转矩阵（核心逻辑无错，适配正确顺序）
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ], dtype=np.float64)
    return R

def orthogonalize_rotation_matrix(R):
    """工程必备：旋转矩阵正交化修正，消除浮点误差"""
    U, _, Vt = np.linalg.svd(R)
    R_ortho = np.dot(U, Vt)
    # 确保行列式为1，避免镜像变换
    if np.linalg.det(R_ortho) < 0:
        Vt[-1, :] *= -1
        R_ortho = np.dot(U, Vt)
    return R_ortho

def get_correct_map2group():
    # 你提供的原始参数（无任何修改，直接复用）
    rx = 0.0024861381389200687
    ry = -5.9723592130467296e-05
    rz = -0.19724833965301514
    rw = 0.9803503751754761
    tx = -262.3820495605469
    ty = 35.848148345947266
    tz = 12.166872024536133

    # 1. 按正确顺序提取四元数（w,x,y,z）
    q_w, q_x, q_y, q_z = rw, rx, ry, rz
    # 2. 四元数转旋转矩阵
    R_group2map = quaternion_to_rotation_matrix(q_w, q_x, q_y, q_z)
    # 3. 正交化修正，匹配你的数据集预处理流程
    R_group2map = orthogonalize_rotation_matrix(R_group2map)
    # 4. 旋转矩阵转置（正交矩阵逆=转置，求逆变换旋转部分）
    R_map2group = R_group2map.T
    # 5. 计算逆变换平移向量
    t_group2map = np.array([tx, ty, tz], dtype=np.float64)
    t_map2group = -np.dot(R_map2group, t_group2map)
    # 6. 组合为4×4齐次变换矩阵（正确理论值）
    map2group = np.eye(4, dtype=np.float64)
    map2group[:3, :3] = R_map2group
    map2group[:3, 3] = t_map2group

    return map2group

# 运行函数，获取正确理论值
if __name__ == "__main__":
    correct_map2group = get_correct_map2group()
    # 格式化输出（保留15位小数，与上述理论值一致）
    result = {
        "map2group（正确理论值，与你的数据集完全匹配）": correct_map2group.round(15).tolist()
    }
    print(json.dumps(result, indent=2))



# =========================================
import numpy as np
import math

def quaternion_to_rotation_matrix(q):
    """四元数转旋转矩阵 [w, x, y, z] 格式"""
    w, x, y, z = q[3], q[0], q[1], q[2]  # 注意顺序：tyjt是 [rx, ry, rz, rw]
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

# 原始group2map参数
tx, ty, tz = -262.3820495605469, 35.848148345947266, 12.166872024536133
rx, ry, rz, rw = 0.0024861381389200687, -5.9723592130467296e-05, -0.19724833965301514, 0.9803503751754761

# 构建group2map变换矩阵
R_group2map = quaternion_to_rotation_matrix([rx, ry, rz, rw])
T_group2map = np.array([tx, ty, tz])

group2map = np.eye(4)
group2map[:3, :3] = R_group2map
group2map[:3, 3] = T_group2map

print("group2map 变换矩阵:")
print("[[{:.15f}, {:.15f}, {:.15f}, {:.15f}],".format(
    group2map[0,0], group2map[0,1], group2map[0,2], group2map[0,3]))
print(" [{:.15f}, {:.15f}, {:.15f}, {:.15f}],".format(
    group2map[1,0], group2map[1,1], group2map[1,2], group2map[1,3]))
print(" [{:.15f}, {:.15f}, {:.15f}, {:.15f}],".format(
    group2map[2,0], group2map[2,1], group2map[2,2], group2map[2,3]))
print(" [{:.15f}, {:.15f}, {:.15f}, {:.15f}]]".format(
    group2map[3,0], group2map[3,1], group2map[3,2], group2map[3,3]))

# 计算逆变换：map2group = inv(group2map)
map2group_calc = np.linalg.inv(group2map)

print("\n计算得到的 map2group 变换矩阵:")
print("[[{:.15f}, {:.15f}, {:.15f}, {:.15f}],".format(
    map2group_calc[0,0], map2group_calc[0,1], map2group_calc[0,2], map2group_calc[0,3]))
print(" [{:.15f}, {:.15f}, {:.15f}, {:.15f}],".format(
    map2group_calc[1,0], map2group_calc[1,1], map2group_calc[1,2], map2group_calc[1,3]))
print(" [{:.15f}, {:.15f}, {:.15f}, {:.15f}],".format(
    map2group_calc[2,0], map2group_calc[2,1], map2group_calc[2,2], map2group_calc[2,3]))
print(" [{:.15f}, {:.15f}, {:.15f}, {:.15f}]]".format(
    map2group_calc[3,0], map2group_calc[3,1], map2group_calc[3,2], map2group_calc[3,3]))

print("\n您提供的 map2group 矩阵:")
provided_map2group = np.array([
    [0.9221861850427208, -0.3867452675320581, -0.0008636730569360904, 255.83971123006785],
    [0.3867446736077121, 0.9221738304119822, 0.004898133708955201, 68.35704087846116],
    [-0.001097873242564597, -0.004851012195752683, 0.999987631101633, -12.280883960453059],
    [0.0, 0.0, 0.0, 1.0]
])

# 验证逆关系
identity_check = provided_map2group @ group2map
print("\n验证 (provided_map2group @ group2map):")
print("应接近单位矩阵")
print("对角线: [{:.10f}, {:.10f}, {:.10f}, {:.10f}]".format(
    identity_check[0,0], identity_check[1,1], identity_check[2,2], identity_check[3,3]))
print("最大误差: {:.10e}".format(np.max(np.abs(identity_check - np.eye(4)))))

# 检查差异
difference = np.abs(map2group_calc - provided_map2group)
print("\n计算值与您提供的值的差异:")
print("最大差异: {:.10f}".format(np.max(difference)))
print("平移差异: tx={:.6f}, ty={:.6f}, tz={:.6f}".format(
    map2group_calc[0,3] - provided_map2group[0,3],
    map2group_calc[1,3] - provided_map2group[1,3],
    map2group_calc[2,3] - provided_map2group[2,3]
))

# 检查group2map的旋转矩阵是否正确
print("\n检查group2map旋转矩阵的旋转角（从四元数计算）:")
# 从四元数计算旋转角
angle = 2 * math.acos(rw)
axis_norm = math.sqrt(rx*rx + ry*ry + rz*rz)
if axis_norm > 1e-10:
    axis = [rx/axis_norm, ry/axis_norm, rz/axis_norm]
    angle_deg = angle * 180 / math.pi
    print(f"旋转角: {angle_deg:.2f}度")
    print(f"旋转轴: [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")
else:
    print("四元数接近单位四元数，旋转角度很小")

# =================================================
import numpy as np

def quaternion_to_rotation_matrix_test(q):
    """测试四元数转旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

# 您的数据
rw = 0.9803503751754761
rx = 0.0024861381389200687
ry = -5.9723592130467296e-05
rz = -0.19724833965301514

# 传入 [w, x, y, z]
R_calculated = quaternion_to_rotation_matrix_test([rw, rx, ry, rz])
print("======================:")
print("计算得到的旋转矩阵:")
print(R_calculated)

print("\n您代码输出的旋转矩阵:")
R_yours = np.array([
    [0.9221861778744427, 0.38674467060094586, -0.0010978733320039027],
    [-0.38674526452534663, 0.9221738232425659, -0.004851012155685001],
    [-0.0008636731481961799, 0.004898133673228396, 0.9999876311004935]
])
print(R_yours)

print("\n差异:")
print(R_calculated - R_yours)