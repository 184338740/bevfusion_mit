import os
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from tqdm import tqdm

# ==================== 配置参数（完全适配你的数据） ====================
DATASET_PATH = "./output/step1/nuscenes_tyjt"
VERSION = "v1.0-tyjt"
BEV_RANGE = [-50, 50, -50, 50]  # [x_min, x_max, y_min, y_max]（自车系）
POINT_SIZE = 0.5
COLOR_BY_HEIGHT = True

# ==================== 初始化数据集 ====================
def init_nuscenes():
    try:
        nusc = NuScenes(version=VERSION, dataroot=DATASET_PATH, verbose=False)
        print(f"✅ 成功加载数据集：{DATASET_PATH}，共{len(nusc.scene)}个场景")
        return nusc
    except Exception as e:
        print(f"❌ 数据集加载失败：{e}")
        exit(1)

# ==================== 读取 xyzirt 格式.npy点云（核心适配） ====================
def load_xyzirt_pointcloud(file_path):
    """
    读取你的点云格式：xyzirt（6维）
    输出：SDK兼容的LidarPointCloud实例（4维：x,y,z,i）
    """
    # 1. 读取.npy点云（形状：(N, 6)，列顺序：x,y,z,i,r,t）
    pc_npy = np.load(file_path)
    assert pc_npy.shape[1] == 6, f"点云格式错误！应为6维（xyzirt），实际为{pc_npy.shape[1]}维"
    
    # 2. 提取有效维度：x,y,z,i（前4维），丢弃r（反射率）和t（时间戳）
    points_xyz_i = pc_npy[:, :4].T  # 转置为(4, N)，符合SDK要求（x,y,z,i顺序）
    
    # 3. 创建LidarPointCloud实例（直接使用小组坐标系，无需额外转换）
    pc = LidarPointCloud(points_xyz_i)
    return pc

# ==================== BEV可视化（适配小组坐标系点云） ====================
def visualize_bev(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_file_path = os.path.join(nusc.dataroot, lidar_data['filename'])

    # 加载适配xyzirt格式的点云
    if lidar_file_path.endswith('.npy'):
        pc = load_xyzirt_pointcloud(lidar_file_path)
    else:
        pc = LidarPointCloud.from_file(lidar_file_path)

    # 点云已在小组坐标系（自车系），直接提取x,y,z
    points = pc.points[:3, :].T  # (N, 3)，自车系坐标（x向前，y向左，z向上）

    # 过滤BEV范围内的点
    x_min, x_max, y_min, y_max = BEV_RANGE
    mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
           (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    points_bev = points[mask]
    if len(points_bev) == 0:
        print(f"⚠️ 样本{sample_token[:8]}在BEV范围内无点云")
        return

    # 绘图
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m) → 前方（小组坐标系）')
    ax.set_ylabel('Y (m) → 左方（小组坐标系）')
    ax.set_title(f"BEV Visualization (Sample: {sample_token[:8]})")

    # 绘制点云（按高度着色）
    if COLOR_BY_HEIGHT:
        z = points_bev[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
        ax.scatter(points_bev[:, 0], points_bev[:, 1], s=POINT_SIZE, c=z_norm, cmap='viridis')
    else:
        ax.scatter(points_bev[:, 0], points_bev[:, 1], s=POINT_SIZE, c='blue')

    # 绘制标注框（SDK自动加载小组坐标系的标注）
    annotations = nusc.get_boxes(lidar_token)
    for box in annotations:
        x, y, _ = box.center  # 标注框中心（自车系）
        length = box.wlh[1]  # 长度（x轴方向）
        width = box.wlh[0]   # 宽度（y轴方向）
        yaw = box.orientation.yaw_pitch_roll[0]  # 绕z轴旋转角

        # 计算旋转框顶点
        half_l, half_w = length/2, width/2
        corners = np.array([
            [x + half_l, y + half_w],
            [x + half_l, y - half_w],
            [x - half_l, y - half_w],
            [x - half_l, y + half_w]
        ])
        # 旋转矩阵（绕z轴）
        rot_mat = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        # 应用旋转（绕中心点）
        corners_rotated = (rot_mat @ (corners - [x, y]).T).T + [x, y]
        corners_rotated = np.vstack([corners_rotated, corners_rotated[0]])  # 闭合
        ax.plot(corners_rotated[:, 0], corners_rotated[:, 1], 'r-', linewidth=2)

    # 保存结果
    save_path = f"bev_xyzirt_vis_{sample_token[:8]}.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"💾 保存BEV图：{save_path}")
    plt.close()

# ==================== 主函数 ====================
def main():
    nusc = init_nuscenes()
    # 取前5个样本可视化
    num_vis = 5
    sample_tokens = []
    for scene in nusc.scene[:min(num_vis, len(nusc.scene))]:
        current_token = scene['first_sample_token']
        while current_token and len(sample_tokens) < num_vis:
            sample_tokens.append(current_token)
            current_sample = nusc.get('sample', current_token)
            current_token = current_sample['next']

    # 批量可视化
    for token in tqdm(sample_tokens, desc="BEV可视化进度"):
        visualize_bev(nusc, token)

    print("✅ 所有可视化完成")

if __name__ == "__main__":
    main()