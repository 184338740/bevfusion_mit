#!/usr/bin/env python3
"""
TYJT 数据集可视化工具函数，用于投影点云、绘制 3D 框、生成 BEV 图等。
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 导入标定工具（根据实际路径调整）
from tyjt_utils.tyjt_calib_utils import CalibrationProcessor

# ==================== 全局常量 ====================
BEV_POINT_CLOUD_DOWNSAMPLE = 30000  # BEV 图中最多显示的点数
BEV_POINT_SIZE = 0.5                # 点的大小
BEV_POINT_ALPHA = 0.6                # 点云透明度

# ==================== 核心工具函数 ====================

def quaternion_to_rotation_matrix(q):
    """四元数 [w,x,y,z] 转旋转矩阵"""
    return CalibrationProcessor.quaternion_to_rotation_matrix(q)


def project_points_to_image(points: np.ndarray, cam_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    将点云投影到相机图像平面。
    参数：
        points: (N, 3) 点云数组
        cam_info: 相机信息字典，包含 'cam2ego_translation', 'cam2ego_rotation', 'cam_intrinsic'
    返回：
        uv: (N, 2) 投影后的图像坐标
        valid: (N,) 布尔数组，表示投影是否有效（深度 > 0）
    """
    t_cam2ego = np.array(cam_info['sensor2ego_translation'], dtype=np.float32)
    q = np.array(cam_info['sensor2ego_rotation'], dtype=np.float32) # [w,x,y,z]
    R_cam2ego = quaternion_to_rotation_matrix(q)
    intrinsic = np.array(cam_info['cam_intrinsic'], dtype=np.float32)

    R_ego2cam = R_cam2ego.T
    t_ego2cam = -R_ego2cam @ t_cam2ego

    points_cam = (R_ego2cam @ points.T).T + t_ego2cam
    z = points_cam[:, 2]
    valid = z > 0
    uv = np.zeros((points.shape[0], 2), dtype=np.float32)
    if np.any(valid):
        x = points_cam[valid, 0]
        y = points_cam[valid, 1]
        z_valid = z[valid]
        u = intrinsic[0, 0] * x / z_valid + intrinsic[0, 2]
        v = intrinsic[1, 1] * y / z_valid + intrinsic[1, 2]
        uv[valid, 0] = u
        uv[valid, 1] = v
    return uv, valid


def project_box_to_image(box: Dict, cam_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    将单个 3D 框投影到图像平面。
    参数：
        box: 包含 'center', 'size', 'yaw' 的字典
        cam_info: 相机信息字典
    返回：
        uv: (8, 2) 8 个角点的图像坐标
        valid: (8,) 布尔数组，表示每个角点是否有效（深度 > 0）
    """
    cx, cy, cz = box['center']
    l, w, h = box['size']
    yaw = box['yaw']

    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    dx = np.array([l/2, w/2, h/2])
    offsets = np.array([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]
    ]) * dx

    corners_ego = (R_yaw @ offsets.T).T + np.array([cx, cy, cz])

    t_cam2ego = np.array(cam_info['sensor2ego_translation'], dtype=np.float32)
    q = np.array(cam_info['sensor2ego_rotation'], dtype=np.float32)
    R_cam2ego = quaternion_to_rotation_matrix(q)
    R_ego2cam = R_cam2ego.T
    t_ego2cam = -R_ego2cam @ t_cam2ego

    corners_cam = (R_ego2cam @ corners_ego.T).T + t_ego2cam

    intrinsic = np.array(cam_info['cam_intrinsic'], dtype=np.float32)
    z = corners_cam[:, 2]
    valid = z > 0
    uv = np.zeros((8, 2), dtype=np.float32)
    if np.any(valid):
        x = corners_cam[valid, 0]
        y = corners_cam[valid, 1]
        u = intrinsic[0, 0] * x / z[valid] + intrinsic[0, 2]
        v = intrinsic[1, 1] * y / z[valid] + intrinsic[1, 2]
        uv[valid, 0] = u
        uv[valid, 1] = v
    return uv, valid


def draw_projected_points(image: np.ndarray, uv: np.ndarray, valid: np.ndarray,
                          color=(0, 255, 0), point_size=2) -> np.ndarray:
    """在图像上绘制点云投影（绿色圆点）"""
    img_copy = image.copy()
    h, w = img_copy.shape[:2]
    u, v = uv[valid, 0], uv[valid, 1]
    in_image = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[in_image].astype(int)
    v = v[in_image].astype(int)
    for x, y in zip(u, v):
        cv2.circle(img_copy, (x, y), point_size, color, -1)
    return img_copy


def draw_projected_box(image: np.ndarray, uv: np.ndarray, valid: np.ndarray,
                       color=(255, 0, 0), thickness=2) -> np.ndarray:
    """在图像上绘制投影的 3D 框（红色线框）"""
    if not np.any(valid):
        return image
    img_copy = image.copy()
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    for i, j in edges:
        if valid[i] and valid[j]:
            pt1 = (int(uv[i,0]), int(uv[i,1]))
            pt2 = (int(uv[j,0]), int(uv[j,1]))
            cv2.line(img_copy, pt1, pt2, color, thickness)
    return img_copy


def generate_bev_image(points, boxes_3d, base_name, out_dir,
                       x_range=(-100,100), y_range=(-100,100),
                       point_size=0.5, alpha=0.6,
                       downsample=30000, ids=None):
    """
    生成 BEV 点云图，带网格和刻度，可选择绘制目标 ID。
    ids: 与 boxes_3d 对应的目标 ID 列表，若提供则在框中心绘制 ID。
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=150)

    # 过滤点云范围
    mask = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) & \
           (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
    points_f = points[mask]

    # 降采样
    if len(points_f) > downsample:
        idx = np.random.choice(len(points_f), downsample, replace=False)
        points_f = points_f[idx]

    # 绘制点云
    ax.scatter(points_f[:, 0], points_f[:, 1], s=point_size, c='blue', alpha=alpha)

    # 绘制 3D 框并添加 ID
    for i, box in enumerate(boxes_3d):
        cx, cy, cz = box['center']
        l, w, h = box['size']
        yaw = box['yaw']
        corners = np.array([
            [-l/2, -w/2],
            [ l/2, -w/2],
            [ l/2,  w/2],
            [-l/2,  w/2]
        ])
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw),  np.cos(yaw)]])
        corners = (R @ corners.T).T + np.array([cx, cy])
        rect = plt.Polygon(corners, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        front = np.array([l/2, 0]) @ R.T + np.array([cx, cy])
        ax.plot([cx, front[0]], [cy, front[1]], color='red', linewidth=2)

        # 绘制 ID（如果提供），无背景框，黑色文字
        if ids is not None and i < len(ids):
            ax.text(cx, cy, str(ids[i]), fontsize=8, color='black',
                    ha='center', va='center')

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # 设置刻度（每10米一个）
    xticks = np.arange(x_range[0], x_range[1] + 1, 10)
    yticks = np.arange(y_range[0], y_range[1] + 1, 10)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.tick_params(axis='both', labelsize=8)

    out_path = out_dir / f"{base_name}_BEV.jpg"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
