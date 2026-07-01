#!/usr/bin/env python3
"""
TYJT pkl 可视化 - 完全采用 nuscenes-devkit 官方渲染逻辑（不依赖数据库）
直接复制官方的点云投影、颜色映射、3D框渲染参数。
"""

import os
import json
import pickle
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
from pyquaternion import Quaternion

# 导入 nuscenes-devkit 的核心几何函数
from nuscenes.utils.geometry_utils import view_points, BoxVisibility

# 相机顺序
CAM_NAMES = ['CAM_A', 'CAM_B', 'CAM_C', 'CAM_D']

# 官方类别颜色 (RGB)
COLOR_MAP = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "bus": (255, 69, 0),
    "pedestrian": (0, 0, 230),
    "bicycle": (220, 20, 60),
    "motorcycle": (255, 61, 99),
    "traffic_cone": (47, 79, 79),
    "barrier": (112, 128, 144),
}

def quaternion_to_rotation_matrix(q):
    """四元数 [w,x,y,z] -> 3x3 旋转矩阵"""
    return Quaternion(q).rotation_matrix

def load_sample(info: Dict):
    """加载点云、图像、标注"""
    points = np.load(info['lidar_path']).astype(np.float32)
    if points.shape[1] > 3:
        points = points[:, :3]

    images = {}
    for cam in CAM_NAMES:
        cam_data = info['cams'].get(cam)
        if cam_data is None:
            images[cam] = None
            continue
        img = cv2.imread(cam_data['data_path'])
        if img is not None:
            images[cam] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            images[cam] = None

    with open(info['label_path'], 'r') as f:
        label_data = json.load(f)
    objects = label_data.get('objects', []) if isinstance(label_data, dict) else label_data

    boxes_3d = []
    for obj in objects:
        if 'box3d' not in obj or 'rotation' not in obj:
            continue
        boxes_3d.append({
            'center': obj['box3d'][:3],
            'size': obj['box3d'][3:6],
            'yaw': obj['rotation'][0],
            'type': obj.get('type', 'unknown'),
            'id': obj.get('id', -1)
        })
    return points, images, boxes_3d

def ego_to_sensor(points: np.ndarray, ego2sensor: np.ndarray) -> np.ndarray:
    """将点云从 ego 坐标系变换到传感器坐标系，返回 (N,3) 和深度"""
    pts_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    pts_sensor = (ego2sensor @ pts_hom.T).T
    return pts_sensor[:, :3], pts_sensor[:, 2]  # 坐标, 深度


def render_camera_official(info: Dict, points: np.ndarray, boxes_3d: List[Dict],
                           cam_name: str, out_path: Path):
    """
    官方风格相机渲染：
    - 点云投影：使用 view_points，颜色按深度映射（近红远蓝），点大小固定1像素
    - 3D框：手动投影12条边，线宽1，抗锯齿
    """
    cam_data = info['cams'][cam_name]
    intrinsic = np.array(cam_data['cam_intrinsic'], dtype=np.float32)
    # 构建 ego2cam 矩阵
    R_cam2ego = quaternion_to_rotation_matrix(cam_data['sensor2ego_rotation'])
    t_cam2ego = np.array(cam_data['sensor2ego_translation'])
    cam2ego = np.eye(4)
    cam2ego[:3, :3] = R_cam2ego
    cam2ego[:3, 3] = t_cam2ego
    ego2cam = np.linalg.inv(cam2ego)

    # 点云投影
    pts_sensor, depths = ego_to_sensor(points, ego2cam)
    valid = depths > 0
    uv = np.zeros((points.shape[0], 2), dtype=np.float32)
    if np.any(valid):
        # view_points 返回 (3, M)，转置后为 (M, 3)，取前两列
        uv_valid = view_points(pts_sensor[valid].T, intrinsic, normalize=True).T[:, :2]
        uv[valid] = uv_valid

    img = cv2.imread(cam_data['data_path'])
    if img is None:
        return
    h, w = img.shape[:2]

    # 绘制点云
    for i in range(uv.shape[0]):
        if not valid[i]:
            continue
        u, v = uv[i]
        if 0 <= u < w and 0 <= v < h:
            d = depths[i]
            d_norm = min(1.0, d / 80.0)
            color = (int((1 - d_norm) * 255), 0, int(d_norm * 255))  # BGR
            cv2.circle(img, (int(u), int(v)), 1, color, -1, lineType=cv2.LINE_AA)

    # 绘制3D框
    for box in boxes_3d:
        cx, cy, cz = box['center']
        l, w_box, h_box = box['size']
        yaw = box['yaw']
        R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw),  np.cos(yaw), 0],
                      [0, 0, 1]])
        offsets = np.array([
            [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
            [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]
        ]) * np.array([l/2, w_box/2, h_box/2])
        corners_ego = (R @ offsets.T).T + np.array([cx, cy, cz])
        corners_hom = np.hstack((corners_ego, np.ones((8, 1))))
        corners_cam = (ego2cam @ corners_hom.T).T
        valid_corners = corners_cam[:, 2] > 0
        uv_corners = np.zeros((8, 2), dtype=np.float32)
        for i in range(8):
            if valid_corners[i]:
                pt = corners_cam[i, :3].reshape(3, 1)
                uv_corners[i] = view_points(pt, intrinsic, normalize=True).flatten()[:2]
        edges = [(0,1),(1,2),(2,3),(3,0),
                 (4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        color_rgb = COLOR_MAP.get(box['type'], (255, 0, 0))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        for i, j in edges:
            if valid_corners[i] and valid_corners[j]:
                pt1 = (int(uv_corners[i, 0]), int(uv_corners[i, 1]))
                pt2 = (int(uv_corners[j, 0]), int(uv_corners[j, 1]))
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    cv2.line(img, pt1, pt2, color_bgr, 1, lineType=cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def render_bev_official(points: np.ndarray, boxes_3d: List[Dict],
                        out_path: Path, xlim=(-50,50), ylim=(-50,50)):
    """
    官方 BEV 风格：黑色背景，点云按高度（或强度）映射 plasma，点大小固定，
    3D框采用细线（1.5pt）+ 前向箭头。
    """
    fig = plt.figure(figsize=(10, 10), dpi=150)
    ax = fig.add_subplot(111)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.axis('off')

    # 降采样
    if len(points) > 30000:
        idx = np.random.choice(len(points), 30000, replace=False)
        points = points[idx]

    # 高度着色
    z = points[:, 2]
    vmin, vmax = np.percentile(z, [1, 99])
    if vmax - vmin < 1e-3:
        vmin, vmax = -2, 4
    norm = (z - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)
    colors = plt.cm.plasma(norm)
    ax.scatter(points[:, 0], points[:, 1], s=0.8, c=colors, alpha=0.8, edgecolors='none')

    for box in boxes_3d:
        cx, cy, cz = box['center']
        l, w_box, h_box = box['size']
        yaw = box['yaw']
        corners = np.array([
            [-l/2, -w_box/2], [ l/2, -w_box/2], [ l/2,  w_box/2], [-l/2,  w_box/2]
        ])
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw), np.cos(yaw)]])
        corners = (R @ corners.T).T + np.array([cx, cy])
        corners = np.vstack([corners, corners[0]])  # 闭合
        color_rgb = COLOR_MAP.get(box['type'], (255,0,0))
        color_norm = np.array(color_rgb) / 255.0
        ax.plot(corners[:,0], corners[:,1], linewidth=1.5, color=color_norm)
        # 前向线
        front = np.array([l/2, 0]) @ R.T + np.array([cx, cy])
        ax.plot([cx, front[0]], [cy, front[1]], linewidth=1.5, color=color_norm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, facecolor='black', bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_sample(idx: int, info: Dict, out_dir: Path):
    pkg_name = info.get('package_name', 'unknown')
    road_id = info.get('road_id', 'unknown')
    timestamp = info['timestamp']
    sub_packet = Path(info['lidar_path']).parent.parent.parent.name
    base_name = f"{pkg_name}_{road_id}_{sub_packet}_{timestamp}"
    print(f"\n处理样本 {idx}: {base_name}")

    points, images, boxes = load_sample(info)
    if not boxes:
        print("  无标注，跳过")
        return

    for cam in CAM_NAMES:
        if images.get(cam) is None:
            continue
        out_path = out_dir / f"{base_name}_{cam}.png"
        render_camera_official(info, points, boxes, cam, out_path)

    bev_path = out_dir / f"{base_name}_bev.png"
    render_bev_official(points, boxes, bev_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', required=True)
    parser.add_argument('--out_dir', default='./vis_official')
    parser.add_argument('--num_samples', type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos'] if isinstance(data, dict) else data
    print(f"加载 {len(infos)} 个样本")

    total = len(infos)
    indices = np.linspace(0, total-1, min(args.num_samples, total), dtype=int)
    for i, idx in enumerate(indices):
        visualize_sample(i, infos[idx], out_dir)

    print("完成")

if __name__ == '__main__':
    main()