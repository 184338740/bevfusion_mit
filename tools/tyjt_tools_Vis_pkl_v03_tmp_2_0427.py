#!/usr/bin/env python3
"""
TYJT 数据可视化工具 - nuScenes 官方风格
基于 tyjt_tools_Vis_pkl_v03.2_0427.py 全面重构

- BEV 图: 黑色背景，点云按高度(plasma)着色，框细线(1.5) + 前向线
- 相机投影: 点云大小/颜色随深度变化(近红远蓝)，框按类别细线(1pt)
- 不显示任何 ID/类别文字，保持画面整洁
- 输出 PNG 格式
"""

import os
import json
import pickle
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil

# 导入标定工具（与 tyjt_converter 共用）
from tyjt_utils.tyjt_calib_utils import CalibrationProcessor

# 相机顺序映射（与 info 中的键一致）
CAM_NAMES = ['CAM_A', 'CAM_B', 'CAM_C', 'CAM_D']

# nuScenes 官方类别颜色 (RGB)
OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "bus": (255, 69, 0),
    "pedestrian": (0, 0, 230),
    "bicycle": (220, 20, 60),
    "motorcycle": (255, 61, 99),
    # 可根据需要添加更多类别
}

def quaternion_to_rotation_matrix(q):
    """四元数 [w,x,y,z] 转旋转矩阵"""
    return CalibrationProcessor.quaternion_to_rotation_matrix(q)

def load_sample(info: Dict):
    """加载点云、图像、标注（适配新版 info 格式）"""
    # 加载点云
    points = np.load(info['lidar_path']).astype(np.float32)
    if points.shape[1] >= 3:
        points = points[:, :3]  # 只取 xyz
    else:
        raise ValueError(f"点云维度不足: {points.shape}")

    # 加载图像
    images = {}
    for cam in CAM_NAMES:
        cam_data = info['cams'].get(cam)
        if cam_data is None:
            images[cam] = None
            continue
        img_path = cam_data['data_path']
        img = cv2.imread(img_path)
        if img is None:
            print(f"  警告：无法读取图像 {img_path}")
            images[cam] = None
        else:
            images[cam] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 加载标注
    with open(info['label_path'], 'r') as f:
        label_data = json.load(f)

    objects = []
    if isinstance(label_data, dict):
        objects = label_data.get('objects', [])
    elif isinstance(label_data, list):
        objects = label_data
    else:
        print(f"  警告：未知的标注格式，类型为 {type(label_data)}")

    boxes_3d = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        box3d = obj.get('box3d')
        rotation = obj.get('rotation')
        if box3d is None or rotation is None:
            continue
        if len(box3d) >= 6 and len(rotation) >= 1:
            boxes_3d.append({
                'center': box3d[:3],
                'size': box3d[3:6],
                'yaw': rotation[0],
                'type': obj.get('type', 'unknown'),
                'id': obj.get('id', -1)
            })
    return points, images, boxes_3d

def project_points_to_image(points: np.ndarray, cam_info: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将点云投影到相机图像平面。
    返回 UV 坐标 (N,2)、有效标志 (N,) 和深度 z (N,)。
    """
    t_cam2ego = np.array(cam_info['cam2ego_translation'], dtype=np.float32)
    q = np.array(cam_info['cam2ego_rotation'], dtype=np.float32)
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
    return uv, valid, z

def project_box_to_image(box: Dict, cam_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """将 3D 框投影到图像平面，返回 8 个角点的 UV 坐标 (8,2) 及其有效性"""
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

    t_cam2ego = np.array(cam_info['cam2ego_translation'], dtype=np.float32)
    q = np.array(cam_info['cam2ego_rotation'], dtype=np.float32)
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
                          depths: Optional[np.ndarray] = None, point_size_base: int = 1) -> np.ndarray:
    """
    在图像上绘制点云投影，大小/颜色随深度变化（近红远蓝）
    """
    img_copy = image.copy()
    h, w = img_copy.shape[:2]
    u, v = uv[valid, 0], uv[valid, 1]
    in_image = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[in_image].astype(int)
    v = v[in_image].astype(int)

    if depths is not None:
        d = depths[valid][in_image]
        # 深度归一化 0~80 米（可根据场景调整）
        d_norm = np.clip(d / 80.0, 0, 1)
        # 颜色：近红 (255,0,0) -> 远蓝 (0,0,255)
        colors = (np.stack([(1 - d_norm) * 255, np.zeros_like(d_norm), d_norm * 255], axis=1)).astype(np.uint8)
        # 点大小：近大远小
        point_sizes = np.clip(point_size_base * (2.0 - d_norm), 1, point_size_base * 2).astype(int)
    else:
        colors = [(0, 255, 0)] * len(u)
        point_sizes = [point_size_base] * len(u)

    for i, (x, y) in enumerate(zip(u, v)):
        cv2.circle(img_copy, (x, y), point_sizes[i], colors[i].tolist(), -1, lineType=cv2.LINE_AA)
    return img_copy

def draw_projected_box(image: np.ndarray, uv: np.ndarray, valid: np.ndarray,
                       box_type: Optional[str] = None, thickness: int = 1) -> np.ndarray:
    """
    绘制投影的 3D 框，线细且抗锯齿，颜色按类别
    """
    if not np.any(valid):
        return image
    # 获取颜色 (BGR)
    if box_type and box_type in OBJECT_PALETTE:
        rgb = OBJECT_PALETTE[box_type]
        color = (rgb[2], rgb[1], rgb[0])   # RGB -> BGR
    else:
        color = (0, 0, 255)   # 红色 BGR

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
            cv2.line(img_copy, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
    return img_copy

def visualize_bev_nuscenes_style(out_path: Path, points: np.ndarray,
                                 boxes_3d: List[Dict], xlim=(-50,50), ylim=(-50,50)):
    """
    nuScenes 官方风格 BEV 图
    - 黑色背景，点云按高度 plasma 色图
    - 3D 框线宽 1.5，带前向线
    """
    fig = plt.figure(figsize=(10, 10), dpi=200)
    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.set_axis_off()

    # 降采样
    max_points = 30000
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]

    # 按高度着色（动态范围）
    z = points[:, 2]
    z_min, z_max = np.percentile(z, [1, 99])
    if z_max - z_min < 0.1:
        z_min, z_max = -2, 4
    z_norm = (z - z_min) / (z_max - z_min)
    z_norm = np.clip(z_norm, 0, 1)
    colors = plt.cm.plasma(z_norm)
    ax.scatter(points[:, 0], points[:, 1], s=0.8, c=colors, alpha=0.8, edgecolors='none')

    # 绘制 3D 框
    for box in boxes_3d:
        cx, cy, cz = box['center']
        l, w, h = box['size']
        yaw = box['yaw']
        # 底面四个角点（闭合）
        corners = np.array([
            [-l/2, -w/2], [ l/2, -w/2], [ l/2,  w/2], [-l/2,  w/2], [-l/2, -w/2]
        ])
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw),  np.cos(yaw)]])
        corners = (R @ corners.T).T + np.array([cx, cy])

        obj_type = box.get('type', 'car')
        color_rgb = OBJECT_PALETTE.get(obj_type, (255, 0, 0))
        color_norm = np.array(color_rgb) / 255.0

        ax.plot(corners[:,0], corners[:,1], linewidth=1.5, color=color_norm, solid_capstyle='round')
        # 前向线（短箭头）
        front = np.array([l/2, 0]) @ R.T + np.array([cx, cy])
        ax.plot([cx, front[0]], [cy, front[1]], linewidth=1.5, color=color_norm)

    os.makedirs(out_path.parent, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor='black', bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_sample(sample_idx: int, info: Dict, out_dir: Path):
    """
    为单个样本生成可视化：
    - 每个相机一张图（点云投影 + 3D 框投影）保存为 PNG
    - 一张 BEV 图
    """
    # 提取元信息
    pkg_name = info.get('package_name', 'unknown_pkg')
    road_id = info.get('road_id', 'unknown_road')
    timestamp = info.get('timestamp', sample_idx)
    lidar_path = info['lidar_path']
    sub_packet = Path(lidar_path).parent.parent.parent.name
    base_name = f"{pkg_name}_{road_id}_{sub_packet}_{timestamp}"

    # 加载数据
    points, images, boxes_3d = load_sample(info)
    print(f"\n>>> 样本 {sample_idx}: {base_name} 总标注数: {len(boxes_3d)}")

    # --- 相机图像投影 ---
    for cam in CAM_NAMES:
        if images[cam] is None:
            continue
        img = images[cam].copy()

        # 获取相机标定
        cam_data = info['cams'][cam]
        cam_info_proj = {
            'cam2ego_translation': cam_data['sensor2ego_translation'],
            'cam2ego_rotation': cam_data['sensor2ego_rotation'],
            'cam_intrinsic': cam_data['cam_intrinsic']
        }

        # 投影点云（按深度着色 + 动态点大小）
        uv_pts, valid_pts, depths = project_points_to_image(points, cam_info_proj)
        img = draw_projected_points(img, uv_pts, valid_pts, depths=depths, point_size_base=1)

        # 投影并绘制 3D 框
        for box in boxes_3d:
            uv_box, valid_box = project_box_to_image(box, cam_info_proj)
            img = draw_projected_box(img, uv_box, valid_box, box_type=box['type'], thickness=1)

        # 保存为 PNG
        out_path = out_dir / f"{base_name}_{cam}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # --- BEV 图 ---
    bev_out_path = out_dir / f"{base_name}_bev.png"
    # 范围可根据需要调整 (xlim, ylim)
    visualize_bev_nuscenes_style(bev_out_path, points, boxes_3d, xlim=(-50,50), ylim=(-50,50))

    # 可选：复制原始 JSON 标注文件（用于对照）
    json_src = info['label_path']
    json_dst = out_dir / f"{base_name}.json"
    shutil.copy2(json_src, json_dst)

def main():
    parser = argparse.ArgumentParser(description="TYJT pkl 可视化 - nuScenes 官方风格")
    parser.add_argument('--pkl', type=str, required=True,
                        help='生成的 info pkl 文件路径')
    parser.add_argument('--out_dir', type=str, default='./vis_output_nusc',
                        help='输出图像目录')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='要采样的样本数')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'infos' in data:
        infos = data['infos']
    else:
        infos = data

    print(f"加载了 {len(infos)} 个样本")

    total = len(infos)
    if total < args.num_samples:
        indices = list(range(total))
        print(f"样本总数小于 {args.num_samples}，将使用全部样本")
    else:
        indices = np.linspace(0, total-1, args.num_samples, dtype=int)

    for i, idx in enumerate(indices):
        info = infos[idx]
        visualize_sample(i, info, out_dir)

    print(f"可视化完成，结果保存在 {out_dir}")

if __name__ == '__main__':
    main()