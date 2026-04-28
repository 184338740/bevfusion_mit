#!/usr/bin/env python3
"""
验证 TYJT 数据预处理生成的 pkl 文件，检查标定准确性。

版本v04.0 (0427)：
    - 相机投影点云：按深度渐变色（近红远蓝），点尺寸 1 像素，透明度 0.3
    - 无点云降采样，保留原始全部点云（通过半透明减少遮挡）
    - BEV 参数说明（修改鸟瞰图效果）：
        - BEV_POINT_CLOUD_DOWNSAMPLE：BEV中点云最大显示点数，超过则随机降采样（设极大值如1e9可禁用）
        - BEV_POINT_SIZE：BEV每个点的大小，默认0.5
        - BEV_POINT_ALPHA：BEV点云透明度，默认0.6

版本 v03.1 (0316)：
    - BEV 图降采样点云，提高清晰度
    - 文件名包含元信息，打印标注数量

使用说明：
    python tyjt_tools_Vis_Gt_pkl.py --pkl ./tyjt_infos_train.pkl --out_dir ./vis --num_samples=20
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

from tyjt_utils.tyjt_calib_utils import CalibrationProcessor

CAM_NAMES = ['CAM_A', 'CAM_B', 'CAM_C', 'CAM_D']

# BEV 参数保持不变
BEV_POINT_CLOUD_DOWNSAMPLE = 30000
BEV_POINT_SIZE = 0.5
BEV_POINT_ALPHA = 0.6

def quaternion_to_rotation_matrix(q):
    return CalibrationProcessor.quaternion_to_rotation_matrix(q)

def load_sample(info: Dict):
    # 与原版相同
    points = np.load(info['lidar_path']).astype(np.float32)
    if points.shape[1] >= 3:
        points = points[:, :3]
    else:
        raise ValueError(f"点云维度不足: {points.shape}")

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
    返回 (uv, valid, depths)
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
    return uv, valid, z   # 返回深度

def project_box_to_image(box: Dict, cam_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
    # 与原版相同
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
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
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

def draw_projected_points_blend(image: np.ndarray, uv: np.ndarray, valid: np.ndarray,
                               depths: np.ndarray, point_size: int = 1, alpha: float = 0.6):
    """
    按深度渐变颜色绘制点，并支持透明度 (alpha)
    - 深度范围: 0~80 米，近处红色，远处蓝色
    - 透明度: 0完全透明，1完全不透明
    """
    img_copy = image.copy()
    h, w = img_copy.shape[:2]
    # 获取有效点的投影坐标和深度
    u_vals = uv[valid, 0]
    v_vals = uv[valid, 1]
    # 过滤图像内点
    in_image = (u_vals >= 0) & (u_vals < w) & (v_vals >= 0) & (v_vals < h)
    u_vals = u_vals[in_image].astype(int)
    v_vals = v_vals[in_image].astype(int)
    depths_valid = depths[valid][in_image]

    # 为每个点计算颜色（BGR）
    colors = []
    for d in depths_valid:
        d_norm = np.clip(d / 80.0, 0, 1)   # 深度归一化
        # 红色 (0,0,255) -> 蓝色 (255,0,0) 在BGR中是 (0,0,255) -> (255,0,0)
        b = int(d_norm * 255)
        g = 0
        r = int((1 - d_norm) * 255)
        colors.append((b, g, r))   # (B, G, R)

    # 半透明绘制：创建一个小ROI，将圆形以alpha比例融合
    # 为提高性能，对每个点单独处理（因为点数量不会特别大，例如几万个）
    for (x, y), color in zip(zip(u_vals, v_vals), colors):
        # 定义圆形区域半径
        r = point_size
        # 获取感兴趣区域（略大于圆）
        x1, x2 = max(x - r, 0), min(x + r + 1, w)
        y1, y2 = max(y - r, 0), min(y + r + 1, h)
        roi = img_copy[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        # 创建同尺寸的圆形掩码（单通道）
        mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
        cv2.circle(mask, (x - x1, y - y1), r, 255, -1)
        # 前景颜色填充
        color_np = np.array(color, dtype=np.uint8)
        foreground = np.full_like(roi, color_np)
        # 混合
        mask_float = mask.astype(np.float32) / 255.0 * alpha
        mask_float = mask_float[..., np.newaxis]  # (H,W,1)
        blended = roi * (1 - mask_float) + foreground * mask_float
        img_copy[y1:y2, x1:x2] = blended.astype(np.uint8)

    return img_copy

def draw_projected_box(image: np.ndarray, uv: np.ndarray, valid: np.ndarray, color=(255, 0, 0), thickness=2):
    # 与原版相同
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

def visualize_sample(sample_idx: int, info: Dict, out_dir: Path):
    DEBUG = True
    pkg_name = info.get('package_name', 'unknown_pkg')
    road_id = info.get('road_id', 'unknown_road')
    timestamp = info.get('timestamp', sample_idx)
    lidar_path = info['lidar_path']
    label_path = info['label_path']
    sub_packet = Path(lidar_path).parent.parent.parent.name
    base_name = f"{pkg_name}_{road_id}_{sub_packet}_{timestamp}"

    points, images, boxes_3d = load_sample(info)
    total_labels = len(boxes_3d)
    if DEBUG:
        print(f"\n>>> 样本 {sample_idx}: {base_name} 总标注数: {total_labels} label_path: {label_path}")

    cam_stats = {cam: {'total': total_labels, 'full': 0, 'partial': 0, 'none': 0} for cam in CAM_NAMES}

    for cam in CAM_NAMES:
        if images[cam] is None:
            continue
        img = images[cam].copy()
        h, w = img.shape[:2]

        cam_data = info['cams'][cam]
        cam_info_proj = {
            'cam2ego_translation': cam_data['sensor2ego_translation'],
            'cam2ego_rotation': cam_data['sensor2ego_rotation'],
            'cam_intrinsic': cam_data['cam_intrinsic']
        }

        # 投影点云（获得深度）
        uv_pts, valid_pts, depths = project_points_to_image(points, cam_info_proj)
        # 使用升级后的绘制函数：点大小1，透明度0.6，渐变色
        img = draw_projected_points_blend(img, uv_pts, valid_pts, depths, point_size=1, alpha=0.6)

        # 绘制3D框（与原版相同）
        for i, box in enumerate(boxes_3d):
            uv_box, valid_box = project_box_to_image(box, cam_info_proj)
            # 可见性统计（与原版相同）
            in_image = np.zeros(8, dtype=bool)
            for j in range(8):
                if valid_box[j]:
                    u, v = uv_box[j]
                    if 0 <= u < w and 0 <= v < h:
                        in_image[j] = True
            num_in_image = np.sum(in_image)
            if num_in_image == 8:
                cam_stats[cam]['full'] += 1
                visible = '完全可见'
            elif num_in_image >= 2:
                cam_stats[cam]['partial'] += 1
                visible = '部分可见'
            else:
                cam_stats[cam]['none'] += 1
                visible = '不可见'

            img = draw_projected_box(img, uv_box, valid_box, color=(255,0,0), thickness=2)

            # 可选：绘制中心点ID（保留原逻辑）
            center_ego = np.array([box['center']])
            t_cam2ego = np.array(cam_info_proj['cam2ego_translation'])
            q = np.array(cam_info_proj['cam2ego_rotation'])
            R_cam2ego = quaternion_to_rotation_matrix(q)
            R_ego2cam = R_cam2ego.T
            t_ego2cam = -R_ego2cam @ t_cam2ego
            center_cam = (R_ego2cam @ center_ego.T).T + t_ego2cam
            z = center_cam[0, 2]
            if z > 0:
                intrinsic = np.array(cam_info_proj['cam_intrinsic'])
                u = intrinsic[0,0] * center_cam[0,0] / z + intrinsic[0,2]
                v = intrinsic[1,1] * center_cam[0,1] / z + intrinsic[1,2]
                if 0 <= u < w and 0 <= v < h:
                    cv2.putText(img, str(box['id']), (int(u), int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if DEBUG:
                print(f"  相机 {cam}, 框 {i}: 有效角点={np.sum(valid_box)}, 图像内角点={num_in_image}, 状态={visible}")

        out_path = out_dir / f"{base_name}_{cam}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if DEBUG:
            stat = cam_stats[cam]
            print(f"  相机 {cam} 统计: 完全可见={stat['full']}, 部分可见={stat['partial']}, 不可见={stat['none']} (总框={total_labels})")

    # BEV 图保持原状
    fig, ax = plt.subplots(1, 1, figsize=(16, 16), dpi=200)
    x_range = [-100, 100]
    y_range = [-100, 100]
    mask = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) & \
           (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
    points_filtered = points[mask]
    if len(points_filtered) > BEV_POINT_CLOUD_DOWNSAMPLE:
        idx = np.random.choice(len(points_filtered), BEV_POINT_CLOUD_DOWNSAMPLE, replace=False)
        points_plot = points_filtered[idx]
    else:
        points_plot = points_filtered

    ax.scatter(points_plot[:, 0], points_plot[:, 1],
               s=BEV_POINT_SIZE, c='blue', alpha=BEV_POINT_ALPHA)

    for box in boxes_3d:
        cx, cy, cz = box['center']
        l, w, h = box['size']
        yaw = box['yaw']
        corners = np.array([
            [-l/2, -w/2], [ l/2, -w/2], [ l/2,  w/2], [-l/2,  w/2]
        ])
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw),  np.cos(yaw)]])
        corners = (R @ corners.T).T + np.array([cx, cy])
        rect = plt.Polygon(corners, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        front = np.array([l/2, 0]) @ R.T + np.array([cx, cy])
        ax.plot([cx, front[0]], [cy, front[1]], color='red', linewidth=2)
        plt.text(cx, cy, str(box['id']), fontsize=8, color='white', bbox=dict(facecolor='red', alpha=0.5))

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title(f'{base_name} BEV (ann={total_labels}, pts={len(points_plot)}/{len(points)})')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    out_path = out_dir / f"{base_name}_bev.jpg"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

    if DEBUG:
        total_full = sum(cam_stats[cam]['full'] for cam in CAM_NAMES)
        total_partial = sum(cam_stats[cam]['partial'] for cam in CAM_NAMES)
        total_none = sum(cam_stats[cam]['none'] for cam in CAM_NAMES)
        print(f"  汇总: 完全可见框总数={total_full}, 部分可见框总数={total_partial}, 不可见框总数={total_none}")

    json_src = info['label_path']
    json_dst = out_dir / f"{base_name}.json"
    shutil.copy2(json_src, json_dst)

def main():
    parser = argparse.ArgumentParser(description="TYJT pkl 可视化升级版 v04 (点云渐变+半透明)")
    parser.add_argument('--pkl', type=str, required=True, help='info pkl 文件')
    parser.add_argument('--out_dir', type=str, default='./vis_output', help='输出目录')
    parser.add_argument('--num_samples', type=int, default=20, help='样本数量')
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