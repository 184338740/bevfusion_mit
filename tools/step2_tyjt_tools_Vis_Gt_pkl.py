#!/usr/bin/env python3
"""
验证 TYJT 数据预处理生成的 pkl 文件，检查标定准确性。

版本v04.3 (0428)：
    - 引入任务（tasks）机制，支持一次运行生成多种不同布局的组合图
    - 每个任务可独立定义输出后缀（suffix）和布局（layout），支持任意行、列组合
    - 布局单元格支持 "CAM_X_boxes", "CAM_X_points", "CAM_X_both", "BEV"
    - 新增全局参数:
        - COMBINED_IMG_HEIGHT: 统一控制相机图像高度（像素），BEV 图自动适配该高度，解决图像压缩问题
    - 所有相机图像仅渲染一次，多种模式（points/boxes/both）复用，提升性能
    - 优化组合图排版：按固定高度缩放，保持宽高比，自动对齐行列

版本v04.2 (0428)：
    - 支持两种可视化模式：'boxes'（只显示3D框）、'points'（只显示点云）
    - 自动生成组合图（4相机+BEV）
    - 全局配置集中管理

版本v04.1 (0428): 
    - 在 v04.0 基础上为每个3D框添加朝向箭头 (相机投影和BEV图)
    - 适配 OpenCV 4.12.0 的 arrowedLine 调用 (不使用关键字参数)
    - 将相机投影的点云大小和透明度提取为全局变量 (CAM_PROJ_POINT_SIZE, CAM_PROJ_ALPHA), 便于统一修改

    - 相机投影点云参数说明 (修改点云样式): 
        - CAM_PROJ_POINT_SIZE: 点大小 (像素), 默认1
        - CAM_PROJ_ALPHA: 透明度, 默认0.3, 建议范围0.2~0.6
        - CAM_PROJ_COLOR_MODE: 颜色模式, 'depth' 深度渐变色 或 'fixed' 固定颜色
        - CAM_PROJ_FIXED_COLOR: 固定颜色 (BGR), 例如 (0,255,0) 绿色
        - CAM_PROJ_ID_FONTSIZE: 字体大小 (默认0.5)
        - CAM_PROJ_ID_COLOR: 文字颜色 (BGR, 默认红色)
        - CAM_PROJ_BOX_COLOR: 相机投影中3D框的颜色

    - BEV 参数说明 (修改鸟瞰图效果): 
        - BEV_POINT_CLOUD_DOWNSAMPLE: BEV中点云最大显示点数, 超过则随机降采样 (设极大值如1e9可禁用)
        - BEV_POINT_SIZE: BEV每个点的大小, 默认0.5
        - BEV_POINT_ALPHA: BEV点云透明度, 默认0.6
        - BEV_ID_FONTSIZE: 目标ID文字大小, 默认6
        - BEV_ID_COLOR: 目标ID文字颜色, 默认'white'
        - BEV_ID_BBOX: 背景框样式, 默认None (无背景)；如需要可设为 dict(facecolor='red', alpha=0.2)
        
版本v04.0 (0427): 
    - 相机投影点云: 按深度渐变色 (近红远蓝), 点尺寸 1 像素, 透明度 0.3
    - 无点云降采样, 保留原始全部点云 (通过半透明减少遮挡)
    - BEV 参数说明 (修改鸟瞰图效果): 
        - BEV_POINT_CLOUD_DOWNSAMPLE: BEV中点云最大显示点数, 超过则随机降采样 (设极大值如1e9可禁用)
        - BEV_POINT_SIZE: BEV每个点的大小, 默认0.5
        - BEV_POINT_ALPHA: BEV点云透明度, 默认0.6

版本 v03.1 (0316): 
    - BEV 图降采样点云, 提高清晰度
    - 文件名包含元信息, 打印标注数量

使用说明(v04.3): 
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

# ==================== 全局配置参数 ====================
# 相机名称列表（固定，请勿修改）
CAM_NAMES = ['CAM_A', 'CAM_B', 'CAM_C', 'CAM_D']

# -------------------- 模块1：相机投影点云样式 --------------------
CAM_PROJ_POINT_SIZE = 1          # 点云投影点半径（像素），1为最小，避免遮挡
CAM_PROJ_ALPHA = 0.3             # 点云透明度，0完全透明，1不透明，建议0.2~0.6
CAM_PROJ_COLOR_MODE = 'depth'    # 颜色模式：'depth'（深度渐变色，近红远蓝）或 'fixed'（固定颜色）
CAM_PROJ_FIXED_COLOR = (0, 255, 0)  # 固定颜色（BGR），当 COLOR_MODE='fixed' 时生效，此处为绿色

# -------------------- 模块2：相机投影3D框样式 --------------------
CAM_PROJ_BOX_COLOR = 'orange'    # 3D框颜色，支持颜色名（如'red'）或BGR元组
CAM_PROJ_BOX_THICKNESS = 2       # 框线宽（像素）

# -------------------- 模块3：相机投影目标ID文字样式 --------------------
CAM_PROJ_ID_FONTSIZE = 0.5       # 字体大小
CAM_PROJ_ID_COLOR = 'green'      # 文字颜色，支持颜色名或BGR元组
CAM_PROJ_ID_THICKNESS = 1        # 文字粗细（笔画宽度）

# -------------------- 模块4：BEV（鸟瞰图）整体样式 --------------------
BEV_POINT_CLOUD_DOWNSAMPLE = 30000   # BEV中最大显示点数，超过则随机降采样（设极大值如1e9可禁用）
BEV_POINT_SIZE = 0.5                # BEV点云散点大小
BEV_POINT_ALPHA = 0.6               # BEV点云透明度
BEV_BOX_COLOR = 'red'               # BEV中3D框的颜色
BEV_BOX_LINEWIDTH = 1.5             # BEV中3D框的线宽
BEV_FRONT_LINEWIDTH = 1.5           # BEV中前向线（车头方向指示线）线宽

# -------------------- 模块5：BEV中目标ID文字样式 --------------------
BEV_ID_FONTSIZE = 6                 # 文字大小
BEV_ID_COLOR = 'white'              # 文字颜色
BEV_ID_BBOX = None                  # 文字背景框，None表示无背景；如需背景设为 dict(facecolor='red', alpha=0.2)

# -------------------- 模块6：朝向箭头参数 --------------------
ARROW_LEN = 2.0                     # 箭头长度（米），仅用于相机投影中的3D框朝向指示

# -------------------- 模块7：输出布局尺寸 --------------------
# COMBINED_IMG_WIDTH = 640          # 旧版按宽度控制（已废弃）
COMBINED_IMG_HEIGHT = 1080          # 组合图中每个相机图像的固定高度（像素），BEV图自动适配此高度
                                    # 建议值：1080（适配4K高度），720（适中），540（紧凑）

# ==================== 辅助函数 ====================
def quaternion_to_rotation_matrix(q):
    return CalibrationProcessor.quaternion_to_rotation_matrix(q)

def get_bgr_color(color):
    if isinstance(color, tuple) and len(color) == 3:
        return color
    if isinstance(color, str):
        color_map = {
            'red': (0,0,255), 'green': (0,255,0), 'blue': (255,0,0),
            'yellow': (0,255,255), 'cyan': (255,255,0), 'magenta': (255,0,255),
            'white': (255,255,255), 'black': (0,0,0), 'orange': (0,165,255),
            'purple': (128,0,128), 'pink': (203,192,255), 'gray': (128,128,128)
        }
        return color_map.get(color.lower(), (0,0,255))
    return (0,0,255)

def resize_image_to_height(img, target_height):
    h, w = img.shape[:2]
    if h == target_height:
        return img
    ratio = target_height / h
    new_w = int(w * ratio)
    return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)

def resize_image_to_width(img, target_width):
    h, w = img.shape[:2]
    if w == target_width:
        return img
    ratio = target_width / w
    new_h = int(h * ratio)
    return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_LINEAR)

def fig_to_cv2(fig):
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

# ==================== 核心功能函数（与 v04.4 相同） ====================
def load_sample(info: Dict):
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
    objects = label_data.get('objects', []) if isinstance(label_data, dict) else label_data

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
                               depths: np.ndarray, point_size: int = 1, alpha: float = 0.3,
                               color_mode: str = 'depth', fixed_color: Tuple[int, int, int] = (0,255,0)):
    img_copy = image.copy()
    h, w = img_copy.shape[:2]
    u_vals = uv[valid, 0]
    v_vals = uv[valid, 1]
    in_image = (u_vals >= 0) & (u_vals < w) & (v_vals >= 0) & (v_vals < h)
    u_vals = u_vals[in_image].astype(int)
    v_vals = v_vals[in_image].astype(int)
    depths_valid = depths[valid][in_image]

    colors = []
    if color_mode == 'depth':
        for d in depths_valid:
            d_norm = np.clip(d / 80.0, 0, 1)
            b = int(d_norm * 255)
            g = 0
            r = int((1 - d_norm) * 255)
            colors.append((b, g, r))
    else:
        for _ in depths_valid:
            colors.append(fixed_color)

    for (x, y), color in zip(zip(u_vals, v_vals), colors):
        r = point_size
        x1, x2 = max(x - r, 0), min(x + r + 1, w)
        y1, y2 = max(y - r, 0), min(y + r + 1, h)
        roi = img_copy[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
        cv2.circle(mask, (x - x1, y - y1), r, 255, -1)
        color_np = np.array(color, dtype=np.uint8)
        foreground = np.full_like(roi, color_np)
        mask_float = mask.astype(np.float32) / 255.0 * alpha
        mask_float = mask_float[..., np.newaxis]
        blended = roi * (1 - mask_float) + foreground * mask_float
        img_copy[y1:y2, x1:x2] = blended.astype(np.uint8)
    return img_copy

def draw_projected_box(image: np.ndarray, uv: np.ndarray, valid: np.ndarray, color=(255,0,0), thickness=2):
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
            cv2.line(img_copy, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
    return img_copy

def draw_arrow_on_image(image: np.ndarray, start_uv, end_uv, color=(0, 255, 0), thickness=2):
    pt1 = (int(start_uv[0]), int(start_uv[1]))
    pt2 = (int(end_uv[0]), int(end_uv[1]))
    cv2.arrowedLine(image, pt1, pt2, color, thickness, cv2.LINE_AA)
    return image

def project_point_to_image(point_ego: np.ndarray, cam_info_proj: Dict, intrinsic: np.ndarray, ego2cam: np.ndarray) -> Optional[Tuple[float, float]]:
    pt_hom = np.append(point_ego, 1.0)
    pt_cam = ego2cam @ pt_hom
    z = pt_cam[2]
    if z <= 0:
        return None
    u = intrinsic[0,0] * pt_cam[0] / z + intrinsic[0,2]
    v = intrinsic[1,1] * pt_cam[1] / z + intrinsic[1,2]
    return (u, v)

def create_bev_figure(points: np.ndarray, boxes_3d: List[Dict], base_name: str, total_labels: int):
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
        rect = plt.Polygon(corners, fill=False, edgecolor=BEV_BOX_COLOR, linewidth=BEV_BOX_LINEWIDTH)
        ax.add_patch(rect)
        front = np.array([l/2, 0]) @ R.T + np.array([cx, cy])
        ax.plot([cx, front[0]], [cy, front[1]], color=BEV_BOX_COLOR, linewidth=BEV_FRONT_LINEWIDTH)
        plt.text(cx, cy, str(box['id']), fontsize=BEV_ID_FONTSIZE, color=BEV_ID_COLOR,
                 bbox=BEV_ID_BBOX, ha='center', va='center')

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title(f'{base_name} BEV (ann={total_labels}, pts={len(points_plot)}/{len(points)})')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    return fig

# ==================== 渲染工厂函数 ====================
def render_camera_points(cam_info_proj, intrinsic, ego2cam, img_rgb, points):
    uv_pts, valid_pts, depths = project_points_to_image(points, cam_info_proj)
    img = draw_projected_points_blend(img_rgb, uv_pts, valid_pts, depths,
                                      point_size=CAM_PROJ_POINT_SIZE,
                                      alpha=CAM_PROJ_ALPHA,
                                      color_mode=CAM_PROJ_COLOR_MODE,
                                      fixed_color=CAM_PROJ_FIXED_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def render_camera_boxes(cam_info_proj, intrinsic, ego2cam, img_rgb, boxes_3d):
    h, w = img_rgb.shape[:2]
    img = img_rgb.copy()
    for box in boxes_3d:
        uv_box, valid_box = project_box_to_image(box, cam_info_proj)
        img = draw_projected_box(img, uv_box, valid_box,
                                 color=get_bgr_color(CAM_PROJ_BOX_COLOR),
                                 thickness=CAM_PROJ_BOX_THICKNESS)
        # 箭头和 ID
        cx, cy, cz = box['center']
        yaw = box['yaw']
        arrow_end_ego = np.array([cx + ARROW_LEN * np.cos(yaw),
                                   cy + ARROW_LEN * np.sin(yaw),
                                   cz])
        start_uv = project_point_to_image(np.array([cx, cy, cz]), cam_info_proj, intrinsic, ego2cam)
        end_uv = project_point_to_image(arrow_end_ego, cam_info_proj, intrinsic, ego2cam)
        if start_uv is not None and end_uv is not None:
            if 0 <= start_uv[0] < w and 0 <= start_uv[1] < h and 0 <= end_uv[0] < w and 0 <= end_uv[1] < h:
                img = draw_arrow_on_image(img, start_uv, end_uv, color=(0,255,0), thickness=2)
        center_ego = np.array(box['center'])
        center_ego_4d = np.append(center_ego, 1.0)
        center_cam = (ego2cam @ center_ego_4d)[:3]
        z = center_cam[2]
        if z > 0:
            u = intrinsic[0,0] * center_cam[0] / z + intrinsic[0,2]
            v = intrinsic[1,1] * center_cam[1] / z + intrinsic[1,2]
            if 0 <= u < w and 0 <= v < h:
                cv2.putText(img, str(box['id']), (int(u), int(v)), cv2.FONT_HERSHEY_SIMPLEX,
                            CAM_PROJ_ID_FONTSIZE, get_bgr_color(CAM_PROJ_ID_COLOR), CAM_PROJ_ID_THICKNESS)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def render_camera_both(cam_info_proj, intrinsic, ego2cam, img_rgb, points, boxes_3d):
    img = render_camera_points(cam_info_proj, intrinsic, ego2cam, img_rgb, points)
    img = render_camera_boxes(cam_info_proj, intrinsic, ego2cam, img_rgb, boxes_3d)
    return img

def get_camera_image(cam: str, mode: str, points, boxes_3d, cam_info_proj, intrinsic, ego2cam, img_rgb):
    """返回对应模式的相机图像 (BGR)"""
    if mode == 'points':
        return render_camera_points(cam_info_proj, intrinsic, ego2cam, img_rgb, points)
    elif mode == 'boxes':
        return render_camera_boxes(cam_info_proj, intrinsic, ego2cam, img_rgb, boxes_3d)
    else:  # both
        return render_camera_both(cam_info_proj, intrinsic, ego2cam, img_rgb, points, boxes_3d)

# ==================== 布局渲染 ====================
def render_combined_image(layout: List[List[str]], cam_images: Dict[str, Dict[str, np.ndarray]],
                          bev_cv: np.ndarray, target_height: int = 540) -> Optional[np.ndarray]:
    rows = []
    for row_cells in layout:
        row_imgs = []
        for cell in row_cells:
            if cell == "BEV":
                row_imgs.append(('bev', bev_cv))
            elif cell.startswith("CAM_"):
                parts = cell.split('_')
                if len(parts) >= 3:
                    cam_short = parts[1]
                    mode = parts[2]
                    cam_full = f"CAM_{cam_short}"
                    if cam_full in cam_images and mode in cam_images[cam_full]:
                        img = cam_images[cam_full][mode]
                        # 先缩放到目标高度
                        img_resized = resize_image_to_height(img, target_height)
                        row_imgs.append(('cam', img_resized))
                    else:
                        print(f"  警告：缺少相机 {cam_full} 模式 {mode} 的图像")
                else:
                    print(f"  警告：无效的相机单元格格式: {cell}")
            else:
                print(f"  警告：未知布局项 '{cell}'，跳过")
        if not row_imgs:
            continue
        # 处理该行：将 BEV 图缩放到与相机图像相同高度（取第一个相机图像的高度）
        # 找出该行中第一个相机图像的高度（所有相机图像高度已统一）
        first_cam_height = None
        for typ, img in row_imgs:
            if typ == 'cam':
                first_cam_height = img.shape[0]
                break
        if first_cam_height is None:
            # 整行只有 BEV，则 BEV 高度可设为 target_height（或保持原样）
            first_cam_height = target_height
        final_row_imgs = []
        for typ, img in row_imgs:
            if typ == 'bev':
                h, w = img.shape[:2]
                ratio = first_cam_height / h
                new_w = int(w * ratio)
                img_resized = cv2.resize(img, (new_w, first_cam_height), interpolation=cv2.INTER_LINEAR)
                final_row_imgs.append(img_resized)
            else:
                final_row_imgs.append(img)
        # 水平拼接
        row_combined = cv2.hconcat(final_row_imgs)
        rows.append(row_combined)
    if rows:
        # 垂直拼接时，统一所有行的宽度（取最小宽度，避免拉伸）
        widths = [row.shape[1] for row in rows]
        target_w = min(widths)
        resized_rows = []
        for row in rows:
            h, w = row.shape[:2]
            if w != target_w:
                ratio = target_w / w
                new_h = int(h * ratio)
                row = cv2.resize(row, (target_w, new_h), interpolation=cv2.INTER_LINEAR)
            resized_rows.append(row)
        return cv2.vconcat(resized_rows)
    return None

# ==================== 主函数 ====================
def visualize_sample(sample_idx: int, info: Dict, out_dir: Path, tasks: List[Dict]):
    pkg_name = info.get('package_name', 'unknown_pkg')
    road_id = info.get('road_id', 'unknown_road')
    timestamp = info.get('timestamp', sample_idx)
    lidar_path = info['lidar_path']
    sub_packet = Path(lidar_path).parent.parent.parent.name
    base_name = f"{pkg_name}_{road_id}_{sub_packet}_{timestamp}"
    print(f"\n>>> 样本 {sample_idx}: {base_name}")

    points, images, boxes_3d = load_sample(info)
    total_labels = len(boxes_3d)

    # 预先计算每个相机的三种模式图像（BGR）
    cam_images = {cam: {} for cam in CAM_NAMES}
    for cam in CAM_NAMES:
        if images[cam] is None:
            continue
        cam_data = info['cams'][cam]
        cam_info_proj = {
            'cam2ego_translation': cam_data['sensor2ego_translation'],
            'cam2ego_rotation': cam_data['sensor2ego_rotation'],
            'cam_intrinsic': cam_data['cam_intrinsic']
        }
        intrinsic = np.array(cam_info_proj['cam_intrinsic'], dtype=np.float32)
        R_cam2ego = quaternion_to_rotation_matrix(cam_info_proj['cam2ego_rotation'])
        t_cam2ego = np.array(cam_info_proj['cam2ego_translation'], dtype=np.float32)
        cam2ego = np.eye(4)
        cam2ego[:3, :3] = R_cam2ego
        cam2ego[:3, 3] = t_cam2ego
        ego2cam = np.linalg.inv(cam2ego)
        img_rgb = images[cam].copy()
        try:
            cam_images[cam]['points'] = get_camera_image(cam, 'points', points, boxes_3d, cam_info_proj, intrinsic, ego2cam, img_rgb)
            cam_images[cam]['boxes'] = get_camera_image(cam, 'boxes', points, boxes_3d, cam_info_proj, intrinsic, ego2cam, img_rgb)
            cam_images[cam]['both'] = get_camera_image(cam, 'both', points, boxes_3d, cam_info_proj, intrinsic, ego2cam, img_rgb)
        except Exception as e:
            print(f"  渲染相机 {cam} 失败: {e}")

    # BEV 图
    fig_bev = create_bev_figure(points, boxes_3d, base_name, total_labels)
    bev_cv = fig_to_cv2(fig_bev)
    plt.close(fig_bev)

    # 循环处理每个任务
    for task in tasks:
        suffix = task.get('suffix', 'output')
        layout = task['layout']
        # combined = render_combined_image(layout, cam_images, bev_cv, COMBINED_IMG_WIDTH)
        combined = render_combined_image(layout, cam_images, bev_cv, COMBINED_IMG_HEIGHT)
        if combined is not None:
            out_path = out_dir / f"{base_name}_{suffix}.jpg"
            cv2.imwrite(str(out_path), combined)
            print(f"  已保存组合图: {out_path} (suffix={suffix})")
        else:
            print(f"  警告：任务 '{suffix}' 未生成有效图像")

    # 复制 JSON
    shutil.copy2(info['label_path'], out_dir / f"{base_name}.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', required=True)
    parser.add_argument('--out_dir', default='./vis_output')
    parser.add_argument('--num_samples', type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos'] if isinstance(data, dict) else data
    print(f"加载了 {len(infos)} 个样本")

    # ========== 定义多个可视化任务 ==========
    # 每个任务包含:
    #   suffix: 输出文件名后缀 (e.g., "CamA_boxes")
    #   layout: 二维列表，单元格格式 "CAM_X_mode" 或 "BEV"
    TASKS = [
        {
            "suffix": "BEV",
            "layout": [["BEV"]]
        },
        {
            "suffix": "3DBoxes",
            "layout": [
                ["CAM_A_boxes", "CAM_B_boxes"],
                ["CAM_C_boxes", "CAM_D_boxes"]
            ]
        },
        {
            "suffix": "3DPoints",
            "layout": [
                ["CAM_A_points", "CAM_B_points"],
                ["CAM_C_points", "CAM_D_points"]
            ]
        }
    ]
    # TASKS = [
    #     {
    #         "suffix": "CamA_boxes",
    #         "layout": [["CAM_A_boxes"]]
    #     },
    #     {
    #         "suffix": "CamA_points",
    #         "layout": [["CAM_A_points"]]
    #     },
    #     {
    #         "suffix": "BEV",
    #         "layout": [["BEV"]]
    #     },
    #     {
    #         "suffix": "boxes_vs_points",
    #         "layout": [["CAM_A_points", "CAM_A_boxes"], ["BEV"]]
    #     },
    #     {
    #         "suffix": "mixed_layout",
    #         "layout": [
    #             ["CAM_A_points", "CAM_B_points"],
    #             ["CAM_C_boxes", "CAM_D_both"],
    #             ["BEV"]
    #         ]
    #     },
    #     {
    #         "suffix": "extra",
    #         "layout": [
    #             ["CAM_A_points", "CAM_B_points"],
    #             ["CAM_C_boxes", "CAM_D_both"],
    #             ["BEV", "CAM_C_both"]
    #         ]
    #     }
    # ]
    # =======================================

    total = len(infos)
    indices = np.linspace(0, total-1, min(args.num_samples, total), dtype=int)
    for i, idx in enumerate(indices):
        visualize_sample(i, infos[idx], out_dir, TASKS)

    print("完成")

if __name__ == '__main__':
    main()