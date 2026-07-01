#!/usr/bin/env python3
"""
TYJT 数据集异常帧对检测工具（v03）
基于 info pkl 文件，检测相邻帧目标数量突变（增加或减少）超过阈值的帧对，
并为每个异常帧对生成可视化图像（相机带框图 + BEV 图）和 info.txt 记录。

改进点：
- 支持 x_range、y_range 筛选目标，统计基于筛选后的数量
- 统一相机投影函数（中心点投影）
- 增强错误处理（文件不存在时跳过并警告）

使用说明：
python tyjt_tools_valid_detector_v03_0325.py --info-pkl <pkl文件> --detect-out <输出目录> [可选参数]

用例：
# tag-val
python tyjt_tools_valid_detector_v03_0325.py \
    --info-pkl ./tyjt_data_infos_v0942/tyjt_infos_val.pkl \
    --detect-out ./output/my_detectResults_Error_val_v03 \
    --thresh-ratio 0.2 \
    --thresh-abs 10 \
    --x-range -60 60 \
    --y-range -60 60

# tag-train
python tyjt_tools_valid_detector_v03_0325.py \
    --info-pkl ./tyjt_data_infos_v0942/tyjt_infos_train.pkl \
    --detect-out ./output/my_detectResults_Error_train_v03 \
    --thresh-ratio 0.2 \
    --thresh-abs 10 \
    --x-range -60 60 \
    --y-range -60 60
"""

import argparse
import pickle
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import warnings

# 导入公共工具模块
from tyjt_utils.tyjt_vis_utils import (
    quaternion_to_rotation_matrix,
    project_box_to_image,
    draw_projected_box,
    generate_bev_image,
)
from tyjt_utils.tyjt_data_utils import get_base_name, load_points


def project_center_to_image(center_ego, cam_data):
    """
    将 ego 坐标系下的三维点投影到相机图像平面。
    参数：
        center_ego: np.ndarray, shape (3,), ego 坐标系下的点
        cam_data: dict, 包含 'sensor2ego_translation', 'sensor2ego_rotation', 'cam_intrinsic'
    返回：
        (u, v) 图像坐标，若投影失败（z <= 0）则返回 None
    """
    t_cam2ego = np.array(cam_data['sensor2ego_translation'], dtype=np.float32)
    q = np.array(cam_data['sensor2ego_rotation'], dtype=np.float32)
    R_cam2ego = quaternion_to_rotation_matrix(q)
    R_ego2cam = R_cam2ego.T
    t_ego2cam = -R_ego2cam @ t_cam2ego
    center_cam = R_ego2cam @ center_ego + t_ego2cam
    if center_cam[2] <= 0:
        return None
    intrinsic = np.array(cam_data['cam_intrinsic'], dtype=np.float32)
    u = intrinsic[0, 0] * center_cam[0] / center_cam[2] + intrinsic[0, 2]
    v = intrinsic[1, 1] * center_cam[1] / center_cam[2] + intrinsic[1, 2]
    return u, v


def filter_objects_by_range(objects, x_range, y_range):
    """
    根据 x_range, y_range 筛选目标。
    参数：
        objects: list, 每个元素为标注中的 object dict
        x_range: (min, max) 或 None
        y_range: (min, max) 或 None
    返回：
        filtered_objects: list
    """
    if x_range is None and y_range is None:
        return objects
    filtered = []
    for obj in objects:
        center = obj['box3d'][:3]
        x, y = center[0], center[1]
        in_x = (x_range is None) or (x_range[0] <= x <= x_range[1])
        in_y = (y_range is None) or (y_range[0] <= y <= y_range[1])
        if in_x and in_y:
            filtered.append(obj)
    return filtered


def load_and_filter_labels(label_path, x_range, y_range):
    """
    读取标注文件，应用范围筛选，返回筛选后的目标列表、总数量、类别统计。
    参数：
        label_path: 标注文件路径
        x_range, y_range: 筛选范围
    返回：
        (objects_filtered, count, stats)
    """
    try:
        with open(label_path, 'r') as f:
            data = json.load(f)
        objs = data.get('objects', []) if isinstance(data, dict) else data
    except Exception as e:
        warnings.warn(f"无法读取标注文件 {label_path}: {e}")
        return [], 0, {}

    # 应用范围筛选
    filtered = filter_objects_by_range(objs, x_range, y_range)

    # 统计总数和类别
    count = len(filtered)
    stats = defaultdict(int)
    for obj in filtered:
        cat = obj.get('type', 'unknown')
        stats[cat] += 1

    return filtered, count, dict(stats)


def detect_missing_labels(info_pkl, out_dir, thresh_ratio, thresh_abs,
                          x_range=None, y_range=None):
    """
    检测标注缺失的帧对（objects 数量突变），并生成可视化文件。
    参数：
        info_pkl: 输入的 info pkl 文件路径
        out_dir: 输出目录（存放异常帧对文件夹）
        thresh_ratio: 变化比例阈值（如 0.5 表示 50%）
        thresh_abs: 变化绝对值阈值（如 3 表示至少相差 3 个目标）
        x_range, y_range: 目标过滤范围，格式为 [min, max]，若为 None 则不过滤
    """
    # 加载 info 列表
    with open(info_pkl, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'infos' in data:
        infos = data['infos']
    else:
        infos = data

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 预处理：为每个 info 计算筛选后的目标列表、数量和类别统计
    print("正在预处理所有帧的标注数据（应用范围筛选）...")
    total_samples = len(infos)
    for idx, info in enumerate(infos):
        if idx % 1000 == 0:
            print(f"预处理进度: {idx}/{total_samples}")
        label_path = info.get('label_path')
        if label_path is None or not Path(label_path).exists():
            warnings.warn(f"帧 {idx} 的标注文件不存在或未提供，跳过筛选统计")
            info['_filtered_objects'] = []
            info['_filtered_count'] = 0
            info['_filtered_stats'] = {}
            continue
        objects, cnt, stats = load_and_filter_labels(label_path, x_range, y_range)
        info['_filtered_objects'] = objects
        info['_filtered_count'] = cnt
        info['_filtered_stats'] = stats

    print("预处理完成，开始检测异常帧对...")

    problem_pairs = []
    valid_prev = sum(1 for info in infos if info.get('prev', -1) != -1)
    print(f"总样本数: {total_samples}, 具有有效 prev 的样本数: {valid_prev}")
    print(f"目标过滤范围: x∈{x_range}, y∈{y_range}")

    # 遍历所有样本，检测异常
    for idx, info in enumerate(infos):
        if idx % 1000 == 0:
            print(f"检测进度: {idx}/{total_samples}, 当前发现 {len(problem_pairs)} 个异常帧对")

        prev_idx = info.get('prev', -1)
        if prev_idx == -1:
            continue
        prev_info = infos[prev_idx]

        # 使用筛选后的统计
        prev_num = prev_info.get('_filtered_count', 0)
        curr_num = info.get('_filtered_count', 0)

        # 检测变化
        if prev_num > 0 or curr_num > 0:
            max_num = max(prev_num, curr_num)
            diff = abs(prev_num - curr_num)
            ratio = diff / max_num if max_num > 0 else 0
            if ratio >= thresh_ratio and diff >= thresh_abs:
                direction = 'increase' if curr_num > prev_num else 'decrease'
                problem_pairs.append((prev_idx, idx, prev_num, curr_num, direction,
                                      prev_info, info))

    print(f"发现 {len(problem_pairs)} 个异常帧对，开始生成可视化文件...")

    # 处理每个异常帧对
    for i, (prev_idx, curr_idx, prev_num, curr_num, direction,
            prev_info, curr_info) in enumerate(problem_pairs):
        pair_dir = out_root / f"pair_{i:04d}_prev{prev_idx}_curr{curr_idx}"
        pair_dir.mkdir(exist_ok=True)

        prev_base = get_base_name(prev_info)
        curr_base = get_base_name(curr_info)

        # 为前一帧和当前帧生成可视化文件
        for role, info, base in [('prev', prev_info, prev_base), ('curr', curr_info, curr_base)]:
            role_dir = pair_dir / role
            role_dir.mkdir(exist_ok=True)

            # 加载点云（用于 BEV 图）
            points = None
            try:
                points = load_points(info['lidar_path'])
            except Exception as e:
                warnings.warn(f"无法加载点云 {info['lidar_path']}: {e}")

            # 获取该帧筛选后的目标列表
            objs_filtered = info.get('_filtered_objects', [])

            # 为每个相机生成带框图像（并绘制目标 ID）
            for cam in ['CAM_A', 'CAM_B', 'CAM_C', 'CAM_D']:
                if cam in info['cams']:
                    src_img = info['cams'][cam].get('data_path')
                    if src_img is None or not Path(src_img).exists():
                        warnings.warn(f"相机图像 {src_img} 不存在，跳过")
                        continue
                    try:
                        img = cv2.imread(src_img)
                        if img is None:
                            warnings.warn(f"无法读取图像 {src_img}")
                            continue
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        warnings.warn(f"读取图像 {src_img} 时出错: {e}")
                        continue

                    cam_data = info['cams'][cam]

                    # 投影每个框
                    for obj in objs_filtered:
                        box = {
                            'center': obj['box3d'][:3],
                            'size': obj['box3d'][3:6],
                            'yaw': obj['rotation'][0],
                            'type': obj.get('type', 'unknown')
                        }
                        uv_box, valid_box = project_box_to_image(box, cam_data)
                        img = draw_projected_box(img, uv_box, valid_box, color=(255,0,0), thickness=2)

                        # 绘制目标 ID（使用统一投影函数）
                        center_ego = np.array(box['center'])
                        uv_center = project_center_to_image(center_ego, cam_data)
                        if uv_center is not None:
                            u, v = int(uv_center[0]), int(uv_center[1])
                            h_img, w_img = img.shape[:2]
                            if 0 <= u < w_img and 0 <= v < h_img:
                                cv2.putText(img, str(obj.get('id', -1)), (u, v),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    out_img = role_dir / f"{base}_{cam}.jpg"
                    cv2.imwrite(str(out_img), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # 生成 BEV 图（带目标 ID）
            if points is not None:
                boxes_bev = []
                ids_list = []
                for obj in objs_filtered:
                    boxes_bev.append({
                        'center': obj['box3d'][:3],
                        'size': obj['box3d'][3:6],
                        'yaw': obj['rotation'][0]
                    })
                    ids_list.append(obj.get('id', -1))
                try:
                    generate_bev_image(points, boxes_bev, base, role_dir,
                                       x_range=(-100,100), y_range=(-100,100),
                                       point_size=0.5, alpha=0.6, ids=ids_list)
                except Exception as e:
                    warnings.warn(f"生成 BEV 图失败: {e}")

        # 保存 info.txt，包含类别统计对比
        prev_stats = prev_info.get('_filtered_stats', {})
        curr_stats = curr_info.get('_filtered_stats', {})

        with open(pair_dir / "info.txt", 'w') as f:
            f.write(f"异常类型: {direction}\n")
            f.write(f"prev_idx: {prev_idx}, curr_idx: {curr_idx}\n")
            f.write(f"prev_objects: {prev_num}, curr_objects: {curr_num}\n")
            f.write(f"变化比例: {abs(prev_num-curr_num)/max(prev_num, curr_num) if max(prev_num, curr_num)>0 else 0:.2f}, 变化绝对值: {abs(prev_num-curr_num)}\n\n")

            f.write("【类别统计对比】（基于范围筛选）\n")
            all_cats = set(prev_stats.keys()) | set(curr_stats.keys())
            for cat in sorted(all_cats):
                prev_cnt = prev_stats.get(cat, 0)
                curr_cnt = curr_stats.get(cat, 0)
                if prev_cnt != curr_cnt:
                    f.write(f"  {cat}: {prev_cnt} → {curr_cnt} (变化 {curr_cnt - prev_cnt:+d})\n")
                else:
                    f.write(f"  {cat}: {prev_cnt} (不变)\n")
            f.write("\n")

            f.write("【前一帧】\n")
            f.write(f"  base_name: {prev_base}\n")
            f.write(f"  human_time: {prev_info.get('human_time', 'N/A')}\n")
            f.write(f"  label_path: {prev_info['label_path']}\n")
            for cam in ['CAM_A', 'CAM_B', 'CAM_C', 'CAM_D']:
                cam_data = prev_info['cams'].get(cam, {})
                path = cam_data.get('data_path', 'N/A')
                f.write(f"  {cam}_path: {path}\n")
            f.write("\n【当前帧】\n")
            f.write(f"  base_name: {curr_base}\n")
            f.write(f"  human_time: {curr_info.get('human_time', 'N/A')}\n")
            f.write(f"  label_path: {curr_info['label_path']}\n")
            for cam in ['CAM_A', 'CAM_B', 'CAM_C', 'CAM_D']:
                cam_data = curr_info['cams'].get(cam, {})
                path = cam_data.get('data_path', 'N/A')
                f.write(f"  {cam}_path: {path}\n")

    if len(problem_pairs) == 0:
        print("未发现异常帧对。")
    else:
        print(f"发现 {len(problem_pairs)} 个异常帧对，处理完成。")
        print(f"检测完成，结果保存在: {out_root.resolve()}")
    return problem_pairs


def main():
    parser = argparse.ArgumentParser(description="TYJT 数据集异常帧对检测工具（v03）")
    parser.add_argument('--info-pkl', type=str, required=True,
                        help='输入的 info pkl 文件路径')
    parser.add_argument('--detect-out', type=str, default='./detect_results',
                        help='异常帧对输出根目录')
    parser.add_argument('--thresh-ratio', type=float, default=0.5,
                        help='目标数量变化比例阈值，默认0.5')
    parser.add_argument('--thresh-abs', type=int, default=3,
                        help='目标数量变化绝对值阈值，默认3')
    parser.add_argument('--x-range', nargs=2, type=float, default=None,
                        help='X 方向过滤范围(min max)，默认不过滤')
    parser.add_argument('--y-range', nargs=2, type=float, default=None,
                        help='Y 方向过滤范围(min max)，默认不过滤')
    args = parser.parse_args()

    detect_missing_labels(
        info_pkl=args.info_pkl,
        out_dir=args.detect_out,
        thresh_ratio=args.thresh_ratio,
        thresh_abs=args.thresh_abs,
        x_range=args.x_range,
        y_range=args.y_range
    )


if __name__ == '__main__':
    main()