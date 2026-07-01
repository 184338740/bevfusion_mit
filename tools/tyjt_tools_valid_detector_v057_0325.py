#!/usr/bin/env python3
"""
TYJT 数据集异常检测工具（v061）
- 相邻帧突变检测（可视化 + info.txt）
- 数目异常段检测：仅根据突变方向（increase/decrease）判定，不使用数值统计
- 目标数量曲线图生成（可选）
- 详细日志输出
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
import logging
import sys
from typing import List, Tuple, Dict, Any

# 导入公共工具模块
from tyjt_utils.tyjt_vis_utils import (
    quaternion_to_rotation_matrix,
    project_box_to_image,
    draw_projected_box,
    generate_bev_image,
)
from tyjt_utils.tyjt_data_utils import get_base_name, load_points

# 设置日志
def setup_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def log_warning(msg):
    logging.warning(msg)
    warnings.warn(msg)

def log_error(msg):
    logging.error(msg)

def log_info(msg):
    logging.info(msg)

# ========================== 辅助函数 ==========================
def project_center_to_image(center_ego, cam_data):
    try:
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
    except Exception as e:
        log_warning(f"投影中心点失败: {e}")
        return None

def filter_objects_by_range(objects, x_range, y_range):
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
    try:
        with open(label_path, 'r') as f:
            data = json.load(f)
        objs = data.get('objects', []) if isinstance(data, dict) else data
    except Exception as e:
        log_warning(f"无法读取标注文件 {label_path}: {e}")
        return [], 0, {}

    filtered = filter_objects_by_range(objs, x_range, y_range)
    count = len(filtered)
    stats = defaultdict(int)
    for obj in filtered:
        cat = obj.get('type', 'unknown')
        stats[cat] += 1
    return filtered, count, dict(stats)

def get_primary_change_category(prev_stats, curr_stats):
    all_cats = set(prev_stats.keys()) | set(curr_stats.keys())
    max_abs_change = 0
    primary_cat = None
    change_val = 0
    change_type = None
    for cat in all_cats:
        prev = prev_stats.get(cat, 0)
        curr = curr_stats.get(cat, 0)
        diff = curr - prev
        if abs(diff) > max_abs_change:
            max_abs_change = abs(diff)
            primary_cat = cat
            change_val = diff
            change_type = 'increase' if diff > 0 else 'decrease' if diff < 0 else None
    return primary_cat, change_val, change_type

def visualize_frame_pair(pair_dir, prev_info, curr_info, prev_idx, curr_idx,
                         direction, prev_num, curr_num, primary_change):
    """为单个突变帧对生成可视化"""
    pair_dir = Path(pair_dir)
    try:
        pair_dir.mkdir(parents=True, exist_ok=True)
        log_info(f"创建目录: {pair_dir}")
    except Exception as e:
        log_error(f"无法创建目录 {pair_dir}: {e}")
        return

    prev_base = get_base_name(prev_info)
    curr_base = get_base_name(curr_info)

    for role, info, base in [('prev', prev_info, prev_base), ('curr', curr_info, curr_base)]:
        role_dir = pair_dir / role
        try:
            role_dir.mkdir(exist_ok=True)
        except Exception as e:
            log_error(f"无法创建子目录 {role_dir}: {e}")
            continue

        points = None
        try:
            points = load_points(info['lidar_path'])
        except Exception as e:
            log_warning(f"无法加载点云 {info['lidar_path']}: {e}")

        objs_filtered = info.get('_filtered_objects', [])

        for cam in ['CAM_A', 'CAM_B', 'CAM_C', 'CAM_D']:
            if cam not in info['cams']:
                continue
            cam_info = info['cams'][cam]
            src_img = cam_info.get('data_path')
            if not src_img or not Path(src_img).exists():
                log_warning(f"相机图像 {src_img} 不存在，跳过")
                continue
            try:
                img = cv2.imread(src_img)
                if img is None:
                    log_warning(f"无法读取图像 {src_img}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                log_warning(f"读取图像 {src_img} 时出错: {e}")
                continue

            for obj in objs_filtered:
                try:
                    box = {
                        'center': obj['box3d'][:3],
                        'size': obj['box3d'][3:6],
                        'yaw': obj['rotation'][0],
                        'type': obj.get('type', 'unknown')
                    }
                    uv_box, valid_box = project_box_to_image(box, cam_info)
                    img = draw_projected_box(img, uv_box, valid_box, color=(255,0,0), thickness=2)

                    center_ego = np.array(box['center'])
                    uv_center = project_center_to_image(center_ego, cam_info)
                    if uv_center is not None:
                        u, v = int(uv_center[0]), int(uv_center[1])
                        h_img, w_img = img.shape[:2]
                        if 0 <= u < w_img and 0 <= v < h_img:
                            cv2.putText(img, str(obj.get('id', -1)), (u, v),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                except Exception as e:
                    log_warning(f"处理目标 {obj.get('id', -1)} 时出错: {e}")
                    continue

            out_img = role_dir / f"{base}_{cam}.jpg"
            try:
                cv2.imwrite(str(out_img), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                log_info(f"保存图像: {out_img}")
            except Exception as e:
                log_warning(f"保存图像 {out_img} 失败: {e}")

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
                log_info(f"生成 BEV 图: {role_dir}")
            except Exception as e:
                log_warning(f"生成 BEV 图失败: {e}")

    prev_stats = prev_info.get('_filtered_stats', {})
    curr_stats = curr_info.get('_filtered_stats', {})
    primary_cat, change_val, change_type = primary_change
    info_txt = pair_dir / "info.txt"
    try:
        with open(info_txt, 'w') as f:
            f.write(f"异常类型: {direction}\n")
            f.write(f"prev_idx: {prev_idx}, curr_idx: {curr_idx}\n")
            f.write(f"prev_objects: {prev_num}, curr_objects: {curr_num}\n")
            f.write(f"变化比例: {abs(prev_num-curr_num)/max(prev_num, curr_num) if max(prev_num, curr_num)>0 else 0:.2f}, 变化绝对值: {abs(prev_num-curr_num)}\n\n")

            if primary_cat is not None:
                f.write(f"【主要突变类别】\n")
                f.write(f"  类别: {primary_cat}\n")
                f.write(f"  变化: {change_val:+d} ({change_type})\n\n")
            else:
                f.write("【主要突变类别】无显著类别变化\n\n")

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
        log_info(f"保存 info.txt: {info_txt}")
    except Exception as e:
        log_error(f"写入 info.txt 失败: {e}")

def visualize_abnormal_segment(out_root, segment_idx, start_idx, end_idx,
                               prev_info, start_info, end_info, next_info,
                               reason):
    """为数目异常段生成边界帧可视化"""
    seg_dir = out_root / f"abnormal_segment_{segment_idx:04d}_start{start_idx}_end{end_idx}"
    try:
        seg_dir.mkdir(parents=True, exist_ok=True)
        log_info(f"创建目录: {seg_dir}")
    except Exception as e:
        log_error(f"无法创建目录 {seg_dir}: {e}")
        return

    info_txt = seg_dir / "info.txt"
    try:
        with open(info_txt, 'w') as f:
            f.write("数目异常段检测报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"段索引: {segment_idx}\n")
            f.write(f"起始帧索引: {start_idx}, 结束帧索引: {end_idx}\n")
            f.write(f"段长度: {end_idx - start_idx + 1} 帧\n")
            f.write(f"异常原因: {reason}\n")
            f.write("\n边界帧说明:\n")
            f.write("  start-1: 段前一帧（若存在）\n")
            f.write("  start: 段起始帧\n")
            f.write("  end: 段结束帧\n")
            f.write("  end+1: 段后一帧（若存在）\n")
            f.write("\n")
        log_info(f"保存 info.txt: {info_txt}")
    except Exception as e:
        log_error(f"写入 info.txt 失败: {e}")

    frames = []
    if prev_info is not None:
        frames.append(('start-1', start_idx-1, prev_info))
    frames.append(('start', start_idx, start_info))
    frames.append(('end', end_idx, end_info))
    if next_info is not None:
        frames.append(('end+1', end_idx+1, next_info))

    for role, idx, info in frames:
        role_dir = seg_dir / role
        try:
            role_dir.mkdir(exist_ok=True)
        except Exception as e:
            log_error(f"无法创建子目录 {role_dir}: {e}")
            continue
        base = get_base_name(info)

        points = None
        try:
            points = load_points(info['lidar_path'])
        except Exception as e:
            log_warning(f"无法加载点云 {info['lidar_path']}: {e}")

        objs_filtered = info.get('_filtered_objects', [])

        for cam in ['CAM_A', 'CAM_B', 'CAM_C', 'CAM_D']:
            if cam not in info['cams']:
                continue
            cam_info = info['cams'][cam]
            src_img = cam_info.get('data_path')
            if not src_img or not Path(src_img).exists():
                log_warning(f"相机图像 {src_img} 不存在，跳过")
                continue
            try:
                img = cv2.imread(src_img)
                if img is None:
                    log_warning(f"无法读取图像 {src_img}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                log_warning(f"读取图像 {src_img} 时出错: {e}")
                continue

            for obj in objs_filtered:
                try:
                    box = {
                        'center': obj['box3d'][:3],
                        'size': obj['box3d'][3:6],
                        'yaw': obj['rotation'][0],
                        'type': obj.get('type', 'unknown')
                    }
                    uv_box, valid_box = project_box_to_image(box, cam_info)
                    img = draw_projected_box(img, uv_box, valid_box, color=(255,0,0), thickness=2)

                    center_ego = np.array(box['center'])
                    uv_center = project_center_to_image(center_ego, cam_info)
                    if uv_center is not None:
                        u, v = int(uv_center[0]), int(uv_center[1])
                        h_img, w_img = img.shape[:2]
                        if 0 <= u < w_img and 0 <= v < h_img:
                            cv2.putText(img, str(obj.get('id', -1)), (u, v),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                except Exception as e:
                    log_warning(f"处理目标 {obj.get('id', -1)} 时出错: {e}")
                    continue

            out_img = role_dir / f"{base}_{cam}.jpg"
            try:
                cv2.imwrite(str(out_img), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                log_info(f"保存图像: {out_img}")
            except Exception as e:
                log_warning(f"保存图像 {out_img} 失败: {e}")

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
                log_info(f"生成 BEV 图: {role_dir}")
            except Exception as e:
                log_warning(f"生成 BEV 图失败: {e}")

def plot_counts_curve(out_root, counts):
    plt.figure(figsize=(12, 6))
    plt.plot(counts, marker='.', linestyle='-', markersize=2, linewidth=0.8)
    plt.xlabel('Frame Index')
    plt.ylabel('Number of Objects (filtered)')
    plt.title('Object Count per Frame')
    plt.grid(True, alpha=0.3)
    min_idx = np.argmin(counts)
    plt.axvline(x=min_idx, color='r', linestyle='--', alpha=0.5, label=f'Min: {counts[min_idx]}')
    plt.legend()
    out_path = out_root / 'counts_curve.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    log_info(f"目标数量曲线已保存至: {out_path}")

def generate_summary(out_root, problem_pairs, abnormal_segments):
    summary_path = out_root / "summary.txt"
    try:
        with open(summary_path, 'w') as f:
            f.write("TYJT 数据集异常检测汇总报告\n")
            f.write("=" * 80 + "\n\n")

            # 突变检测统计
            f.write("【突变检测结果】\n")
            f.write(f"总计异常帧对数量: {len(problem_pairs)}\n")
            if problem_pairs:
                direction_counts = defaultdict(int)
                for _, _, _, _, direction, _, _, _ in problem_pairs:
                    direction_counts[direction] += 1
                f.write("按异常类型统计:\n")
                for d, cnt in direction_counts.items():
                    f.write(f"  {d}: {cnt}\n")
                f.write("\n")

                primary_cat_counts = defaultdict(int)
                for _, _, _, _, _, _, _, (primary_cat, _, _) in problem_pairs:
                    if primary_cat is not None:
                        primary_cat_counts[primary_cat] += 1
                if primary_cat_counts:
                    f.write("按主要突变类别统计:\n")
                    for cat, cnt in sorted(primary_cat_counts.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  {cat}: {cnt}\n")
                f.write("\n")

                f.write("详细异常帧对列表:\n")
                f.write("-" * 80 + "\n")
                for i, (prev_idx, curr_idx, prev_num, curr_num, direction,
                        prev_info, curr_info, (primary_cat, change_val, change_type)) in enumerate(problem_pairs):
                    prev_base = get_base_name(prev_info)
                    curr_base = get_base_name(curr_info)
                    f.write(f"\n【异常帧对 #{i:04d}】\n")
                    f.write(f"  方向: {direction}\n")
                    f.write(f"  前一帧: idx={prev_idx} | base_name={prev_base}\n")
                    f.write(f"  当前帧: idx={curr_idx} | base_name={curr_base}\n")
                    f.write(f"  目标数量变化: {prev_num} → {curr_num} (变化比例: {abs(prev_num-curr_num)/max(prev_num, curr_num) if max(prev_num, curr_num)>0 else 0:.2f}, 绝对值: {abs(prev_num-curr_num)})\n")
                    if primary_cat is not None:
                        f.write(f"  主要突变类别: {primary_cat} ({change_val:+d}, {change_type})\n")
                    else:
                        f.write(f"  主要突变类别: 无\n")
                    f.write("  " + "-" * 76 + "\n")
            else:
                f.write("未发现突变异常帧对。\n")
            f.write("\n" + "=" * 80 + "\n\n")

            # 数目异常段统计
            f.write("【数目异常段检测结果】\n")
            f.write(f"总计数目异常段数量: {len(abnormal_segments)}\n")
            if abnormal_segments:
                f.write("详细数目异常段列表:\n")
                f.write("-" * 80 + "\n")
                for i, (start, end, reason) in enumerate(abnormal_segments):
                    f.write(f"\n【数目异常段 #{i:04d}】\n")
                    f.write(f"  起始帧索引: {start}, 结束帧索引: {end}\n")
                    f.write(f"  段长度: {end - start + 1} 帧\n")
                    f.write(f"  异常原因: {reason}\n")
                    f.write("  " + "-" * 76 + "\n")
            else:
                f.write("未发现数目异常段。\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("报告结束\n")
        log_info(f"整体汇总报告已保存至: {summary_path}")
    except Exception as e:
        log_error(f"生成汇总报告失败: {e}")

def detect_missing_labels(info_pkl, out_dir, thresh_ratio, thresh_abs,
                          x_range=None, y_range=None,
                          plot_counts=True):
    # 设置日志
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(out_root / "detect.log")
    log_info("开始检测...")

    # 加载 info
    with open(info_pkl, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'infos' in data:
        infos = data['infos']
    else:
        infos = data
    log_info(f"加载 {len(infos)} 帧数据")

    # 预处理
    log_info("预处理标注（应用范围筛选）...")
    for idx, info in enumerate(infos):
        if idx % 1000 == 0:
            log_info(f"预处理进度: {idx}/{len(infos)}")
        label_path = info.get('label_path')
        if not label_path or not Path(label_path).exists():
            log_warning(f"帧 {idx} 标注文件不存在，跳过")
            info['_filtered_objects'] = []
            info['_filtered_count'] = 0
            info['_filtered_stats'] = {}
            continue
        objects, cnt, stats = load_and_filter_labels(label_path, x_range, y_range)
        info['_filtered_objects'] = objects
        info['_filtered_count'] = cnt
        info['_filtered_stats'] = stats

    # 提取目标数量序列（仅用于曲线图，不参与判断）
    counts = [info.get('_filtered_count', 0) for info in infos]

    # 曲线图
    if plot_counts:
        plot_counts_curve(out_root, counts)

    # 突变检测
    log_info("检测相邻帧突变...")
    problem_pairs = []  # 每个元素: (prev_idx, curr_idx, prev_num, curr_num, direction, prev_info, curr_info, primary_change)
    for idx, info in enumerate(infos):
        if idx % 1000 == 0:
            log_info(f"突变检测进度: {idx}/{len(infos)}, 已发现 {len(problem_pairs)}")
        prev_idx = info.get('prev', -1)
        if prev_idx == -1:
            continue
        prev_info = infos[prev_idx]
        prev_num = prev_info.get('_filtered_count', 0)
        curr_num = info.get('_filtered_count', 0)
        if prev_num == 0 and curr_num == 0:
            continue
        max_num = max(prev_num, curr_num)
        diff = abs(prev_num - curr_num)
        ratio = diff / max_num if max_num > 0 else 0
        if ratio >= thresh_ratio and diff >= thresh_abs:
            direction = 'increase' if curr_num > prev_num else 'decrease'
            prev_stats = prev_info.get('_filtered_stats', {})
            curr_stats = info.get('_filtered_stats', {})
            primary_cat, change_val, change_type = get_primary_change_category(prev_stats, curr_stats)
            problem_pairs.append((prev_idx, idx, prev_num, curr_num, direction,
                                  prev_info, info, (primary_cat, change_val, change_type)))

    log_info(f"发现 {len(problem_pairs)} 个突变帧对")
    for i, (prev_idx, curr_idx, prev_num, curr_num, direction,
            prev_info, curr_info, primary_change) in enumerate(problem_pairs):
        pair_dir = out_root / f"pair_{i:04d}_prev{prev_idx}_curr{curr_idx}"
        log_info(f"处理突变帧对 {i}: {pair_dir}")
        visualize_frame_pair(pair_dir, prev_info, curr_info, prev_idx, curr_idx,
                             direction, prev_num, curr_num, primary_change)

    # ==================== 数目异常段检测 ====================
    log_info("基于突变点分段检测数目异常段...")

    # 收集所有突变点及其方向
    # 突变点由 (curr_idx, direction) 组成，direction 是突变发生的“后一帧”相对于前一帧的变化
    mutations = []
    for prev_idx, curr_idx, _, _, direction, _, _, _ in problem_pairs:
        mutations.append((curr_idx, direction))
    # 按位置排序
    mutations.sort(key=lambda x: x[0])
    log_info(f"突变点列表（位置，方向）: {mutations}")

    # 如果没有突变，则没有数目异常段
    if not mutations:
        log_info("无突变点，无法分段")
        abnormal_segments = []
    else:
        # 划分区间：区间不包含突变点本身，突变点作为前一段的结束
        # 第一个区间: [0, mutations[0][0] - 1]
        # 中间区间: [mutations[i-1][0], mutations[i][0] - 1]
        # 最后一个区间: [mutations[-1][0], n-1]
        intervals = []
        # 第一个区间
        start = 0
        end = mutations[0][0] - 1
        if start <= end:
            intervals.append((start, end))
        # 中间区间
        for i in range(1, len(mutations)):
            start = mutations[i-1][0]
            end = mutations[i][0] - 1
            if start <= end:
                intervals.append((start, end))
        # 最后一个区间
        start = mutations[-1][0]
        end = len(infos) - 1
        if start <= end:
            intervals.append((start, end))

        log_info(f"划分的区间: {intervals}")

        # 现在 intervals 的长度等于 mutations 长度 + 1
        # 每个区间的前一个突变方向是 mutations[i-1][1]（对于 i>0）
        # 每个区间的后一个突变方向是 mutations[i][1]（对于 i < len(mutations)）
        abnormal_segments = []
        for i, (start, end) in enumerate(intervals):
            # 前一个突变方向（如果 i > 0）
            prev_direction = mutations[i-1][1] if i > 0 else None
            # 后一个突变方向（如果 i < len(mutations)）
            next_direction = mutations[i][1] if i < len(mutations) else None
            reasons = []
            # 规则1：若前一个突变是 decrease，则该段相对于前一段数目减少，是异常段
            if prev_direction == 'decrease':
                reasons.append("前一个突变是 decrease（数目突然变少）")
            # 规则2：若后一个突变是 increase，则该段相对于后一段数目减少，是异常段
            if next_direction == 'increase':
                reasons.append("后一个突变是 increase（数目突然变多）")
            if reasons:
                reason_str = "；".join(reasons)
                abnormal_segments.append((start, end, reason_str))
                log_info(f"数目异常段: [{start},{end}] 原因: {reason_str}")

        log_info(f"发现 {len(abnormal_segments)} 个数目异常段")

        # 生成可视化
        for seg_idx, (start, end, reason) in enumerate(abnormal_segments):
            prev_info = infos[start-1] if start > 0 else None
            start_info = infos[start]
            end_info = infos[end]
            next_info = infos[end+1] if end+1 < len(infos) else None
            log_info(f"处理数目异常段 {seg_idx}: 帧 {start}-{end}")
            visualize_abnormal_segment(out_root, seg_idx, start, end,
                                       prev_info, start_info, end_info, next_info,
                                       reason)

    # 汇总报告
    generate_summary(out_root, problem_pairs, abnormal_segments)
    log_info("检测完成")

def main():
    parser = argparse.ArgumentParser(description="TYJT 数据集异常检测工具（v061）")
    parser.add_argument('--info-pkl', required=True, help='info pkl 文件')
    parser.add_argument('--detect-out', default='./detect_results', help='输出目录')
    parser.add_argument('--thresh-ratio', type=float, default=0.5, help='突变比例阈值')
    parser.add_argument('--thresh-abs', type=int, default=3, help='突变绝对值阈值')
    parser.add_argument('--x-range', nargs=2, type=float, default=None, help='X 范围')
    parser.add_argument('--y-range', nargs=2, type=float, default=None, help='Y 范围')
    parser.add_argument('--no-plot', action='store_true', help='不生成目标数量曲线图')
    args = parser.parse_args()

    detect_missing_labels(
        info_pkl=args.info_pkl,
        out_dir=args.detect_out,
        thresh_ratio=args.thresh_ratio,
        thresh_abs=args.thresh_abs,
        x_range=args.x_range,
        y_range=args.y_range,
        plot_counts=not args.no_plot
    )

if __name__ == '__main__':
    main()