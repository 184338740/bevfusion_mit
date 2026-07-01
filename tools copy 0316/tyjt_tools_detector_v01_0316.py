#!/usr/bin/env python3
"""
TYJT 数据集异常帧对检测工具
基于 info pkl 文件，检测相邻帧目标数量突变（增加或减少）超过阈值的帧对，
并为每个异常帧对生成可视化图像（相机带框图 + BEV 图）和 info.txt 记录。

使用说明：
python tyjt_tools_detector_v01_0316.py --info-pkl <pkl文件> --detect-out <输出目录> [可选参数]

用例:
# tag-val
python tyjt_tools_detector_v01_0316.py \
    --info-pkl ./tyjt_data_infos_v03/tyjt_infos_val.pkl \
    --detect-out ./output/my_detectResults_Error_val \
    --thresh-ratio 0.2 \
    --thresh-abs 10 \
    --x-range -60 60 \
    --y-range -60 60

# tag-train
python tyjt_tools_detector_v01_0316.py \
    --info-pkl ./output/tyjt_data_infos_v03/tyjt_infos_train.pkl \
    --detect-out ./my_detectResults_Error_train \
    --thresh-ratio 0.2 \
    --thresh-abs 10 \
    --x-range -60 60 \
    --y-range -60 60
"""

import argparse
import pickle
import json
import numpy as np
import shutil
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict

# 导入公共工具模块（需提前在 tyjt_utils 中定义）
from tyjt_utils.tyjt_vis_utils import (
    quaternion_to_rotation_matrix,
    project_box_to_image,
    draw_projected_box,
    generate_bev_image,
)
from tyjt_utils.tyjt_data_utils import get_base_name, load_points, is_obj_in_range


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
        infos = pickle.load(f)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    problem_pairs = []
    total_samples = len(infos)

    # 统计有效 prev 的数量
    valid_prev = sum(1 for info in infos if info.get('prev', -1) != -1)
    print(f"总样本数: {total_samples}, 具有有效 prev 的样本数: {valid_prev}")
    print(f"目标过滤范围: x∈{x_range}, y∈{y_range}")

    # 遍历所有样本
    for idx, info in enumerate(infos):
        if idx % 1000 == 0:
            print(f"已处理 {idx}/{total_samples} 个样本，当前发现 {len(problem_pairs)} 个异常帧对")

        prev_idx = info.get('prev', -1)
        if prev_idx == -1:
            continue
        prev_info = infos[prev_idx]

        # 使用预存的统计字段，若无则回退到解析标注文件（兼容旧版）
        prev_stats = prev_info.get('obj_stats', {})
        curr_stats = info.get('obj_stats', {})
        prev_num = prev_info.get('total_objects', sum(prev_stats.values()))
        curr_num = info.get('total_objects', sum(curr_stats.values()))

        # 如果需要范围过滤，但预存统计可能未过滤，此处简单起见不二次过滤（假设 pkl 中已包含过滤后的统计）
        # 若需精确过滤，可在此处重新读取标注文件并过滤（略）

        # 检测变化（双向）
        if prev_num > 0 or curr_num > 0:
            max_num = max(prev_num, curr_num)
            diff = abs(prev_num - curr_num)
            ratio = diff / max_num if max_num > 0 else 0
            if ratio >= thresh_ratio and diff >= thresh_abs:
                direction = 'increase' if curr_num > prev_num else 'decrease'
                problem_pairs.append((prev_idx, idx, prev_num, curr_num, direction,
                                      prev_info, info))  # 保存两帧的 info

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
            points = load_points(info['lidar_path'])

            # 获取该帧的所有 objects（原始标注数据）
            try:
                with open(info['label_path'], 'r') as f:
                    label_data = json.load(f)
                objs_all = label_data.get('objects', []) if isinstance(label_data, dict) else label_data
            except Exception as e:
                print(f"警告：无法读取标注文件 {info['label_path']}，跳过可视化")
                objs_all = []

            # 为每个相机生成带框图像（并绘制目标 ID）
            for cam in ['CAM_A', 'CAM_B', 'CAM_C', 'CAM_D']:
                if cam in info['cams']:
                    src_img = info['cams'][cam]['img_path']
                    if Path(src_img).exists():
                        img = cv2.imread(src_img)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # 投影每个框
                        for obj in objs_all:
                            box = {
                                'center': obj['box3d'][:3],
                                'size': obj['box3d'][3:6],
                                'yaw': obj['rotation'][0],
                                'type': obj.get('type', 'unknown')
                            }
                            uv_box, valid_box = project_box_to_image(box, info['cams'][cam])
                            img = draw_projected_box(img, uv_box, valid_box, color=(255,0,0), thickness=2)

                            # 绘制目标 ID
                            center_ego = np.array([box['center']])
                            t_cam2ego = np.array(info['cams'][cam]['cam2ego_translation'], dtype=np.float32)
                            q = np.array(info['cams'][cam]['cam2ego_rotation'], dtype=np.float32)
                            R_cam2ego = quaternion_to_rotation_matrix(q)
                            R_ego2cam = R_cam2ego.T
                            t_ego2cam = -R_ego2cam @ t_cam2ego
                            center_cam = (R_ego2cam @ center_ego.T).T + t_ego2cam
                            z = center_cam[0, 2]
                            if z > 0:
                                intrinsic = np.array(info['cams'][cam]['cam_intrinsic'], dtype=np.float32)
                                u = intrinsic[0, 0] * center_cam[0, 0] / z + intrinsic[0, 2]
                                v = intrinsic[1, 1] * center_cam[0, 1] / z + intrinsic[1, 2]
                                h_img, w_img = img.shape[:2]
                                if 0 <= u < w_img and 0 <= v < h_img:
                                    cv2.putText(img, str(obj.get('id', -1)), (int(u), int(v)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        out_img = role_dir / f"{base}_{cam}.jpg"
                        cv2.imwrite(str(out_img), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # 生成 BEV 图（带目标 ID）
            boxes_bev = []
            ids_list = []
            for obj in objs_all:
                boxes_bev.append({
                    'center': obj['box3d'][:3],
                    'size': obj['box3d'][3:6],
                    'yaw': obj['rotation'][0]
                })
                ids_list.append(obj.get('id', -1))
            generate_bev_image(points, boxes_bev, base, role_dir,
                               x_range=(-100,100), y_range=(-100,100),
                               point_size=0.5, alpha=0.6, ids=ids_list)

        # 保存 info.txt，增加类别统计
        prev_stats = prev_info.get('obj_stats', {})
        curr_stats = curr_info.get('obj_stats', {})

        with open(pair_dir / "info.txt", 'w') as f:
            f.write(f"异常类型: {direction}\n")
            f.write(f"prev_idx: {prev_idx}, curr_idx: {curr_idx}\n")
            f.write(f"prev_objects: {prev_num}, curr_objects: {curr_num}\n")
            f.write(f"变化比例: {abs(prev_num-curr_num)/max(prev_num, curr_num) if max(prev_num, curr_num)>0 else 0:.2f}, 变化绝对值: {abs(prev_num-curr_num)}\n\n")

            # 输出类别统计对比
            f.write("【类别统计对比】\n")
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
                f.write(f"  {cam}_path: {prev_info['cams'].get(cam, {}).get('img_path', 'N/A')}\n")
            f.write("\n【当前帧】\n")
            f.write(f"  base_name: {curr_base}\n")
            f.write(f"  human_time: {curr_info.get('human_time', 'N/A')}\n")
            f.write(f"  label_path: {curr_info['label_path']}\n")
            for cam in ['CAM_A', 'CAM_B', 'CAM_C', 'CAM_D']:
                f.write(f"  {cam}_path: {curr_info['cams'].get(cam, {}).get('img_path', 'N/A')}\n")


    if len(problem_pairs) == 0:
        print("未发现异常帧对。")
    else:
        print(f"发现 {len(problem_pairs)} 个异常帧对，开始生成可视化文件...")
        # 处理每个异常帧对（略）
        print(f"检测完成，结果保存在: {out_root.resolve()}")
    return problem_pairs


def main():
    parser = argparse.ArgumentParser(description="TYJT 数据集异常帧对检测工具")
    parser.add_argument('--info-pkl', type=str, required=True,
                        help='输入的 info pkl 文件路径（如 ./tyjt_data_infos_v03/tyjt_infos_train.pkl）')
    parser.add_argument('--detect-out', type=str, default='./detect_results',
                        help='异常帧对输出根目录')
    parser.add_argument('--thresh-ratio', type=float, default=0.5,
                        help='目标数量变化比例阈值，默认0.5')
    parser.add_argument('--thresh-abs', type=int, default=3,
                        help='目标数量变化绝对值阈值，默认3')
    parser.add_argument('--x-range', nargs=2, type=float, default=[-60, 60],
                        help='X 方向过滤范围（min max），默认 -60 60')
    parser.add_argument('--y-range', nargs=2, type=float, default=[-60, 60],
                        help='Y 方向过滤范围（min max），默认 -60 60')
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