#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
========================================================================
File: step3_3_tool_analyze_nusc_pkl_v1.0.py
Version: v1.2 (摘要版)
Date: 2026-02-13
Author: xmy
========================================================================
Description:
    分析单个 nuScenes 格式的 info pkl 文件（如 *_infos_train.pkl）。
    输出 metadata、样本总数、场景列表、类别分布等**摘要信息**。
    控制台以树形结构打印，列表/字典过长时自动截断。
    支持将分析摘要导出为 JSON 文件（--output）。

Usage:
    python step3_3_tool_analyze_nusc_pkl_v1.0.py <pkl_path> [--output JSON_FILE]

Example:
    python step3_3_tool_analyze_nusc_pkl_v1.0.py ./tyjt_infos_train.pkl
    python step3_3_tool_analyze_nusc_pkl_v1.0.py ./tyjt_infos_val.pkl --output summary.json

    python step3_3_tool_analyze_nusc_infos_pkl_v1.0.py /data2/xmy/01_project/bevfusion_mit_xmy_1205/xmy_tools/tyjt2nusc/1219_A100/output-1226-v9.2.3-all/step1/nuscenes_tyjt/tyjt_infos_train.pkl
    python step3_3_tool_analyze_nusc_infos_pkl_v1.0.py /data2/xmy/01_project/bevfusion_mit_xmy_1205/xmy_tools/tyjt2nusc/1219_A100/output-1226-v9.2.3-all/step1/nuscenes_tyjt/tyjt_infos_val.pkl
========================================================================
"""

import pickle
import json
import argparse
from pathlib import Path
from collections import Counter


def truncate_value(obj, max_len=5):
    """
    对列表/字典进行截断，返回适合打印的简化版本。
    """
    if isinstance(obj, dict):
        items = list(obj.items())
        if len(items) > max_len:
            truncated = dict(items[:max_len])
            truncated[f"... (and {len(items)-max_len} more)"] = ""
            return truncated
        return obj
    elif isinstance(obj, (list, tuple)):
        if len(obj) > max_len:
            truncated = list(obj[:max_len])
            truncated.append(f"... (and {len(obj)-max_len} more)")
            return truncated
        return obj
    else:
        return obj


def print_tree(data, indent=0, prefix="", is_last=True, max_preview=5):
    """
    递归打印树形结构，对容器进行截断。
    """
    if isinstance(data, dict):
        items = list(data.items())
        # 如果当前是根节点且字典很大，先打印键名再递归
        for i, (key, value) in enumerate(items):
            cur_last = (i == len(items) - 1)
            line_prefix = prefix + ("└─ " if cur_last else "├─ ")
            print(f"{line_prefix}{key}: ", end="")
            if isinstance(value, (dict, list)):
                print()
                new_prefix = prefix + ("    " if cur_last else "│   ")
                print_tree(value, indent + 1, new_prefix, cur_last, max_preview)
            else:
                print(value)
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            cur_last = (i == len(data) - 1)
            line_prefix = prefix + ("└─ " if cur_last else "├─ ")
            if isinstance(item, (dict, list)):
                print(f"{line_prefix}[{i}]:")
                new_prefix = prefix + ("    " if cur_last else "│   ")
                print_tree(item, indent + 1, new_prefix, cur_last, max_preview)
            else:
                print(f"{line_prefix}[{i}]: {item}")
    else:
        # 标量值，由上层处理
        pass


def analyze_nusc_pkl(pkl_path: str, output_json: str = None):
    pkl_file = Path(pkl_path)
    if not pkl_file.exists():
        print(f"❌ 文件不存在: {pkl_file}")
        return

    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ 加载 pkl 失败: {e}")
        return

    # 解析数据结构
    if isinstance(data, dict):
        metadata = data.get('metadata', {})
        infos = data.get('infos', [])
        # 兼容旧格式：只有一个键且该键是列表
        if not infos and len(data) == 1:
            first_key = next(iter(data))
            infos = data[first_key]
    elif isinstance(data, list):
        metadata = {}
        infos = data
    else:
        print("❌ 无法解析的数据格式")
        return

    samples = len(infos)

    # 统计场景和标注
    scene_counter = Counter()
    total_gt = 0
    class_counter = Counter()
    sample_with_gt = 0

    for info in infos:
        # 场景名
        scene = info.get('scene_name') or info.get('scene_token', 'unknown')
        if scene:
            scene_counter[scene] += 1
        # 标注
        gt_names = info.get('gt_names')
        if gt_names is not None and len(gt_names) > 0:
            sample_with_gt += 1
            total_gt += len(gt_names)
            class_counter.update(gt_names)

    avg_gt = total_gt / samples if samples else 0

    # 构建摘要字典（准备用于打印和导出）
    summary = {
        "file": str(pkl_file),
        "metadata": metadata,
        "samples": samples,
        "scenes": {
            "count": len(scene_counter),
            "top_5": dict(scene_counter.most_common(5))
        },
        "annotations": {
            "samples_with_gt": sample_with_gt,
            "total_gt_boxes": total_gt,
            "gt_per_sample_avg": round(avg_gt, 2),
            "top_10_classes": dict(class_counter.most_common(10))
        }
    }

    # 如果场景数超过5，添加提示
    if len(scene_counter) > 5:
        summary["scenes"]["note"] = f"... (and {len(scene_counter)-5} more scenes)"

    # 如果类别数超过10，添加提示
    if len(class_counter) > 10:
        summary["annotations"]["note"] = f"... (and {len(class_counter)-10} more classes)"

    # 树形打印
    print(f"\n📁 分析文件: {pkl_file.name}")
    print("=" * 60)
    print_tree(summary, max_preview=5)
    print("=" * 60)
    print("✅ 分析完成")

    # 导出摘要 JSON
    if output_json:
        out_path = Path(output_json)
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"📄 分析摘要已导出至: {out_path}")
        except Exception as e:
            print(f"❌ 导出 JSON 失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="分析 nuScenes 格式 info pkl 文件（摘要版）",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pkl_path", help="要分析的 pkl 文件路径")
    parser.add_argument("--output", "-o", help="将分析摘要导出为 JSON 文件")
    args = parser.parse_args()

    analyze_nusc_pkl(args.pkl_path, args.output)


if __name__ == "__main__":
    main()