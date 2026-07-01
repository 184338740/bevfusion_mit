#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
========================================================================
File: step3_3_analyze_dbinfos_pkl_v1.1.py
Version: v1.1
Date: 2026-02-26
Author: xmy
========================================================================
Description:
    分析 nuScenes 格式的数据库文件（*_dbinfos_train.pkl）。
    增强版统计：
      - 每个类别的实例数、占比
      - 点数的 min, max, mean, median, p95, p99
      - 单独统计点数为 0、1 的实例数
      - 点数区间分布（0, 1, 2-5, 6-10, 11-50, 51-100, 101-500, 501-1000, 1001-5000, >5000）
    支持导出 JSON 摘要。

Usage:
    python step3_3_analyze_dbinfos_pkl_v1.1.py <dbinfos.pkl> [--output JSON_FILE]

Example:
    python step3_3_analyze_dbinfos_pkl_v1.1.py ./tyjt_dbinfos_train.pkl
    python step3_3_analyze_dbinfos_pkl_v1.1.py ./tyjt_dbinfos_train.pkl --output dbinfo_summary.json
========================================================================
"""

import pickle
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter


def print_tree(data, prefix="", is_last=True):
    """递归打印树形结构"""
    if isinstance(data, dict):
        items = list(data.items())
        for i, (key, value) in enumerate(items):
            cur_last = (i == len(items) - 1)
            line_prefix = prefix + ("└─ " if cur_last else "├─ ")
            print(f"{line_prefix}{key}: ", end="")
            if isinstance(value, (dict, list)):
                print()
                new_prefix = prefix + ("    " if cur_last else "│   ")
                print_tree(value, new_prefix, cur_last)
            else:
                print(value)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            cur_last = (i == len(data) - 1)
            line_prefix = prefix + ("└─ " if cur_last else "├─ ")
            if isinstance(item, (dict, list)):
                print(f"{line_prefix}[{i}]:")
                new_prefix = prefix + ("    " if cur_last else "│   ")
                print_tree(item, new_prefix, cur_last)
            else:
                print(f"{line_prefix}[{i}]: {item}")
    else:
        print(data)


def compute_point_stats(points_list):
    """
    输入：points_list (list of int) 每个目标的点数
    返回：包含各种统计指标的字典
    """
    if not points_list:
        return {}

    arr = np.array(points_list)
    stats = {
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "mean": round(float(np.mean(arr)), 2),
        "median": int(np.median(arr)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
    }

    # 单独统计点数为 0、1 的数量
    count_0 = np.sum(arr == 0)
    count_1 = np.sum(arr == 1)

    # 自定义区间分布
    bins = [0, 1, 2, 6, 11, 51, 101, 501, 1001, 5001, float('inf')]
    bin_labels = ["0", "1", "2-5", "6-10", "11-50", "51-100", "101-500", "501-1000", "1001-5000", ">5000"]
    hist, _ = np.histogram(arr, bins=bins)
    hist_dict = {label: int(hist[i]) for i, label in enumerate(bin_labels) if hist[i] > 0}

    stats.update({
        "count_0": int(count_0),
        "count_1": int(count_1),
        "histogram": hist_dict
    })
    return stats


def analyze_dbinfos_pkl(pkl_path: str, output_json: str = None):
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

    if not isinstance(data, dict):
        print("❌ 错误：数据库文件应为字典类型（类别名 -> 实例列表）")
        return

    total_instances = 0
    class_stats = {}

    for cls_name, instances in data.items():
        if not isinstance(instances, list):
            print(f"⚠️ 类别 {cls_name} 的值不是列表，跳过")
            continue

        num_instances = len(instances)
        total_instances += num_instances

        # 收集每个实例的点数
        points_list = []
        for inst in instances:
            if isinstance(inst, dict):
                pts = inst.get('num_points_in_gt', 0)
                points_list.append(pts)
            else:
                points_list.append(0)

        point_stats = compute_point_stats(points_list)

        class_stats[cls_name] = {
            "instances": num_instances,
            "points": point_stats
        }

    # 计算每个类别的占比（基于总实例数）
    for cls_name, stats in class_stats.items():
        stats["percentage"] = round(stats["instances"] / total_instances * 100, 2)

    # 按实例数降序排序
    sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]["instances"], reverse=True)

    # 构建最终摘要
    summary = {
        "file": str(pkl_file),
        "total_instances": total_instances,
        "num_classes": len(class_stats),
        "classes": {}
    }
    for cls_name, stats in sorted_classes:
        summary["classes"][cls_name] = stats

    # 打印树形结构
    print(f"\n📁 分析数据库文件: {pkl_file.name}")
    print("=" * 60)
    print_tree(summary)
    print("=" * 60)
    print("✅ 分析完成")

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
        description="分析 nuScenes 数据库文件（*_dbinfos_train.pkl）- 增强版"
    )
    parser.add_argument("pkl_path", help="数据库 pkl 文件路径")
    parser.add_argument("--output", "-o", help="将分析摘要导出为 JSON 文件")
    args = parser.parse_args()

    analyze_dbinfos_pkl(args.pkl_path, args.output)


if __name__ == "__main__":
    main()