#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
========================================================================
File: step3_3_tool_analyze_nusc_dbinfos_pkl_v1.0.py
Version: v1.0
Date: 2026-02-26
Author: xmy
========================================================================
Description:
    分析 nuScenes 格式的数据库文件（*_dbinfos_train.pkl）。
    输出每个类别的实例数、总框数、平均点云数、点云数范围等统计信息。
    控制台以树形结构打印，支持导出为 JSON 文件。

Usage:
    python step3_3_tool_analyze_nusc_dbinfos_pkl_v1.0.py <dbinfos.pkl> [--output JSON_FILE]

Example:
    python step3_3_tool_analyze_nusc_dbinfos_pkl_v1.0.py ./tyjt_dbinfos_train.pkl
    python step3_3_tool_analyze_nusc_dbinfos_pkl_v1.0.py ./tyjt_dbinfos_train.pkl --output dbinfo_summary.json
========================================================================
"""

import pickle
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter


def print_tree(data, prefix="", is_last=True):
    """
    递归打印树形结构（简化版，仅用于此脚本）。
    """
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

    # 检查数据结构：应为字典，键为类别名，值为列表
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

        # 收集点数信息（每个实例的 'num_points_in_gt' 字段）
        points_list = []
        for inst in instances:
            if isinstance(inst, dict):
                pts = inst.get('num_points_in_gt', 0)
                points_list.append(pts)
            else:
                points_list.append(0)

        if points_list:
            avg_points = np.mean(points_list)
            max_points = np.max(points_list)
            min_points = np.min(points_list)
            # 统计点数分布（可选分箱）
            hist, bins = np.histogram(points_list, bins=5)
            points_dist = {
                "min": int(min_points),
                "max": int(max_points),
                "avg": round(float(avg_points), 2),
                "histogram": [int(h) for h in hist],
                "bins": [round(float(b), 2) for b in bins]
            }
        else:
            points_dist = {}

        class_stats[cls_name] = {
            "instances": num_instances,
            "percentage": round(num_instances / total_instances * 100, 2) if total_instances else 0,
            "points": points_dist
        }

    # 按实例数排序
    sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]["instances"], reverse=True)

    # 构建摘要字典
    summary = {
        "file": str(pkl_file),
        "total_instances": total_instances,
        "num_classes": len(class_stats),
        "classes": {}
    }
    for cls_name, stats in sorted_classes:
        summary["classes"][cls_name] = stats

    # 树形打印
    print(f"\n📁 分析数据库文件: {pkl_file.name}")
    print("=" * 60)
    print_tree(summary)
    print("=" * 60)
    print("✅ 分析完成")

    # 导出 JSON
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
        description="分析 nuScenes 数据库文件（*_dbinfos_train.pkl）"
    )
    parser.add_argument("pkl_path", help="数据库 pkl 文件路径")
    parser.add_argument("--output", "-o", help="将分析摘要导出为 JSON 文件")
    args = parser.parse_args()

    analyze_dbinfos_pkl(args.pkl_path, args.output)


if __name__ == "__main__":
    main()