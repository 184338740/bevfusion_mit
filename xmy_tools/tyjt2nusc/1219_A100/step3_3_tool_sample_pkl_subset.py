#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
========================================================================
File: step3_3_tool_sample_pkl_subset.py
Version: v1.0
Date: 2026-02-13
Author: xmy
========================================================================
Description:
    从 nuScenes 格式的 info pkl 文件中抽取前 N 个样本，生成一个较小的子集 pkl。
    用于快速调试模型（避免完整验证集耗时过长）。

Usage:
    python step3_3_tool_sample_pkl_subset.py <src_pkl> <dst_pkl> [--num NUM]

Example:
    python step3_3_tool_sample_pkl_subset.py \
        /path/to/tyjt_infos_val.pkl \
        /path/to/tyjt_infos_val_subset.pkl \
        --num 100
========================================================================
"""

import pickle
import argparse
from pathlib import Path


def sample_pkl_subset(src_path: str, dst_path: str, num_samples: int = 100):
    """
    从源 pkl 文件中抽取前 num_samples 个样本，保存到目标 pkl。
    """
    src_file = Path(src_path)
    if not src_file.exists():
        print(f"❌ 源文件不存在: {src_file}")
        return

    # 加载原始 pkl
    with open(src_file, 'rb') as f:
        data = pickle.load(f)

    # 判断数据结构
    if isinstance(data, dict):
        if 'infos' in data:
            infos = data['infos']
            # 抽取前 num_samples 个
            sampled_infos = infos[:num_samples]
            # 创建新数据，保留其他字段
            new_data = data.copy()
            new_data['infos'] = sampled_infos
            # 可选：在 metadata 中记录子集信息
            if 'metadata' in new_data:
                new_data['metadata']['subset_info'] = f"sampled_first_{num_samples}_from_{len(infos)}"
        else:
            # 如果 dict 没有 'infos'，则假设其值为 infos 列表（例如 {'infos': [...]} 或其他键）
            # 取第一个键作为样本列表
            key = next(iter(data))
            if isinstance(data[key], list):
                infos = data[key]
                sampled_infos = infos[:num_samples]
                new_data = data.copy()
                new_data[key] = sampled_infos
                print(f"⚠️ 使用键 '{key}' 作为样本列表")
            else:
                print("❌ 无法识别的 dict 结构，期望包含 'infos' 列表或单个列表键")
                return
    elif isinstance(data, list):
        # 直接是列表，则抽取前 num_samples 个
        sampled_infos = data[:num_samples]
        new_data = sampled_infos
        print("⚠️ 源文件为纯列表格式，抽取后仍为列表")
    else:
        print(f"❌ 不支持的数据类型: {type(data)}")
        return

    # 保存新 pkl
    dst_file = Path(dst_path)
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_file, 'wb') as f:
        pickle.dump(new_data, f)

    print(f"✅ 已生成子集 pkl: {dst_file}")
    print(f"   原始样本数: {len(infos) if 'infos' in locals() else 'unknown'}, 抽取样本数: {num_samples}")


def main():
    parser = argparse.ArgumentParser(
        description="从 nuScenes 格式的 info pkl 文件中抽取前 N 个样本，生成子集 pkl。"
    )
    parser.add_argument("src_pkl", help="源 pkl 文件路径")
    parser.add_argument("dst_pkl", help="目标 pkl 文件路径")
    parser.add_argument("--num", type=int, default=100,
                        help="抽取的样本数量（默认 100）")
    args = parser.parse_args()

    sample_pkl_subset(args.src_pkl, args.dst_pkl, args.num)


if __name__ == "__main__":
    main()