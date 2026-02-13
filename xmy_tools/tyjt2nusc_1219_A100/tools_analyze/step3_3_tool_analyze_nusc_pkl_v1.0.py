#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
========================================================================
File: step3_3_tool_analyze_nusc_pkl_v1.0.py
Version: v1.0
Date: 2026-02-13
Author: [xmy]
========================================================================
Description:
    快速分析单个 nuScenes 格式的 info pkl 文件（如 *_infos_train.pkl）。
    输出 metadata、样本总数、场景列表、标注类别分布等基础信息。
    适用于标准 nuScenes 及通过 TYJT 转换生成的兼容格式。

Usage:
    python step3_3_tool_analyze_nusc_pkl_v1.0.py <pkl_path> [--verbose]

Example:
    python step3_3_tool_analyze_nusc_pkl_v1.0.py ./tyjt_infos_train.pkl
    python step3_3_tool_analyze_nusc_pkl_v1.0.py ./tyjt_infos_val.pkl --verbose

Changelog:
    v1.0 (2026-02-13):
        - 初始版本
        - 支持从命令行读取 pkl 文件
        - 输出 metadata、样本数、场景数、类别统计
        - 添加 --verbose 选项显示详细场景列表
========================================================================
"""

import pickle
import argparse
from pathlib import Path
from collections import defaultdict, Counter

def analyze_nusc_pkl(pkl_path: str, verbose: bool = False):
    """
    分析单个 nuScenes 格式的 info pkl 文件，打印关键信息。
    
    Args:
        pkl_path (str): pkl 文件路径
        verbose (bool): 是否显示详细场景列表
    """
    pkl_file = Path(pkl_path)
    if not pkl_file.exists():
        print(f"❌ 文件不存在: {pkl_file}")
        return

    print("\n" + "="*70)
    print(f"📁 分析文件: {pkl_file.name}")
    print("="*70)

    # 加载 pkl
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # ----- 1. 顶层结构 -----
    if isinstance(data, dict):
        print(f"📌 顶层类型: dict, keys: {list(data.keys())}")
        metadata = data.get('metadata', {})
        infos = data.get('infos', [])
        if not infos and len(data) == 1:
            # 兼容某些只有单个 key 的格式
            first_key = next(iter(data))
            infos = data[first_key]
            print(f"⚠️  使用首个键 '{first_key}' 作为 infos 列表")
    elif isinstance(data, list):
        print(f"📌 顶层类型: list (可能是旧格式), 长度: {len(data)}")
        metadata = {}
        infos = data
    else:
        print(f"❌ 不支持的数据类型: {type(data)}")
        return

    print(f"📊 样本数 (infos): {len(infos)}")

    # ----- 2. Metadata -----
    print("\n📋 Metadata:")
    if metadata:
        for k, v in metadata.items():
            print(f"   {k}: {v}")
        # 检查 version 是否官方
        ver = metadata.get('version', '')
        if ver and ver not in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']:
            print(f"   ⚠️  version 不是官方版本: {ver}")
    else:
        print("   (无 metadata)")

    if len(infos) == 0:
        print("\n⚠️  样本列表为空，分析结束。")
        return

    # ----- 3. 场景统计 -----
    scene_names = set()
    scene_sample_cnt = defaultdict(int)
    sample_tokens = []
    
    for info in infos:
        # 尝试从不同字段获取场景名
        scene = info.get('scene_name') or info.get('scene_token', 'unknown')
        if isinstance(scene, str) and scene.startswith('scene-'):
            scene_names.add(scene)
            scene_sample_cnt[scene] += 1
        sample_tokens.append(info.get('token', 'N/A'))

    print(f"\n🌍 场景数: {len(scene_names)}")
    if verbose and scene_names:
        print("   场景列表 (样本数):")
        for scene, cnt in sorted(scene_sample_cnt.items())[:20]:  # 最多显示20个
            print(f"     {scene}: {cnt}")
        if len(scene_sample_cnt) > 20:
            print(f"     ... 还有 {len(scene_sample_cnt)-20} 个场景未显示")
    elif scene_names:
        print("   (使用 --verbose 查看详细场景列表)")

    # ----- 4. 标注信息统计 -----
    print("\n🏷️ 标注统计:")
    total_gt = 0
    class_counter = Counter()
    sample_with_gt = 0
    gt_per_sample = []

    for info in infos:
        gt_names = info.get('gt_names')
        if gt_names is not None and len(gt_names) > 0:
            sample_with_gt += 1
            total_gt += len(gt_names)
            gt_per_sample.append(len(gt_names))
            class_counter.update(gt_names)

    print(f"   有标注的样本数: {sample_with_gt} / {len(infos)} ({sample_with_gt/len(infos)*100:.1f}%)")
    print(f"   总 GT 框数: {total_gt}")
    if gt_per_sample:
        print(f"   每样本 GT 数: 平均 {total_gt/len(infos):.1f}, "
              f"范围 [{min(gt_per_sample)}-{max(gt_per_sample)}]")

    if class_counter:
        print("\n   类别分布:")
        # 按数量降序排列
        for cls, cnt in class_counter.most_common():
            pct = cnt / total_gt * 100
            print(f"     {cls:20s}: {cnt:6d} ({pct:5.1f}%)")
    else:
        print("   (无标注信息，可能是测试集)")

    # ----- 5. 其他信息（可选）-----
    # 检查是否有 prev/next 链接
    has_prev = any('prev' in info and info['prev'] not in ('', -1) for info in infos[:100])
    if has_prev:
        print("\n🔗 包含时序链接 (prev/next)")

    # 检查是否有相机/雷达数据
    has_cam = any('cams' in info and info['cams'] for info in infos[:100])
    has_radar = any('radars' in info and info['radars'] for info in infos[:100])
    if has_cam:
        print("📷 包含相机数据")
    if has_radar:
        print("📡 包含雷达数据")

    print("\n" + "="*70)
    print("✅ 分析完成")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="快速分析 nuScenes 格式的 info pkl 文件 (v1.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例: %(prog)s ./tyjt_infos_train.pkl --verbose"
    )
    parser.add_argument("pkl_path", type=str, help="要分析的 pkl 文件路径")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示详细场景列表")
    args = parser.parse_args()

    analyze_nusc_pkl(args.pkl_path, args.verbose)


if __name__ == "__main__":
    main()