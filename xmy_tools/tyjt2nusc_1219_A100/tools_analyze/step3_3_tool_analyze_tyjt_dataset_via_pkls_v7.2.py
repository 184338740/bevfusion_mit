#!/usr/bin/env python3
import mmcv
import numpy as np
from collections import Counter

def analyze_pkl(pkl_path, name="数据集"):
    print(f"\n=== 分析 {name} ===")
    
    data = mmcv.load(pkl_path)
    infos = data['infos']
    
    print(f"样本总数: {len(infos)}")
    
    # 统计所有GT类别
    all_gt_names = []
    total_boxes = 0
    
    for i, info in enumerate(infos):
        if 'gt_names' in info:
            all_gt_names.extend(info['gt_names'])
            total_boxes += len(info['gt_names'])
    
    # 类别统计
    class_counts = Counter(all_gt_names)
    
    print(f"GT框总数: {total_boxes}")
    print("类别分布:")
    for cls, count in class_counts.most_common():
        percentage = count / total_boxes * 100
        print(f"  {cls:20s}: {count:6d} ({percentage:.1f}%)")
    
    # 检查样本级别的分布
    samples_with_class = {}
    for cls in class_counts.keys():
        samples_with_class[cls] = 0
    
    for info in infos:
        if 'gt_names' in info:
            unique_classes = set(info['gt_names'])
            for cls in unique_classes:
                samples_with_class[cls] = samples_with_class.get(cls, 0) + 1
    
    print(f"\n包含各类别的样本数:")
    for cls, count in sorted(samples_with_class.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(infos) * 100
        print(f"  {cls:20s}: {count:6d}/{len(infos)} ({percentage:.1f}%)")
    
    return class_counts

if __name__ == "__main__":
    base_path = "/mnt/bevfusion_mit_xmy_RunningA100_1205_doing/xmy_tools/tyjt2nusc/1219_A100/output-1226-v1.9/step1/nuscenes_tyjt"
    
    print("🔵 开始分析pkl文件")
    print("="*80)
    
    # 分析训练集
    train_counts = analyze_pkl(f"{base_path}/tyjt_infos_train.pkl", "训练集")
    
    print("\n" + "="*80)
    
    # 分析验证集  
    val_counts = analyze_pkl(f"{base_path}/tyjt_infos_val.pkl", "验证集")
    
    print("\n" + "="*80)
    print("=== 训练集 vs 验证集对比 ===")
    
    all_classes = set(train_counts.keys()) | set(val_counts.keys())
    for cls in sorted(all_classes):
        train_count = train_counts.get(cls, 0)
        val_count = val_counts.get(cls, 0)
        print(f"{cls:20s}: 训练{train_count:6d} | 验证{val_count:6d}")

