#!/usr/bin/env python3
"""
TYJT 数据集 pkl 分析工具
基于 tyjt_converter_v9.4.4 生成的 pkl 文件，统计样本分布、类别分布、距离分布等。
支持训练集、验证集、数据库文件的分析与对比

# 版本v1.1
    - 三个pkl,允许缺省

# 使用说明:
    python tools_analyze_TYJTdataset_PlanB_tyjt2pkl_via_pkls_v1.0_0402.py \
        --train-pkl /path/to/tyjt_infos_train.pkl \
        --val-pkl /path/to/tyjt_infos_val.pkl \
        --dbinfos-pkl /path/to/tyjt_dbinfos_train.pkl \
        --output-json stats.json

## 示例-Local
python tools_analyze_TYJTdataset_PlanB_tyjt2pkl_via_pkls_v1.0_0402.py \
    --train-pkl /mnt/bevfusion_mit_xmy/datasets/tyjt2pkl/Local_V031/tyjt_infos_train.pkl \
    --val-pkl /mnt/bevfusion_mit_xmy/datasets/tyjt2pkl/Local_V031/tyjt_infos_val.pkl \
    --dbinfos-pkl /mnt/bevfusion_mit_xmy/datasets/tyjt2pkl/Local_V031/tyjt_dbinfos_train.pkl \
    --output-json /mnt/bevfusion_mit_xmy/datasets/tyjt2pkl/Local_V031/pkls_stats.json

"""

import argparse
import pickle
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """用于将 numpy 数组转换为列表的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        return super().default(obj)


def load_infos(pkl_path):
    """加载 pkl 文件，返回 infos 列表和 metadata"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        if 'infos' in data:
            return data['infos'], data.get('metadata', {})
        else:
            # 可能是以其他键存储，取第一个列表
            for v in data.values():
                if isinstance(v, list):
                    return v, {}
            return [], {}
    elif isinstance(data, list):
        return data, {}
    else:
        raise TypeError(f"不支持的数据类型: {type(data)}")


def basic_stats(infos):
    """基础统计：样本数、有标注样本数、GT框总数等"""
    total_samples = len(infos)
    sample_with_gt = 0
    total_boxes = 0
    boxes_per_sample = []
    for info in infos:
        if 'gt_names' in info and len(info['gt_names']) > 0:
            sample_with_gt += 1
            num = len(info['gt_names'])
            total_boxes += num
            boxes_per_sample.append(num)
    return {
        'total_samples': total_samples,
        'sample_with_gt': sample_with_gt,
        'total_boxes': total_boxes,
        'avg_boxes': total_boxes / total_samples if total_samples > 0 else 0,
        'min_boxes': min(boxes_per_sample) if boxes_per_sample else 0,
        'max_boxes': max(boxes_per_sample) if boxes_per_sample else 0,
    }


def category_stats(infos):
    """类别统计：每个类别的数量、出现在多少样本中"""
    class_counts = Counter()
    samples_with_class = defaultdict(int)
    for info in infos:
        if 'gt_names' in info and len(info['gt_names']) > 0:
            gt_names = info['gt_names']
            # 将 numpy 数组转换为 Python 列表
            if isinstance(gt_names, np.ndarray):
                gt_names = gt_names.tolist()
            class_counts.update(gt_names)
            unique_classes = set(gt_names)
            for cls in unique_classes:
                samples_with_class[cls] += 1
    return class_counts, samples_with_class


def distance_distribution(infos, distance_bins=None, name="数据集"):
    """
    分析各类别的距离分布
    distance_bins: 距离区间列表，例如 [0,10,20,30,50,100]
    """
    if distance_bins is None:
        distance_bins = [0, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200]

    bin_labels = [f"{distance_bins[i]}-{distance_bins[i+1]}" for i in range(len(distance_bins)-1)]
    bin_labels.append(f">{distance_bins[-1]}")

    class_total = defaultdict(int)
    class_distance_avg = defaultdict(list)
    class_distance_min = defaultdict(lambda: float('inf'))
    class_distance_max = defaultdict(lambda: float('-inf'))
    class_distance_stats = defaultdict(lambda: defaultdict(int))

    sample_count = 0
    for info in infos:
        if 'gt_names' not in info or 'gt_boxes' not in info:
            continue
        gt_names = info['gt_names']
        gt_boxes = info['gt_boxes']
        if len(gt_names) == 0 or len(gt_boxes) == 0:
            continue

        sample_count += 1
        for name, box in zip(gt_names, gt_boxes):
            # 计算平面距离
            distance = np.sqrt(box[0]**2 + box[1]**2)

            class_total[name] += 1
            class_distance_avg[name].append(distance)
            class_distance_min[name] = min(class_distance_min[name], distance)
            class_distance_max[name] = max(class_distance_max[name], distance)

            # 分配到距离区间
            bin_idx = 0
            for i in range(len(distance_bins)-1):
                if distance_bins[i] <= distance < distance_bins[i+1]:
                    bin_idx = i
                    break
            else:
                bin_idx = len(bin_labels) - 1
            class_distance_stats[name][bin_idx] += 1

    # 打印距离分布
    print(f"\n📊 {name} - 距离分布分析")
    print("-" * 60)
    print(f"样本数: {sample_count}")
    print(f"总标注框数: {sum(class_total.values())}")

    if not class_total:
        print("⚠️ 未找到标注数据")
        return

    sorted_classes = sorted(class_total.items(), key=lambda x: x[1], reverse=True)

    print(f"\n📈 各类别距离分布:")
    print("类别              总数  平均距离 最小距离 最大距离  主要距离区间")
    print("-" * 70)
    for cls, total in sorted_classes:
        avg_dist = np.mean(class_distance_avg[cls]) if class_distance_avg[cls] else 0
        min_dist = class_distance_min[cls] if class_distance_min[cls] != float('inf') else 0
        max_dist = class_distance_max[cls] if class_distance_max[cls] != float('-inf') else 0

        dist_stats = class_distance_stats[cls]
        if dist_stats:
            max_bin = max(dist_stats.items(), key=lambda x: x[1])
            if max_bin[0] < len(bin_labels):
                main_bin = bin_labels[max_bin[0]]
                main_percent = max_bin[1] / total * 100
                main_dist_str = f"{main_bin}m({main_percent:.0f}%)"
            else:
                main_dist_str = "超出范围"
        else:
            main_dist_str = "无数据"

        print(f"{cls:15} {total:6d}  {avg_dist:7.1f}m  {min_dist:7.1f}m  {max_dist:7.1f}m  {main_dist_str:15}")

    # 详细距离分布表格
    print(f"\n📋 详细距离分布表（单位：米）:")
    header = "类别              "
    for i in range(len(bin_labels)):
        label = bin_labels[i]
        if len(label) > 7:
            label = label[:7]
        header += f" {label:>7}"
    print(header)

    separator_len = 15 + len(bin_labels) * 8
    print("-" * separator_len)

    for cls, total in sorted_classes:
        row = f"{cls:15} "
        for i in range(len(bin_labels)):
            count = class_distance_stats[cls].get(i, 0)
            if count > 0:
                percent = count / total * 100
                row += f" {percent:6.1f}%"
            else:
                row += f" {'-':>7}"
        print(row)

    return class_total, class_distance_stats


def compare_train_val(train_infos, val_infos, train_counts, val_counts):
    """对比训练集和验证集"""
    print(f"\n{'='*80}")
    print(f"🔍 训练集 vs 验证集对比")
    print(f"{'='*80}")

    # 基础对比
    print(f"📊 数据集大小对比:")
    print(f"  训练集样本数: {len(train_infos):6d}")
    print(f"  验证集样本数: {len(val_infos):6d}")
    if len(val_infos) > 0:
        ratio = len(train_infos) / len(val_infos)
        print(f"  训练/验证比例: {ratio:.2f}:1")
    else:
        print("  验证集为空，无法计算比例")

    # 类别数量对比
    all_classes = set(train_counts.keys()) | set(val_counts.keys())

    print(f"\n📈 类别数量对比:")
    print("类别              训练集     验证集     训练/验证比   状态")
    print("-" * 70)

    for cls in sorted(all_classes):
        train_count = train_counts.get(cls, 0)
        val_count = val_counts.get(cls, 0)
        if val_count > 0:
            ratio = train_count / val_count
        elif train_count > 0:
            ratio = float('inf')
        else:
            ratio = 0

        if train_count == 0 and val_count == 0:
            status = "❌ 都无"
        elif train_count == 0:
            status = "🔴 仅验证有"
        elif val_count == 0:
            status = "🔴 仅训练有"
        elif ratio < 0.5:
            status = "⚠️ 训练过少"
        elif ratio > 2.0:
            status = "⚠️ 验证过少"
        else:
            status = "✅ 平衡"

        print(f"{cls:15} {train_count:8d} {val_count:8d} {ratio:12.1f}   {status}")

    # 稀有类别警告
    print(f"\n⚠️ 稀有类别警告（样本数<10）:")
    rare_in_train = [cls for cls, count in train_counts.items() if count < 10]
    rare_in_val = [cls for cls, count in val_counts.items() if count < 10]
    if rare_in_train:
        print(f"  训练集中稀有类别: {', '.join(rare_in_train)}")
    if rare_in_val:
        print(f"  验证集中稀有类别: {', '.join(rare_in_val)}")

    # 缺失类别
    missing_in_train = set(val_counts.keys()) - set(train_counts.keys())
    missing_in_val = set(train_counts.keys()) - set(val_counts.keys())
    if missing_in_train:
        print(f"  🔴 验证集有但训练集无的类别: {', '.join(missing_in_train)}")
    if missing_in_val:
        print(f"  🔴 训练集有但验证集无的类别: {', '.join(missing_in_val)}")

    # 样本分布对比（各类别出现在多少样本中）
    def count_samples_with_class(infos, cls_name):
        cnt = 0
        for info in infos:
            if 'gt_names' in info and len(info['gt_names']) > 0:
                gt_names = info['gt_names']
                if isinstance(gt_names, np.ndarray):
                    gt_names = gt_names.tolist()
                if cls_name in gt_names:
                    cnt += 1
        return cnt

    print(f"\n📋 样本分布对比（各类别出现在多少样本中）:")
    print("类别              训练集样本 验证集样本 训练/验证比")
    print("-" * 50)
    for cls in sorted(all_classes):
        train_samples = count_samples_with_class(train_infos, cls)
        val_samples = count_samples_with_class(val_infos, cls)
        if val_samples > 0:
            ratio = train_samples / val_samples
        elif train_samples > 0:
            ratio = float('inf')
        else:
            ratio = 0
        print(f"{cls:15} {train_samples:10d} {val_samples:10d} {ratio:12.1f}")


def analyze_dbinfos(dbinfos_path, train_infos=None, name="数据库文件"):
    """分析 GT 数据库文件"""
    print(f"\n{'='*80}")
    print(f"📁 分析 {name}")
    print(f"文件路径: {dbinfos_path}")
    print(f"{'='*80}")

    if not Path(dbinfos_path).exists():
        print(f"❌ 文件不存在: {dbinfos_path}")
        return None

    with open(dbinfos_path, 'rb') as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        print(f"❌ 数据结构错误: 应该是dict, 实际是{type(data)}")
        return None

    print(f"📊 数据库统计:")
    print(f"  包含的类别数: {len(data)}")

    total_samples = 0
    class_stats = {}
    for category, samples in data.items():
        if isinstance(samples, list):
            count = len(samples)
        elif isinstance(samples, np.ndarray):
            count = len(samples)
        else:
            count = 0
        class_stats[category] = count
        total_samples += count

    print(f"  总样本数: {total_samples}")

    sorted_classes = sorted(class_stats.items(), key=lambda x: x[1], reverse=True)
    print(f"\n📈 各类别数据库样本分布:")
    for cls, count in sorted_classes:
        percentage = count / total_samples * 100 if total_samples > 0 else 0
        print(f"  {cls:20s}: {count:6d} ({percentage:6.1f}%)")

    # 距离分布分析
    print(f"\n📏 数据库距离分布分析:")
    distance_bins = [0, 10, 20, 30, 40, 50, 60, 80, 100]
    bin_labels = [f"{distance_bins[i]}-{distance_bins[i+1]}" for i in range(len(distance_bins)-1)]
    bin_labels.append(f">{distance_bins[-1]}")

    class_distance_stats = defaultdict(lambda: defaultdict(int))
    for category, samples in data.items():
        if not isinstance(samples, list) or len(samples) == 0:
            continue
        # 尝试获取距离信息
        # 数据库中每个样本通常是字典，包含 'box3d_lidar' 或 'gt_box' 或 'bbox' 等字段
        # 根据 tyjt_converter 生成的数据库结构，可能包含 'box3d_lidar' 字段
        sample = samples[0]
        if isinstance(sample, dict):
            # 查找坐标字段
            box_key = None
            for key in ['box3d_lidar', 'gt_box', 'bbox']:
                if key in sample:
                    box_key = key
                    break
            if box_key is None:
                # 尝试任意第一个列表或数组
                for val in sample.values():
                    if isinstance(val, (list, np.ndarray)) and len(val) >= 2:
                        box_key = val
                        break
            if box_key is None:
                continue

            # 计算距离
            for sample in samples:
                box = sample[box_key] if isinstance(box_key, str) else box_key
                if len(box) >= 2:
                    distance = np.sqrt(box[0]**2 + box[1]**2)
                    # 分配区间
                    bin_idx = 0
                    for i in range(len(distance_bins)-1):
                        if distance_bins[i] <= distance < distance_bins[i+1]:
                            bin_idx = i
                            break
                    else:
                        bin_idx = len(bin_labels) - 1
                    class_distance_stats[category][bin_idx] += 1

    if class_distance_stats:
        print("类别              总数  主要距离区间")
        print("-" * 50)
        for category, count in sorted_classes[:10]:
            if category in class_distance_stats:
                dist_stats = class_distance_stats[category]
                max_bin = max(dist_stats.items(), key=lambda x: x[1])
                if max_bin[0] < len(bin_labels):
                    main_bin = bin_labels[max_bin[0]]
                    main_percent = max_bin[1] / count * 100
                    print(f"{category:15} {count:6d}  {main_bin}m({main_percent:.0f}%)")
                else:
                    print(f"{category:15} {count:6d}  超出范围")
            else:
                print(f"{category:15} {count:6d}  无距离数据")

    # 与训练集对比（如果提供了训练集）
    if train_infos is not None:
        train_counts, _ = category_stats(train_infos)
        print(f"\n🔍 训练集 vs 数据库对比")
        print("类别              训练集GT数  数据库样本数  数据库/训练集比")
        print("-" * 60)
        all_classes = set(train_counts.keys()) | set(class_stats.keys())
        for cls in sorted(all_classes):
            train_count = train_counts.get(cls, 0)
            db_count = class_stats.get(cls, 0)
            if train_count > 0 and db_count > 0:
                ratio = db_count / train_count
                if ratio >= 0.5:
                    status = "✅ 充足"
                elif ratio >= 0.2:
                    status = "⚠️ 一般"
                else:
                    status = "🔴 不足"
                print(f"{cls:15} {train_count:10d} {db_count:12d} {ratio:12.2f}   {status}")
            elif train_count > 0 and db_count == 0:
                print(f"{cls:15} {train_count:10d} {'缺失':>12} {'N/A':>12}   🔴 缺失")
            elif db_count > 0 and train_count == 0:
                print(f"{cls:15} {'0':>10} {db_count:12d} {'inf':>12}   ⚠️ 仅数据库有")

    return class_stats


def main():
    parser = argparse.ArgumentParser(description="TYJT 数据集 pkl 分析工具")
    parser.add_argument("--train-pkl", type=str, required=True,
                        help="训练集 info pkl 文件路径（必需）")
    parser.add_argument("--val-pkl", type=str, default=None,
                        help="验证集 info pkl 文件路径（可选）")
    parser.add_argument("--dbinfos-pkl", type=str, default=None,
                        help="数据库文件 tyjt_dbinfos_train.pkl 路径（可选）")
    parser.add_argument("--output-json", type=str, default=None,
                        help="将统计结果保存为 JSON 文件（可选）")
    args = parser.parse_args()

    # --------------------- 检查文件有效性 ---------------------
    train_path = Path(args.train_pkl)
    if not train_path.exists():
        print(f"❌ 错误：训练集文件不存在: {train_path}")
        print("程序终止。")
        return

    # 验证集检查
    val_path = Path(args.val_pkl) if args.val_pkl else None
    val_exists = False
    if val_path is None:
        print("🔵 未提供验证集文件路径，将跳过验证集分析。")
    elif not val_path.exists():
        print(f"🔴 警告：验证集文件不存在: {val_path}，将跳过验证集分析。")
    else:
        val_exists = True

    # 数据库检查
    dbinfos_path = Path(args.dbinfos_pkl) if args.dbinfos_pkl else None
    dbinfos_exists = False
    if dbinfos_path is None:
        print("🔵 未提供数据库文件路径，将跳过数据库分析。")
    elif not dbinfos_path.exists():
        print(f"🔴 警告：数据库文件不存在: {dbinfos_path}，将跳过数据库分析。")
    else:
        dbinfos_exists = True

    print("\n" + "=" * 80)
    print("🔵 开始分析训练集")
    # 加载训练集
    train_infos, train_meta = load_infos(train_path)
    train_basic = basic_stats(train_infos)
    train_counts, train_samples_with_class = category_stats(train_infos)

    print(f"样本总数: {train_basic['total_samples']}")
    print(f"有标注的样本数: {train_basic['sample_with_gt']}/{train_basic['total_samples']} "
          f"({train_basic['sample_with_gt']/train_basic['total_samples']*100:.1f}%)")
    print(f"GT框总数: {train_basic['total_boxes']}")
    print(f"平均每样本标注数: {train_basic['avg_boxes']:.1f}")
    print(f"每样本标注数范围: {train_basic['min_boxes']} - {train_basic['max_boxes']}")

    print(f"\n📈 类别分布:")
    for cls, count in train_counts.most_common():
        percentage = count / train_basic['total_boxes'] * 100 if train_basic['total_boxes'] > 0 else 0
        print(f"  {cls:20s}: {count:6d} ({percentage:6.1f}%)")

    print(f"\n📋 包含各类别的样本数:")
    for cls, count in sorted(train_samples_with_class.items(), key=lambda x: x[1], reverse=True):
        percentage = count / train_basic['total_samples'] * 100
        print(f"  {cls:20s}: {count:6d}/{train_basic['total_samples']} ({percentage:6.1f}%)")

    # 训练集距离分布
    distance_distribution(train_infos, name="训练集")

    # --------------------- 验证集分析 ---------------------
    if val_exists:
        print("\n" + "=" * 80)
        print("🔵 分析验证集")
        val_infos, val_meta = load_infos(val_path)
        val_basic = basic_stats(val_infos)
        val_counts, val_samples_with_class = category_stats(val_infos)

        print(f"样本总数: {val_basic['total_samples']}")
        print(f"有标注的样本数: {val_basic['sample_with_gt']}/{val_basic['total_samples']} "
              f"({val_basic['sample_with_gt']/val_basic['total_samples']*100:.1f}%)")
        print(f"GT框总数: {val_basic['total_boxes']}")
        print(f"平均每样本标注数: {val_basic['avg_boxes']:.1f}")
        print(f"每样本标注数范围: {val_basic['min_boxes']} - {val_basic['max_boxes']}")

        print(f"\n📈 类别分布:")
        for cls, count in val_counts.most_common():
            percentage = count / val_basic['total_boxes'] * 100 if val_basic['total_boxes'] > 0 else 0
            print(f"  {cls:20s}: {count:6d} ({percentage:6.1f}%)")

        print(f"\n📋 包含各类别的样本数:")
        for cls, count in sorted(val_samples_with_class.items(), key=lambda x: x[1], reverse=True):
            percentage = count / val_basic['total_samples'] * 100
            print(f"  {cls:20s}: {count:6d}/{val_basic['total_samples']} ({percentage:6.1f}%)")

        distance_distribution(val_infos, name="验证集")

        # 训练集与验证集对比
        compare_train_val(train_infos, val_infos, train_counts, val_counts)
    else:
        val_infos = None  # 确保变量存在

    # --------------------- 数据库分析 ---------------------
    if dbinfos_exists:
        analyze_dbinfos(dbinfos_path, train_infos=train_infos)

    # --------------------- 保存 JSON 结果 ---------------------
    if args.output_json:
        result = {
            "train": {
                "basic": train_basic,
                "category_counts": dict(train_counts),
                "samples_with_class": dict(train_samples_with_class),
            }
        }
        if val_exists:
            result["val"] = {
                "basic": val_basic,
                "category_counts": dict(val_counts),
                "samples_with_class": dict(val_samples_with_class),
            }
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
        print(f"\n统计结果已保存至 {args.output_json}")

    print("\n" + "=" * 80)
    print("🎯 分析完成")



if __name__ == "__main__":
    main()