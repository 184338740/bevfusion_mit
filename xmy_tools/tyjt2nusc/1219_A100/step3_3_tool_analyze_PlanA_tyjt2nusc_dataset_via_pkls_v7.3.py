#!/usr/bin/env python3
"""
增强版PKL分析工具：分析训练集、验证集的距离分布和类别分布
"""

import mmcv
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_distance_distribution(infos, name="数据集"):
    """
    分析各类别的距离分布
    """
    print(f"\n📊 {name} - 距离分布分析")
    print("-" * 60)
    
    # 定义距离区间（单位：米）
    distance_bins = [0, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200]

    bin_labels = [f"{distance_bins[i]}-{distance_bins[i+1]}" 
                for i in range(len(distance_bins)-1)]
    bin_labels.append(f">{distance_bins[-1]}")
    
    # 统计每个类别的距离分布
    class_distance_stats = defaultdict(lambda: defaultdict(int))
    class_total = defaultdict(int)
    
    # 统计距离相关的其他信息
    class_distance_min = defaultdict(lambda: float('inf'))
    class_distance_max = defaultdict(lambda: float('-inf'))
    class_distance_avg = defaultdict(list)
    
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
            # 计算2D距离（x,y平面）
            distance = np.sqrt(box[0]**2 + box[1]**2)  # xy平面距离
            
            # 更新距离统计
            class_total[name] += 1
            class_distance_avg[name].append(distance)
            class_distance_min[name] = min(class_distance_min[name], distance)
            class_distance_max[name] = max(class_distance_max[name], distance)
            
            # 分配到距离区间
            bin_index = 0
            for i in range(len(distance_bins)-1):
                if distance_bins[i] <= distance < distance_bins[i+1]:
                    bin_index = i
                    break
            else:
                bin_index = len(bin_labels) - 1  # 超过最大范围
                
            class_distance_stats[name][bin_index] += 1
    
    # 打印距离分布
    print(f"样本数: {sample_count}")
    print(f"总标注框数: {sum(class_total.values())}")
    
    if not class_total:
        print("⚠️ 未找到标注数据")
        return
    
    # 按数量排序
    sorted_classes = sorted(class_total.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n📈 各类别距离分布:")
    print("类别              总数  平均距离 最小距离 最大距离  主要距离区间")
    print("-" * 70)
    
    for cls, total in sorted_classes:
        if total == 0:
            continue
            
        # 计算距离统计
        avg_dist = np.mean(class_distance_avg[cls]) if class_distance_avg[cls] else 0
        min_dist = class_distance_min[cls] if class_distance_min[cls] != float('inf') else 0
        max_dist = class_distance_max[cls] if class_distance_max[cls] != float('-inf') else 0
        
        # 找出主要距离区间
        dist_stats = class_distance_stats[cls]
        if dist_stats:
            # 找到最多的距离区间
            max_bin = max(dist_stats.items(), key=lambda x: x[1])
            if max_bin[1] > 0:
                if max_bin[0] < len(bin_labels):
                    main_bin = bin_labels[max_bin[0]]
                else:
                    main_bin = "超出范围"
                    
                # 计算该区间的占比
                main_percent = max_bin[1] / total * 100
                main_dist_str = f"{main_bin}m({main_percent:.0f}%)"
            else:
                main_dist_str = "无数据"
        else:
            main_dist_str = "无数据"
        
        print(f"{cls:15} {total:6d}  {avg_dist:7.1f}m  {min_dist:7.1f}m  {max_dist:7.1f}m  {main_dist_str:15}")
    
    # 详细距离分布表格
    print(f"\n📋 详细距离分布表（单位：米）:")
    header = "类别              "
    for i in range(len(bin_labels)):
        # 裁剪标签到7个字符（留1个给空格）
        label = bin_labels[i]
        if len(label) > 7:
            label = label[:7]
        header += f" {label:>7}"  # 固定7字符右对齐，加1空格=8字符
    print(header)
    
    # 计算分隔线长度：15(类别列) + len(bin_labels)*8
    separator_len = 15 + len(bin_labels) * 8
    print("-" * separator_len)
    
    # 打印数据行
    for cls, total in sorted_classes:
        if total == 0:
            continue
            
        row = f"{cls:15} "
        for i in range(len(bin_labels)):
            count = class_distance_stats[cls].get(i, 0)
            if count > 0:
                percent = count / total * 100
                # 固定格式：总宽度8字符，包括空格
                row += f" {percent:6.1f}%"  # 6.1f=6字符，%是1字符，前面空格=1字符，总共8字符
            else:
                row += f" {'-':>7}"  # 7字符右对齐的'-'，加前面空格=8字符
        print(row)
    
    return class_total, class_distance_stats

def analyze_pkl_comprehensive(pkl_path, name="数据集"):
    """
    综合分析pkl文件
    """
    print(f"\n{'='*80}")
    print(f"🔵 分析 {name}")
    print(f"文件路径: {pkl_path}")
    print(f"{'='*80}")
    
    if not Path(pkl_path).exists():
        print(f"❌ 文件不存在: {pkl_path}")
        return None, None
    
    try:
        data = mmcv.load(pkl_path)
    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        return None, None
    
    # 处理不同的数据结构
    if isinstance(data, dict):
        if 'infos' in data:
            infos = data['infos']
            print("📁 数据结构: dict with 'infos' field")
        else:
            infos = list(data.values())[0] if data else []
            print("📁 数据结构: dict (其他格式)")
    elif isinstance(data, list):
        infos = data
        print("📁 数据结构: list")
    else:
        print(f"❌ 不支持的数据结构: {type(data)}")
        return None, None
    
    print(f"样本总数: {len(infos)}")
    
    # 基础统计
    all_gt_names = []
    total_boxes = 0
    sample_with_gt = 0
    boxes_per_sample = []
    
    for i, info in enumerate(infos):
        if 'gt_names' in info and len(info['gt_names']) > 0:
            all_gt_names.extend(info['gt_names'])
            num_boxes = len(info['gt_names'])
            total_boxes += num_boxes
            boxes_per_sample.append(num_boxes)
            sample_with_gt += 1
    
    # 类别统计
    class_counts = Counter(all_gt_names)
    
    print(f"📊 基础统计:")
    print(f"  有标注的样本数: {sample_with_gt}/{len(infos)} ({sample_with_gt/len(infos)*100:.1f}%)")
    print(f"  GT框总数: {total_boxes}")
    print(f"  平均每样本标注数: {total_boxes/len(infos):.1f}" if len(infos) > 0 else "平均每样本标注数: 0")
    if boxes_per_sample:
        print(f"  每样本标注数范围: {min(boxes_per_sample)} - {max(boxes_per_sample)}")
    
    print(f"\n📈 类别分布:")
    for cls, count in class_counts.most_common():
        percentage = count / total_boxes * 100 if total_boxes > 0 else 0
        print(f"  {cls:20s}: {count:6d} ({percentage:6.1f}%)")
    
    # 样本级别的类别分布
    samples_with_class = {}
    for cls in class_counts.keys():
        samples_with_class[cls] = 0
    
    for info in infos:
        if 'gt_names' in info and len(info['gt_names']) > 0:
            unique_classes = set(info['gt_names'])
            for cls in unique_classes:
                samples_with_class[cls] = samples_with_class.get(cls, 0) + 1
    
    print(f"\n📋 包含各类别的样本数:")
    for cls, count in sorted(samples_with_class.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(infos) * 100 if len(infos) > 0 else 0
        print(f"  {cls:20s}: {count:6d}/{len(infos)} ({percentage:6.1f}%)")
    
    # 距离分布分析
    if total_boxes > 0:
        class_total, class_distance_stats = analyze_distance_distribution(infos, name)
    else:
        class_total = class_counts
        class_distance_stats = {}
    
    return class_counts, class_distance_stats

def compare_train_val(train_counts, val_counts, train_path, val_path):
    """
    对比训练集和验证集
    """
    print(f"\n{'='*80}")
    print(f"🔍 训练集 vs 验证集对比")
    print(f"{'='*80}")
    
    # 加载详细信息用于距离对比
    train_data = mmcv.load(train_path)
    val_data = mmcv.load(val_path)
    
    train_infos = train_data['infos'] if isinstance(train_data, dict) and 'infos' in train_data else train_data
    val_infos = val_data['infos'] if isinstance(val_data, dict) and 'infos' in val_data else val_data
    
    # 基础对比
    print(f"📊 数据集大小对比:")
    print(f"  训练集样本数: {len(train_infos):6d}")
    print(f"  验证集样本数: {len(val_infos):6d}")
    print(f"  训练/验证比例: {len(train_infos)/len(val_infos):.2f}:1")
    
    # 类别数量对比
    all_classes = set(train_counts.keys()) | set(val_counts.keys())
    
    print(f"\n📈 类别数量对比:")
    print("类别              训练集     验证集     训练/验证比   状态")
    print("-" * 70)
    
    for cls in sorted(all_classes):
        train_count = train_counts.get(cls, 0)
        val_count = val_counts.get(cls, 0)
        
        # 计算比例
        if val_count > 0:
            ratio = train_count / val_count
        elif train_count > 0:
            ratio = float('inf')
        else:
            ratio = 0
        
        # 状态标记
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
    
    # 计算稀有类别
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
    
    # 样本分布对比
    print(f"\n📋 样本分布对比（各类别出现在多少样本中）:")
    
    def count_samples_with_class(infos, cls_name):
        count = 0
        for info in infos:
            if 'gt_names' in info and cls_name in info['gt_names']:
                count += 1
        return count
    
    print("类别              训练集样本 验证集样本 训练/验证比")
    print("-" * 50)
    
    for cls in sorted(all_classes)[:10]:  # 只显示前10个
        train_samples = count_samples_with_class(train_infos, cls)
        val_samples = count_samples_with_class(val_infos, cls)
        
        ratio = train_samples / val_samples if val_samples > 0 else float('inf')
        print(f"{cls:15} {train_samples:10d} {val_samples:10d} {ratio:12.1f}")

def analyze_dbinfos(pkl_path, name="数据库文件"):
    """
    分析dbinfos数据库文件
    """
    print(f"\n{'='*80}")
    print(f"📁 分析 {name}")
    print(f"文件路径: {pkl_path}")
    print(f"{'='*80}")
    
    if not Path(pkl_path).exists():
        print(f"❌ 文件不存在: {pkl_path}")
        return None
    
    try:
        data = mmcv.load(pkl_path)
    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        return None
    
    if not isinstance(data, dict):
        print(f"❌ 数据结构错误: 应该是dict, 实际是{type(data)}")
        return None
    
    # 数据库文件通常是按类别组织的
    print(f"📊 数据库统计:")
    print(f"  包含的类别数: {len(data)}")
    
    total_samples = 0
    class_stats = {}
    
    # 统计每个类别的样本数
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
    
    # 按样本数排序
    sorted_classes = sorted(class_stats.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n📈 各类别数据库样本分布:")
    for cls, count in sorted_classes:
        percentage = count / total_samples * 100 if total_samples > 0 else 0
        print(f"  {cls:20s}: {count:6d} ({percentage:6.1f}%)")
    
    # 分析数据库中的距离分布
    print(f"\n📏 数据库距离分布分析:")
    
    distance_bins = [0, 10, 20, 30, 40, 50, 60, 80, 100]
    bin_labels = [f"{distance_bins[i]}-{distance_bins[i+1]}" 
                  for i in range(len(distance_bins)-1)]
    bin_labels.append(f">{distance_bins[-1]}")
    
    class_distance_stats = defaultdict(lambda: defaultdict(int))
    
    for category, samples in data.items():
        if not isinstance(samples, list) or len(samples) == 0:
            continue
            
        # 检查样本结构
        sample = samples[0]
        if isinstance(sample, dict):
            # 检查常见的坐标字段
            if 'box3d_lidar' in sample:
                # 格式: [x, y, z, dx, dy, dz, heading]
                box = sample['box3d_lidar']
                if len(box) >= 2:
                    distance = np.sqrt(box[0]**2 + box[1]**2)
                else:
                    continue
            elif 'gt_box' in sample:
                # 其他可能的字段名
                box = sample['gt_box']
                if len(box) >= 2:
                    distance = np.sqrt(box[0]**2 + box[1]**2)
                else:
                    continue
            else:
                # 尝试找到坐标字段
                for key, value in sample.items():
                    if isinstance(value, (list, np.ndarray)) and len(value) >= 2:
                        try:
                            distance = np.sqrt(value[0]**2 + value[1]**2)
                            break
                        except:
                            continue
                else:
                    continue
            
            # 分配到距离区间
            bin_index = 0
            for i in range(len(distance_bins)-1):
                if distance_bins[i] <= distance < distance_bins[i+1]:
                    bin_index = i
                    break
            else:
                bin_index = len(bin_labels) - 1
                
            class_distance_stats[category][bin_index] += len(samples)
    
    # 打印距离分布
    if class_distance_stats:
        print("类别              总数  主要距离区间")
        print("-" * 50)
        
        for category, count in sorted_classes[:10]:  # 只显示前10个
            if category in class_distance_stats:
                dist_stats = class_distance_stats[category]
                if dist_stats:
                    max_bin = max(dist_stats.items(), key=lambda x: x[1])
                    if max_bin[0] < len(bin_labels):
                        main_bin = bin_labels[max_bin[0]]
                        main_percent = max_bin[1] / count * 100
                        print(f"{category:15} {count:6d}  {main_bin}m({main_percent:.0f}%)")
                    else:
                        print(f"{category:15} {count:6d}  超出范围")
                else:
                    print(f"{category:15} {count:6d}  无距离数据")
    
    # 检查与训练集的匹配情况
    print(f"\n🔍 数据库使用建议:")
    
    # 常见类别的数据库建议大小
    recommendation = {
        'car': 2000, 'truck': 1000, 'pedestrian': 2000,
        'motorcycle': 500, 'bicycle': 300, 'traffic_cone': 500,
        'barrier': 500, 'bus': 500, 'trailer': 300
    }
    
    for cls, rec_count in recommendation.items():
        actual_count = class_stats.get(cls, 0)
        if actual_count > 0:
            if actual_count >= rec_count:
                status = "✅ 充足"
            elif actual_count >= rec_count * 0.5:
                status = "⚠️ 一般"
            else:
                status = "🔴 不足"
            print(f"  {cls:15}: {actual_count:4d}个样本 ({status}, 建议{rec_count}+)")
        elif cls in ['car', 'truck', 'pedestrian']:  # 关键类别
            print(f"  {cls:15}: 缺失 🔴 (建议添加{rec_count}+个样本)")
    
    return class_stats


def main():
    """主函数"""
    # 配置路径
    # nusc
    base_path = "/data/nuscenes/"
    train_pkl = f"{base_path}/nuscenes_infos_train.pkl"
    val_pkl = f"{base_path}/nuscenes_infos_val.pkl"
    test_pkl = f"{base_path}/nuscenes_infos_test.pkl"
    dbinfos_pkl = f"{base_path}/nuscenes_dbinfos_train.pkl"
    # tyjt
    # base_path = "/mnt/bevfusion_mit_xmy_RunningA100_1205_doing/xmy_tools/tyjt2nusc/1219_A100/output-1226-v1.9/step1/nuscenes_tyjt"  # 本地 demo 数据集
    # base_path = "/data2/xmy/01_project/bevfusion_mit_xmy_1205/xmy_tools/tyjt2nusc/1219_A100/output-1204-sub/step1/nuscenes_tyjt/"   # A100 Sub 数据集
    # base_path = "/data2/xmy/01_project/bevfusion_mit_xmy_1205/xmy_tools/tyjt2nusc/1219_A100/output-1226-v9.2.3-all/step1/nuscenes_tyjt/V02-TYJT/pkl_tmp/"
    # base_path = "/data2/xmy/01_project/bevfusion_mit_xmy_1205/xmy_tools/tyjt2nusc/1219_A100/output-1226-v9.2.3-all/step1/nuscenes_tyjt"   # A100 All 数据集
    # train_pkl = f"{base_path}/tyjt_infos_train.pkl"
    # val_pkl = f"{base_path}/tyjt_infos_val.pkl"
    # test_pkl = f"{base_path}/tyjt_infos_test.pkl"
    # dbinfos_pkl = f"{base_path}/tyjt_dbinfos_train.pkl"
    
    print(f"{'='*80}")
    print(f"🔵 开始分析TYJT数据集PKL文件")
    print(f"基础路径: {base_path}")
    print(f"{'='*80}")
    
    # 分析训练集
    train_counts, train_dist_stats = analyze_pkl_comprehensive(train_pkl, "训练集")
    
    # 分析验证集
    val_counts, val_dist_stats = analyze_pkl_comprehensive(val_pkl, "验证集")
    
    # 分析测试集（如果有）
    if Path(test_pkl).exists():
        test_counts, test_dist_stats = analyze_pkl_comprehensive(test_pkl, "测试集")
    
    # 分析数据库文件
    if Path(dbinfos_pkl).exists():
        db_stats = analyze_dbinfos(dbinfos_pkl, "数据库文件 (dbinfos_train)")
        
        # 对比训练集和数据库
        if train_counts and db_stats:
            print(f"\n{'='*80}")
            print(f"🔍 训练集 vs 数据库对比")
            print(f"{'='*80}")
            
            print("类别              训练集GT数  数据库样本数  数据库/训练集比")
            print("-" * 60)
            
            all_classes = set(train_counts.keys()) | set(db_stats.keys())
            for cls in sorted(all_classes):
                train_count = train_counts.get(cls, 0)
                db_count = db_stats.get(cls, 0)
                
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
    
    # 对比训练集和验证集
    if train_counts and val_counts:
        compare_train_val(train_counts, val_counts, train_pkl, val_pkl)
    
    print(f"\n{'='*80}")
    print(f"🎯 分析完成")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()