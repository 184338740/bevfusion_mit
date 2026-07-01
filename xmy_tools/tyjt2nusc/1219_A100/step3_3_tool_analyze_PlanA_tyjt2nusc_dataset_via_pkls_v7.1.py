#!/usr/bin/env python3
"""
分析tyjt数据集pkl文件 - 最终修复版
"""

import mmcv
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict



def analyze_pkl_files():
    """分析所有pkl文件"""
    
    # 路径配置
    Mode = "A100"
    if Mode == "Local":
        nusc_root = "./output-1205-weiyuan_3/step1/nuscenes_tyjt"
        pkl_root = "./output-1205-weiyuan_3/step1/nuscenes_tyjt"
    elif Mode == "A100":
        # nusc_root = "./output-1205-weiyuan_3/step1/nuscenes_tyjt"
        # pkl_root = "./output-1205-weiyuan_3/step3_pkl"
        # nusc_root = "./output-1226-v9.2.3-all/step1/nuscenes_tyjt"
        # pkl_root = "./output-1226-v9.2.3-all/step1/nuscenes_tyjt"
        nusc_root = "/mnt/bevfusion_mit_xmy_RunningA100_1205_doing/xmy_tools/tyjt2nusc/1219_A100/output-1226-v1.9/step1/nuscenes_tyjt"
        pkl_root = "/mnt/bevfusion_mit_xmy_RunningA100_1205_doing/xmy_tools/tyjt2nusc/1219_A100/output-1226-v1.9/step1/nuscenes_tyjt"
        
    else:
        print("❌ Mode配置错误")
        return
    
    print("=" * 80)
    print("🔵[XMY 调试] 开始分析tyjt数据集pkl文件")
    print("=" * 80)
    
    # 要分析的文件列表
    pkl_files = {
        'infos_train': Path(pkl_root) / "tyjt_infos_train.pkl",
        'infos_val': Path(pkl_root) / "tyjt_infos_val.pkl", 
        'infos_test': Path(pkl_root) / "tyjt_infos_test.pkl",
        'dbinfos_train': Path(nusc_root) / "tyjt_dbinfos_train.pkl"
    }
    
    # 标准NuScenes类别
    standard_categories = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
        'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]
    
    for file_name, file_path in pkl_files.items():
        print("\n" + "-" * 50)
        print(f"🔵[XMY 调试] 分析文件: {file_name}")
        print("-" * 50)
        
        if not file_path.exists():
            print(f"  ❌ 文件不存在: {file_path}")
            continue
            
        try:
            data = mmcv.load(file_path)
            
            # 修复：确保正确的函数被调用
            if file_name == 'dbinfos_train':
                analyze_dbinfos_file(data, file_name, standard_categories)
            elif 'infos' in file_name:
                analyze_infos_file(data, file_name, standard_categories)
            else:
                print(f"  ⚠️  未知文件类型，使用通用分析")
                quick_analyze_file(data, file_name)
                
        except Exception as e:
            print(f"  ❌ 加载文件失败: {e}")
            import traceback
            traceback.print_exc()


def quick_analyze_file(data, file_name):
    """快速分析任意文件"""
    print(f"  数据类型: {type(data)}")
    
    if isinstance(data, dict):
        print(f"  📁 字典结构，键: {list(data.keys())}")
        # 如果是dbinfos，显示类别统计
        if any(key in file_name for key in ['dbinfo', 'db_info']):
            total = 0
            for key, value in data.items():
                if isinstance(value, list):
                    print(f"    {key}: {len(value)}个样本")
                    total += len(value)
                else:
                    print(f"    {key}: 非列表格式 ({type(value)})")
            print(f"  总样本数: {total}")
    elif isinstance(data, list):
        print(f"  📁 列表结构，长度: {len(data)}")
        if len(data) > 0:
            print(f"  第一个元素类型: {type(data[0])}")
    else:
        print(f"  ⚠️  其他数据类型: {type(data)}")


def analyze_infos_file(data, file_name, standard_categories):
    """修复版分析infos文件"""
    print(f"  数据类型: {type(data)}")
    
    # 修复：直接处理数据结构，不进入错误分支
    if isinstance(data, dict) and 'infos' in data:
        infos = data['infos']
        print(f"  📁 数据结构: dict with 'infos' field")
    elif isinstance(data, list):
        infos = data
        print(f"  📁 数据结构: list")
    else:
        # 修复：不报告错误，直接返回
        print(f"  ⚠️  非常见数据结构: {type(data)}")
        return
        
    print(f"  样本数量: {len(infos)}")
    
    # 检查第一个样本的结构
    if infos and isinstance(infos[0], dict):
        first_info = infos[0]
        print(f"  样本字段: {list(first_info.keys())}")
    
    # 统计场景分布
    scenes = Counter()
    for info in infos:
        if isinstance(info, dict):
            if 'scene_name' in info:
                scenes[info['scene_name']] += 1
            elif 'scene_token' in info:
                scenes[info['scene_token']] += 1
            elif 'scene_id' in info:
                scenes[info['scene_id']] += 1
    
    if scenes:
        print(f"  场景数量: {len(scenes)}")
        print(f"  场景样本分布: {dict(scenes.most_common(5))}")
    else:
        print(f"  场景信息: 未找到场景字段")
    
    # 统计标注类别分布
    gt_names_all = []
    gt_counts = Counter()
    sample_with_ann = 0
    
    for info in infos:
        if isinstance(info, dict) and 'gt_names' in info:
            gt_names = info['gt_names']
            
            # 安全检查：判断是否有标注
            has_annotations = False
            if isinstance(gt_names, (list, np.ndarray)):
                if (isinstance(gt_names, np.ndarray) and gt_names.size > 0) or \
                   (isinstance(gt_names, list) and len(gt_names) > 0):
                    has_annotations = True
            
            if has_annotations:
                sample_with_ann += 1
                # 安全地处理gt_names
                if isinstance(gt_names, np.ndarray):
                    gt_names_list = gt_names.tolist()
                else:
                    gt_names_list = gt_names
                
                gt_names_all.extend(gt_names_list)
                gt_counts.update(gt_names_list)
    
    print(f"  有标注的样本: {sample_with_ann}/{len(infos)} ({sample_with_ann/len(infos)*100:.1f}%)")
    print(f"  总标注框数量: {len(gt_names_all)}")
    
    if gt_counts:
        print(f"  标注类别分布:")
        # 按标准类别统计
        for category in standard_categories:
            count = 0
            for cat in gt_counts:
                cat_str = str(cat).lower()
                if category in cat_str or cat_str in category:
                    count += gt_counts[cat]
            status = "✅" if count > 0 else "❌"
            print(f"    {status} {category}: {count}")
        
        print(f"  实际类别分布:")
        for cat, count in gt_counts.most_common():
            print(f"    '{cat}': {count}")
    else:
        print(f"  未找到标注信息")
    
    # 统计时间戳范围
    if infos and isinstance(infos[0], dict) and 'timestamp' in infos[0]:
        try:
            timestamps = [info['timestamp'] for info in infos if isinstance(info['timestamp'], (int, float))]
            if timestamps:
                print(f"  时间戳范围: {min(timestamps)} -> {max(timestamps)}")
        except:
            print(f"  时间戳信息: 格式异常")


def analyze_dbinfos_file(data, file_name, standard_categories):
    """修复版分析dbinfos文件"""
    print(f"  数据库信息:")
    
    if not isinstance(data, dict):
        print(f"  ❌ 数据库文件格式错误: 应该是dict, 实际是{type(data)}")
        return
    
    print(f"  ✅ 正确的数据结构: dict")
    print(f"  数据库包含的类别: {list(data.keys())}")
    
    total_samples = 0
    for category, samples in data.items():
        if isinstance(samples, list):
            print(f"    {category}: {len(samples)}个样本")
            total_samples += len(samples)
        else:
            print(f"    {category}: 非list格式 - {type(samples)}")
    
    print(f"  总数据库样本: {total_samples}")
    
    # 检查标准类别覆盖
    print(f"  标准类别覆盖检查:")
    for category in standard_categories:
        if category in data:
            count = len(data[category]) if isinstance(data[category], list) else 0
            status = "✅" if count > 0 else "⚠️"
            print(f"    {status} {category}: {count}个样本")
        else:
            print(f"    ❌ {category}: 缺失")


# def check_data_issues_final():
#     """最终修复版检查数据问题"""
#     print(f"\n🔧 数据问题检查")
#     print("=" * 50)
    
#     # 路径配置
#     Mode = "Local"
#     if Mode == "Local":
#         pkl_root = "./output-1205-weiyuan_3/step1/nuscenes_tyjt"
    
#     train_file = Path(pkl_root) / "tyjt_infos_train.pkl"
    
#     if not train_file.exists():
#         print("❌ 训练文件不存在")
#         return
    
#     try:
#         data = mmcv.load(train_file)
        
#         # 确定数据结构
#         if isinstance(data, dict) and 'infos' in data:
#             infos = data['infos']
#         else:
#             infos = data
        
#         print(f"训练集样本数量: {len(infos)}")
        
#         # 统计每个类别的样本数量
#         class_sample_count = defaultdict(int)
        
#         for info in infos:
#             if isinstance(info, dict) and 'gt_names' in info:
#                 gt_names = info['gt_names']
                
#                 # 安全处理gt_names
#                 if isinstance(gt_names, np.ndarray) and gt_names.size > 0:
#                     gt_names_list = gt_names.tolist()
#                 elif isinstance(gt_names, list) and len(gt_names) > 0:
#                     gt_names_list = gt_names
#                 else:
#                     continue
                
#                 for gt_name in gt_names_list:
#                     class_sample_count[gt_name] += 1
        
#         if class_sample_count:
#             print("训练集类别样本统计:")
#             for category, count in sorted(class_sample_count.items(), key=lambda x: x[1], reverse=True):
#                 status = "❌" if count == 0 else "✅"
#                 print(f"  {status} '{category}': {count}")
                    
#         else:
#             print("❌ 未找到标注信息")
            
#     except Exception as e:
#         print(f"❌ 检查失败: {e}")
#         import traceback
#         traceback.print_exc()

if __name__ == "__main__":
    analyze_pkl_files()
    # check_data_issues_final()