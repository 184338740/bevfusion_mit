#!/usr/bin/env python3
"""
检查Step1 JSON文件中的真实类别
"""

import json
from pathlib import Path
from collections import Counter
import numpy as np

def check_step1_json_categories():
    """检查Step1 JSON文件中的类别"""
    
    nusc_root = Path("./output-1205-weiyuan_3/step1/nuscenes_tyjt")
    version_dir = nusc_root / "v1.0-tyjt"
    
    print("🔍 检查Step1 JSON文件类别")
    print("=" * 60)
    
    # 1. 检查category.json
    print("1. 检查category.json:")
    category_file = version_dir / "category.json"
    if category_file.exists():
        with open(category_file, 'r') as f:
            categories = json.load(f)
        print(f"  category.json中的类别:")
        for cat in categories:
            print(f"    {cat['name']}: {cat.get('description', '')}")
    
    # 2. 检查sample_annotation.json
    print("\n2. 检查sample_annotation.json:")
    ann_file = version_dir / "sample_annotation.json"
    if ann_file.exists():
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        
        category_counts = Counter()
        for ann in annotations[:10]:  # 前10个标注
            if 'category_name' in ann:
                category_counts[ann['category_name']] += 1
                print(f"  标注: category_name='{ann['category_name']}'")
        
        print(f"\n  前10个标注的类别分布: {dict(category_counts)}")
    
    # 3. 检查完整的类别分布
    print("\n3. 完整的类别分布:")
    if ann_file.exists():
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        
        all_categories = Counter()
        for ann in annotations:
            if 'category_name' in ann:
                all_categories[ann['category_name']] += 1
        
        print(f"  总标注数: {len(annotations)}")
        print(f"  类别分布:")
        for category, count in all_categories.most_common():
            print(f"    '{category}': {count}")
    
    # 4. 检查sample.json中的标注引用
    print("\n4. 检查sample.json:")
    sample_file = version_dir / "sample.json"
    if sample_file.exists():
        with open(sample_file, 'r') as f:
            samples = json.load(f)
        
        print(f"  样本数量: {len(samples)}")
        if samples:
            first_sample = samples[0]
            print(f"  第一个样本的anns: {first_sample.get('anns', [])[:3]}")  # 前3个标注token

# def check_pkl_vs_json():
#     """对比pkl和json文件的类别差异"""
#     print(f"\n🔍 对比pkl和json文件类别差异")
#     print("=" * 60)
    
#     nusc_root = Path("./output/step1/nuscenes_tyjt")
    
#     # 检查pkl文件中的类别
#     import mmcv
#     pkl_file = nusc_root / "tyjt_infos_train.pkl"
#     if pkl_file.exists():
#         pkl_data = mmcv.load(pkl_file)
#         if 'infos' in pkl_data:
#             pkl_categories = Counter()
#             for info in pkl_data['infos']:
#                 if 'gt_names' in info:
#                     gt_names = info['gt_names']
#                     if isinstance(gt_names, (list, np.ndarray)):
#                         if isinstance(gt_names, np.ndarray):
#                             gt_names = gt_names.tolist()
#                         pkl_categories.update(gt_names)
            
#             print("train: pkl文件中的类别分布:")
#             for category, count in pkl_categories.most_common():
#                 print(f"  '{category}': {count}")

#     pkl_file = nusc_root / "tyjt_infos_val.pkl"
#     if pkl_file.exists():
#         pkl_data = mmcv.load(pkl_file)
#         if 'infos' in pkl_data:
#             pkl_categories = Counter()
#             for info in pkl_data['infos']:
#                 if 'gt_names' in info:
#                     gt_names = info['gt_names']
#                     if isinstance(gt_names, (list, np.ndarray)):
#                         if isinstance(gt_names, np.ndarray):
#                             gt_names = gt_names.tolist()
#                         pkl_categories.update(gt_names)
            
#             print("val: pkl文件中的类别分布:")
#             for category, count in pkl_categories.most_common():
#                 print(f"  '{category}': {count}")

#     pkl_file = nusc_root / "tyjt_infos_test.pkl"
#     if pkl_file.exists():
#         pkl_data = mmcv.load(pkl_file)
#         if 'infos' in pkl_data:
#             pkl_categories = Counter()
#             for info in pkl_data['infos']:
#                 if 'gt_names' in info:
#                     gt_names = info['gt_names']
#                     if isinstance(gt_names, (list, np.ndarray)):
#                         if isinstance(gt_names, np.ndarray):
#                             gt_names = gt_names.tolist()
#                         pkl_categories.update(gt_names)
            
#             print("test: pkl文件中的类别分布:")
#             for category, count in pkl_categories.most_common():
#                 print(f"  '{category}': {count}")




def check_pkl_file_detailed():
    """详细检查pkl文件内容"""
    print(f"\n🔍 详细检查pkl文件")
    print("=" * 60)
    
    import mmcv
    import numpy as np
    
    nusc_root = Path("./output-1205-weiyuan_3/step1/nuscenes_tyjt")
    
    pkl_files = [
        ("train", nusc_root / "tyjt_infos_train.pkl"),
        ("val", nusc_root / "tyjt_infos_val.pkl"),
        ("test", nusc_root / "tyjt_infos_test.pkl"),
    ]
    
    for split_name, pkl_path in pkl_files:
        if not pkl_path.exists():
            print(f"❌ {split_name}: pkl文件不存在: {pkl_path}")
            continue
            
        print(f"\n📊 {split_name} pkl文件检查:")
        pkl_data = mmcv.load(pkl_path)
        
        # 检查文件结构
        print(f"  pkl文件keys: {list(pkl_data.keys())}")
        
        if 'infos' in pkl_data:
            infos = pkl_data['infos']
            print(f"  样本数量: {len(infos)}")
            
            if len(infos) > 0:
                # 检查第一个样本的结构
                first_info = infos[0]
                print(f"  第一个样本的keys: {list(first_info.keys())[:10]}...")  # 前10个key
                
                # 检查是否有gt_names
                if 'gt_names' in first_info:
                    gt_names = first_info['gt_names']
                    print(f"  第一个样本的gt_names类型: {type(gt_names)}")
                    if isinstance(gt_names, (list, np.ndarray)):
                        print(f"  第一个样本的gt_names: {gt_names[:5]}...")  # 前5个
                else:
                    print("  ⚠️ 第一个样本没有gt_names")
                
                # 统计所有样本的类别
                pkl_categories = Counter()
                valid_samples = 0
                for i, info in enumerate(infos):
                    if 'gt_names' in info:
                        gt_names = info['gt_names']
                        if isinstance(gt_names, np.ndarray):
                            gt_names = gt_names.tolist()
                        if isinstance(gt_names, list) and len(gt_names) > 0:
                            pkl_categories.update(gt_names)
                            valid_samples += 1
                
                print(f"  有标注的样本数: {valid_samples}/{len(infos)}")
                if pkl_categories:
                    print(f"  类别分布:")
                    for category, count in pkl_categories.most_common(15):  # 最多显示15个
                        print(f"    '{category}': {count}")
                else:
                    print("  ⚠️ 没有找到任何类别信息")
        
        if 'metadata' in pkl_data:
            print(f"  元数据: {pkl_data['metadata']}")

def check_database_info():
    """检查GT数据库信息"""
    print(f"\n🔍 检查GT数据库信息")
    print("=" * 60)
    
    import mmcv
    
    db_info_path = Path("./output-1205-weiyuan_3/step1/nuscenes_tyjt/tyjt_dbinfos_train.pkl")
    if db_info_path.exists():
        db_info = mmcv.load(db_info_path)
        print(f"数据库信息keys: {list(db_info.keys())}")
        
        for class_name, class_infos in db_info.items():
            print(f"  类别 '{class_name}': {len(class_infos)} 个样本")

# 在主函数中调用
if __name__ == "__main__":
    check_step1_json_categories()
    check_pkl_file_detailed()
    check_database_info()