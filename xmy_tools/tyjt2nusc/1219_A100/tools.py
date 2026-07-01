import json
import pickle
import numpy as np
from collections import Counter
import os

def analyze_tyjt_data_discrepancy(dataset_root):
    """
    统一分析TYJT数据集在NuScenes数据库和PKL文件中的样本与标注分布。
    
    参数:
        dataset_root: TYJT数据集根目录路径 (包含v1.0-tyjt-trainval/和pkl文件)
    """
    
    # 1. 分析PKL文件 (训练/验证源)
    print("="*60)
    print("📁 分析 PKL 文件 (tyjt_infos_val.pkl)")
    print("="*60)
    
    pkl_path = os.path.join(dataset_root, 'tyjt_infos_val.pkl')
    try:
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        
        # 兼容不同结构的pkl
        infos = pkl_data['infos'] if isinstance(pkl_data, dict) and 'infos' in pkl_data else pkl_data
        
        pkl_sample_count = len(infos)
        pkl_ann_counter = Counter()
        pkl_sample_tokens = set()
        
        for info in infos:
            token = info.get('token')
            if token:
                pkl_sample_tokens.add(token)
            
            if 'gt_names' in info:
                for name in info['gt_names']:
                    pkl_ann_counter[str(name)] += 1
        
        print(f"样本数量: {pkl_sample_count}")
        print(f"唯一Token数量: {len(pkl_sample_tokens)}")
        print(f"标注总数: {sum(pkl_ann_counter.values())}")
        print("\n标注类别分布 (PKL):")
        for cls, count in sorted(pkl_ann_counter.items()):
            print(f"  {cls:25s}: {count:4d}")
            
    except Exception as e:
        print(f"❌ 读取PKL文件失败: {e}")
        return
    
    # 2. 分析NuScenes数据库 (评测源)
    print("\n" + "="*60)
    print("🗃️  分析 NuScenes 数据库 (sample_annotation.json)")
    print("="*60)
    
    db_anno_path = os.path.join(dataset_root, 'v1.0-tyjt-trainval', 'sample_annotation.json')
    try:
        with open(db_anno_path, 'r') as f:
            db_annotations = json.load(f)
        
        # 统计数据库标注
        db_ann_counter = Counter()
        db_sample_tokens = set()
        
        for anno in db_annotations:
            # 注意：你的数据库使用'category_name'而非标准的'category_token'
            cls_name = anno.get('category_name', 'unknown')
            db_ann_counter[cls_name] += 1
            db_sample_tokens.add(anno['sample_token'])
        
        print(f"样本数量: {len(db_sample_tokens)}")
        print(f"标注总数: {len(db_annotations)}")
        print("\n标注类别分布 (数据库):")
        for cls, count in sorted(db_ann_counter.items()):
            print(f"  {cls:25s}: {count:4d}")
            
    except Exception as e:
        print(f"❌ 读取数据库文件失败: {e}")
        return
    
    # 3. 关键对比分析
    print("\n" + "="*60)
    print("🔍 关键差异对比分析")
    print("="*60)
    
    # 3.1 样本Token一致性
    common_tokens = pkl_sample_tokens.intersection(db_sample_tokens)
    only_in_pkl = pkl_sample_tokens - db_sample_tokens
    only_in_db = db_sample_tokens - pkl_sample_tokens
    
    print(f"样本Token一致性检查:")
    print(f"  ✅ 共同样本: {len(common_tokens)} 个")
    print(f"  ⚠️  仅出现在PKL: {len(only_in_pkl)} 个")
    print(f"  ⚠️  仅出现在数据库: {len(only_in_db)} 个")
    
    # 3.2 类别标注差异
    print(f"\n类别标注数量对比:")
    all_classes = set(pkl_ann_counter.keys()) | set(db_ann_counter.keys())
    
    mismatch_found = False
    for cls in sorted(all_classes):
        pkl_count = pkl_ann_counter.get(cls, 0)
        db_count = db_ann_counter.get(cls, 0)
        
        if pkl_count != db_count:
            mismatch_found = True
            diff = pkl_count - db_count
            status = "❌ 不匹配" if abs(diff) > 0 else ""
            print(f"  {cls:25s}: PKL={pkl_count:4d}, DB={db_count:4d}, 差异={diff:+d} {status}")
    
    if not mismatch_found:
        print("  ✅ 所有类别标注数量一致")
    
    # 3.3 特别关注缺失类别
    print(f"\n特别关注 - 缺失类别检查:")
    critical_classes = ['bicycle', 'barrier', 'traffic_cone', 'construction_vehicle', 'trailer']
    for cls in critical_classes:
        # 在PKL和数据库中查找包含该关键词的类别名
        pkl_cls_keys = [k for k in pkl_ann_counter.keys() if cls in k.lower()]
        db_cls_keys = [k for k in db_ann_counter.keys() if cls in k.lower()]
        
        pkl_total = sum(pkl_ann_counter[k] for k in pkl_cls_keys)
        db_total = sum(db_ann_counter[k] for k in db_cls_keys)
        
        if pkl_total == 0 and db_total == 0:
            status = "⚠️  两者皆无"
        elif pkl_total > 0 and db_total == 0:
            status = f"❌ 数据库缺失 (PKL中有{pkl_total}个)"
        elif pkl_total == 0 and db_total > 0:
            status = f"⚠️  PKL缺失 (DB中有{db_total}个)"
        else:
            status = f"✅ 两者都有 (PKL:{pkl_total}, DB:{db_total})"
        
        print(f"  {cls:20s}: {status}")

# 使用示例 - 将路径替换为你的实际路径
dataset_root = "/mnt/bevfusion_mit_xmy_RunningA100_1205_doing/xmy_tools/tyjt2nusc/1219_A100/output-1226-v1.9/step1/nuscenes_tyjt"
analyze_tyjt_data_discrepancy(dataset_root)