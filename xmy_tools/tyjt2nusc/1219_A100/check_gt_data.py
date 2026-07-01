# xmy_tools/check_gt_data.py
import pickle
import numpy as np

def check_gt_data(ann_file):
    """检查GT数据文件"""
    
    data = pickle.load(open(ann_file, 'rb'))
    infos = data['infos']
    
    print(f"🔍 检查GT数据文件: {ann_file}")
    print(f"样本数量: {len(infos)}")
    print(f"metadata: {data['metadata']}")
    
    # 检查前5个样本
    for i, info in enumerate(infos[:5]):
        print(f"\n样本 {i} (token: {info['token'][:20]}...):")
        print(f"  gt_names: {info['gt_names'][:10]}...")  # 前10个类别
        print(f"  gt_names类型: {type(info['gt_names'])}")
        print(f"  gt_names长度: {len(info['gt_names'])}")
        
        # 统计类别分布
        unique, counts = np.unique(info['gt_names'], return_counts=True)
        print(f"  类别统计: {dict(zip(unique, counts))}")
    
    # 总体统计
    print(f"\n📊 总体类别分布:")
    all_categories = []
    for info in infos[:100]:  # 只统计前100个样本
        all_categories.extend(info['gt_names'])
    
    unique, counts = np.unique(all_categories, return_counts=True)
    for cat, count in zip(unique, counts):
        print(f"  {cat}: {count}")

if __name__ == '__main__':
    # 检查验证集GT
    val_file = "/mnt/bevfusion_mit_xmy_RunningA100_1205_doing/xmy_tools/tyjt2nusc/1219_A100/output-1226-v1.9/step1/nuscenes_tyjt/tyjt_infos_val.pkl"
    check_gt_data(val_file)