# step1_check_raw_data.py
import pickle
import numpy as np

# 1. 直接检查pkl文件
val_pkl = '/mnt/bevfusion_mit_xmy_RunningA100_1205_doing/xmy_tools/tyjt2nusc/1219_A100/output-1226-v1.9/step1/nuscenes_tyjt/tyjt_infos_train.pkl'
print(f"=== 1. 检查原始pkl文件: {val_pkl} ===")

with open(val_pkl, 'rb') as f:
    val_data = pickle.load(f)

print(f"数据类型: {type(val_data)}")
print(f"字典键: {list(val_data.keys())}")

if 'infos' in val_data:
    print(f"infos长度: {len(val_data['infos'])}")
    print(f"metadata: {val_data['metadata']}")
    
    # 检查前3个样本的结构
    print("\n=== 前3个样本结构 ===")
    for i in range(min(3, len(val_data['infos']))):
        info = val_data['infos'][i]
        print(f"\n样本{i}:")
        print(f"  token: {info.get('token', 'N/A')}")
        print(f"  字段: {list(info.keys())}")
        if 'gt_names' in info:
            gt_names = info['gt_names']
            print(f"  gt_names类型: {type(gt_names)}, 长度: {len(gt_names)}")
            print(f"  前3个gt_names: {gt_names[:3] if len(gt_names) > 3 else gt_names}")
            
            # 统计类别分布
            unique, counts = np.unique(gt_names, return_counts=True)
            print(f"  类别分布: {dict(zip(unique, counts))}")
            
            # 特别检查bicycle
            bicycle_count = np.sum(gt_names == 'bicycle')
            if bicycle_count > 0:
                print(f"  🚴 有{bicycle_count}个bicycle!")
else:
    print("没有'infos'键，直接是列表？")
    if isinstance(val_data, list):
        print(f"列表长度: {len(val_data)}")

