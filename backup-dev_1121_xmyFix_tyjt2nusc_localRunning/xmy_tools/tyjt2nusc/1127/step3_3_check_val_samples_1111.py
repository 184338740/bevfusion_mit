import mmcv
import os

# 加载验证集信息
val_info_path = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/tyjt_infos_val.pkl"
val_infos = mmcv.load(val_info_path)['infos']

print(f"验证集样本数量: {len(val_infos)}")
print("验证集样本token:")
val_tokens = [info['token'] for info in val_infos]
print(f"前5个token: {val_tokens[:5]}")
print(f"后5个token: {val_tokens[-5:]}")

# 检查是否有重复或缺失的token
unique_tokens = set(val_tokens)
print(f"唯一token数量: {len(unique_tokens)}")
print(f"是否有重复token: {len(val_tokens) != len(unique_tokens)}")

# 检查样本的完整性
print("\n检查样本完整性:")
for i, info in enumerate(val_infos[:3]):  # 只检查前3个样本
    print(f"样本 {i}: token={info['token']}, lidar_path={info.get('lidar_path', 'N/A')}")
    if 'gt_boxes' in info:
        print(f"  标注框数量: {len(info['gt_boxes'])}")
    else:
        print("  无标注框")
