import mmcv

# 使用正确的路径
val_info_path = '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/tyjt_infos_val.pkl'
train_info_path = '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/tyjt_infos_train.pkl'

print(f"验证集文件存在: {mmcv.is_filepath(val_info_path)}")
print(f"训练集文件存在: {mmcv.is_filepath(train_info_path)}")

# 加载验证集信息
val_info = mmcv.load(val_info_path)
train_info = mmcv.load(train_info_path)

print(f"训练集样本数: {len(train_info['infos'])}")
print(f"验证集样本数: {len(val_info['infos'])}")

# 检查样本token
val_tokens = [info['token'] for info in val_info['infos']]
train_tokens = [info['token'] for info in train_info['infos']]

print(f"验证集样本token示例: {val_tokens[:3]}")
print(f"训练集样本token示例: {train_tokens[:3]}")

# 检查重叠
overlap = set(val_tokens) & set(train_tokens)
print(f"训练验证重叠样本数: {len(overlap)}")

if len(overlap) > 0:
    print(f"重叠样本示例: {list(overlap)[:3]}")
