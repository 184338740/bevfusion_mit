import mmcv

train_info_path = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/tyjt_infos_train.pkl"
train_infos = mmcv.load(train_info_path)['infos']
print(f"训练集样本数量: {len(train_infos)}")


val_info_path = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/tyjt_infos_val.pkl"
val_infos = mmcv.load(val_info_path)['infos']
print(f"验证集样本数量: {len(val_infos)}")

print(f"总样本数量: {len(train_infos) + len(val_infos)}")
