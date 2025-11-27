import mmcv
import shutil

# 备份原文件
db_path = '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/tyjt_dbinfos_train.pkl'
backup_path = db_path + '.backup'
shutil.copy(db_path, backup_path)
print(f"已备份原文件到: {backup_path}")

# 加载现有的database
db_infos = mmcv.load(db_path)
print("现有的类别:", list(db_infos.keys()))

# 标准nuScenes 10个类别
standard_categories = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

# 确保所有标准类别都存在，缺失的创建空列表
for category in standard_categories:
    if category not in db_infos:
        db_infos[category] = []
        print(f"添加空类别: {category}")

print("修正后的类别:", list(db_infos.keys()))

# 保存修正后的database
mmcv.dump(db_infos, db_path)
print("Database修正完成!")
