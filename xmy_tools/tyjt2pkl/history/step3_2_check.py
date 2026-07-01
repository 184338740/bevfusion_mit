import pickle
import json

# 验证训练集PKL
with open('output/step3_1/tyjt_infos_train.pkl', 'rb') as f:
    train_infos = pickle.load(f)
print(f"训练集样本数: {len(train_infos)}")
print(f"第一个样本的键: {train_infos[0].keys()}")

# 验证GT数据库
with open('output/step3_1/tyjt_dbinfos_train.pkl', 'rb') as f:
    db_infos = pickle.load(f)
print(f"GT数据库类别: {list(db_infos.keys())}")
for cls, infos in db_infos.items():
    print(f"  {cls}: {len(infos)} 个实例")

# 验证统计信息
with open('output/step3_1/conversion_stats.json', 'r') as f:
    stats = json.load(f)
print(f"总帧数: {stats['total_frames']}")