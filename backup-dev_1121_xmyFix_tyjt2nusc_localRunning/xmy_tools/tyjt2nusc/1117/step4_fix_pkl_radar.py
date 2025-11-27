# fix_pkl_radar.py
import pickle

# 读取pkl文件
with open('/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/tyjt_infos_train.pkl', 'rb') as f:
    data = pickle.load(f)

# 修复radars字段
if 'infos' in data:
    for info in data['infos']:
        if 'radars' in info and info['radars'] == {}:
            # 方法1：移除radars字段
            del info['radars']
            # 或者方法2：设置为None
            # info['radars'] = None

# 保存修复后的文件
with open('/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/tyjt_infos_train_fixed.pkl', 'wb') as f:
    pickle.dump(data, f)

print("pkl文件修复完成！")