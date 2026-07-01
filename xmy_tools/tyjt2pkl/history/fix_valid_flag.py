# 修复pkl文件，添加valid_flag并检查类别分布
import pickle
import numpy as np

def fix_pkl_files():
    # 修复训练文件
    with open('/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_train_correct.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    print(f"训练样本数量: {len(train_data['infos'])}")
    
    # 统计类别分布
    class_counts = {}
    for info in train_data['infos']:
        # 添加valid_flag
        info['valid_flag'] = True
        
        # 统计类别
        for gt_name in info.get('gt_names', []):
            class_counts[gt_name] = class_counts.get(gt_name, 0) + 1
    
    print("训练集类别分布:", class_counts)
    
    with open('/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_train_correct.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    
    # 修复验证文件
    with open('/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_val_correct.pkl', 'rb') as f:
        val_data = pickle.load(f)
    
    print(f"验证样本数量: {len(val_data['infos'])}")
    
    for info in val_data['infos']:
        info['valid_flag'] = True
    
    with open('/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_val_correct.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    
    print("修复完成！")

if __name__ == '__main__':
    fix_pkl_files()