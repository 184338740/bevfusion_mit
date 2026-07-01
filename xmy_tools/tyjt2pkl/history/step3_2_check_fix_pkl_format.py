# fix_pkl_format.py
import pickle
import os

def fix_pkl_format():
    data_dir = '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/'
    
    # 修复训练文件
    with open(os.path.join(data_dir, 'tyjt_infos_train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    
    # 转换为NuScenes期望的格式
    train_fixed = {'infos': train_data, 'metadata': {}}
    
    with open(os.path.join(data_dir, 'tyjt_infos_train_fixed.pkl'), 'wb') as f:
        pickle.dump(train_fixed, f)
    
    # 修复验证文件
    with open(os.path.join(data_dir, 'tyjt_infos_val.pkl'), 'rb') as f:
        val_data = pickle.load(f)
    
    val_fixed = {'infos': val_data, 'metadata': {}}
    
    with open(os.path.join(data_dir, 'tyjt_infos_val_fixed.pkl'), 'wb') as f:
        pickle.dump(val_fixed, f)
    
    print("修复完成！生成的文件：tyjt_infos_train_fixed.pkl, tyjt_infos_val_fixed.pkl")

if __name__ == '__main__':
    fix_pkl_format()