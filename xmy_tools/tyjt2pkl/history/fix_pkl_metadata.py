import pickle
import os

def fix_pkl_metadata():
    data_dir = '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/'
    
    # 修复训练文件
    with open(os.path.join(data_dir, 'tyjt_infos_train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    
    # 转换为NuScenes期望的完整格式
    train_fixed = {
        'infos': train_data,
        'metadata': {
            'version': 'tyjt-v1.0'  # 添加version字段
        }
    }
    
    with open(os.path.join(data_dir, 'tyjt_infos_train_fixed2.pkl'), 'wb') as f:
        pickle.dump(train_fixed, f)
    print(f"训练文件修复完成，样本数量: {len(train_data)}")
    
    # 修复验证文件
    with open(os.path.join(data_dir, 'tyjt_infos_val.pkl'), 'rb') as f:
        val_data = pickle.load(f)
    
    val_fixed = {
        'infos': val_data,
        'metadata': {
            'version': 'tyjt-v1.0'  # 添加version字段
        }
    }
    
    with open(os.path.join(data_dir, 'tyjt_infos_val_fixed2.pkl'), 'wb') as f:
        pickle.dump(val_fixed, f)
    print(f"验证文件修复完成，样本数量: {len(val_data)}")
    
    # 检查修复后的文件
    print("\n检查修复后的文件:")
    with open(os.path.join(data_dir, 'tyjt_infos_train_fixed2.pkl'), 'rb') as f:
        check_data = pickle.load(f)
        print(f"修复后训练文件类型: {type(check_data)}")
        print(f"修复后训练文件keys: {check_data.keys()}")
        print(f"metadata内容: {check_data['metadata']}")

if __name__ == '__main__':
    fix_pkl_metadata()