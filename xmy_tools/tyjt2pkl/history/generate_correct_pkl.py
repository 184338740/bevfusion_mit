# generate_correct_pkl.py
import pickle
import os

def generate_correct_pkl():
    data_dir = '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/'
    
    print("=== 生成正确的pkl文件格式 ===")
    
    # 1. 生成训练文件
    print("处理训练文件...")
    with open(os.path.join(data_dir, 'tyjt_infos_train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
        print(f"原始训练数据: 类型={type(train_data)}, 长度={len(train_data)}")
    
    train_fixed = {
        'infos': train_data,
        'metadata': {
            'version': 'tyjt-v1.0'
        }
    }
    
    with open(os.path.join(data_dir, 'tyjt_infos_train_correct.pkl'), 'wb') as f:
        pickle.dump(train_fixed, f)
    print(f"生成训练文件: tyjt_infos_train_correct.pkl, 样本数量: {len(train_data)}")
    
    # 2. 生成验证文件
    print("处理验证文件...")
    with open(os.path.join(data_dir, 'tyjt_infos_val.pkl'), 'rb') as f:
        val_data = pickle.load(f)
        print(f"原始验证数据: 类型={type(val_data)}, 长度={len(val_data)}")
    
    val_fixed = {
        'infos': val_data,
        'metadata': {
            'version': 'tyjt-v1.0'
        }
    }
    
    with open(os.path.join(data_dir, 'tyjt_infos_val_correct.pkl'), 'wb') as f:
        pickle.dump(val_fixed, f)
    print(f"生成验证文件: tyjt_infos_val_correct.pkl, 样本数量: {len(val_data)}")
    
    # 3. 验证生成的文件
    print("\n=== 验证生成的文件 ===")
    with open(os.path.join(data_dir, 'tyjt_infos_train_correct.pkl'), 'rb') as f:
        check_data = pickle.load(f)
        print(f"训练文件格式: 类型={type(check_data)}")
        print(f"训练文件keys: {check_data.keys()}")
        print(f"训练文件metadata: {check_data['metadata']}")
        print(f"训练文件infos长度: {len(check_data['infos'])}")

if __name__ == '__main__':
    generate_correct_pkl()