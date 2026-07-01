#!/usr/bin/env python3
import pickle
import os

def fix_info_pkl_format(input_file, output_file):
    """修复infos_train.pkl和infos_val.pkl的格式"""
    print(f"修复文件: {input_file} -> {output_file}")
    
    # 加载原始数据
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"原始数据类型: {type(data)}")
    print(f"原始数据长度: {len(data)}")
    
    # 修复为标准的NuScenes格式
    if isinstance(data, list):
        print("检测到列表格式，转换为标准字典格式")
        fixed_data = {
            'infos': data,
            'metadata': {
                'version': 'tyjt_v1.0',
                'database': 'TYJT',
                'map_version': 'tyjt_v1.0'
            }
        }
    else:
        print(f"未知格式: {type(data)}")
        return
    
    # 保存修复后的文件
    with open(output_file, 'wb') as f:
        pickle.dump(fixed_data, f)
    
    print("修复完成!")

def fix_dbinfo_pkl_format(input_file, output_file):
    """修复dbinfos_train.pkl的格式"""
    print(f"修复文件: {input_file} -> {output_file}")
    
    # 加载原始数据
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"原始数据类型: {type(data)}")
    
    # dbinfos应该是字典格式，如果已经是字典则直接保存
    if isinstance(data, dict):
        print("dbinfos格式正确，直接保存")
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
    else:
        print(f"未知的dbinfos格式: {type(data)}")

def main():
    base_path = '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1'
    
    # 修复训练集信息文件
    fix_info_pkl_format(
        os.path.join(base_path, 'tyjt_infos_train.pkl'),
        os.path.join(base_path, 'tyjt_infos_train_fixed.pkl')
    )
    
    print("\n" + "="*50 + "\n")
    
    # 修复验证集信息文件
    fix_info_pkl_format(
        os.path.join(base_path, 'tyjt_infos_val.pkl'),
        os.path.join(base_path, 'tyjt_infos_val_fixed.pkl')
    )
    
    print("\n" + "="*50 + "\n")
    
    # 修复数据库信息文件
    fix_dbinfo_pkl_format(
        os.path.join(base_path, 'tyjt_dbinfos_train.pkl'),
        os.path.join(base_path, 'tyjt_dbinfos_train_fixed.pkl')
    )
    
    print("\n" + "="*50)
    print("所有文件修复完成！")

if __name__ == "__main__":
    main()
