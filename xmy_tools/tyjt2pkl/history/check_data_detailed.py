import pickle
import numpy as np
import os

def check_data_detailed():
    # 检查文件是否存在
    train_file = '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_train_fixed.pkl'
    val_file = '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_val_fixed.pkl'
    
    print("=== 检查文件是否存在 ===")
    print(f"训练文件存在: {os.path.exists(train_file)}")
    print(f"验证文件存在: {os.path.exists(val_file)}")
    
    if not os.path.exists(train_file):
        print("训练文件不存在，请检查路径")
        return
    
    # 加载训练数据
    print("\n=== 加载训练数据 ===")
    with open(train_file, 'rb') as f:
        train_info = pickle.load(f)
    
    print(f"训练数据类型: {type(train_info)}")
    print(f"训练数据长度: {len(train_info)}")
    
    # 检查数据结构
    if isinstance(train_info, dict):
        print("\n=== 字典结构分析 ===")
        print(f"字典键: {list(train_info.keys())}")
        
        # 如果是字典，可能有metadata和data_list
        if 'data_list' in train_info:
            data_list = train_info['data_list']
            print(f"data_list长度: {len(data_list)}")
            if len(data_list) > 0:
                sample = data_list[0]
                analyze_sample(sample, 0)
        elif 'infos' in train_info:
            data_list = train_info['infos']
            print(f"infos长度: {len(data_list)}")
            if len(data_list) > 0:
                sample = data_list[0]
                analyze_sample(sample, 0)
        else:
            # 尝试直接使用第一个键值对
            first_key = list(train_info.keys())[0]
            first_value = train_info[first_key]
            print(f"第一个键: {first_key}")
            print(f"第一个值类型: {type(first_value)}")
            if isinstance(first_value, list) and len(first_value) > 0:
                analyze_sample(first_value[0], 0)
                
    elif isinstance(train_info, list):
        print("\n=== 列表结构分析 ===")
        if len(train_info) > 0:
            sample = train_info[0]
            analyze_sample(sample, 0)
        else:
            print("训练数据列表为空!")
    else:
        print(f"未知的数据结构: {type(train_info)}")

def analyze_sample(sample, index):
    print(f"\n=== 分析样本 {index} ===")
    print(f"样本类型: {type(sample)}")
    
    if isinstance(sample, dict):
        print(f"样本键: {list(sample.keys())}")
        
        # 检查关键字段
        for key in ['token', 'timestamp', 'cams', 'lidar_path', 'sweeps', 'annos']:
            if key in sample:
                value = sample[key]
                print(f"{key}: {type(value)} - {str(value)[:100]}...")
            else:
                print(f"{key}: 不存在")
        
        # 详细检查cams
        if 'cams' in sample:
            cams = sample['cams']
            print(f"\n相机数量: {len(cams)}")
            for cam_name, cam_info in cams.items():
                print(f"  相机 {cam_name}:")
                for k, v in cam_info.items():
                    if k in ['data_path', 'cam_intrinsic', 'lidar2cam']:
                        print(f"    {k}: {type(v)} - {str(v)[:50]}...")
        
        # 详细检查annos
        if 'annos' in sample:
            annos = sample['annos']
            print(f"\n标注信息:")
            for k, v in annos.items():
                if isinstance(v, (np.ndarray, list)):
                    print(f"  {k}: {type(v)} - 形状: {get_shape(v)}")
                else:
                    print(f"  {k}: {type(v)} - {str(v)[:100]}...")
                    
    else:
        print(f"样本内容: {str(sample)[:200]}...")

def get_shape(data):
    if isinstance(data, np.ndarray):
        return data.shape
    elif isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], (np.ndarray, list)):
            return f"list of {len(data)}, shapes: {[get_shape(x) for x in data[:3]]}"
        else:
            return f"list of {len(data)}"
    else:
        return "unknown"

def check_dbinfo():
    print("\n=== 检查数据库信息 ===")
    dbinfo_file = '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_dbinfos_train_fixed.pkl'
    
    if os.path.exists(dbinfo_file):
        with open(dbinfo_file, 'rb') as f:
            dbinfo = pickle.load(f)
        
        print(f"数据库信息类型: {type(dbinfo)}")
        if isinstance(dbinfo, dict):
            print(f"数据库类别: {list(dbinfo.keys())}")
            for class_name, class_info in dbinfo.items():
                if isinstance(class_info, list):
                    print(f"  {class_name}: {len(class_info)} 个样本")
                else:
                    print(f"  {class_name}: {type(class_info)}")
    else:
        print("数据库信息文件不存在")

if __name__ == '__main__':
    check_data_detailed()
    check_dbinfo()