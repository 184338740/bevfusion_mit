import pickle
import os

def check_pkl_format(file_path):
    print(f"检查文件: {file_path}")
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"数据类型: {type(data)}")
    print(f"数据长度: {len(data) if hasattr(data, '__len__') else 'N/A'}")
    
    if isinstance(data, dict):
        print("字典的keys:", list(data.keys()))
        if 'infos' in data:
            infos = data['infos']
            print(f"infos类型: {type(infos)}")
            print(f"infos长度: {len(infos) if hasattr(infos, '__len__') else 'N/A'}")
            if len(infos) > 0:
                print("第一个info的keys:", list(infos[0].keys()) if isinstance(infos, list) else list(infos.keys()))
    elif isinstance(data, list):
        print("列表的第一个元素类型:", type(data[0]) if len(data) > 0 else "空列表")
        if len(data) > 0:
            print("第一个元素的keys:", list(data[0].keys()) if isinstance(data[0], dict) else "不是字典")

# 检查训练和验证文件
pkl_files = [
    '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_train.pkl',
    '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_val.pkl',
    '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_dbinfos_train.pkl'
]

for pkl_file in pkl_files:
    check_pkl_format(pkl_file)
    print("-" * 50)