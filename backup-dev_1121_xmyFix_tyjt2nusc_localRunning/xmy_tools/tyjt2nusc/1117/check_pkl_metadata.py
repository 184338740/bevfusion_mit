import mmcv
import os

def check_pkl_metadata():
    """检查pkl文件中的metadata"""
    dataset_root = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt"
    
    pkl_files = [
        "tyjt_infos_train.pkl",
        "tyjt_infos_val.pkl", 
        "tyjt_infos_test.pkl"
    ]
    
    print("=== 检查pkl文件metadata ===")
    
    for pkl_file in pkl_files:
        pkl_path = os.path.join(dataset_root, pkl_file)
        if os.path.exists(pkl_path):
            data = mmcv.load(pkl_path)
            metadata = data.get('metadata', {})
            version = metadata.get('version', '❌ 未设置')
            print(f"{pkl_file}: version = {version}")
        else:
            print(f"{pkl_file}: ❌ 文件不存在")

if __name__ == "__main__":
    check_pkl_metadata()
