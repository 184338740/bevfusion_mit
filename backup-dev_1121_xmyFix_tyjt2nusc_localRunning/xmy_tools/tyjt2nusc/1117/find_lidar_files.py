import os
import mmcv
from pathlib import Path

def find_lidar_files():
    """查找Lidar文件的实际位置"""
    val_info_path = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/tyjt_infos_val.pkl"
    
    if not os.path.exists(val_info_path):
        print(f"❌ 验证集信息文件不存在: {val_info_path}")
        return None
        
    val_infos = mmcv.load(val_info_path)
    
    # 检查数据结构
    if 'infos' not in val_infos:
        print("❌ 数据格式错误，没有找到'infos'键")
        print(f"数据键值: {val_infos.keys()}")
        return None
        
    infos = val_infos['infos']
    print(f"验证集样本数: {len(infos)}")
    
    if len(infos) == 0:
        print("❌ 验证集为空")
        return None
        
    # 检查第一个样本
    sample = infos[0]
    print(f"样本键值: {sample.keys()}")
    
    if 'lidar_path' not in sample:
        print("❌ 样本中没有lidar_path字段")
        return None
        
    lidar_path = sample['lidar_path']
    print(f"Lidar路径: {lidar_path}")
    
    # 检查可能的路径
    possible_paths = [
        f"/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/{lidar_path}",
        f"/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/{lidar_path}",
        lidar_path,  # 绝对路径
    ]
    
    for test_path in possible_paths:
        if os.path.exists(test_path):
            print(f"✅ 找到Lidar文件: {test_path}")
            return test_path
    
    # 搜索文件系统
    print("在文件系统中搜索Lidar文件...")
    lidar_filename = Path(lidar_path).name
    print(f"搜索文件名: {lidar_filename}")
    
    search_dirs = [
        "/mnt/bevfusion_mit_xmy",
        "/mnt",
        "/home",
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            print(f"搜索目录: {search_dir}")
            for root, dirs, files in os.walk(search_dir):
                if lidar_filename in files:
                    full_path = os.path.join(root, lidar_filename)
                    print(f"✅ 找到Lidar文件: {full_path}")
                    return full_path
                # 限制搜索深度
                if root.count(os.sep) - search_dir.count(os.sep) > 3:
                    dirs[:] = []  # 不继续深入
    
    print("❌ 未找到Lidar文件")
    return None

if __name__ == "__main__":
    find_lidar_files()
