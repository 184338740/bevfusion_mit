# debug_nuscenes_api_correct.py
import json
from pathlib import Path
from nuscenes.nuscenes import NuScenes

def debug_correct_api_usage():
    """调试正确的NuScenes API使用方法"""
    data_root = Path("/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/output/step1/nuscenes_tyjt")
    version = "v1.0-tyjt"
    
    print("=== 正确的NuScenes API使用方法 ===")
    
    try:
        nusc = NuScenes(version=version, dataroot=str(data_root), verbose=False)
        print("✅ NuScenes初始化成功!")
        
        sample_token = "eaceae02-b27d-b019-8472-68e0948cbcd2"
        
        # 方法1: 使用get_sample_data的正确方式
        print(f"\n🔍 方法1: 使用get_sample_data")
        sample = nusc.get('sample', sample_token)
        print(f"样本数据: {sample.keys()}")
        
        # 检查样本中的传感器数据引用
        print(f"样本中的data tokens:")
        for key, token in sample['data'].items():
            print(f"  {key}: {token}")
            
        # 通过样本的data tokens获取传感器数据
        for channel, data_token in sample['data'].items():
            try:
                print(f"\n  尝试获取 {channel} (token: {data_token})")
                # 正确的get_sample_data调用方式
                points, coloring, im = nusc.get_sample_data(data_token)
                print(f"  ✅ {channel}: 成功获取数据")
                if points is not None:
                    print(f"     点云/图像数据点数: {len(points)}")
            except Exception as e:
                print(f"  ❌ {channel}: 错误 - {e}")
        
        # 方法2: 直接通过sample_data token获取
        print(f"\n🔍 方法2: 直接通过sample_data token获取")
        for channel in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'LIDAR_TOP']:
            # 查找对应的sample_data token
            data_token = None
            for sd in nusc.sample_data:
                if sd['sample_token'] == sample_token and channel in sd['channel']:
                    data_token = sd['token']
                    break
            
            if data_token:
                try:
                    points, coloring, im = nusc.get_sample_data(data_token)
                    print(f"  ✅ {channel}: 成功获取 (token: {data_token})")
                except Exception as e:
                    print(f"  ❌ {channel}: 错误 - {e}")
            else:
                print(f"  ⚠️  {channel}: 未找到对应的sample_data token")
                
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_correct_api_usage()