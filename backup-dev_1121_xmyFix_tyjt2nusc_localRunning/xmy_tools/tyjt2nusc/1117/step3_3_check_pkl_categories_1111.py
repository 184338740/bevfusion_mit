import mmcv
import pickle

def check_pkl_categories():
    # 检查所有相关的pkl文件
    pkl_files = [
        '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/tyjt_dbinfos_train.pkl',
        '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/tyjt_infos_train.pkl',
        '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/tyjt_infos_val.pkl'
    ]
    
    # nuScenes标准类别
    nus_categories = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                     'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
    
    for pkl_path in pkl_files:
        print(f"\n🔍 检查文件: {pkl_path}")
        
        try:
            # 尝试用mmcv加载
            data = mmcv.load(pkl_path)
            
            if 'dbinfos' in pkl_path:
                # 检查database文件
                print("📊 Database类别统计:")
                for cat in nus_categories:
                    if cat in data:
                        count = len(data[cat])
                        status = "✅" if count > 0 else "⚠️"
                        print(f"  {status} {cat:20s}: {count:3d} samples")
                    else:
                        print(f"  ❌ {cat:20s}: 缺失")
                        
                # 检查是否有额外类别
                extra_cats = [cat for cat in data.keys() if cat not in nus_categories]
                if extra_cats:
                    print(f"  🔵 额外类别: {extra_cats}")
                    
            else:
                # 检查info文件中的类别分布
                print("📊 Info文件类别统计:")
                if 'infos' in data:
                    category_counts = {}
                    for info in data['infos']:
                        if 'gt_names' in info:
                            for name in info['gt_names']:
                                category_counts[name] = category_counts.get(name, 0) + 1
                    
                    for cat in nus_categories:
                        count = category_counts.get(cat, 0)
                        status = "✅" if count > 0 else "⚠️"
                        print(f"  {status} {cat:20s}: {count:3d} samples")
                        
        except Exception as e:
            print(f"❌ 加载失败: {e}")

if __name__ == "__main__":
    check_pkl_categories()
