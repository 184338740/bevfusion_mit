# check_tyjt_files.py

import os
import pickle

def check_generated_files(output_dir):
    """检查生成的pkl文件"""
    expected_files = [
        'tyjt_infos_train.pkl',
        'tyjt_infos_val.pkl', 
        'tyjt_infos_test.pkl',
        'tyjt_dbinfos_train.pkl'
    ]
    
    print("🔵[TYJT验证]>>> 检查生成的文件...")
    
    for filename in expected_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / 1024 / 1024  # MB
            print(f"✅ {filename} ({file_size:.2f} MB)")
            
            # 加载并检查内容
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                if 'infos' in data:
                    print(f"   - 包含 {len(data['infos'])} 个样本")
                    if len(data['infos']) > 0:
                        sample = data['infos'][0]
                        if 'gt_names' in sample:
                            unique_categories = set()
                            for info in data['infos']:
                                if 'gt_names' in info:
                                    unique_categories.update(info['gt_names'])
                            print(f"   - 包含类别: {sorted(unique_categories)}")
                
                elif isinstance(data, dict):
                    total_instances = sum(len(infos) for infos in data.values())
                    print(f"   - 包含 {len(data)} 个类别, {total_instances} 个实例")
                    for category, infos in data.items():
                        print(f"     {category}: {len(infos)} 个实例")
                        
            except Exception as e:
                print(f"   - 加载失败: {e}")
        else:
            print(f"❌ {filename} - 文件不存在")

if __name__ == "__main__":
    output_dir = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1121/output/step1-v6/nuscenes_tyjt"
    check_generated_files(output_dir)