# tmp_debug_fixed.py
import os
import json
from nuscenes import NuScenes

def debug_sample_tokens():
    """调试样本token"""
    dataroot = "./output/step1/nuscenes_tyjt_fixed/v1.0-tyjt"
    version = "v1.0-tyjt"
    
    print("🔍 调试样本token...")
    
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    
    # 列出所有样本的前8位token
    print(f"\n所有样本token (前10个):")
    for i, sample in enumerate(nusc.sample[:10]):
        print(f"样本{i+1}: {sample['token'][:8]} (完整token: {sample['token']})")
        print(f"  标注数量: {len(sample['anns'])}")
    
    # 直接检查sample_annotation.json文件
    print(f"\n🔍 直接检查标注文件...")
    ann_file = os.path.join(dataroot, "v1.0-tyjt", "sample_annotation.json")
    
    if os.path.exists(ann_file):
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        
        print(f"标注文件中的记录数: {len(annotations)}")
        
        # 按样本分组统计
        sample_ann_count = {}
        for ann in annotations:
            sample_token = ann['sample_token']
            if sample_token not in sample_ann_count:
                sample_ann_count[sample_token] = 0
            sample_ann_count[sample_token] += 1
        
        # 找到标注数量异常的样本
        print(f"\n标注数量异常的样本:")
        for sample_token, count in sample_ann_count.items():
            if count <= 3:
                print(f"样本 {sample_token[:8]}: {count} 个标注")
                
                # 显示这个样本的所有标注
                sample_anns = [ann for ann in annotations if ann['sample_token'] == sample_token]
                for i, ann in enumerate(sample_anns):
                    print(f"  标注{i+1}:")
                    print(f"    位置: {ann['translation']}")
                    print(f"    尺寸: {ann['size']}")
                    print(f"    旋转: {ann['rotation']}")
                    if 'category_token' in ann:
                        try:
                            category = nusc.get('category', ann['category_token'])
                            print(f"    类别: {category['name']}")
                        except:
                            print(f"    类别token: {ann['category_token']}")
                    else:
                        print(f"    ⚠️ 无category_token字段")
    
    # 检查数据完整性
    print(f"\n🔍 检查数据完整性...")
    
    # 检查必要的JSON文件
    required_files = [
        "sample.json", "sample_annotation.json", "category.json",
        "ego_pose.json", "calibrated_sensor.json", "sample_data.json"
    ]
    
    for file in required_files:
        file_path = os.path.join(dataroot, "v1.0-tyjt", file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            print(f"✅ {file}: {len(data)} 条记录")
        else:
            print(f"❌ {file}: 文件不存在")

def check_annotation_structure():
    """检查标注结构"""
    print(f"\n🔍 检查标注结构...")
    
    dataroot = "./output/step1/nuscenes_tyjt_fixed/v1.0-tyjt"
    ann_file = os.path.join(dataroot, "v1.0-tyjt", "sample_annotation.json")
    
    if os.path.exists(ann_file):
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        
        if annotations:
            first_ann = annotations[0]
            print(f"标注字段: {list(first_ann.keys())}")
            
            # 检查前几个标注的类别信息
            print(f"\n前3个标注的类别信息:")
            for i, ann in enumerate(annotations[:3]):
                print(f"标注{i+1}:")
                for key in ['category_token', 'instance_token', 'sample_token']:
                    if key in ann:
                        print(f"  {key}: {ann[key][:8]}...")
                    else:
                        print(f"  ⚠️ 缺少 {key}")

def find_real_problem_samples():
    """查找真正的问题样本"""
    print(f"\n🔍 查找真正的问题样本...")
    
    dataroot = "./output/step1/nuscenes_tyjt_fixed/v1.0-tyjt"
    
    # 加载数据
    ann_file = os.path.join(dataroot, "v1.0-tyjt", "sample_annotation.json")
    sample_file = os.path.join(dataroot, "v1.0-tyjt", "sample.json")
    category_file = os.path.join(dataroot, "v1.0-tyjt", "category.json")
    
    if not all(os.path.exists(f) for f in [ann_file, sample_file, category_file]):
        print("❌ 缺少必要的JSON文件")
        return
    
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    with open(sample_file, 'r') as f:
        samples = json.load(f)
    with open(category_file, 'r') as f:
        categories = json.load(f)
    
    # 创建映射
    sample_map = {s['token']: s for s in samples}
    category_map = {cat['token']: cat for cat in categories}
    
    # 按样本分组统计标注
    sample_ann_count = {}
    for ann in annotations:
        sample_token = ann['sample_token']
        if sample_token not in sample_ann_count:
            sample_ann_count[sample_token] = []
        sample_ann_count[sample_token].append(ann)
    
    print(f"\n问题样本列表:")
    
    # 查找标注数量少的样本
    low_annotation_samples = []
    for sample_token, anns in sample_ann_count.items():
        if len(anns) <= 3:
            low_annotation_samples.append((sample_token, anns))
    
    # 显示找到的问题样本
    for sample_token, anns in low_annotation_samples[:10]:  # 只显示前10个
        sample_short = sample_token[:8]
        print(f"\n样本 {sample_short}: {len(anns)} 个标注")
        
        for i, ann in enumerate(anns):
            if 'category_token' in ann and ann['category_token'] in category_map:
                category_name = category_map[ann['category_token']]['name']
            else:
                category_name = "未知"
            
            print(f"  标注{i+1}: {category_name}")
            print(f"    位置: {ann['translation']}")
            print(f"    尺寸: {ann['size']}")
            
            # 检查尺寸是否异常
            size = ann['size']
            if category_name == 'traffic_cone' and max(size) > 1.0:
                print(f"    ⚠️ 交通锥尺寸异常: {size}")
            if category_name == 'car' and (size[0] < 2.0 or size[0] > 8.0):
                print(f"    ⚠️ 车辆尺寸异常: {size}")

if __name__ == '__main__':
    debug_sample_tokens()
    check_annotation_structure()
    find_real_problem_samples()