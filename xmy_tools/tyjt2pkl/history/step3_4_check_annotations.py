# check_annotations.py
import json
from pathlib import Path

def check_annotations():
    dataset_root = "/mnt/dataset/tyjt_RawData"
    subscene = "2d3d4d_20250728_weiyuan"
    subdataset = "G51102400001M00_20250728191726_2.0HZ"
    
    dataset_path = Path(dataset_root) / subscene / "datasets" / subdataset
    label_dir = dataset_path / "lidar" / "label"
    
    print(f"标注目录: {label_dir}")
    
    # 检查标注文件
    json_files = list(label_dir.glob("*.json"))
    print(f"找到 {len(json_files)} 个标注文件")
    
    # 检查前5个标注文件内容
    for i, json_file in enumerate(json_files[:5]):
        print(f"\n--- 标注文件 {i+1}: {json_file.name} ---")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'objects' in data:
                print(f"  对象数量: {len(data['objects'])}")
                for j, obj in enumerate(data['objects'][:3]):  # 显示前3个对象
                    print(f"    对象 {j}: {obj.get('type', 'unknown')} - {obj.get('box3d', [])}")
            else:
                print("  没有objects字段")
                
        except Exception as e:
            print(f"  读取失败: {e}")

check_annotations()