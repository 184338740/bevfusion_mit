# check_class_distribution.py
import pickle

def check_class_distribution():
    # 检查训练文件
    with open('/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_train_correct.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    print("=== 训练集类别分布 ===")
    class_counts = {}
    total_objects = 0
    
    for i, info in enumerate(train_data['infos']):
        gt_names = info.get('gt_names', [])
        for gt_name in gt_names:
            class_counts[gt_name] = class_counts.get(gt_name, 0) + 1
            total_objects += 1
    
    print(f"总样本数: {len(train_data['infos'])}")
    print(f"总物体数: {total_objects}")
    print("类别分布:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")
    
    # 检查配置中的类别哪些没有样本
    config_classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 
                     'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
    
    print("\n=== 配置类别 vs 实际数据 ===")
    for cls in config_classes:
        if cls in class_counts:
            print(f"  {cls}: ✓ ({class_counts[cls]}个样本)")
        else:
            print(f"  {cls}: ✗ (无样本)")

if __name__ == '__main__':
    check_class_distribution()