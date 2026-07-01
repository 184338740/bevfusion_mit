# step3_3_check_tyjt_data.py
import pickle
import numpy as np

def check_tyjt_data():
    """检查TYJT PKL数据格式"""
    try:
        with open('./step3_1_output/tyjt_infos_train.pkl', 'rb') as f:
            data = pickle.load(f)
        
        print("✅ TYJT PKL文件加载成功")
        print(f"样本数量: {len(data['infos'])}")
        
        # 检查第一个样本
        sample = data['infos'][0]
        print(f"\n样本键: {sample.keys()}")
        
        # 检查关键字段
        print(f"token: {sample.get('token')}")
        print(f"相机数量: {len(sample.get('cams', {}))}")
        print(f"相机名称: {list(sample.get('cams', {}).keys())}")
        
        # 检查标注字段
        print(f"gt_boxes形状: {sample.get('gt_boxes', np.array([])).shape}")
        print(f"gt_names: {sample.get('gt_names', [])}")
        print(f"valid_flag形状: {sample.get('valid_flag', np.array([])).shape}")
        
        # 统计标注信息
        total_boxes = 0
        for i, s in enumerate(data['infos'][:5]):
            boxes = s.get('gt_boxes', [])
            if len(boxes) > 0:
                total_boxes += len(boxes)
                print(f"样本 {i}: {len(boxes)} 个标注框")
        
        print(f"\n前5个样本总标注框: {total_boxes}")
        
        # 检查元数据
        if 'metadata' in data:
            print(f"\n元数据: {data['metadata']}")
            
    except Exception as e:
        print(f"❌ 检查失败: {e}")

if __name__ == "__main__":
    check_tyjt_data()