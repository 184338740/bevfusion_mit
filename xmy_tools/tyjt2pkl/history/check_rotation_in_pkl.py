import pickle
import os
from tqdm import tqdm

def check_rotation_fields(pkl_path):
    """
    检查PKL文件中旋转参数的格式（是否为4元素四元数）
    Args:
        pkl_path: PKL文件路径（如tyjt_infos_train_fixed.pkl）
    """
    if not os.path.exists(pkl_path):
        print(f"Error: PKL文件不存在 - {pkl_path}")
        return

    # 加载PKL文件
    print(f"加载文件: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data_infos = pickle.load(f)
    import pdb; pdb.set_trace()
    
    # 需检查的旋转字段（根据BEVFusion需求）
    rotation_fields = [
        "lidar2ego_rotation",  # 激光雷达到自车的旋转
        "camera2ego_rotation",  # 相机到自车的旋转（若有）
        "lidar2camera_rotation"  # 激光雷达到相机的旋转（若有）
    ]

    # 统计异常字段
    error_counts = {field: 0 for field in rotation_fields}
    total_samples = len(data_infos)

    # 遍历所有样本检查
    for idx, info in tqdm(enumerate(data_infos), total=total_samples, desc="检查进度"):
        for field in rotation_fields:
            if field not in info:
                print(f"警告: 样本{idx}缺少字段 {field}")
                error_counts[field] += 1
                continue

            rot = info[field]
            # 检查元素数量（四元数应为4元素）
            if len(rot) != 4:
                error_counts[field] += 1
                # 打印前10个异常样本详情
                if error_counts[field] <= 10:
                    print(f"样本{idx}的{field}格式错误: 长度={len(rot)}, 值={rot}")

    # 输出检查结果
    print("\n===== 旋转参数检查结果 =====")
    print(f"总样本数: {total_samples}")
    for field in rotation_fields:
        error_rate = error_counts[field] / total_samples * 100
        print(f"{field}: 异常样本数={error_counts[field]} ({error_rate:.2f}%)")
        if error_counts[field] == 0:
            print(f"  ✅ 所有样本均为4元素四元数")
        else:
            print(f"  ❌ 存在{error_counts[field]}个样本格式错误（需转为4元素四元数）")

if __name__ == "__main__":
    # 替换为你的PKL文件路径
    pkl_files = [
        "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_train_fixed.pkl",
        "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_val_fixed.pkl"
    ]

    for pkl_file in pkl_files:
        check_rotation_fields(pkl_file)
        print("-" * 50)