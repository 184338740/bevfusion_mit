import os
import json
import argparse
from collections import defaultdict

def load_json(official_path, tyjt_path, json_file):
    """适配v1.0-mini版本，加载JSON文件"""
    # 优先查找v1.0-mini版本目录（匹配你的数据集结构）
    official_json_path = os.path.join(official_path, "v1.0-mini", json_file)
    if not os.path.exists(official_json_path):
        # 兼容v1.0-trainval版本（双层目录）
        official_json_path = os.path.join(official_path, "v1.0-trainval", "v1.0-trainval", json_file)
        if not os.path.exists(official_json_path):
            # 兼容v1.0-trainval版本（单层目录）
            official_json_path = os.path.join(official_path, "v1.0-trainval", json_file)
    
    # 加载官方数据
    try:
        with open(official_json_path, 'r', encoding='utf-8') as f:
            official_data = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ 官方数据集未找到文件: {json_file}（已尝试v1.0-mini和v1.0-trainval路径）")
        return None, None
    except json.JSONDecodeError:
        print(f"⚠️ 官方数据集文件格式错误: {json_file}")
        return None, None
    
    # 加载转换后数据（固定v1.0-tyjt版本目录）
    tyjt_json_path = os.path.join(tyjt_path, "v1.0-tyjt", json_file)
    try:
        with open(tyjt_json_path, 'r', encoding='utf-8') as f:
            tyjt_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 转换后数据集缺失文件: {json_file}")
        return official_data, None
    except json.JSONDecodeError:
        print(f"❌ 转换后数据集文件格式错误: {json_file}")
        return official_data, None
    
    return official_data, tyjt_data

def compare_json_structure(official_data, tyjt_data, path=""):
    """递归对比两个JSON数据的结构和元素差异"""
    diffs = []
    path_prefix = f"{path}." if path else ""

    # 类型不同
    if type(official_data) != type(tyjt_data):
        diffs.append(f"{path}: 类型不匹配 (官方: {type(official_data).__name__}, 转换后: {type(tyjt_data).__name__})")
        return diffs

    # 处理字典类型
    if isinstance(official_data, dict):
        # 对比键集合
        official_keys = set(official_data.keys())
        tyjt_keys = set(tyjt_data.keys())
        
        missing_keys = official_keys - tyjt_keys
        for key in missing_keys:
            diffs.append(f"{path_prefix}{key}: 转换后缺失该键")
        
        extra_keys = tyjt_keys - official_keys
        for key in extra_keys:
            diffs.append(f"{path_prefix}{key}: 转换后存在额外键")
        
        # 对比共同键的值
        for key in official_keys & tyjt_keys:
            diffs.extend(compare_json_structure(
                official_data[key], 
                tyjt_data[key], 
                f"{path_prefix}{key}"
            ))

    # 处理列表类型
    elif isinstance(official_data, list):
        # 列表长度对比（仅提示差异，不强制一致）
        official_len = len(official_data)
        tyjt_len = len(tyjt_data)
        if official_len != tyjt_len:
            diffs.append(f"{path}: 列表长度不匹配 (官方: {official_len}, 转换后: {tyjt_len})")
        
        # 只对比前3个元素的结构（避免大规模数据对比效率问题）
        min_len = min(official_len, tyjt_len, 3)
        for i in range(min_len):
            diffs.extend(compare_json_structure(
                official_data[i], 
                tyjt_data[i], 
                f"{path}[{i}]"
            ))

    return diffs

def main():
    parser = argparse.ArgumentParser(description="对比官方NuScenes与转换后的nuscenes_tyjt数据集结构")
    parser.add_argument("--official", type=str, default="/mnt/bevfusion_mit_xmy/data/nuscenes_tmp",
                      help="官方NuScenes数据集路径")
    parser.add_argument("--tyjt", type=str, default="/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/output/step1/nuscenes_tyjt/v1.0-tyjt",
                      help="转换后的nuscenes_tyjt数据集路径")
    args = parser.parse_args()

    # 定义需要对比的JSON文件（NuScenes核心文件）
    json_files = [
        "attribute.json", "calibrated_sensor.json", "category.json",
        "ego_pose.json", "instance.json", "log.json", "map.json",
        "sample.json", "sample_annotation.json", "sample_data.json",
        "scene.json", "sensor.json", "visibility.json"
    ]

    # 存储所有差异
    all_diffs = defaultdict(list)
    missing_files = []

    print("===== NuScenes数据集结构对比工具 =====")
    print(f"官方数据集路径: {args.official}")
    print(f"转换后数据集路径: {args.tyjt}\n")

    # 对比每个JSON文件
    for json_file in json_files:
        print(f"=== 对比文件: {json_file} ===")
        
        # 加载JSON数据（适配v1.0-mini版本）
        official_data, tyjt_data = load_json(args.official, args.tyjt, json_file)

        if official_data is None:
            print()
            continue
        if tyjt_data is None:
            missing_files.append(json_file)
            print()
            continue

        # 对比结构
        diffs = compare_json_structure(official_data, tyjt_data)
        if diffs:
            all_diffs[json_file] = diffs
            for diff in diffs[:5]:  # 只显示前5个差异
                print(f"  {diff}")
            if len(diffs) > 5:
                print(f"  ... 还有 {len(diffs)-5} 个差异未显示")
        else:
            print("✅ 结构完全匹配")
        print()

    # 生成总结报告
    print("===== 对比总结 =====")
    if not all_diffs and not missing_files:
        print("🎉 所有JSON文件结构与官方数据集完全一致！")
    else:
        if missing_files:
            print(f"❌ 转换后数据集缺失文件: {', '.join(missing_files)}")
        for file, diffs in all_diffs.items():
            print(f"⚠️ {file} 存在 {len(diffs)} 处结构差异")
    
    # 导出详细差异报告
    with open("dataset_diff_report.txt", "w", encoding="utf-8") as f:
        f.write("===== NuScenes与nuscenes_tyjt差异报告 =====\n")
        f.write(f"官方路径: {args.official}\n")
        f.write(f"转换后路径: {args.tyjt}\n\n")
        
        if missing_files:
            f.write("=== 转换后数据集缺失文件 ===\n")
            for file in missing_files:
                f.write(f"- {file}\n")
            f.write("\n")
        
        for file, diffs in all_diffs.items():
            f.write(f"=== {file} 差异 ===\n")
            for i, diff in enumerate(diffs, 1):
                f.write(f"{i}. {diff}\n")
            f.write("\n")
    print("\n📄 详细差异已导出至: dataset_diff_report.txt")

if __name__ == "__main__":
    main()