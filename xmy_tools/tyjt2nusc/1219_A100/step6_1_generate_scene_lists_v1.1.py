#!/usr/bin/env python3
"""
generate_scene_lists_actual.py - 修复版（带详细统计）
根据实际scene.json格式生成场景列表，修正包名匹配逻辑，并生成详细统计
"""

import json
from pathlib import Path
import argparse
from collections import defaultdict
from datetime import datetime

# 评测数据集配置 - 使用正确的包名格式
EVAL_PACKAGES = {
    "2d3d_20250114",           # hikvision格式：2d3d_20250114
    "2d3d4d_20250728_weiyuan"  # all_in_one格式：2d3d4d_20250728_weiyuan
}

def extract_package_name(scene_name: str) -> str:
    """从scene_name中正确提取包名"""
    parts = scene_name.split('_')
    
    if len(parts) < 2:
        return scene_name
    
    # 检查是否是2d3d4d_xxxxxx_weiyuan格式（all_in_one）
    if parts[0] == "2d3d4d" and len(parts) >= 3 and "weiyuan" in parts[2]:
        # 格式：2d3d4d_20250728_weiyuan
        return f"{parts[0]}_{parts[1]}_{parts[2]}"
    elif parts[0] == "2d3d":
        # 格式：2d3d_20250114 (hikvision)
        return f"{parts[0]}_{parts[1]}"
    else:
        # 默认返回前两部分
        return f"{parts[0]}_{parts[1]}"

def is_eval_scene(scene_name: str) -> bool:
    """判断场景是否属于评测数据集"""
    # 方法1：提取包名检查
    package_name = extract_package_name(scene_name)
    if package_name in EVAL_PACKAGES:
        return True
    
    # 方法2：直接包含检查（备用）
    for eval_pkg in EVAL_PACKAGES:
        if eval_pkg in scene_name:
            return True
    
    return False

def main():
    parser = argparse.ArgumentParser(description='根据评测数据集生成场景列表（修复版）')
    parser.add_argument('--nuscenes-dir', type=str, required=True,
                       help='nuscenes_tyjt数据集目录')
    parser.add_argument('--output-dir', type=str, default='./scene_lists_v02',
                       help='输出目录')
    
    args = parser.parse_args()
    
    nuscenes_dir = Path(args.nuscenes_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载scene.json
    scene_file = nuscenes_dir / "v1.0-tyjt" / "scene.json"
    if not scene_file.exists():
        print(f"❌ 找不到scene.json: {scene_file}")
        return
    
    with open(scene_file, 'r', encoding='utf-8') as f:
        scenes = json.load(f)
    
    print(f"✅ 加载 {len(scenes)} 个场景")
    
    # 2. 分离场景（使用修复的逻辑）
    train_scenes = []
    val_scenes = []
    
    for scene in scenes:
        scene_name = scene['name']
        
        if is_eval_scene(scene_name):
            val_scenes.append(scene_name)
        else:
            train_scenes.append(scene_name)
    
    # 3. 详细统计每个包的数据
    print(f"\n📊 场景划分统计（修复后）:")
    print(f"  总场景数: {len(scenes)}")
    print(f"  训练集场景: {len(train_scenes)} ({len(train_scenes)/len(scenes)*100:.1f}%)")
    print(f"  验证集场景: {len(val_scenes)} ({len(val_scenes)/len(scenes)*100:.1f}%)")
    
    # 按包名详细统计
    print("\n📦 按包详细统计:")
    
    package_stats = defaultdict(lambda: {'total': 0, 'train': 0, 'val': 0, 'scenes': []})
    
    for scene in scenes:
        scene_name = scene['name']
        package = extract_package_name(scene_name)
        
        package_stats[package]['total'] += 1
        package_stats[package]['scenes'].append(scene_name)
        
        if is_eval_scene(scene_name):
            package_stats[package]['val'] += 1
        else:
            package_stats[package]['train'] += 1
    
    # 按包名排序输出
    for package in sorted(package_stats.keys()):
        stats = package_stats[package]
        status = "🔵 验证集" if package in EVAL_PACKAGES else "🟢 训练集"
        train_percent = stats['train'] / stats['total'] * 100 if stats['total'] > 0 else 0
        val_percent = stats['val'] / stats['total'] * 100 if stats['total'] > 0 else 0
        
        print(f"  {package}:")
        print(f"    总场景数: {stats['total']}")
        print(f"    训练集: {stats['train']} ({train_percent:.1f}%)")
        print(f"    验证集: {stats['val']} ({val_percent:.1f}%)")
        print(f"    状态: {status}")
    
    # 4. 生成详细的统计文件
    stats_file = output_dir / "split_statistics.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("                     场景划分详细统计\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("📊 总体统计\n")
        f.write("-" * 40 + "\n")
        f.write(f"总场景数: {len(scenes)}\n")
        f.write(f"训练集场景数: {len(train_scenes)} ({len(train_scenes)/len(scenes)*100:.1f}%)\n")
        f.write(f"验证集场景数: {len(val_scenes)} ({len(val_scenes)/len(scenes)*100:.1f}%)\n\n")
        
        f.write("🎯 评测数据集配置\n")
        f.write("-" * 40 + "\n")
        for eval_pkg in sorted(EVAL_PACKAGES):
            f.write(f"- {eval_pkg}\n")
        f.write("\n")
        
        f.write("📦 按包详细统计\n")
        f.write("-" * 40 + "\n")
        
        # 首先列出验证集的包
        f.write("验证集包:\n")
        for package in sorted(package_stats.keys()):
            if package in EVAL_PACKAGES:
                stats = package_stats[package]
                f.write(f"  {package}:\n")
                f.write(f"    总场景数: {stats['total']}\n")
                f.write(f"    训练集: {stats['train']} ({stats['train']/stats['total']*100:.1f}%)\n")
                f.write(f"    验证集: {stats['val']} ({stats['val']/stats['total']*100:.1f}%)\n")
        
        # 然后列出训练集的包
        f.write("\n训练集包:\n")
        for package in sorted(package_stats.keys()):
            if package not in EVAL_PACKAGES and package_stats[package]['total'] > 0:
                stats = package_stats[package]
                f.write(f"  {package}:\n")
                f.write(f"    总场景数: {stats['total']}\n")
                f.write(f"    训练集: {stats['train']} ({stats['train']/stats['total']*100:.1f}%)\n")
                f.write(f"    验证集: {stats['val']} ({stats['val']/stats['total']*100:.1f}%)\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("=" * 60 + "\n")
    
    print(f"\n📄 详细统计已保存到: {stats_file}")
    
    # 5. 保存场景列表
    train_file = output_dir / "train_scenes.txt"
    val_file = output_dir / "val_scenes.txt"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write("# 训练集场景列表（修复版）\n")
        f.write("# 生成规则：不属于评测数据集的所有场景\n")
        f.write(f"# 总场景数: {len(train_scenes)}\n")
        f.write("# 评测数据集: 2d3d_20250114, 2d3d4d_20250728_weiyuan\n")
        f.write("#\n")
        for scene in sorted(train_scenes):
            f.write(f"{scene}\n")
    
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write("# 验证集场景列表（修复版）\n")
        f.write("# 生成规则：评测数据集的所有场景\n")
        f.write(f"# 总场景数: {len(val_scenes)}\n")
        f.write("# 评测数据集: 2d3d_20250114, 2d3d4d_20250728_weiyuan\n")
        f.write("#\n")
        for scene in sorted(val_scenes):
            f.write(f"{scene}\n")
    
    # 6. 生成CSV格式的统计（便于导入Excel）
    csv_file = output_dir / "split_statistics.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("包名,总场景数,训练集,训练集百分比,验证集,验证集百分比,数据集类型\n")
        
        for package in sorted(package_stats.keys()):
            stats = package_stats[package]
            train_percent = stats['train'] / stats['total'] * 100 if stats['total'] > 0 else 0
            val_percent = stats['val'] / stats['total'] * 100 if stats['total'] > 0 else 0
            dataset_type = "验证集" if package in EVAL_PACKAGES else "训练集"
            
            f.write(f"{package},{stats['total']},{stats['train']},{train_percent:.1f}%,{stats['val']},{val_percent:.1f}%,{dataset_type}\n")
    
    print(f"📊 CSV统计已保存到: {csv_file}")
    
    # 7. 输出验证结果
    print(f"\n✅ 验证结果:")
    total_val_scenes = sum(stats['val'] for stats in package_stats.values())
    total_train_scenes = sum(stats['train'] for stats in package_stats.values())
    
    print(f"  训练集场景总数验证: {total_train_scenes} (应为: {len(train_scenes)}) {'✅' if total_train_scenes == len(train_scenes) else '❌'}")
    print(f"  验证集场景总数验证: {total_val_scenes} (应为: {len(val_scenes)}) {'✅' if total_val_scenes == len(val_scenes) else '❌'}")
    
    # 检查评测数据包是否完全在验证集
    print(f"\n🔍 评测数据集检查:")
    for eval_pkg in sorted(EVAL_PACKAGES):
        if eval_pkg in package_stats:
            stats = package_stats[eval_pkg]
            if stats['val'] == stats['total']:
                print(f"  ✅ {eval_pkg}: 所有 {stats['total']} 个场景都在验证集")
            else:
                print(f"  ❌ {eval_pkg}: {stats['val']}/{stats['total']} 个场景在验证集")
    
    print(f"\n📁 输出文件:")
    print(f"  训练集场景列表: {train_file}")
    print(f"  验证集场景列表: {val_file}")
    print(f"  详细统计文件: {stats_file}")
    print(f"  CSV统计文件: {csv_file}")

if __name__ == "__main__":
    main()