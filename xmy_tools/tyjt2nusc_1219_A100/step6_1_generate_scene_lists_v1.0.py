#!/usr/bin/env python3
"""
generate_scene_lists_actual.py - 修复版
根据实际scene.json格式生成场景列表，修正包名匹配逻辑
"""

import json
from pathlib import Path
import argparse

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
    parser.add_argument('--output-dir', type=str, default='./scene_lists_v02_fixed',
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
    
    # 3. 详细统计
    print(f"\n📊 场景划分统计（修复后）:")
    print(f"  总场景数: {len(scenes)}")
    print(f"  训练集场景: {len(train_scenes)}")
    print(f"  验证集场景: {len(val_scenes)}")
    
    # 按包名统计（使用正确的提取方法）
    print("\n📦 按包统计（修复后）:")
    package_stats = {}
    
    for scene in scenes:
        scene_name = scene['name']
        package = extract_package_name(scene_name)
        
        if package not in package_stats:
            package_stats[package] = {'total': 0, 'train': 0, 'val': 0}
        
        package_stats[package]['total'] += 1
        
        if is_eval_scene(scene_name):
            package_stats[package]['val'] += 1
        else:
            package_stats[package]['train'] += 1
    
    for package in sorted(package_stats.keys()):
        stats = package_stats[package]
        status = "🔵 验证集" if package in EVAL_PACKAGES else "🟢 训练集"
        print(f"  {package}: {stats['total']} 场景 ({stats['train']}训练/{stats['val']}验证) {status}")
    
    # 4. 特别检查评测数据包的场景
    print("\n🔍 评测数据包详细检查:")
    for eval_pkg in sorted(EVAL_PACKAGES):
        # 找出所有属于这个包但不在验证集的场景
        missing_scenes = []
        for scene in scenes:
            scene_name = scene['name']
            if eval_pkg in scene_name and scene_name not in val_scenes:
                missing_scenes.append(scene_name)
        
        if missing_scenes:
            print(f"  ❌ {eval_pkg}: 有 {len(missing_scenes)} 个场景被错误分配")
            print(f"     示例: {missing_scenes[0][:50]}...")
        else:
            # 统计正确分配的
            correct_count = sum(1 for s in val_scenes if eval_pkg in s)
            print(f"  ✅ {eval_pkg}: {correct_count} 个场景正确分配到了验证集")
    
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
    
    # 6. 验证完整性
    print(f"\n✅ 验证完整性:")
    print(f"  训练集文件: {train_file}")
    print(f"  验证集文件: {val_file}")
    
    # 快速检查
    with open(val_file, 'r') as f:
        val_lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # 检查验证集中是否包含两个评测数据集
    has_2d3d_20250114 = any("2d3d_20250114" in line for line in val_lines)
    has_2d3d4d_20250728_weiyuan = any("2d3d4d_20250728_weiyuan" in line for line in val_lines)
    
    print(f"  验证集包含 2d3d_20250114: {'✅' if has_2d3d_20250114 else '❌'}")
    print(f"  验证集包含 2d3d4d_20250728_weiyuan: {'✅' if has_2d3d4d_20250728_weiyuan else '❌'}")

if __name__ == "__main__":
    main()