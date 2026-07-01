#!/usr/bin/env python3
"""
统计tyjt数据集各文件夹样本数量
"""

import os
from pathlib import Path
from collections import defaultdict

def count_samples_statistics():
    """统计各文件夹样本数量"""
    tyjt_root = "/mnt/dataset/tyjt_RawData"
    root_path = Path(tyjt_root)
    
    print("📊 TYJT数据集样本统计")
    print("=" * 60)
    
    total_stats = {
        'packages': 0,
        'sub_packages': 0,
        'samples': 0,
        'cameras': defaultdict(int),
        'lidar_files': 0
    }
    
    packages = [p for p in root_path.iterdir() if p.is_dir() and not p.name.startswith('.')]
    total_stats['packages'] = len(packages)
    
    for package in packages:
        print(f"\n📦 数据包: {package.name}")
        package_stats = {
            'sub_packages': 0,
            'samples': 0,
            'cameras': defaultdict(int),
            'lidar_files': 0
        }
        
        # 检查datasets目录
        datasets_dir = package / "datasets"
        if not datasets_dir.exists():
            print("  ❌ 无datasets目录")
            continue
            
        # 统计子数据包
        sub_packages = [d for d in datasets_dir.iterdir() if d.is_dir()]
        package_stats['sub_packages'] = len(sub_packages)
        total_stats['sub_packages'] += len(sub_packages)
        
        print(f"  📁 子数据包: {len(sub_packages)} 个")
        
        for sub_pkg in sub_packages:
            print(f"    🗂️  子包: {sub_pkg.name}")
            
            # 统计LiDAR文件
            lidar_dir = sub_pkg / "lidar" / "pcd"
            if lidar_dir.exists():
                lidar_files = list(lidar_dir.glob("*.npy"))
                package_stats['lidar_files'] += len(lidar_files)
                total_stats['lidar_files'] += len(lidar_files)
                print(f"      🗂️  LiDAR文件: {len(lidar_files)} 个")
                
                # 以LiDAR文件数为样本数
                package_stats['samples'] += len(lidar_files)
                total_stats['samples'] += len(lidar_files)
            else:
                print(f"      ❌ 无LiDAR目录")
            
            # 统计相机
            sc_dirs = [d for d in sub_pkg.iterdir() if d.is_dir() and d.name.startswith('SC_')]
            print(f"      🗂️  相机文件夹: {len(sc_dirs)} 个")
            
            for sc_dir in sc_dirs:
                # 只统计CamR相机
                if sc_dir.name.endswith('CamR'):
                    image_dc_dir = sc_dir / "image_dc"
                    if image_dc_dir.exists():
                        images = list(image_dc_dir.glob("*.jpg"))
                        package_stats['cameras'][sc_dir.name] = len(images)
                        total_stats['cameras'][sc_dir.name] += len(images)
                        print(f"        ✅ {sc_dir.name}: {len(images)} 图像")
                    else:
                        print(f"        ❌ {sc_dir.name}: 无image_dc目录")
                else:
                    print(f"        ⚠️  {sc_dir.name}: 非CamR相机，跳过")
        
        # 打印包统计
        print(f"  📊 包统计:")
        print(f"    子包: {package_stats['sub_packages']} 个")
        print(f"    样本: {package_stats['samples']} 个") 
        print(f"    LiDAR文件: {package_stats['lidar_files']} 个")
        print(f"    相机统计:")
        for cam, count in package_stats['cameras'].items():
            print(f"      {cam}: {count} 图像")
    
    # 打印总统计
    print("\n" + "=" * 60)
    print("📊 总统计:")
    print(f"数据包: {total_stats['packages']} 个")
    print(f"子数据包: {total_stats['sub_packages']} 个")
    print(f"总样本数: {total_stats['samples']} 个")
    print(f"总LiDAR文件: {total_stats['lidar_files']} 个")
    print(f"相机统计:")
    for cam, count in total_stats['cameras'].items():
        print(f"  {cam}: {count} 图像")

if __name__ == "__main__":
    count_samples_statistics()