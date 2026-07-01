#!/usr/bin/env python3
"""
TYJT数据集统计分析工具
优化版本：类封装，支持多种统计维度
"""

import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

class TYJTDataAnalyzer:
    """TYJT数据集统计分析器"""
    
    def __init__(self, data_root: str = "/mnt/dataset/tyjt_RawData"):
        self.data_root = Path(data_root)
        self.stats = {}
        
    def analyze_all(self) -> Dict:
        """执行完整的数据分析"""
        print("📊 TYJT数据集统计分析")
        print("=" * 60)
        
        # 执行各项分析
        package_stats = self._analyze_packages()
        sample_stats = self._analyze_samples()
        camera_stats = self._analyze_cameras()
        
        # 汇总统计
        self.stats = {
            'overview': {
                'total_packages': len(package_stats['valid_packages']),
                'total_sub_packages': package_stats['total_sub_packages'],
                'total_samples': sample_stats['total_samples'],
                'total_lidar_files': sample_stats['total_lidar_files'],
                'total_cameras': len(camera_stats['camera_summary']),
                'total_images': camera_stats['total_images']
            },
            'packages': package_stats,
            'samples': sample_stats,
            'cameras': camera_stats
        }
        
        self._print_summary()
        return self.stats
    
    def _analyze_packages(self) -> Dict:
        """分析数据包结构"""
        print("\n🔍 分析数据包结构...")
        
        packages = [p for p in self.data_root.iterdir() if p.is_dir() and not p.name.startswith('.')]
        valid_packages = []
        package_details = {}
        
        for package in packages:
            package_info = {
                'name': package.name,
                'has_calib': (package / "calib").exists(),
                'has_datasets': (package / "datasets").exists(),
                'sub_packages': []
            }
            
            datasets_dir = package / "datasets"
            if datasets_dir.exists():
                sub_packages = [d for d in datasets_dir.iterdir() if d.is_dir()]
                package_info['sub_packages'] = [sp.name for sp in sub_packages]
                package_info['sub_package_count'] = len(sub_packages)
                
                if sub_packages:
                    valid_packages.append(package.name)
                    package_details[package.name] = package_info
        
        total_sub_packages = sum(info['sub_package_count'] for info in package_details.values())
        
        return {
            'all_packages': [p.name for p in packages],
            'valid_packages': valid_packages,
            'package_details': package_details,
            'total_sub_packages': total_sub_packages
        }
    
    def _analyze_samples(self) -> Dict:
        """分析样本数据"""
        print("🔍 分析样本数据...")
        
        sample_stats = {
            'total_samples': 0,
            'total_lidar_files': 0,
            'package_samples': defaultdict(int),
            'subpackage_samples': defaultdict(int)
        }
        
        packages = [p for p in self.data_root.iterdir() if p.is_dir() and not p.name.startswith('.')]
        
        for package in packages:
            datasets_dir = package / "datasets"
            if not datasets_dir.exists():
                continue
                
            sub_packages = [d for d in datasets_dir.iterdir() if d.is_dir()]
            
            for sub_pkg in sub_packages:
                lidar_dir = sub_pkg / "lidar" / "pcd"
                if lidar_dir.exists():
                    lidar_files = list(lidar_dir.glob("*.npy"))
                    sample_count = len(lidar_files)
                    
                    sample_stats['total_samples'] += sample_count
                    sample_stats['total_lidar_files'] += sample_count
                    sample_stats['package_samples'][package.name] += sample_count
                    sample_stats['subpackage_samples'][f"{package.name}/{sub_pkg.name}"] = sample_count
        
        return sample_stats
    
    def _analyze_cameras(self) -> Dict:
        """分析相机数据"""
        print("🔍 分析相机数据...")
        
        camera_stats = {
            'camera_summary': defaultdict(int),
            'camera_details': defaultdict(dict),
            'total_images': 0
        }
        
        packages = [p for p in self.data_root.iterdir() if p.is_dir() and not p.name.startswith('.')]
        
        for package in packages:
            datasets_dir = package / "datasets"
            if not datasets_dir.exists():
                continue
                
            # 取第一个子包进行分析（假设同一数据包内相机配置相同）
            sub_packages = [d for d in datasets_dir.iterdir() if d.is_dir()]
            if not sub_packages:
                continue
                
            sample_sub_pkg = sub_packages[0]
            sc_dirs = [d for d in sample_sub_pkg.iterdir() if d.is_dir() and d.name.startswith('SC_')]
            
            for sc_dir in sc_dirs:
                if sc_dir.name.endswith('CamR'):  # 只分析CamR相机
                    image_dc_dir = sc_dir / "image_dc"
                    if image_dc_dir.exists():
                        images = list(image_dc_dir.glob("*.jpg"))
                        image_count = len(images)
                        
                        camera_stats['camera_summary'][sc_dir.name] += image_count
                        camera_stats['camera_details'][sc_dir.name] = {
                            'image_count': image_count,
                            'package': package.name,
                            'has_image_dc': True
                        }
                        camera_stats['total_images'] += image_count
                    else:
                        camera_stats['camera_details'][sc_dir.name] = {
                            'image_count': 0,
                            'package': package.name,
                            'has_image_dc': False
                        }
        
        return camera_stats
    
    def _print_summary(self):
        """打印统计摘要"""
        overview = self.stats['overview']
        packages = self.stats['packages']
        samples = self.stats['samples']
        cameras = self.stats['cameras']
        
        print("\n" + "=" * 60)
        print("📈 数据统计摘要")
        print("=" * 60)
        
        print(f"\n📦 数据包统计:")
        print(f"  总数据包: {overview['total_packages']} 个")
        print(f"  有效数据包: {len(packages['valid_packages'])} 个")
        print(f"  总子数据包: {overview['total_sub_packages']} 个")
        
        print(f"\n🗂️  样本统计:")
        print(f"  总样本数: {overview['total_samples']} 个")
        print(f"  总LiDAR文件: {overview['total_lidar_files']} 个")
        
        print(f"\n📷 相机统计:")
        print(f"  总相机数: {overview['total_cameras']} 个")
        print(f"  总图像数: {overview['total_images']} 张")
        
        # 按数据包显示样本分布
        print(f"\n📊 数据包样本分布:")
        for pkg_name, sample_count in samples['package_samples'].items():
            print(f"  {pkg_name}: {sample_count} 样本")
        
        # 相机详情
        print(f"\n📸 相机详情:")
        for cam_name, details in cameras['camera_details'].items():
            status = "✅" if details['has_image_dc'] and details.get('image_count', 0) > 0 else "❌"
            count = details.get('image_count', 0)
            print(f"  {status} {cam_name}: {count} 图像 ({details['package']})")
    
    def export_statistics(self, output_file: str = "tyjt_statistics.json"):
        """导出统计结果到JSON文件"""
        import json
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 统计结果已导出到: {output_path}")
    
    def get_data_quality_report(self) -> Dict:
        """生成数据质量报告"""
        if not self.stats:
            self.analyze_all()
        
        quality_report = {
            'data_completeness': {},
            'potential_issues': [],
            'recommendations': []
        }
        
        overview = self.stats['overview']
        packages = self.stats['packages']
        cameras = self.stats['cameras']
        
        # 数据完整性评估
        total_potential_samples = overview['total_sub_packages'] * 100  # 假设每个子包约100样本
        completeness_ratio = overview['total_samples'] / total_potential_samples if total_potential_samples > 0 else 0
        
        quality_report['data_completeness'] = {
            'sample_completeness': f"{completeness_ratio:.1%}",
            'camera_coverage': f"{len(cameras['camera_summary']) / (len(packages['valid_packages']) * 4):.1%}",
            'valid_package_ratio': f"{len(packages['valid_packages']) / len(packages['all_packages']):.1%}"
        }
        
        # 潜在问题检测
        for cam_name, details in cameras['camera_details'].items():
            if not details['has_image_dc'] or details.get('image_count', 0) == 0:
                quality_report['potential_issues'].append(f"相机 {cam_name} 无有效图像数据")
        
        if overview['total_samples'] == 0:
            quality_report['potential_issues'].append("未找到任何样本数据")
        
        # 建议
        if completeness_ratio < 0.8:
            quality_report['recommendations'].append("建议检查数据完整性，部分样本可能缺失")
        
        if len(quality_report['potential_issues']) > 0:
            quality_report['recommendations'].append("建议修复标识的问题后再进行转换")
        
        return quality_report

def main():
    """主函数"""
    analyzer = TYJTDataAnalyzer()
    
    # 执行完整分析
    stats = analyzer.analyze_all()
    
    # 生成质量报告
    quality_report = analyzer.get_data_quality_report()
    
    print("\n" + "=" * 60)
    print("🔍 数据质量报告")
    print("=" * 60)
    
    print(f"\n📋 数据完整性:")
    for metric, value in quality_report['data_completeness'].items():
        print(f"  {metric}: {value}")
    
    if quality_report['potential_issues']:
        print(f"\n⚠️  潜在问题:")
        for issue in quality_report['potential_issues']:
            print(f"  • {issue}")
    
    if quality_report['recommendations']:
        print(f"\n💡 建议:")
        for recommendation in quality_report['recommendations']:
            print(f"  • {recommendation}")
    
    # 导出统计结果
    analyzer.export_statistics()

if __name__ == "__main__":
    main()