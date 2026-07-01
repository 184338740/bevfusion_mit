import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class DataStatistics:
    def __init__(self, data_root: str, output_root: str = "output"):
        self.data_root = Path(data_root).expanduser()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_root) / "02_DataStatistics" / current_time
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_path = self.data_root / "datasets" / "G51102400001M00_20250728191726_2.0HZ"
        
        # 相机文件夹列表
        self.camera_folders = ['SC_1A_CamR', 'SC_1B_CamR', 'SC_1C_CamR', 'SC_1D_CamR']
        
        print(f"数据根目录: {self.data_root}")
        print(f"输出目录: {self.output_dir}")
    
    def load_labels(self, file_path: Path) -> List[Dict]:
        """加载3D标签数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            
            # 处理不同的标签格式
            if isinstance(labels, list):
                return labels
            elif isinstance(labels, dict):
                if 'objects' in labels:
                    return labels['objects']
                elif 'labels' in labels:
                    return labels['labels']
                else:
                    return []
            else:
                return []
                
        except Exception as e:
            print(f"加载标签文件 {file_path} 时出错: {e}")
            return []
    
    def run_statistics_analysis(self):
        """运行数据统计分析"""
        print("开始数据统计分析...")
        
        stats = {
            'camera_counts': {},
            'point_cloud_counts': 0,
            'label_counts': 0,
            'object_statistics': {
                'total_objects': 0,
                'type_distribution': {},
                'size_statistics': {}
            }
        }
        
        # 统计相机图像数量
        print("统计相机图像数量...")
        for cam_folder in self.camera_folders:
            image_dir = self.dataset_path / cam_folder / "image_dc"
            if image_dir.exists():
                image_count = len(list(image_dir.glob("*.jpg")))
                stats['camera_counts'][cam_folder] = image_count
                print(f"  {cam_folder}: {image_count} 张图像")
        
        # 统计点云数量
        print("统计点云数量...")
        lidar_dir = self.dataset_path / "lidar" / "pcd"
        if lidar_dir.exists():
            pcd_count = len(list(lidar_dir.glob("*.pcd")))
            npy_count = len(list(lidar_dir.glob("*.npy")))
            stats['point_cloud_counts'] = pcd_count
            print(f"  点云PCD文件: {pcd_count} 个")
            print(f"  点云NPY文件: {npy_count} 个")
        
        # 统计标签和对象信息
        print("统计标签和对象信息...")
        label_dir = self.dataset_path / "lidar" / "label"
        if label_dir.exists():
            label_files = list(label_dir.glob("*.json"))
            stats['label_counts'] = len(label_files)
            print(f"  标签文件: {len(label_files)} 个")
            
            all_objects = []
            sample_count = min(200, len(label_files))  # 限制样本数量
            
            for i, label_file in enumerate(label_files[:sample_count]):
                if i % 50 == 0:
                    print(f"  处理标签文件: {i}/{sample_count}")
                try:
                    labels = self.load_labels(label_file)
                    if isinstance(labels, list):
                        all_objects.extend(labels)
                except Exception as e:
                    continue
            
            if all_objects:
                stats['object_statistics']['total_objects'] = len(all_objects)
                print(f"  总物体数量: {len(all_objects)}")
                
                # 类型分布
                type_counts = {}
                for obj in all_objects:
                    obj_type = obj.get('type', 'unknown')
                    type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
                stats['object_statistics']['type_distribution'] = type_counts
                
                print("  物体类型分布:")
                for obj_type, count in type_counts.items():
                    print(f"    {obj_type}: {count}")
                
                # 尺寸统计
                sizes = {'length': [], 'width': [], 'height': []}
                positions = {'x': [], 'y': [], 'z': []}
                
                for obj in all_objects:
                    if 'l' in obj and 'w' in obj and 'h' in obj:
                        sizes['length'].append(obj['l'])
                        sizes['width'].append(obj['w'])
                        sizes['height'].append(obj['h'])
                    if 'x' in obj and 'y' in obj and 'z' in obj:
                        positions['x'].append(obj['x'])
                        positions['y'].append(obj['y'])
                        positions['z'].append(obj['z'])
                
                stats['object_statistics']['size_statistics'] = {
                    dim: {
                        'mean': float(np.mean(vals)) if vals else 0,
                        'std': float(np.std(vals)) if vals else 0,
                        'min': float(np.min(vals)) if vals else 0,
                        'max': float(np.max(vals)) if vals else 0,
                        'count': len(vals)
                    }
                    for dim, vals in sizes.items()
                }
                
                stats['object_statistics']['position_statistics'] = {
                    dim: {
                        'mean': float(np.mean(vals)) if vals else 0,
                        'std': float(np.std(vals)) if vals else 0,
                        'min': float(np.min(vals)) if vals else 0,
                        'max': float(np.max(vals)) if vals else 0,
                    }
                    for dim, vals in positions.items()
                }
                
                print("  物体尺寸统计:")
                for dim, dim_stats in stats['object_statistics']['size_statistics'].items():
                    print(f"    {dim}: 均值={dim_stats['mean']:.2f}m, 标准差={dim_stats['std']:.2f}m, 数量={dim_stats['count']}")
        
        # 保存统计结果
        with open(self.output_dir / "data_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # 生成可视化图表
        self.generate_statistics_plots(stats)
        
        print("数据统计分析完成！")
        print(f"输出目录: {self.output_dir}")
        
        return stats
    
    def generate_statistics_plots(self, stats):
        """生成统计图表"""
        try:
            import matplotlib.pyplot as plt
            
            # 物体类型分布饼图
            if stats['object_statistics']['type_distribution']:
                plt.figure(figsize=(10, 8))
                types = list(stats['object_statistics']['type_distribution'].keys())
                counts = list(stats['object_statistics']['type_distribution'].values())
                
                plt.pie(counts, labels=types, autopct='%1.1f%%', startangle=90)
                plt.title('Object Type Distribution')
                plt.savefig(self.output_dir / "object_type_distribution.png", 
                           bbox_inches='tight', dpi=150)
                plt.close()
            
            # 相机数据量柱状图
            if stats['camera_counts']:
                plt.figure(figsize=(10, 6))
                cameras = list(stats['camera_counts'].keys())
                counts = list(stats['camera_counts'].values())
                
                plt.bar(cameras, counts)
                plt.title('Camera Image Counts')
                plt.xlabel('Camera')
                plt.ylabel('Image Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(self.output_dir / "camera_image_counts.png", 
                           bbox_inches='tight', dpi=150)
                plt.close()
                
            print("统计图表生成完成！")
            
        except ImportError:
            print("Matplotlib未安装，跳过图表生成")

if __name__ == "__main__":
    data_path = "/home/tyjt/ws/02_docker/cv2-tests/dataset/2d3d4d_20250728_weiyuan"
    
    stats_analyzer = DataStatistics(data_path)
    stats = stats_analyzer.run_statistics_analysis()