#!/usr/bin/env python3
"""
BEV可视化工具主程序
版本: v1.0.5
使用: python main.py --help 查看使用说明
"""

import sys
from pathlib import Path
from datetime import datetime

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from config import parse_args, create_config_from_args
from data_loader import DataLoader
from geometry import CalibrationManager
from visualizer import PointCloudVisualizer, LabelVisualizer, BEVVisualizer

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建配置
    config = create_config_from_args(args)
    print(f"BEV可视化工具 v1.0.5")
    print(f"数据根目录: {config.data_root}")
    print(f"输出目录: {config.output_root}")
    print(f"处理时间戳: {config.sample_timestamps[0]}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_root) / config.task_name / timestamp
    
    # 保持与之前版本相同的文件夹结构
    pointcloud_dir = output_dir / "pointcloud_projection"
    label_dir = output_dir / "3d_label_projection" 
    bev_fusion_dir = output_dir / "bev_fusion_visualization"
    
    try:
        # 初始化组件
        data_loader = DataLoader(config)
        calib_data = data_loader.load_calibration()
        calib_manager = CalibrationManager(calib_data)
        
        # 初始化可视化器
        pointcloud_viz = PointCloudVisualizer(data_loader, calib_manager, pointcloud_dir)
        label_viz = LabelVisualizer(data_loader, calib_manager, label_dir)
        bev_viz = BEVVisualizer(data_loader, calib_manager, bev_fusion_dir)
        
        # 运行可视化
        print(f"\n开始处理 {len(config.sample_timestamps)} 个样本")
        
        for i, timestamp_str in enumerate(config.sample_timestamps):
            print(f"\n[{i+1}/{len(config.sample_timestamps)}] 处理: {timestamp_str}")
            
            # 点云投影可视化
            pointcloud_viz.create_visualization(timestamp_str)
            
            # 3D标签投影可视化  
            label_viz.create_visualization(timestamp_str)
            
            # BEV融合可视化
            bev_viz.create_visualization(timestamp_str)
        
        print(f"\n完成! 结果保存在: {output_dir}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()