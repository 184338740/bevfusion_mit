import sys
from pathlib import Path
from datetime import datetime

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

import config
from data_loader import DataLoader
from transformer import Transformer
from projection_visualizer import ProjectionVisualizer
from visualizer import BEVVisualizer

def main():
    # 配置区域 - 与v1.0.3保持一致
    viz_config = config.Config()
    
    # 创建输出目录 - 保持与v1.0.3相同的文件夹结构
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(viz_config.output_root) / viz_config.task_name / timestamp
    
    # 保持与v1.0.3相同的文件夹名称
    pointcloud_dir = output_dir / "pointcloud_projection"
    label_dir = output_dir / "3d_label_projection" 
    bev_fusion_dir = output_dir / "bev_fusion_visualization"
    
    # 初始化组件
    data_loader = DataLoader(viz_config)
    calib_data = data_loader.load_calibration()
    transformer = Transformer(calib_data)
    
    # 初始化可视化器
    projection_viz = ProjectionVisualizer(data_loader, transformer, pointcloud_dir)
    label_viz = ProjectionVisualizer(data_loader, transformer, label_dir)
    bev_viz = BEVVisualizer(data_loader, transformer, bev_fusion_dir)
    
    # 测试时间戳
    sample_timestamps = ["1753701446899737835"]
    
    # 运行可视化 - 保持与v1.0.3相同的调用方式
    print(f"开始处理 {len(sample_timestamps)} 个样本")
    
    for i, timestamp in enumerate(sample_timestamps):
        print(f"\n[{i+1}/{len(sample_timestamps)}] 处理: {timestamp}")
        
        # 点云投影可视化
        projection_viz.create_pointcloud_projection(timestamp)
        
        # 3D标签投影可视化  
        label_viz.create_label_projection(timestamp)
        
        # BEV融合可视化
        bev_viz.create_bev_visualization(timestamp)
    
    print(f"\n完成! 结果保存在: {output_dir}")

if __name__ == "__main__":
    main()