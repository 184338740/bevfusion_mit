"""
BEV可视化工具配置模块
版本: v1.0.5
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import argparse

@dataclass
class BEVConfig:
    """BEV可视化配置类"""
    
    # 数据路径配置
    data_root: str = "/home/tyjt/ws/02_docker/cv2-tests/dataset/2d3d4d_20250728_weiyuan"
    calib_file: str = "sensor2map_calib.json"
    dataset_name: str = "G51102400001M00_20250728191726_2.0HZ"
    image_folder: str = "image_dc"
    
    # 输出配置
    output_root: str = "output"
    task_name: str = "03_visualization"
    
    # 可视化参数
    bev_range: float = 75.0
    point_cloud_max_points: int = 50000
    point_size: int = 3
    point_alpha: float = 0.8
    fov_range: float = 150.0
    fov_angle: float = 60.0
    
    # 数据配置
    sample_timestamps: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.sample_timestamps is None:
            self.sample_timestamps = ["1753701446899737835"]
    
    @property
    def camera_config(self) -> Dict[str, str]:
        """相机配置映射"""
        return {
            'SC_R1_Aw_CpcS_CAMR': 'SC_1A_CamR',
            'SC_R1_Bn_CpcW_CAMR': 'SC_1B_CamR',
            'SC_R1_Ce_CpcN_CAMR': 'SC_1C_CamR', 
            'SC_R1_Ds_CpcE_CAMR': 'SC_1D_CamR'
        }
    
    @property
    def camera_colors(self) -> Dict[str, str]:
        """相机颜色配置"""
        return {
            'SC_R1_Aw_CpcS_CAMR': 'red',
            'SC_R1_Bn_CpcW_CAMR': 'green',  
            'SC_R1_Ce_CpcN_CAMR': 'blue',
            'SC_R1_Ds_CpcE_CAMR': 'orange'
        }
    
    @property
    def camera_labels(self) -> Dict[str, str]:
        """相机标签"""
        return {
            'SC_R1_Aw_CpcS_CAMR': 'Cam A',
            'SC_R1_Bn_CpcW_CAMR': 'Cam B',  
            'SC_R1_Ce_CpcN_CAMR': 'Cam C',
            'SC_R1_Ds_CpcE_CAMR': 'Cam D'
        }

# 标签颜色配置
LABEL_COLORS = {
    'car': (0, 255, 0),        # 绿色 - BGR格式
    'truck': (255, 0, 0),      # 红色
    'cyclist': (0, 0, 255),    # 蓝色
    'pedestrian': (255, 255, 0),  # 黄色
    'default': (255, 255, 255)    # 白色
}

BEV_LABEL_COLORS = {
    'car': 'red',
    'truck': 'blue', 
    'cyclist': 'green',
    'pedestrian': 'orange',
    'default': 'black'
}

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="BEV可视化工具 v1.0.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py  # 使用默认配置
  python main.py --data_root /path/to/data --timestamp 123456789
  python main.py --help  # 查看帮助信息
        """
    )
    
    parser.add_argument('--data_root', type=str, 
                       default="/home/tyjt/ws/02_docker/cv2-tests/dataset/2d3d4d_20250728_weiyuan",
                       help='数据集根目录路径')
    
    parser.add_argument('--calib_file', type=str, default="sensor2map_calib.json",
                       help='标定文件名')
    
    parser.add_argument('--timestamp', type=str, 
                       default="1753701446899737835",
                       help='要处理的时间戳')
    
    parser.add_argument('--output_root', type=str, default="output",
                       help='输出目录根路径')
    
    parser.add_argument('--task_name', type=str, default="03_visualization",
                       help='任务名称')
    
    parser.add_argument('--bev_range', type=float, default=75.0,
                       help='BEV视图范围(米)')
    
    parser.add_argument('--max_points', type=int, default=50000,
                       help='点云投影最大点数')
    
    return parser.parse_args()

def create_config_from_args(args) -> BEVConfig:
    """从命令行参数创建配置"""
    return BEVConfig(
        data_root=args.data_root,
        calib_file=args.calib_file,
        output_root=args.output_root,
        task_name=args.task_name,
        bev_range=args.bev_range,
        point_cloud_max_points=args.max_points,
        sample_timestamps=[args.timestamp]
    )