from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Config:
    # 数据路径
    data_root: str = "/home/tyjt/ws/02_docker/cv2-tests/dataset/2d3d4d_20250728_weiyuan"
    calib_file: str = "sensor2map_calib.json"
    dataset_name: str = "G51102400001M00_20250728191726_2.0HZ"
    image_folder: str = "image_dc"
    
    # 输出配置
    output_root: str = "output"
    task_name: str = "03_visualization"
    
    # 相机配置
    camera_config: Dict[str, str] = None
    
    # 点云配置 - 确保配置正确传递
    point_cloud_config: Dict = None
    
    # 采样时间戳
    sample_timestamps: List[str] = None
    
    def __post_init__(self):
        if self.camera_config is None:
            self.camera_config = {
                'SC_R1_Aw_CpcS_CAMR': 'SC_1A_CamR',
                'SC_R1_Bn_CpcW_CAMR': 'SC_1B_CamR',
                'SC_R1_Ce_CpcN_CAMR': 'SC_1C_CamR', 
                'SC_R1_Ds_CpcE_CAMR': 'SC_1D_CamR'
            }
        
        if self.point_cloud_config is None:
            # 确保这些配置正确传递到点云投影函数
            self.point_cloud_config = {
                'max_points': 5000,  # 点云投影最大点数
                'point_size': 3,      # 点大小
                'alpha': 0.8,         # 透明度
            }
        
        if self.sample_timestamps is None:
            self.sample_timestamps = ["1753701446899737835"]

# 标签颜色配置 (保持与v1.0.3一致)
LABEL_COLORS = {
    'car': (0, 255, 0),      # 绿色 - BGR格式
    'truck': (255, 0, 0),    # 红色
    'cyclist': (0, 0, 255),  # 蓝色
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