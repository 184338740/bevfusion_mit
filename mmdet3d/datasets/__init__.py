from mmdet.datasets.builder import build_dataloader

from .builder import *
from .custom_3d import *
from .nuscenes_dataset import *
from .pipelines import *
from .utils import *

from .tyjt_dataset_v2 import TYJTDatasetV2

# 可选：如果要显式控制 __all__，必须保证名称完全匹配（推荐显式声明，避免隐式导出问题）
__all__ = [
    'build_dataloader',
    # 原有模块/类（保持不变）
    'Custom3DDataset',
    'NuScenesDataset',  # 从 nuscenes_dataset.py 导入的类
    # 新增 TYJT 数据集（名称必须是 TYJTDataset，与类名一致）
    'TYJTDatasetV2',
]