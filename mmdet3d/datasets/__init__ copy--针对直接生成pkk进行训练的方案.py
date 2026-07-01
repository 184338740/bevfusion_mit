from mmdet.datasets.builder import build_dataloader

from .builder import *
from .custom_3d import *
from .nuscenes_dataset import *
from .pipelines import *
from .utils import *


from .tyjt_dataset import TYJTDataset

__all__ = ['TYJTDataset']   # 在单独running 直接

# # 导入注册表
# from .builder import DATASETS

# # 🟡 临时修改：检查是否已注册，避免重复注册
# try:
#     from .tyjt_dataset import TYJTDataset
#     if 'TYJTDataset' not in DATASETS._module_dict:
#         DATASETS.register_module(name='TYJTDataset', module=TYJTDataset)
#         print('✅ TYJTDataset 注册成功')
#     else:
#         print('✅ TYJTDataset 已注册，跳过重复注册')
# except ImportError as e:
#     print(f'❌ 导入TYJTDataset失败: {e}')

# __all__ = [
#     # 其他已有的类...
# ]
