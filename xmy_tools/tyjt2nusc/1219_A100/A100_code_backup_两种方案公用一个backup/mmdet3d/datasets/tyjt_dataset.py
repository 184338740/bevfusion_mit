import tempfile
from os import path as osp
from typing import Any, Dict

import mmcv
import numpy as np
import pyquaternion
import torch
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion

from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
from mmdet3d.datasets.nuscenes_dataset import output_to_nusc_box, lidar_nusc_box_to_global


@DATASETS.register_module('TYJTDataset')  # 注册 TYJTDataset
class TYJTDataset(NuScenesDataset):
    """TYJT数据集类，继承自NuScenesDataset
    
    使用最终10个nuScenes标准类别训练：
    car, truck, bus, pedestrian, motorcycle, bicycle, 
    construction_vehicle, trailer, traffic_cone, barrier
    """
    
    # 最终训练类别（10类，与nuScenes完全一致）
    CLASSES = (
        "car",
        "truck",
        "bus",
        "pedestrian",
        "motorcycle",
        "bicycle",
        "construction_vehicle",
        "trailer",
        "traffic_cone",
        "barrier",
    )
    
    # TYJT到nuScenes的类别映射（15→10）
    TYJT_CATEGORY_MAPPING = {
        # ============ 车辆类 (6类 -> 4类) ============
        "tyjt.car": "car",                    # car + van -> car
        "tyjt.van": "car",                    
        "tyjt.truck": "truck",                # truck + construction_truck -> truck  
        "tyjt.construction_truck": "truck",
        "tyjt.bus": "bus",                    # bus
        "tyjt.robot": "construction_vehicle", # robot -> construction_vehicle
        # ============ 行人及骑行者 (4类 -> 3类) ============
        "tyjt.pedestrian": "pedestrian",      # pedestrian
        "tyjt.cyclist": "pedestrian",         # cyclist -> pedestrian
        "tyjt.tricyclist": "motorcycle",      # tricyclist -> motorcycle
        "tyjt.bicycle": "bicycle",            # bicycle
        "tyjt.tricycle": "motorcycle",        # tricycle -> motorcycle
        # ============ 特殊物体 (1类 -> 1类) ============
        "tyjt.trolley": "trailer",            # trolley -> trailer
        # ============ 道路设施 (3类 -> 2类) ============
        "tyjt.cone": "traffic_cone",          # cone -> traffic_cone
        "tyjt.barrier": "barrier",            # barrier
        "tyjt.other": "barrier",              # other -> barrier
    }
    
    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
    ) -> None:
        """初始化TYJT数据集"""
        # 如果用户未指定object_classes，使用默认的CLASSES
        if object_classes is None:
            object_classes = self.CLASSES
        
        # 设置默认modality
        if modality is None:
            modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        
        # 调用父类初始化
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            dataset_root=dataset_root,
            object_classes=object_classes,
            map_classes=map_classes,
            load_interval=load_interval,
            with_velocity=with_velocity,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            eval_version=eval_version,
            use_valid_flag=use_valid_flag,
        )
        
        print(f"TYJT Dataset initialized:")
        print(f"  - Classes: {self.CLASSES}")
        print(f"  - Dataset Root: {dataset_root}")
        print(f"  - Ann File: {ann_file}")
        print(f"  - Test Mode: {test_mode}")
        print(f"  - Data Infos length: {len(self.data_infos) if hasattr(self, 'data_infos') else 'N/A'}")
    
    # ============ 修复：添加PyTorch Dataset必需的方法 ============
    def __len__(self):
        """返回数据集长度 - PyTorch Dataset必需"""
        if hasattr(self, 'data_infos'):
            return len(self.data_infos)
        return 0
    
    def __getitem__(self, idx):
        """获取数据项 - PyTorch Dataset必需"""
        return self.get_data_info(idx)
    
    def pre_pipeline(self, results):
        """预处理pipeline - 复用父类方法"""
        if hasattr(super(), 'pre_pipeline') and callable(getattr(super(), 'pre_pipeline')):
            super().pre_pipeline(results)
        else:
            # 基本预处理
            results['box_type_3d'] = self.box_type_3d
            results['box_mode_3d'] = self.box_mode_3d
    
    # ============ 修复get_ann_info方法 ============
    def get_ann_info(self, index):
        """获取标注信息"""
        info = self.data_infos[index]
        
        # 处理test模式或无标注情况
        if self.test_mode or 'gt_names' not in info or len(info.get('gt_names', [])) == 0:
            # 返回空的LiDARInstance3DBoxes对象
            empty_boxes = LiDARInstance3DBoxes(
                np.zeros((0, 7), dtype=np.float32),
                box_dim=7,
                origin=(0.5, 0.5, 0)
            ).convert_to(self.box_mode_3d)
            
            return dict(
                gt_bboxes_3d=empty_boxes,
                gt_labels_3d=np.zeros((0,), dtype=np.int64),
                gt_names=np.array([], dtype=str)
            )
        
        # 训练/验证模式：过滤无效标注
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        
        # 验证类别名称是否在最终类别中
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        
        gt_labels_3d = np.array(gt_labels_3d)
        
        # 处理速度信息
        if self.with_velocity and 'gt_velocity' in info:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        
        # 转换box格式 - 确保总是创建LiDARInstance3DBoxes对象
        box_dim = gt_bboxes_3d.shape[-1] if gt_bboxes_3d.shape[0] > 0 else 7
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, 
            box_dim=box_dim, 
            origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)
        
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        return anns_results