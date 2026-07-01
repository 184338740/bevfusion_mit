import pyquaternion
import numpy as np
import mmcv
from mmdet3d.datasets import NuScenesDataset
from mmdet.datasets import DATASETS


@DATASETS.register_module()
class TYJTDataset(NuScenesDataset):
    METAINFO = {
        'classes': ('car', 'truck', 'construction_vehicle', 'bus', 
                   'trailer', 'barrier', 'motorcycle', 'bicycle', 
                   'pedestrian', 'traffic_cone'),
        'palette': [(255, 158, 0), (255, 99, 71), (255, 140, 0),
                   (255, 127, 80), (233, 150, 70), (169, 169, 169),
                   (0, 0, 230), (119, 11, 32), (0, 0, 142), (0, 0, 70)]
    }
    
    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 use_valid_flag=True,
                 **kwargs):
        
        # 设置默认modality
        if modality is None:
            modality = dict(
                use_camera=True,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        
        super().__init__(
            dataset_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            use_valid_flag=use_valid_flag,
            **kwargs
        )
    
    def get_cat_ids(self, idx):
        """修复：处理valid_flag缺失的问题"""
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"] if "valid_flag" in info else np.ones(len(info["gt_names"]), dtype=bool)
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = [self.cat2id[name] for name in gt_names if name in self.CLASSES]
        return cat_ids
    
    def get_ann_info(self, index):
        """修复：处理valid_flag和num_lidar_pts缺失的问题"""
        info = self.data_infos[index]
        # 处理mask逻辑
        if self.use_valid_flag:
            mask = info["valid_flag"] if "valid_flag" in info else np.ones(len(info["gt_names"]), dtype=bool)
        else:
            mask = info["num_lidar_pts"] > 0 if "num_lidar_pts" in info else np.ones(len(info["gt_names"]), dtype=bool)
                
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = np.array([self.CLASSES.index(cat) if cat in self.CLASSES else -1 for cat in gt_names_3d])

        # 处理速度信息
        if self.with_velocity:
            if "gt_velocity" in info:
                gt_velocity = info["gt_velocity"][mask]
                gt_velocity[np.isnan(gt_velocity[:, 0])] = [0.0, 0.0]
            else:
                gt_velocity = np.zeros((len(gt_bboxes_3d), 2))
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # 转换bbox格式
        from ..core.bbox import LiDARInstance3DBoxes
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        return dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
    
    def quaternion_to_rotation_matrix(self, quaternion):
        """将四元数 [w, x, y, z] 转换为3x3旋转矩阵"""
        quaternion = np.array(quaternion, dtype=np.float32)
        return pyquaternion.Quaternion(quaternion).rotation_matrix
    
    def get_data_info(self, index: int):
        """获取数据信息 - 适配PKL中的四元数格式"""
        info = self.data_infos[index]
        
        # 基础数据信息
        data = dict(
            token=info["token"],
            sample_idx=info['token'],
            lidar_path=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"],
        )
        
        # 1. Ego to Global 变换（直接使用四元数）
        ego2global = np.eye(4).astype(np.float32)
        ego2global[:3, :3] = self.quaternion_to_rotation_matrix(info["ego2global_rotation"])
        ego2global[:3, 3] = info["ego2global_translation"]
        data["ego2global"] = ego2global
        
        # 2. LiDAR to Ego 变换（直接使用四元数）
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = self.quaternion_to_rotation_matrix(info["lidar2ego_rotation"])
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        data["lidar2ego"] = lidar2ego
        
        # 3. 相机数据处理
        if self.modality["use_camera"] and "cams" in info:
            data.update({
                "image_paths": [],
                "lidar2camera": [],
                "lidar2image": [],
                "camera2ego": [],
                "camera_intrinsics": [],
                "camera2lidar": []
            })
            
            for cam_name, camera_info in info["cams"].items():
                data["image_paths"].append(camera_info["data_path"])
                
                # 相机内参
                cam_intrinsic = np.eye(4).astype(np.float32)
                cam_intrinsic[:3, :3] = camera_info["cam_intrinsic"]
                data["camera_intrinsics"].append(cam_intrinsic)
                
                # Camera to Ego 变换（四元数直接转换）
                cam2ego = np.eye(4).astype(np.float32)
                cam2ego[:3, :3] = self.quaternion_to_rotation_matrix(camera_info["sensor2ego_rotation"])
                cam2ego[:3, 3] = camera_info["sensor2ego_translation"]
                data["camera2ego"].append(cam2ego)
                
                # Camera to LiDAR 变换（兼容四元数/矩阵输入）
                cam2lidar = np.eye(4).astype(np.float32)
                sensor2lidar_rot = camera_info["sensor2lidar_rotation"]
                if len(sensor2lidar_rot) == 4:
                    # 输入为四元数
                    cam2lidar[:3, :3] = self.quaternion_to_rotation_matrix(sensor2lidar_rot)
                else:
                    # 输入为矩阵
                    cam2lidar[:3, :3] = np.array(sensor2lidar_rot, dtype=np.float32)
                cam2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
                data["camera2lidar"].append(cam2lidar)
                
                # LiDAR to Camera 变换（cam2lidar的逆）
                lidar2camera = np.linalg.inv(cam2lidar)
                data["lidar2camera"].append(lidar2camera)
                
                # LiDAR to Image 变换（内参 × lidar2camera）
                lidar2image = cam_intrinsic @ lidar2camera
                data["lidar2image"].append(lidar2image)
        
        # 添加标注信息
        data["ann_info"] = self.get_ann_info(index)
        return data