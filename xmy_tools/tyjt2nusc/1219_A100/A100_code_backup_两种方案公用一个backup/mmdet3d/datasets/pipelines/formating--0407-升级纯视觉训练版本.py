# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet3d.core.points import BasePoints
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor

import torch

Debug = False

print(f">>>[xmy]🟡[mmdet3d/datasets/pipelines/formating.py] >>> [DataLoader侧]  >>> [Debug Mode = {Debug}] ")

if Debug:
    import time
    # 复用您已有的get_worker_info函数（来自loading.py）
    # 如果没有，则创建一个简单的
    import threading
    import os
    global_print_interval = 10
    def get_worker_info():
        """获取worker信息（简化版）"""
        try:
            import torch
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                return f"DL_Worker{worker_info.id}"
        except:
            pass
        
        # 备选方案
        thread_id = threading.get_ident()
        process_id = os.getpid()
        return f"P{process_id}_T{thread_id%1000:03d}"



@PIPELINES.register_module()
class DebugDataContainer:
    def __call__(self, data):
        print("\n>>>[xmy]🟡[formating.py]>>> === DebugDataContainer ===")
        for key, val in data.items():
            if isinstance(val, DC):
                print(f"Key: {key}")
                print(f"  data type: {type(val.data)}")
                if hasattr(val.data, 'shape'):
                    print(f"  data.shape: {val.data.shape}")
                print(f"  pad_dims: {val.pad_dims}")
                print(f"  stack: {val.stack}")
        return data


@PIPELINES.register_module()
class DebugPrintImg:
    def __call__(self, results):
        if 'img' in results:
            img = results['img']
            print(f">>>[xmy]🟡[formating.py]>>> DebugPrintImg: img type={type(img)}")
            if hasattr(img, 'data'):
                print(f"  img.data type={type(img.data)}")
                if isinstance(img.data, list):
                    print(f"    list length={len(img.data)}")
                    if len(img.data) > 0:
                        print(f"    first element type={type(img.data[0])}")
                        if isinstance(img.data[0], np.ndarray):
                            print(f"      dtype={img.data[0].dtype}, shape={img.data[0].shape}")
                        else:
                            print(f"      value={img.data[0]}")
                elif isinstance(img.data, np.ndarray):
                    print(f"    dtype={img.data.dtype}, shape={img.data.shape}")
            else:
                print("  img has no 'data' attribute")
        else:
            print(">>>[xmy]🟡[formating.py]>>> DebugPrintImg: img not in results")
        return results


@PIPELINES.register_module()
class DefaultFormatBundle3D:
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(
        self,
        classes,
        with_gt: bool = True,
        with_label: bool = True,
    ) -> None:
        super().__init__()
        self.class_names = classes
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if Debug:
            import time
            # 添加计时
            start_time = time.time()
            # 获取样本ID
            sample_id = 'unknown'
            for key in ['sample_idx', 'sample_id', 'token', 'frame_id']:
                if key in results:
                    sample_id = str(results[key])
                    break        
            print(f">>>[xmy]🟡[formating.py]>>> [DEBUG] results keys: {results.keys()}")


        # Format 3D data
        if "points" in results:
            assert isinstance(results["points"], BasePoints)
            results["points"] = DC(results["points"].tensor)

        # 【xmy-临时修改】
        # if "radar" in results:
        if 'radar' in results and results['radar'] is not None and hasattr(results['radar'], 'tensor'):
            results["radar"] = DC(results["radar"].tensor)

        for key in ["voxels", "coors", "voxel_centers", "num_points"]:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]), stack=False)

        if self.with_gt:
            # Clean GT bboxes in the final
            if "gt_bboxes_3d_mask" in results:
                gt_bboxes_3d_mask = results["gt_bboxes_3d_mask"]
                results["gt_bboxes_3d"] = results["gt_bboxes_3d"][gt_bboxes_3d_mask]
                if "gt_names_3d" in results:
                    results["gt_names_3d"] = results["gt_names_3d"][gt_bboxes_3d_mask]
                if "centers2d" in results:
                    results["centers2d"] = results["centers2d"][gt_bboxes_3d_mask]
                if "depths" in results:
                    results["depths"] = results["depths"][gt_bboxes_3d_mask]
            if "gt_bboxes_mask" in results:
                gt_bboxes_mask = results["gt_bboxes_mask"]
                if "gt_bboxes" in results:
                    results["gt_bboxes"] = results["gt_bboxes"][gt_bboxes_mask]
                results["gt_names"] = results["gt_names"][gt_bboxes_mask]
            if self.with_label:
                if "gt_names" in results and len(results["gt_names"]) == 0:
                    results["gt_labels"] = np.array([], dtype=np.int64)
                    results["attr_labels"] = np.array([], dtype=np.int64)
                elif "gt_names" in results and isinstance(results["gt_names"][0], list):
                    # gt_labels might be a list of list in multi-view setting
                    results["gt_labels"] = [
                        np.array(
                            [self.class_names.index(n) for n in res], dtype=np.int64
                        )
                        for res in results["gt_names"]
                    ]
                elif "gt_names" in results:
                    results["gt_labels"] = np.array(
                        [self.class_names.index(n) for n in results["gt_names"]],
                        dtype=np.int64,
                    )

                # 处理 gt_labels_3d
                # we still assume one pipeline for one frame LiDAR
                # thus, the 3D name is list[string]
                # # 源代码要求，有gt_names_3d才能加载gt_labels_3d
                if "gt_names_3d" in results:
                    results["gt_labels_3d"] = np.array(
                        [self.class_names.index(n) for n in results["gt_names_3d"]],
                        dtype=np.int64,
                    )

        if "img" in results:
            results["img"] = DC(torch.stack(results["img"]), stack=True)

        for key in [
            "proposals",
            "gt_bboxes",
            "gt_bboxes_ignore",
            "gt_labels",
            "gt_labels_3d",
            "attr_labels",
            "centers2d",
            "depths",
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))
        if "gt_bboxes_3d" in results:
            if isinstance(results["gt_bboxes_3d"], BaseInstance3DBoxes):
                results["gt_bboxes_3d"] = DC(results["gt_bboxes_3d"], cpu_only=True)
            else:
                results["gt_bboxes_3d"] = DC(to_tensor(results["gt_bboxes_3d"]))

        # 添加计时输出
        if Debug:
            duration = time.time() - start_time
            if duration > 0.000001:  # 只打印耗时>1ms的
                # 获取worker信息
                try:
                    import threading
                    import os
                    thread_id = threading.get_ident()
                    process_id = os.getpid()
                    worker_id = f"P{process_id}_T{thread_id%1000:03d}"
                except:
                    worker_id = "Unknown"
                
                # 统计格式化后的数据结构
                formatted_items = []
                for key in ['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d']:
                    if key in results:
                        if hasattr(results[key], 'data'):
                            if hasattr(results[key].data, 'shape'):
                                formatted_items.append(f"{key}: {results[key].data.shape}")
                            else:
                                formatted_items.append(f"{key}: formatted")
                        else:
                            formatted_items.append(f"{key}: present")
                
                print(f">>>[xmy]🟡[Pipeline] DefaultFormatBundle3D [{worker_id}]: "
                      f"样本={sample_id[:15]} | 耗时={duration:.4f}s | "
                      f"格式化项: {', '.join(formatted_items[:3])}")
        return results


@PIPELINES.register_module()
class Collect3D:
    def __init__(
        self,
        keys,
        meta_keys=(
            "camera_intrinsics",
            "camera2ego",
            "img_aug_matrix",
            "lidar_aug_matrix",
        ),
        meta_lis_keys=(
            "filename",
            "timestamp",
            "ori_shape",
            "img_shape",
            "lidar2image",
            "depth2img",
            "cam2img",
            "pad_shape",
            "scale_factor",
            "flip",
            "pcd_horizontal_flip",
            "pcd_vertical_flip",
            "box_mode_3d",
            "box_type_3d",
            "img_norm_cfg",
            "pcd_trans",
            "token",
            "pcd_scale_factor",
            "pcd_rotation",
            "lidar_path",
            "transformation_3d_flow",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys
        # [fixme] note: need at least 1 meta lis key to perform training.
        self.meta_lis_keys = meta_lis_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``metas``
        """
        # 添加计时
        if Debug:
            import time
            start_time = time.time()
            # 获取样本ID
            sample_id = 'unknown'
            for key in ['sample_idx', 'sample_id', 'token', 'frame_id']:
                if key in results:
                    sample_id = str(results[key])
                    break        
        data = {}
        if Debug: print(f">>>[xmy]🟡[formating.py] >>> Collect3D 开始处理self.keys, self.keys={self.keys}, Start")
        for key in self.keys:
            if Debug: print(f"    >>>[xmy]🟡[formating.py] >>> Collect3D key={key}, Start")
            if key not in self.meta_keys:
                data[key] = results[key]
            if Debug: print(f"    >>>[xmy]🟡[formating.py] >>> Collect3D key={key}, End")

        if Debug: print(f">>>[xmy]🟡[formating.py] >>> Collect3D 开始处理self.meta_keys, self.meta_keys={self.meta_keys}, Start")
        for key in self.meta_keys:
            if key in results:
                if Debug: print(f"    >>>[xmy]🟡[formating.py] >>> Collect3D meta_key_i={key} results[key]")
                val = np.array(results[key])
                if isinstance(results[key], list):
                    data[key] = DC(to_tensor(val), stack=True)
                else:
                    data[key] = DC(to_tensor(val), stack=True, pad_dims=1)
        if Debug: print(f">>>[xmy]🟡[formating.py] >>> Collect3D 开始处理self.meta_keys, self.meta_keys={self.meta_keys}, End")

        metas = {}
        for key in self.meta_lis_keys:
            if key in results:
                metas[key] = results[key]

        data["metas"] = DC(metas, cpu_only=True)

        # 添加计时输出
        if Debug:
            duration = time.time() - start_time
            if duration > 0.000001:  # 只打印耗时>1ms的
                # 获取worker信息
                try:
                    import threading
                    import os
                    thread_id = threading.get_ident()
                    process_id = os.getpid()
                    worker_id = f"P{process_id}_T{thread_id%1000:03d}"
                except:
                    worker_id = "Unknown"
                
                # 统计收集的数据项
                collected_keys = list(data.keys())
                num_keys = len(collected_keys)
                meta_keys_count = len(metas.keys())
                
                print(f">>>[xmy]🟡[Pipeline] Collect3D [{worker_id}]: "
                      f"样本={sample_id[:15]} | 耗时={duration:.4f}s | "
                      f"收集数据项={num_keys}, 元数据项={meta_keys_count}")        
        return data
