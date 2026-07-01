from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS


from .base import Base3DFusionModel

__all__ = ["BEVFusion"]

Debug = True

if Debug:
    import time

@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        if Debug:
            # 添加调试计数器
            self.debug_iter = 0
            self.debug_interval = 50  # 每50个iteration打印一次
            # 时间统计
            self.camera_times = []
            self.forward_times = []


        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if encoders.get("radar") is not None:
            if encoders["radar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["radar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["radar"]["voxelize"])
            self.encoders["radar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["radar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["radar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        # If the camera's vtransform is a BEVDepth version, then we're using depth loss. 
        self.use_depth_loss = ((encoders.get('camera', {}) or {}).get('vtransform', {}) or {}).get('type', '') in ['BEVDepth', 'AwareBEVDepth', 'DBEVDepth', 'AwareDBEVDepth']


        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        radar_points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        gt_depths=None,
    ) -> torch.Tensor:
        
        if Debug:
            camera_start = time.time()  # 修复：定义开始时间    

        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        if Debug:
            # 相机骨干网络计时
            backbone_start = time.time()

        x = self.encoders["camera"]["backbone"](x)

        if Debug:    
            backbone_time = time.time() - backbone_start
            # 颈部网络计时
            neck_start = time.time()

        x = self.encoders["camera"]["neck"](x)

        if Debug:    
            neck_time = time.time() - neck_start


        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        if Debug:    
            # 视图变换计时
            vtransform_start = time.time()
        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            radar_points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            depth_loss=self.use_depth_loss, 
            gt_depths=gt_depths,
        )
        if Debug:        
            vtransform_time = time.time() - vtransform_start
            total_camera_time = time.time() - camera_start
            
            self.camera_times.append(total_camera_time)
            
            self.debug_iter += 1
            if self.debug_iter % self.debug_interval == 0:
                avg_camera = sum(self.camera_times[-10:]) / min(len(self.camera_times), 10)
                
                # 添加时间戳（这是关键修改）
                current_time = time.time()
                print(f"\n[{current_time:.6f}] >>>🟢[xmy][Train-forward] mmdet3d/models/fusion_models/bevfusion.py [BEVFusion调试] Iter {self.debug_iter} - 相机编码器:")
                print(f"  Backbone: {backbone_time:.3f}s")
                print(f"  Neck: {neck_time:.3f}s") 
                print(f"  VTransform: {vtransform_time:.3f}s")
                print(f"  本次相机总耗时: {total_camera_time:.3f}s")
                print(f"  最近10次平均: {avg_camera:.3f}s")

                # ===== 新增：打印数据等待与瓶颈分析 =====

        return x
    

    def extract_features(self, x, sensor) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x, sensor)
        batch_size = coords[-1, 0] + 1
        x = self.encoders[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x
    
    # def extract_lidar_features(self, x) -> torch.Tensor:
    #     feats, coords, sizes = self.voxelize(x)
    #     batch_size = coords[-1, 0] + 1
    #     x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
    #     return x

    # def extract_radar_features(self, x) -> torch.Tensor:
    #     feats, coords, sizes = self.radar_voxelize(x)
    #     batch_size = coords[-1, 0] + 1
    #     x = self.encoders["radar"]["backbone"](feats, coords, batch_size, sizes=sizes)
    #     return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, sensor):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders[sensor]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    # @torch.no_grad()
    # @force_fp32()
    # def radar_voxelize(self, points):
    #     feats, coords, sizes = [], [], []
    #     for k, res in enumerate(points):
    #         ret = self.encoders["radar"]["voxelize"](res)
    #         if len(ret) == 3:
    #             # hard voxelize
    #             f, c, n = ret
    #         else:
    #             assert len(ret) == 2
    #             f, c = ret
    #             n = None
    #         feats.append(f)
    #         coords.append(F.pad(c, (1, 0), mode="constant", value=k))
    #         if n is not None:
    #             sizes.append(n)

    #     feats = torch.cat(feats, dim=0)
    #     coords = torch.cat(coords, dim=0)
    #     if len(sizes) > 0:
    #         sizes = torch.cat(sizes, dim=0)
    #         if self.voxelize_reduce:
    #             feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
    #                 -1, 1
    #             )
    #             feats = feats.contiguous()

    #     return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                depths,
                radar,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if Debug:
            # 记录前向传播开始时间点
            forward_start_time = time.time()

            # ===== 新增：计算数据等待时间 =====
            # 假设你有一个记录“batch数据准备好时刻”的变量。
            # 这里为了演示，我们先定义一个简单的方法：用当前时间减去一个全局的“数据就绪时间戳”。
            # 更精确的做法是在数据加载结束时记录该时间戳。
            if not hasattr(self, '_last_batch_ready_time'):
                self._last_batch_ready_time = forward_start_time

            # 计算数据等待时间（模型等待数据的时间）
            data_wait_time = forward_start_time - self._last_batch_ready_time
            # 更新“数据就绪时间”为本次forward开始时刻
            self._last_batch_ready_time = forward_start_time

            # 存储等待时间用于平均计算
            if not hasattr(self, '_data_wait_times'):
                self._data_wait_times = []
            self._data_wait_times.append(data_wait_time)

            sensor_times = {}  # 存储各传感器耗时

        features = []
        auxiliary_losses = {}


        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if Debug:
                sensor_start_time = time.time() # 添加：开始计时
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    radar,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    gt_depths=depths,
                )
                if self.use_depth_loss:
                    feature, auxiliary_losses['depth'] = feature[0], feature[-1]
            elif sensor == "lidar":
                feature = self.extract_features(points, sensor)
            elif sensor == "radar":
                feature = self.extract_features(radar, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            features.append(feature)

            if Debug:
                sensor_times[sensor] = time.time() - sensor_start_time  # 添加：记录耗时

        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]


        # 添加解码器计时
        if Debug:
            decoder_start_time = time.time()
        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)
        if Debug:
            decoder_time = time.time() - decoder_start_time

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            if self.use_depth_loss:
                if 'depth' in auxiliary_losses:
                    outputs["loss/depth"] = auxiliary_losses['depth']
                else:
                    raise ValueError('Use depth loss is true, but depth loss not found')
            if Debug:
                forward_end_time = time.time()
                forward_duration = forward_end_time - forward_start_time
                
                # 添加传感器耗时打印
                if self.debug_iter % self.debug_interval == 0:
                    # print(f"  --- Forward耗时分析 [start] ---")
                    # print(f"  各传感器耗时:")
                    # for sensor, time_taken in sensor_times.items():
                    #     print(f"    {sensor}: {time_taken:.3f}s")
                    # print(f"  解码器耗时: {decoder_time:.3f}s")
                    # print(f"  总forward耗时: {forward_duration:.3f}s")
                    
                    # # 计算相机占比
                    # if 'camera' in sensor_times:
                    #     camera_ratio = sensor_times['camera'] / forward_duration * 100
                    #     print(f"  相机编码器占比: {camera_ratio:.1f}%")
                    # print(f"  --- Forward耗时分析 [end] ---")

                    # # ===== 关键：瓶颈分析（判断是否在等dataloader）=====
                    # if len(self._data_wait_times) >= 10:
                    #     # 计算最近10次的平均等待时间
                    #     avg_wait = sum(self._data_wait_times[-10:]) / 10
                        
                    #     # ===== 新增：打印当前iteration的等待时间 =====
                    #     current_wait = self._data_wait_times[-1]
                    #     print(f"  --- 数据加载分析 [start] ---")
                    #     print(f"  本次数据等待时间: {current_wait:.3f}s")
                    #     print(f"  平均数据等待时间: {avg_wait:.3f}s")
                    #     print(f"  模型前向计算时间: {forward_duration:.3f}s")
                    #     print(f"  总iteration时间: {current_wait + forward_duration:.3f}s")
                        
                    #     # 瓶颈判断
                    #     if current_wait > forward_duration * 2:
                    #         print(f"  ⚠️🚨 严重瓶颈：DataLoader太慢！")
                    #         print(f"     数据等待({current_wait:.1f}s) >> 模型计算({forward_duration:.1f}s)")
                    #         print(f"     可优化：增加num_workers，使用SSD，优化数据预处理")
                    #     elif current_wait > forward_duration:
                    #         print(f"  ⚠️ 瓶颈：DataLoader较慢")
                    #         print(f"     数据等待({current_wait:.1f}s) > 模型计算({forward_duration:.1f}s)")
                    #     elif current_wait > forward_duration * 0.5:
                    #         print(f"  ⚠️ 轻微瓶颈：DataLoader有点慢")
                    #         print(f"     数据等待({current_wait:.1f}s) 占比较大")
                    #     else:
                    #         print(f"  ✅ DataLoader正常")
                    #         print(f"     模型计算是主要耗时")
                        
                    #     # GPU利用率分析（简单版）
                    #     gpu_utilization = forward_duration / (current_wait + forward_duration) * 100
                    #     print(f"  GPU利用率: {gpu_utilization:.1f}%")
                        
                    #     # 建议
                    #     if gpu_utilization < 30:
                    #         print(f"  💡 建议：显著增加num_workers，减少数据加载时间")
                    #     elif gpu_utilization < 60:
                    #         print(f"  💡 建议：适当增加num_workers")
                    #     elif gpu_utilization > 90:
                    #         print(f"  💡 GPU几乎满载，瓶颈在模型计算")
                    #         print(f"     可考虑：混合精度训练，模型剪枝")
                    #     else:
                    #         print(f"  💡 GPU利用率良好")         
                    #     print(f"  --- 数据加载分析 [end] ---")
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs

