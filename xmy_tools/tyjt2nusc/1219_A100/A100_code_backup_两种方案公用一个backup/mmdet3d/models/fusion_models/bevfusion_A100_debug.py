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
    print(f"\n>>>[xmy]🟢[mmdet3d/models/fusion_models/bevfusion.py] >>> [Train.forward侧] >>> [Debug Mode = True] ")


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
                print(f"\n>>>🟢[xmy][Train-forward] mmdet3d/models/fusion_models/bevfusion.py [BEVFusion调试] [{current_time:.6f}]  第Iter {self.debug_iter} ")
                print(f"-- 相机编码器:")
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
            import time
            # 记录前向传播开始时间点
            forward_start_time = time.time()
            
            # ===== 修正：准确计算两次迭代间的等待时间 =====
            if not hasattr(self, '_last_iter_end_time'):
                # 第一次迭代，没有等待时间
                self._last_iter_end_time = forward_start_time
                data_wait_time = 0.0
            else:
                # 计算从上次迭代结束到这次开始的等待时间
                data_wait_time = forward_start_time - self._last_iter_end_time
            
            # 存储等待时间用于统计
            if not hasattr(self, '_data_wait_times'):
                self._data_wait_times = []
            self._data_wait_times.append(data_wait_time)
            
            # 记录当前处理的样本ID范围（假设metas中包含样本信息）
            sample_ids = []
            if isinstance(metas, list) and len(metas) > 0:
                for meta in metas:
                    # 尝试获取样本ID
                    sample_id = meta.get('sample_idx', meta.get('token', meta.get('frame_id', 'unknown')))
                    sample_ids.append(str(sample_id))
            
            # 初始化其他统计
            sensor_times = {}
            self.debug_iter += 1  # 迭代计数器

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


        # 解码器计时
        if Debug:
            decoder_start_time = time.time()
        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)
        if Debug:
            decoder_time = time.time() - decoder_start_time

        if self.training:
            # 添加检测头计时
            if Debug:
                head_start_time = time.time()
                        
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
            if Debug:
                head_time = time.time() - head_start_time

            if self.use_depth_loss:
                if 'depth' in auxiliary_losses:
                    outputs["loss/depth"] = auxiliary_losses['depth']
                else:
                    raise ValueError('Use depth loss is true, but depth loss not found')
                
            if Debug:
                forward_end_time = time.time()
                forward_duration = forward_end_time - forward_start_time
                
                # ===== 更新：记录本次迭代结束时间 =====
                self._last_iter_end_time = forward_end_time
                
                # 纯净输出：每N个iteration输出一次
                if self.debug_iter % self.debug_interval == 0:
                    current_time = time.time()

                    print(f"--Train forward 耗时 info:")

                    # 1. 迭代信息和样本信息
                    print(f"    [IterInfo] 迭代 | "
                        f"计数={self.debug_iter} | "
                        f"时间戳={current_time:.6f} | "
                        f"batch_size={batch_size} | "
                        f"样本={','.join(sample_ids[:3])}{'...' if len(sample_ids)>3 else ''}")
                    
                    # 2. 时间分解（回答你的问题2）
                    print(f"    [TimeBreakdown] 时间分解 | "
                        f"迭代={self.debug_iter} | "
                        f"迭代间等待={data_wait_time:.6f}s | "
                        f"前向计算={forward_duration:.6f}s | "
                        f"总耗时={data_wait_time + forward_duration:.6f}s")
                    
                    # 3. 模块耗时详情
                    print(f"    [ModuleTime] 模块耗时 | "
                        f"迭代={self.debug_iter} | "
                        f"相机编码={sensor_times.get('camera', 0):.6f}s | "
                        f"激光编码={sensor_times.get('lidar', 0):.6f}s | "
                        f"雷达编码={sensor_times.get('radar', 0):.6f}s | "
                        f"解码器={decoder_time:.6f}s | "
                        f"检测头={head_time:.6f}s")  # 新增
                    
                    # 4. 预取信息提示（回答你的问题3）
                    # # 由于在Model侧不知道DataLoader的预取情况，我们只能提示
                    # print(f"[DataLoaderHint] DataLoader提示 | "
                    #     f"迭代={self.debug_iter} | "
                    #     f"配置: workers_per_gpu={self.data_cfg.get('workers_per_gpu', 'unknown')} | "
                    #     f"prefetch_factor={self.data_cfg.get('prefetch_factor', 'unknown')} | "
                    #     f"理论预取样本数: workers×prefetch={self.data_cfg.get('workers_per_gpu', 1) * self.data_cfg.get('prefetch_factor', 1)}")

                    # 从你的配置我们知道：workers_per_gpu=12, prefetch_factor=2
                    print(f"[DataLoaderHint] DataLoader提示 | "
                        f"迭代={self.debug_iter} | "
                        f"理论预取样本数: 【注意需要根据自己配置的workers_per_gpu, prefetch_factor。来计算】workers×prefetch=12×2=24个 | "
                        f"当前样本可能由DataLoader提前预取")
                    
                    # 5. 平均统计
                    if len(self._data_wait_times) >= 10:
                        avg_wait = sum(self._data_wait_times[-10:]) / 10
                        print(f"[AvgStats] 平均统计 | "
                            f"迭代={self.debug_iter} | "
                            f"最近10次平均等待={avg_wait:.6f}s | "
                            f"平均前向计算={forward_duration:.6f}s")
                    
                    print("")  # 空行分隔           


            return outputs
        else:
            # 测试模式也更新时间戳
            if Debug:
                forward_end_time = time.time()
                self._last_iter_end_time = forward_end_time

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

