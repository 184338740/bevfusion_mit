# projects/bevdepth_plugin/vtransforms/bevdepth_transform.py
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
# 修改1：使用 mmdet3d 原有的 builder 注册机制（与官方保持一致）
from mmdet3d.models.builder import VTRANSFORMS
from mmdet3d.models.vtransforms.base import BaseDepthTransform


@VTRANSFORMS.register_module()   # 修改2：使用 VTRANSFORMS 注册
class BEVDepthLSSTransform_v1(BaseDepthTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
        depth_supervision: bool = True,       # 新增：是否启用深度监督
        depth_loss_weight: float = 1.0,       # 新增：深度损失权重
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        # 新增属性
        self.depth_supervision = depth_supervision
        self.depth_loss_weight = depth_loss_weight

        # 以下三个模块与官方 DepthLSSTransform 完全一致
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

        # 新增：深度损失函数（忽略索引 255）
        self.depth_loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    def get_cam_feats(self, x, d):
        """
        与官方 DepthLSSTransform.get_cam_feats 的计算逻辑完全一致，
        唯一区别：额外返回深度概率分布 depth_pred，用于深度监督。
        """
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        d_feat = self.dtransform(d)
        x = torch.cat([d_feat, x], dim=1)
        x = self.depthnet(x)

        depth_pred = x[:, :self.D].softmax(dim=1)            # 深度概率 (B*N, D, fH, fW)
        img_feat = x[:, self.D:self.D + self.C]             # 图像特征 (B*N, C, fH, fW)

        # 抬升操作（Lift）
        x_lifted = depth_pred.unsqueeze(1) * img_feat.unsqueeze(2)   # (B*N, C, D, fH, fW)
        x_lifted = x_lifted.view(B, N, self.C, self.D, fH, fW)      # (B, N, C, D, fH, fW)
        x_lifted = x_lifted.permute(0, 1, 3, 4, 5, 2)               # (B, N, D, fH, fW, C)

        depth_pred = depth_pred.view(B, N, self.D, fH, fW)           # 恢复 batch 维度

        # 返回抬升特征 + 深度概率（官方仅返回 x_lifted）
        return x_lifted, depth_pred

    def forward(self, *args, **kwargs):
        """
        修改点：
        1. 参数签名改为 *args, **kwargs，与官方完全一致，便于调用。
        2. 从 kwargs 中提取深度监督所需的 gt_depths 和 depth_loss 标志。
        3. 支持返回 (BEV特征, 深度损失) 元组，或仅返回 BEV 特征。
        """
        # 提取深度监督相关参数（如果存在）
        gt_depths = kwargs.pop('gt_depths', None)
        depth_loss = kwargs.pop('depth_loss', False)

        # 调用父类 forward，父类会根据 get_cam_feats 的返回值类型自动处理：
        # - 如果 get_cam_feats 返回 tuple，则父类返回 (bev_feat, depth_pred)
        # - 否则返回 bev_feat
        output = super().forward(*args, **kwargs)

        # 因为我们的 get_cam_feats 返回了两个值，所以 output 是一个元组
        bev_feat, depth_pred = output

        # 下采样 BEV 特征（与官方行为一致）
        bev_feat = self.downsample(bev_feat)

        # 如果需要计算深度损失且具备条件
        if depth_loss and self.depth_supervision and gt_depths is not None:
            # gt_depths 形状: (B, N, H, W)，原始分辨率深度真值（由点云投影生成）
            # depth_pred 形状: (B, N, D, fH, fW)，预测的深度概率
            B, N, H, W = gt_depths.shape
            _, _, fH, fW = depth_pred.shape

            # 将真值深度图下采样到与深度预测相同的空间分辨率
            gt_depth_down = F.interpolate(
                gt_depths.view(B * N, 1, H, W),
                size=(fH, fW),
                mode='nearest'
            ).view(B, N, fH, fW)

            # 将连续深度值离散化为区间索引 (0 到 D-1)
            dbound_min, dbound_max, dbound_step = self.dbound
            D = self.D
            depth_idx = ((gt_depth_down - dbound_min) / dbound_step).long()
            depth_idx = torch.clamp(depth_idx, 0, D - 1)

            # 无效像素（没有点云投影的区域）设为 ignore_index
            invalid_mask = (gt_depth_down <= 0) | (gt_depth_down > dbound_max)
            depth_idx[invalid_mask] = 255

            # 计算交叉熵损失并加权
            loss_depth = self.depth_loss_fn(depth_pred, depth_idx) * self.depth_loss_weight
            return bev_feat, loss_depth

        # 默认情况：仅返回 BEV 特征（与官方行为一致）
        return bev_feat