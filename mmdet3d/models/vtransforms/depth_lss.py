from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.models.builder import VTRANSFORMS

from .base import BaseDepthTransform

__all__ = ["DepthLSSTransform"]


@VTRANSFORMS.register_module()
class DepthLSSTransform(BaseDepthTransform):
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

    @force_fp32()
    def get_cam_feats(self, x, d):
        # 【xmy】Lift操作
        # x：图像特征，形状 (B, N, C, fH, fW)，来自图像骨干网络。
        # d：由 BaseDepthTransform 生成的真实深度图（基于点云投影），形状 (B, N, 1, H, W) 或 (B, N, D, H, W)（取决于 depth_input 模式）。在训练时，d 是真实值；在推理时，d 可能由网络自身生成或置零。
        # self.dtransform：将真实深度图下采样并提取特征，与图像特征空间对齐。
        # self.depthnet：融合深度和图像特征，同时输出深度概率和图像特征
        # 离散的深度概率：softmax:深度概率：通过 softmax 获得，可用于深度监督（但该函数未返回，需修改代码才能在外部使用）。
        # 抬升：用深度概率对图像特征加权，实现“每个深度区间都有特征”的效果。

        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        d = self.dtransform(d)        # 将 depth 通过 dtransform（一个小型 CNN）提取深度特征,用于将深度图下采样到与特征图相同的空间尺寸，并提取深度特征。形状 (B*N, 64, fH, fW)
        x = torch.cat([d, x], dim=1)  # 4. 拼接“深度特征d” & “图像特征 x” 【此处完成了 图像 + 点云 的concat拼接 】
        x = self.depthnet(x)          # 5. 深度预测与特征变换。  拼接后的特征通过 depthnet（一个三层的卷积网络），D 是离散深度区间数（如 119），C 是图像特征通道数。

        depth = x[:, : self.D].softmax(dim=1)  # 6. 提取"深度概率"分布
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)  # 7. 抬升：将图像特征按"深度概率"加权，获取“抬升”后的特征。。 其中，x[:, self.D : (self.D + self.C)] 取输出的后 C 个通道

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x
