from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet3d.models.builder import VTRANSFORMS
from .base import BaseTransform

# ======================
# 1. MLP + SE (官方BEVDepth原版)
# ======================
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

# ======================
# 2. 极简 DepthNet (MLP + SE + 1个BasicBlock + 1x1 conv)
# 完全对齐官方BEVDepth的简化版，无ASPP，无多残差块
# ======================
class BasicBlock(nn.Module):
    """BasicBlock from ResNet (简化版，用于DepthNet)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class DepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, depth_channels, context_channels):
        super().__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.bn = nn.BatchNorm1d(27)

        # Camera-aware 模块
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)

        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)

        # 一个 BasicBlock + 1x1 conv 作为深度预测头
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            nn.Conv2d(mid_channels, depth_channels, 1),
        )
        self.context_conv = nn.Conv2d(mid_channels, context_channels, 1)

    def forward(self, x, mats_dict):
        # 相机参数编码 (与官方BEVDepth一致)
        intrins = mats_dict['intrin_mats'][:, ..., :3, :3]
        B, N = intrins.shape[:2]
        ida = mats_dict['ida_mats'][:, ...]
        sensor2ego = mats_dict['sensor2ego_mats'][:, ..., :3, :]
        bda = mats_dict['bda_mat'].view(B, 1, 4, 4).repeat(1, N, 1, 1)

        mlp_input = torch.cat([
            torch.stack([
                intrins[..., 0, 0], intrins[..., 1, 1], intrins[..., 0, 2], intrins[..., 1, 2],
                ida[..., 0, 0], ida[..., 0, 1], ida[..., 0, 3],
                ida[..., 1, 0], ida[..., 1, 1], ida[..., 1, 3],
                bda[..., 0, 0], bda[..., 0, 1], bda[..., 1, 0], bda[..., 1, 1], bda[..., 2, 2],
            ], dim=-1),
            sensor2ego.view(B, N, -1)
        ], dim=-1)
        mlp_input = self.bn(mlp_input.reshape(-1, 27))

        x = self.reduce_conv(x)

        # Context 分支
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)

        # Depth 分支
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)

        # 限制 logits 范围，防止 softmax 溢出
        depth = torch.clamp(depth, -9, 9)
        return torch.cat([depth, context], dim=1)

# ======================
# 3. AwareLSSTransform (基于 LSSTransform，增加 MLP+SE 和深度监督)
# ======================
@VTRANSFORMS.register_module()
class xmyAwareBEVDepthLSSLiteV1(BaseTransform):
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
        use_points: str = 'lidar',           # 新增，默认 'lidar'
        downsample: int = 1,
        bevdepth_downsample: int = 8,           # 深度真值下采样倍数，应与特征图下采样倍数一致
        depth_loss_factor: float = 0.2,         # 深度损失权重
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
            use_points=use_points,            # 传递给父类
        )
        self.downsample_factor = bevdepth_downsample
        self.depth_loss_factor = depth_loss_factor
        self.depth_channels = self.D   # D = 离散深度区间数

        # 替换原来的 1x1 conv 为 增强DepthNet (MLP+SE)
        self.depthnet = DepthNet(
            in_channels=in_channels,
            mid_channels=in_channels,
            depth_channels=self.D,
            context_channels=self.C
        )

        # 原 LSSTransform 的下采样模块
        if downsample > 1:
            assert downsample == 2
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()
    def get_cam_feats(self, x, mats_dict):
        B, N, C, fH, fW = x.shape
        x = x.view(B * N, C, fH, fW)

        # 使用 DepthNet 生成深度 logits 和 context
        x = self.depthnet(x, mats_dict)
        depth_logits = x[:, :self.D]
        context = x[:, self.D:]

        depth_probs = depth_logits.softmax(dim=1)
        depth_probs = torch.clamp(depth_probs, 1e-4, 1.0 - 1e-4)

        # 抬升操作
        x = depth_probs.unsqueeze(1) * context.unsqueeze(2)
        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        # [xmy][bugfix][0602] 应该返回的是概率 depth_probs ，而不是原始数值 depth_logits
        return x, depth_probs

    def get_downsampled_gt_depth(self, gt_depths):
        """下采样深度真值并转换为 one-hot 标签"""
        B, N, H, W = gt_depths.shape
        ds = self.downsample_factor
        gt_depths = gt_depths.view(B*N, H//ds, ds, W//ds, ds, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, ds*ds)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B*N, H//ds, W//ds)

        gt_depths = (gt_depths - (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where((gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.depth_channels+1)
        gt_depths = gt_depths.view(-1, self.depth_channels+1)[:, 1:].float()
        return gt_depths

    def get_depth_loss(self, depth_labels, depth_probs):
        """计算深度损失 (BCE)"""
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_probs = depth_probs.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with torch.cuda.amp.autocast(enabled=False):
            loss = F.binary_cross_entropy(depth_probs[fg_mask], depth_labels[fg_mask],
                                          reduction='sum')
            loss = loss / max(1.0, fg_mask.sum())
        return self.depth_loss_factor * loss

    def forward(self, *args, **kwargs):
        # super().forward 会调用 get_cam_feats，返回 (bev_feat, depth_probs)
        x = super().forward(*args, **kwargs)
        # [xmy][bugfix][0602] 应该返回的是概率 depth ，而不是原始数值depth_logits
        bev_feat, depth_probs = x[0], x[-1]

        bev_feat = self.downsample(bev_feat)

        if self.training and kwargs.get('depth_loss', False):
            depth_loss = self.get_depth_loss(kwargs['gt_depths'], depth_probs)
            return bev_feat, depth_loss
        else:
            return bev_feat