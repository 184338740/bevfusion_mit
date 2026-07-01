from typing import Tuple

from mmcv.cnn import build_conv_layer
from mmcv.runner import force_fp32

from torch import nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast

from mmdet3d.models.builder import VTRANSFORMS
from mmdet.models.backbones.resnet import BasicBlock

from .base import BaseTransform, BaseDepthTransform

import torch

__all__ = ["AwareBEVDepth"]

Debug = False
DebugWarning = False

print(f"\n>>>[xmy]🟢[mmdet3d/models/vtransforms/aware_bevdepth.py] >>> [BEVDepth] >>> Debug = {Debug}; DebugWarning = {DebugWarning} ")

class DepthRefinement(nn.Module):
    """
    pixel cloud feature extraction
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DepthRefinement, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x):
        # ========== [xmy][bugfix] 强制 DepthRefinement 使用 FP32 计算 ==========
        # 问题背景：
        #   在混合精度（FP16）训练下，DepthRefinement 模块中的卷积和 BN 操作
        #   在前向和反向传播时均使用 FP16。由于输入的 BEV 特征数值范围较大
        #   （例如经过 bev_pool 后可达 ±4000），BN 层的反向计算中需要计算
        #   grad_weight = sum(grad_out * (x - mean) / sqrt(var+eps))。
        #   当 x 的范围很大时，(x - mean) 可能达到数千，乘以 grad_out（可达几十）
        #   后乘积极易超过 FP16 的最大表示范围（65504），产生 inf 梯度，
        #   进而污染整个模型的参数，导致训练崩溃。
        #
        # 解决方案：
        #   1. 使用 @autocast(False) 装饰器，强制该模块的前向和反向均在 FP32
        #      精度下计算，避免 FP16 下的数值溢出。
        #   2. 在 forward 入口处将输入 x 显式转换为 FP32（x.float()），确保
        #      即使输入是 FP16 也能提升精度。
        #
        # 效果：
        #   该模块的梯度计算将在 FP32 下进行，数值范围扩大至约 1e-8 ~ 1e8，
        #   完全容纳 bev_pool 输出的数值范围，消除因 FP16 溢出导致的梯度爆炸。
        #   同时，由于该模块仅占整体计算量的一小部分，对训练速度的影响可忽略。
        #
        # 适用范围：
        #   所有继承自 DepthRefinement 的类似模块，以及任何在 FP16 训练中
        #   处理大数值特征的子网络。
        # =====================================================================
        x = x.float()     # xmy，新加强制FP32计算

        x = self.reduce_conv(x)
        x = self.conv(x) + x
        x = self.out_conv(x)
        return x



class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
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

class DepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels,
                 depth_channels):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_conv_1 = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        )
        self.depth_conv_2 = nn.Sequential(
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='Conv2d',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
            )),
            nn.BatchNorm2d(mid_channels), 
        )
        self.depth_conv_3 = nn.Sequential(
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(depth_channels), 
        )
        self.export = False

    def export_mode(self):
        self.export = True

    @force_fp32()
    def forward(self, x, mats_dict):
        intrins = mats_dict['intrin_mats'][:, ..., :3, :3]
        batch_size = intrins.shape[0]
        num_cams = intrins.shape[1]
        ida = mats_dict['ida_mats'][:, ...]
        sensor2ego = mats_dict['sensor2ego_mats'][:, ..., :3, :]
        bda = mats_dict['bda_mat'].view(batch_size, 1, 4, 4).repeat(1, num_cams, 1, 1)

        # If exporting, cache the MLP input, since it's based on 
        # intrinsics and data augmentation, which are constant at inference time. 
        if not hasattr(self, 'mlp_input') or not self.export:
            mlp_input = torch.cat(
                [
                    torch.stack(
                        [
                            intrins[:, ..., 0, 0],
                            intrins[:, ..., 1, 1],
                            intrins[:, ..., 0, 2],
                            intrins[:, ..., 1, 2],
                            ida[:, ..., 0, 0],
                            ida[:, ..., 0, 1],
                            ida[:, ..., 0, 3],
                            ida[:, ..., 1, 0],
                            ida[:, ..., 1, 1],
                            ida[:, ..., 1, 3],
                            bda[:, ..., 0, 0],
                            bda[:, ..., 0, 1],
                            bda[:, ..., 1, 0],
                            bda[:, ..., 1, 1],
                            bda[:, ..., 2, 2],
                        ],
                        dim=-1,
                    ),
                    sensor2ego.view(batch_size, num_cams, -1),
                ],
                -1,
            )
            self.mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))


        x = self.reduce_conv(x)
        context_se = self.context_mlp(self.mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(self.mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv_1(depth) 
        depth = self.depth_conv_2(depth)
        depth = self.depth_conv_3(depth) 

        # ========== 修改说明 ==========
        # [xmy] bugfix: 将 depth_logits 限制在 [-9, 9] 范围内，防止梯度爆炸
        # 原代码： logits → softmax 内 exp() 数学上 (0, inf)，但工程上因浮点溢出变成 [0, inf]
        #         → softmax 除法后概率变为 [0, 1]（含精确 0 或 1）
        #         → BCE 内 log(p) 计算得到 -inf 或 0，导致 loss=inf → 梯度 NaN
        # 新代码： logits → clamp(-9, 9) → exp() 范围 (exp(-9), exp(9)) 永不溢出
        #         → softmax 概率严格在 (0,1) 开区间（无 0/1）
        #         → BCE 内 log(p) 输出 (-∞,0) 开区间，无 -inf → 梯度稳定
        depth = torch.clamp(depth, min=-9.0, max=9.0)

        if DebugWarning:
            import os, threading
            pid = os.getpid()
            tid = threading.get_ident()
            # ②-2 深度预测（尚未 softmax）
            print(f"\n>>>[xmy]🟢[aware_bevdepth.py] >>>【{pid}:{tid}】【②-1】[LSS-Lift] DepthNet的输出 深度预测 logits 原始值: min={depth.min().item():.4f}, max={depth.max().item():.4f}, "
                f"has_nan={torch.isnan(depth).any()}, has_inf={torch.isinf(depth).any()}")
            # ②-3 上下文特征
            print(f">>>[xmy]🟢[aware_bevdepth.py] >>>【{pid}:{tid}】【②-2】[LSS-Lift] DepthNet的输出 上下文context: min={context.min().item():.4f}, max={context.max().item():.4f}, "
                f"has_nan={torch.isnan(context).any()}, has_inf={torch.isinf(context).any()}")
            
        return torch.cat([depth, context], dim=1)


@VTRANSFORMS.register_module()
class AwareBEVDepth(BaseTransform):
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
        use_points = 'lidar', 
        downsample: int = 1,
        bevdepth_downsample: int = 16, 
        bevdepth_refine: bool = True, 
        depth_loss_factor: float = 3.0, 
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
            use_points=use_points,
        )
        self.depth_loss_factor = depth_loss_factor
        self.downsample_factor = bevdepth_downsample
        self.bevdepth_refine = bevdepth_refine
        if self.bevdepth_refine:
            self.refinement = DepthRefinement(self.C, self.C, self.C)

        self.depth_channels = self.frustum.shape[0]

        mid_channels = in_channels
        self.depthnet = DepthNet(
            in_channels, 
            mid_channels, 
            self.C, 
            self.D
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

    def export_mode(self):
        super().export_mode()
        self.depthnet.export_mode()

    @force_fp32()
    def get_cam_feats(self, x, mats_dict):  # 【xmy】 训练、评测总是 return 元租
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depthnet(x, mats_dict)

        # ========== 检测 depth logits ==========
        depth_logits = x[:, :self.D]

        if DebugWarning:
            import os, threading
            pid = os.getpid()
            tid = threading.get_ident()
            with torch.no_grad():
                print(f">>>[xmy]🟢[aware_bevdepth.py] >>> 【{pid}:{tid}】【②-3.1】[LSS-Lift] [Before softmax] depth_logits: min={depth_logits.min().item():.4f}, max={depth_logits.max().item():.4f}, "
                    f"has_nan={torch.isnan(depth_logits).any()}, has_inf={torch.isinf(depth_logits).any()}")
        # =====================================


        # [xmy][bugfix][0529]: 引入Scale/温度系数软化深度概率（而不是软标签GT），降低深度分布峰值，减小BEV特征数值范围，缓解FP16下梯度爆炸。
        # 温度系数 T=2.0 使 softmax 分布更平滑，同时保留后续概率 clamp(1e-4, 1-1e-4) 防止 log(0)/log(1)。
        # depth = x[:, : self.D].softmax(dim=1)
        scale_temperature = 2.0  # Scale/温度系数，软化深度概率
        depth = (depth_logits / scale_temperature).softmax(dim=1)


        # [xmy][bugfix]: 裁剪概率，避免出现精确 0 或 1，防止 BCE 中的 log(0) 或 log(1)
        # depth = torch.clamp(depth, min=1e-7, max=1-1e-7)
        depth = torch.clamp(depth, min=1e-4, max=1-1e-4)

        # ========== 检测 softmax 输出范围 ==========
        if DebugWarning:
            import os, threading
            pid = os.getpid()
            tid = threading.get_ident()
            with torch.no_grad():
                v = depth.view(-1)
                min_val = v.min().item()
                max_val = v.max().item()
                # 检查是否有超出 [0,1] 的值（理论上 clamp 后不应出现，但保留检测）
                lt0 = (v < 0).sum().item()
                gt1 = (v > 1).sum().item()
                if lt0 > 0 or gt1 > 0:
                    print(f">>>[xmy]🟢[aware_bevdepth.py] >>> 【{pid}:{tid}】【②-3.1】[LSS-Lift] ⚠️ Out of [0,1]: <0: {lt0}, >1: {gt1}")
                print(f">>>[xmy]🟢[aware_bevdepth.py] >>> 【{pid}:{tid}】【②-3.1】[LSS-Lift] depth_pred (softmax) min={min_val:.6f}, max={max_val:.6f}")
        # ===================================================

        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        if self.bevdepth_refine:
            x = x.permute(0, 3, 1, 4, 2).contiguous() # [n, c, d, h, w] -> [n, h, c, w, d]
            n, h, c, w, d = x.shape
            x = x.view(-1, c, w, d)
            x = self.refinement(x)
            x = x.view(n, h, c, w, d).permute(0, 2, 4, 1, 3).contiguous().float()

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        if DebugWarning:
            import os, threading
            pid = os.getpid()
            tid = threading.get_ident()
            print(f">>>[xmy]🟢[aware_bevdepth.py] >>> 【{pid}:{tid}】【②-3】[LSS-Lift] depthnet输出的 深度预测 depth_pred=softmax(depth): min={depth.min().item():.4f}, max={depth.max().item():.4f}, "
                f"has_nan={torch.isnan(depth).any()}, has_inf={torch.isinf(depth).any()}")
            print(f">>>[xmy]🟢[aware_bevdepth.py] >>> 【{pid}:{tid}】【②-4】[LSS-Lift] 外积抬升后的feat: min={x.min().item():.4f}, max={x.max().item():.4f}, "
                f"has_nan={torch.isnan(x).any()}, has_inf={torch.isinf(x).any()}")
        return x, depth


    def get_depth_loss(self, depth_labels, depth_preds):
        if len(depth_labels.shape) == 5:
            # only key-frame will calculate depth loss
            depth_labels = depth_labels[:, 0, ...]

        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        
        # ========== 检测进入 BCE 前的 depth_preds 范围 ==========
        if DebugWarning:
            import os, threading
            pid = os.getpid()
            tid = threading.get_ident()
            with torch.no_grad():
                v = depth_preds.view(-1)
                total = v.numel()
                lt0 = (v < 0).sum().item()
                gt1 = (v > 1).sum().item()
                ratio_lt0 = lt0 / total * 100
                ratio_gt1 = gt1 / total * 100
                if lt0 > 0 or gt1 > 0:
                    print(f">>>[xmy]🟢[aware_bevdepth.py] >>> 【{pid}:{tid}】【②-3】[LSS-Lift] [Before BCE] Out of [0,1]: <0: {lt0} ({ratio_lt0:.4f}%), >1: {gt1} ({ratio_gt1:.4f}%)")
        # =====================================================

        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        if DebugWarning:
            import os, threading
            pid = os.getpid()
            tid = threading.get_ident()
            # 仅在出现异常时打印（避免日志泛滥）
            if torch.isnan(depth_labels).any() or torch.isinf(depth_labels).any():
                # 注意：此处的 sample_id 需要从外部传入或从全局获取，建议通过 kwargs 或类成员变量
                print(f"\n>>>[xmy]🟢[aware_bevdepth.py] >>> class AwareBEVDepth::get_depth_loss() >>>[PID {pid} TID {tid}] >>> [⚠️ Warning] NaN/inf in depth_labels (GT)")
            if torch.isnan(depth_preds).any() or torch.isinf(depth_preds).any():
                print(f"\n>>>[xmy]🟢[aware_bevdepth.py] >>> class AwareBEVDepth::get_depth_loss() >>> [PID {pid} TID {tid}] >>> [⚠️ Warning] NaN/inf in depth_preds")

        if DebugWarning:
            import os, threading
            pid = os.getpid()
            tid = threading.get_ident()
            # ========== 注册 FP16 原始梯度的钩子（无论哪种分支都会保留） ==========
            def grad_hook_fp16(grad):
                if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
                    return grad
                print(f"\n[FP16 Grad] depth_preds (original) grad: "
                    f"min={grad.min().item():.6f}, max={grad.max().item():.6f}, "
                    f"norm={grad.norm().item():.6f}, has_inf={torch.isinf(grad).any()}, has_nan={torch.isnan(grad).any()}")
                return grad
            handle_fp16 = depth_preds.register_hook(grad_hook_fp16)
            # ====================================================================

        # 和BEVDepth官方一致： autocast(enabled=False) 强制深度损失用 FP32 计算
        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return self.depth_loss_factor * depth_loss


    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:]

        return gt_depths.float()


    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)   # 【xmy】 训练、评测总是 super().forward() ==> get_cam_feats（）==> CNN操作后，总是返回 元租(x, depth)
        x, depth_pred = x[0], x[-1]
        if DebugWarning:
            import os, threading
            pid = os.getpid()
            tid = threading.get_ident()
            print(f">>>[xmy]🟢[base.py] >>> 【{pid}:{tid}】【③-1】[LSS-Splat] bev_pool 输出的 feat : min={x.min().item():.4f}, max={x.max().item():.4f}, "
                f"has_nan={torch.isnan(x).any()}, has_inf={torch.isinf(x).any()}")
            print(f">>>[xmy]🟢[base.py] >>> 【{pid}:{tid}】【③-2】[LSS-Splat] 经过bev_pool 输出的的depth_pred: min={depth_pred.min().item():.4f}, max={depth_pred.max().item():.4f}, "
                f"has_nan={torch.isnan(depth_pred).any()}, has_inf={torch.isinf(depth_pred).any()}")
            
        # [xmy][bugfix][0529]: 强制将 downsampl 的输入转为 FP32，避免该模块在 FP16 下反向传播时产生数值溢出（INF/NAN）。
        # x = self.downsample(x)
        x = self.downsample(x.float())         # 强制转为 FP32 再下采样

        if DebugWarning:
            import os, threading
            pid = os.getpid()
            tid = threading.get_ident()
            print(f">>>[xmy]🟢[base.py] >>> 【{pid}:{tid}】【③-3】[LSS-Splat] 经过bev_pool, downsample后的feat: min={x.min().item():.4f}, max={x.max().item():.4f}, "
                f"has_nan={torch.isnan(x).any()}, has_inf={torch.isinf(x).any()}")
                
        #【xmy-bugfix】0507
        # if kwargs.get('depth_loss', False):
        if self.training and kwargs.get('depth_loss', False):  # 【xmy】BEVDepth的深度监督，仅 训练时启用 get_depth_loss()
            depth_loss = self.get_depth_loss(kwargs['gt_depths'], depth_pred)   # 深度监督，用gt_depths 监督 CNN预测的 depth_pred
            if Debug: print(f"\n>>>[xmy]🟢[aware_bevdepth.py] >>> class AwareBEVDepth::forward() >>>  [DEBUG] after downsample: depth_loss.shape = {depth_loss.shape} depth_loss={depth_loss} type(depth_loss)={type(depth_loss)} ")
            return x, depth_loss
        else:
            return x


@VTRANSFORMS.register_module()
class AwareDBEVDepth(BaseDepthTransform):
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
        use_points = 'lidar', 
        depth_input = 'scalar', 
        height_expand = False, 
        downsample: int = 1,
        bevdepth_downsample: int = 16, 
        bevdepth_refine: bool = True, 
        depth_loss_factor: float = 3.0, 
        add_depth_features = False,
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
            use_points=use_points,
            depth_input=depth_input,
            height_expand=height_expand,
            add_depth_features=add_depth_features,
        )
        self.depth_loss_factor = depth_loss_factor
        self.downsample_factor = bevdepth_downsample
        self.bevdepth_refine = bevdepth_refine
        if self.bevdepth_refine:
            self.refinement = DepthRefinement(self.C, self.C, self.C)

        self.depth_channels = self.frustum.shape[0]

        mid_channels = in_channels 
        self.depthnet = DepthNet(
            in_channels+64, 
            mid_channels, 
            self.C, 
            self.D
        )

        dtransform_in_channels = 1 if depth_input=='scalar' else self.D
        if self.add_depth_features:
            dtransform_in_channels += 45

        if depth_input == 'scalar':
            self.dtransform = nn.Sequential(
                nn.Conv2d(dtransform_in_channels, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(8, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
        else:
            self.dtransform = nn.Sequential(
                nn.Conv2d(dtransform_in_channels, 32, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
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
    def get_cam_feats(self, x, d, mats_dict):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        d = self.dtransform(d)

        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x, mats_dict)
        depth = x[:, : self.D].softmax(dim=1)

        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        if self.bevdepth_refine:
            x = x.permute(0, 3, 1, 4, 2).contiguous() # [n, c, d, h, w] -> [n, h, c, w, d]
            n, h, c, w, d = x.shape
            x = x.view(-1, c, w, d)
            x = self.refinement(x)
            x = x.view(n, h, c, w, d).permute(0, 2, 4, 1, 3).contiguous().float()

        # Here, x.shape is [num_cams, num_channels, depth_bins, downsampled_height, downsampled_width]
        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x, depth

    def export_mode(self):
        super().export_mode()
        self.depthnet.export_mode()

    def get_depth_loss(self, depth_labels, depth_preds):
        # if len(depth_labels.shape) == 5:
        #     # only key-frame will calculate depth loss
        #     depth_labels = depth_labels[:, 0, ...]

        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return self.depth_loss_factor * depth_loss

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:]

        return gt_depths.float()


    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        x, depth_pred = x[0], x[-1]
        x = self.downsample(x)
        if kwargs.get('depth_loss', False):
            depth_loss = self.get_depth_loss(kwargs['gt_depths'], depth_pred) 
            return x, depth_loss
        else:
            return x
        

# ===================== 新增：xmyAwareBEVDepthNormalized =====================
# [xmy][feat][0601] 继承AwareBEVDepth，BEV_Pool升级为mean和count（sum/count + 密度通道压缩），解决FP16下梯度爆炸问题
@VTRANSFORMS.register_module()
class xmyAwareBEVDepthNormalized(AwareBEVDepth):
    """AwareBEVDepth with sum/count BEV pooling and density-aware compression.

    This subclass uses `bev_pool_sum_count_xmy` operator to obtain both accumulated sum
    and point count per BEV grid cell. It then computes avg = sum / (count+eps) and
    concatenates log1p(count) as an extra density channel. A learnable 1x1 convolution
    (without BN/ReLU) reduces the channel dimension from C+1 back to C, preserving the
    original output channel count. This design maintains numerical stability (average
    pooling) while retaining density information, and is robust to varying camera
    intrinsics/extrinsics.

    The final BEV feature has shape [B, C*D, H, W], compatible with the original decoder.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 1x1 convolution to compress (C+1) channels back to C, without BN/ReLU
        self.compress_conv = nn.Conv2d(self.C + 1, self.C, kernel_size=1, bias=True)

    @force_fp32()
    def bev_pool(self, geom_feats, x):
        from mmdet3d.ops.bev_pool_sum_count_xmy import bev_pool_sum_count_xmy

        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        x = x.reshape(Nprime, C)
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        kept = (
            (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) &
            (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) &
            (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        # feats_sum: [B, C, D, H, W], feats_count: [B, D, H, W]
        feats_sum, feats_count = bev_pool_sum_count_xmy(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # average feature (numerically stable)
        avg_feat = feats_sum / (feats_count.unsqueeze(1) + 1e-5)          # [B, C, D, H, W]

        # density channel (log1p to compress range)
        count_log = torch.log1p(feats_count).unsqueeze(1)                 # [B, 1, D, H, W]

        # concatenate avg_feat and count_log
        bev_feat = torch.cat([avg_feat, count_log], dim=1)                # [B, C+1, D, H, W]

        # compress channels using 1x1 conv (no BN/ReLU)
        # reshape to merge depth dimension into batch for 2D convolution
        B, C1, D, H, W = bev_feat.shape
        bev_feat = bev_feat.permute(0, 2, 1, 3, 4).reshape(B * D, C1, H, W)   # [B*D, C+1, H, W]
        bev_feat = self.compress_conv(bev_feat)                                 # [B*D, C, H, W]
        bev_feat = bev_feat.reshape(B, D, C, H, W).permute(0, 2, 1, 3, 4)       # [B, C, D, H, W]

        # collapse depth dimension into channels (original behavior)
        final = bev_feat.permute(0, 2, 1, 3, 4).reshape(B, -1, H, W)            # [B, C*D, H, W]
        return final
