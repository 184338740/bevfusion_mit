# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from mmcv.utils import to_2tuple

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..utils.ckpt_convert import swin_converter
from ..utils.transformer import PatchEmbed, PatchMerging


class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):

        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        self.init_cfg = init_cfg

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    # def forward(self, x, mask=None):
    #     """
    #     Args:

    #         x (tensor): input features with shape of (num_windows*B, N, C)
    #         mask (tensor | None, Optional): mask with shape of (num_windows,
    #             Wh*Ww, Wh*Ww), value should be between (-inf, 0].
    #     """
    #     B, N, C = x.shape
    #     qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
    #                               C // self.num_heads).permute(2, 0, 3, 1, 4)
    #     # make torchscript happy (cannot use tensor as tuple)
    #     q, k, v = qkv[0], qkv[1], qkv[2]

    #     q = q * self.scale
    #     attn = (q @ k.transpose(-2, -1))

    #     relative_position_bias = self.relative_position_bias_table[
    #         self.relative_position_index.view(-1)].view(
    #             self.window_size[0] * self.window_size[1],
    #             self.window_size[0] * self.window_size[1],
    #             -1)  # Wh*Ww,Wh*Ww,nH
    #     relative_position_bias = relative_position_bias.permute(
    #         2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    #     attn = attn + relative_position_bias.unsqueeze(0)

    #     if mask is not None:
    #         nW = mask.shape[0]
    #         attn = attn.view(B // nW, nW, self.num_heads, N,
    #                          N) + mask.unsqueeze(1).unsqueeze(0)
    #         attn = attn.view(-1, self.num_heads, N, N)
    #     attn = self.softmax(attn)

    #     attn = self.attn_drop(attn)

    #     x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     return x

    # def forward(self, x, mask=None):
    #     """
    #     Args:
    #         x (tensor): input features with shape of (num_windows*B, N, C)
    #         mask (tensor | None, Optional): mask with shape of (num_windows,
    #             Wh*Ww, Wh*Ww), value should be between (-inf, 0].
    #     """
    #     B, N, C = x.shape
        
    #     # QKV投影
    #     qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    #     q, k, v = qkv[0], qkv[1], qkv[2]
        
    #     # 确保张量连续
    #     q = q.contiguous()
    #     k = k.contiguous()
    #     v = v.contiguous()
        
    #     batch_heads = B * self.num_heads
    #     C_head = C // self.num_heads
        
    #     q_reshaped = q.view(batch_heads, N, C_head)
    #     k_reshaped = k.view(batch_heads, N, C_head)
    #     v_reshaped = v.view(batch_heads, N, C_head)

    #     # ===== 终极修复：完全避免torch.bmm =====
    #     q_scaled = q_reshaped * self.scale
    #     k_t = k_reshaped.transpose(1, 2).contiguous()
        
    #     # 方法1：使用torch.matmul（可能走不同路径）
    #     # attn_reshaped = torch.matmul(q_scaled, k_t)
        
    #     # 方法2：手动实现批量矩阵乘法（最安全）
    #     if batch_heads > 500:  # 大批次时，手动计算避免cublas问题
    #         print(f"[WindowMSA安全模式] B={B}, batch_heads={batch_heads}，使用手动矩阵乘法")
    #         attn_list = []
    #         for i in range(batch_heads):
    #             attn_list.append(torch.mm(q_scaled[i], k_t[i]))
    #         attn_reshaped = torch.stack(attn_list, dim=0)
    #     else:
    #         # 小批次时使用torch.matmul
    #         attn_reshaped = torch.matmul(q_scaled, k_t)
    #     # ===== 修复结束 =====

    #     # 恢复维度
    #     attn = attn_reshaped.view(B, self.num_heads, N, N)

    #     # 相对位置偏置
    #     relative_position_bias = self.relative_position_bias_table[
    #         self.relative_position_index.view(-1)].view(
    #             self.window_size[0] * self.window_size[1],
    #             self.window_size[0] * self.window_size[1], -1)
    #     relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
    #     attn = attn + relative_position_bias.unsqueeze(0)

    #     if mask is not None:
    #         nW = mask.shape[0]
    #         attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    #         attn = attn.view(-1, self.num_heads, N, N)

    #     attn = self.softmax(attn)
    #     attn = self.attn_drop(attn)

    #     # 输出投影也需要修复
    #     attn_reshaped = attn.view(batch_heads, N, N)
        
    #     # 同样避免torch.bmm
    #     if batch_heads > 500:
    #         output_list = []
    #         for i in range(batch_heads):
    #             output_list.append(torch.mm(attn_reshaped[i], v_reshaped[i]))
    #         x = torch.stack(output_list, dim=0)
    #     else:
    #         x = torch.matmul(attn_reshaped, v_reshaped)
        
    #     x = x.view(B, self.num_heads, N, C_head)
    #     x = x.transpose(1, 2).reshape(B, N, C)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)

    #     return x


    # def forward(self, x, mask=None):
    #     B, N, C = x.shape
        
    #     # ===== 简单修复：确保张量连续 + 降级到FP32计算 =====
    #     # 保存原始精度
    #     original_dtype = x.dtype
        
    #     # 在FP32上下文中计算（避免Tensor Core问题）
    #     with torch.cuda.amp.autocast(enabled=False):
    #         # 转换为FP32
    #         x_fp32 = x.float() if original_dtype == torch.float16 else x
            
    #         # 原有计算
    #         qkv = self.qkv(x_fp32).reshape(B, N, 3, self.num_heads,
    #                                     C // self.num_heads).permute(2, 0, 3, 1, 4)
    #         q, k, v = qkv[0], qkv[1], qkv[2]

    #         q = q * self.scale
    #         attn = (q @ k.transpose(-2, -1))

    #         relative_position_bias = self.relative_position_bias_table[
    #             self.relative_position_index.view(-1)].view(
    #                 self.window_size[0] * self.window_size[1],
    #                 self.window_size[0] * self.window_size[1],
    #                 -1)  # Wh*Ww,Wh*Ww,nH
    #         relative_position_bias = relative_position_bias.permute(
    #             2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    #         attn = attn + relative_position_bias.unsqueeze(0)

    #         if mask is not None:
    #             nW = mask.shape[0]
    #             attn = attn.view(B // nW, nW, self.num_heads, N,
    #                             N) + mask.unsqueeze(1).unsqueeze(0)
    #             attn = attn.view(-1, self.num_heads, N, N)
    #         attn = self.softmax(attn)

    #         attn = self.attn_drop(attn)

    #         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    #         x = self.proj(x)
    #         x = self.proj_drop(x)
        
    #     # 转换回原始精度
    #     return x.to(original_dtype)



    # def forward(self, x, mask=None):
    #     """
    #     Args:
    #         x (tensor): input features with shape of (num_windows*B, N, C)
    #         mask (tensor | None, Optional): mask with shape of (num_windows,
    #             Wh*Ww, Wh*Ww), value should be between (-inf, 0].
    #     """
    #     # ========== 基础信息 + 分布式Trace ==========
    #     import torch.distributed as dist
    #     try:
    #         rank = dist.get_rank()
    #         world_size = dist.get_world_size()
    #     except:
    #         rank = -1
    #         world_size = -1
    #     print(f"\n===== [A100 Trace] 分布式环境：rank={rank}, world_size={world_size} =====")
        
    #     B, N_ori, C = x.shape  # 现在N_ori=49（window_size=7），匹配权重
    #     print(f"[A100 Trace] 输入x: shape={x.shape}, 连续={x.is_contiguous()}, device={x.device}")

    #     # ========== 步骤1：QKV计算 + 维度重塑（绕开Strided Batched） ==========
    #     qkv = self.qkv(x).reshape(B, N_ori, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    #     q, k, v = qkv[0].contiguous(), qkv[1].contiguous(), qkv[2].contiguous()  # 强制连续
        
    #     # 重塑维度：将 (B, num_heads, N, C_head) → (B*num_heads, N, C_head)，避免Strided Batched调用
    #     batch_num_heads = B * self.num_heads
    #     C_head = C // self.num_heads
    #     q_reshaped = q.reshape(batch_num_heads, N_ori, C_head).contiguous()  # (B*num_heads, 49, C_head)
    #     k_reshaped = k.reshape(batch_num_heads, N_ori, C_head).contiguous()
    #     v_reshaped = v.reshape(batch_num_heads, N_ori, C_head).contiguous()
    #     print(f"[A100 Trace] 重塑后：q_reshaped.shape={q_reshaped.shape}, 连续={q_reshaped.is_contiguous()}")

    #     # ========== 步骤2：矩阵乘法（普通GEMM，绕开Strided Batched接口） ==========
    #     q_scaled = q_reshaped * self.scale
    #     k_t_reshaped = k_reshaped.transpose(-2, -1).contiguous()  # (B*num_heads, C_head, 49)
        
    #     try:
    #         # 普通GEMM：(B*num_heads, 49, C_head) @ (B*num_heads, C_head, 49) → (B*num_heads, 49, 49)
    #         attn_reshaped = torch.matmul(q_scaled, k_t_reshaped).contiguous()
    #         print(f"[A100 Trace] ✅ CUDA普通GEMM成功！attn_reshaped.shape={attn_reshaped.shape}")
    #     except Exception as e:
    #         print(f"[A100 Trace] ❌ CUDA GEMM失败，降级CPU：{str(e)[:200]}")
    #         # 兜底CPU计算，转回CUDA并强制连续
    #         attn_reshaped = torch.matmul(q_scaled.cpu(), k_t_reshaped.cpu()).to(q_scaled.device).contiguous()
    #         print(f"[A100 Trace] ✅ CPU GEMM成功！attn_reshaped.shape={attn_reshaped.shape}")

    #     # ========== 步骤3：恢复原始维度（逆重塑） ==========
    #     attn = attn_reshaped.reshape(B, self.num_heads, N_ori, N_ori).contiguous()  # (B, num_heads, 49, 49)
    #     print(f"[A100 Trace] 恢复维度后：attn.shape={attn.shape}, 连续={attn.is_contiguous()}")

    #     # ========== 步骤4：原有逻辑（相对位置偏置 + mask + softmax） ==========
    #     # 相对位置偏置：现在window_size=7，relative_position_index是49×49，匹配权重！
    #     relative_position_bias = self.relative_position_bias_table[
    #         self.relative_position_index.view(-1)].view(
    #             self.window_size * self.window_size,
    #             self.window_size * self.window_size, -1)
    #     relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
    #     attn = attn + relative_position_bias.unsqueeze(0)

    #     # Mask处理
    #     if mask is not None:
    #         nW = mask.shape[0]
    #         attn = attn.view(B // nW, nW, self.num_heads, N_ori, N_ori) + mask.unsqueeze(1).unsqueeze(0)
    #         attn = attn.view(-1, self.num_heads, N_ori, N_ori).contiguous()
        
    #     # Softmax + Dropout
    #     attn = self.softmax(attn)
    #     attn = self.attn_drop(attn)

    #     # ========== 步骤5：最终输出计算（确保维度匹配+连续） ==========
    #     attn_reshaped = attn.reshape(batch_num_heads, N_ori, N_ori).contiguous()
    #     x_reshaped = torch.matmul(attn_reshaped, v_reshaped).contiguous()  # (B*num_heads, 49, C_head)
        
    #     # 恢复原始维度
    #     x = x_reshaped.reshape(B, self.num_heads, N_ori, C_head).transpose(1, 2).reshape(B, N_ori, C).contiguous()
    #     x = self.proj(x)
    #     x = self.proj_drop(x)

    #     print(f"[A100 Trace] ✅ 最终输出：x.shape={x.shape}, 连续={x.is_contiguous()}, device={x.device}")
    #     return x


    # def forward(self, x, mask=None):
    #     """
    #     Args:
    #         x (tensor): input features with shape of (num_windows*B, N, C)
    #         mask (tensor | None, Optional): mask with shape of (num_windows,
    #             Wh*Ww, Wh*Ww), value should be between (-inf, 0].
    #     """
    #     # ========== 基础信息 + 分布式Trace ==========
    #     import torch.distributed as dist
    #     try:
    #         rank = dist.get_rank()
    #         world_size = dist.get_world_size()
    #     except:
    #         rank = -1
    #         world_size = -1
    #     print(f"\n===== [A100 Trace] 分布式环境：rank={rank}, world_size={world_size} =====")
        
    #     B, N_ori, C = x.shape  # N_ori=49（window_size=7）
    #     # 关键修复1：强制所有张量转为FP32，避免Half/FP16导致的兼容问题
    #     x = x.float().contiguous()  # 从Half→float32
    #     print(f"[A100 Trace] 输入x: shape={x.shape}, 连续={x.is_contiguous()}, 类型={x.dtype}, device={x.device}")

    #     # ========== 步骤1：QKV计算 + 维度重塑（绕开Strided Batched） ==========
    #     qkv = self.qkv(x).reshape(B, N_ori, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    #     q, k, v = qkv[0].float().contiguous(), qkv[1].float().contiguous(), qkv[2].float().contiguous()  # 强制FP32
        
    #     # 重塑维度：将 (B, num_heads, N, C_head) → (B*num_heads, N, C_head)
    #     batch_num_heads = B * self.num_heads
    #     C_head = C // self.num_heads
    #     q_reshaped = q.reshape(batch_num_heads, N_ori, C_head).float().contiguous()  # 强制FP32
    #     k_reshaped = k.reshape(batch_num_heads, N_ori, C_head).float().contiguous()
    #     v_reshaped = v.reshape(batch_num_heads, N_ori, C_head).float().contiguous()
    #     print(f"[A100 Trace] 重塑后：q_reshaped.shape={q_reshaped.shape}, 连续={q_reshaped.is_contiguous()}, 类型={q_reshaped.dtype}")

    #     # ========== 步骤2：矩阵乘法（强制FP32，绕开Strided Batched接口） ==========
    #     q_scaled = q_reshaped * self.scale
    #     k_t_reshaped = k_reshaped.transpose(-2, -1).float().contiguous()  # 强制FP32
        
    #     try:
    #         # 普通GEMM：FP32计算，避免A100 FP16的接口缺陷
    #         attn_reshaped = torch.matmul(q_scaled, k_t_reshaped).float().contiguous()
    #         print(f"[A100 Trace] ✅ CUDA普通GEMM成功！attn_reshaped.shape={attn_reshaped.shape}, 类型={attn_reshaped.dtype}")
    #     except Exception as e:
    #         print(f"[A100 Trace] ❌ CUDA GEMM失败，降级CPU：{str(e)[:200]}")
    #         # 关键修复2：CPU计算时强制FP32，避免Half不兼容
    #         q_scaled_cpu = q_scaled.cpu().float().contiguous()
    #         k_t_cpu = k_t_reshaped.cpu().float().contiguous()
    #         attn_reshaped = torch.matmul(q_scaled_cpu, k_t_cpu).float().to(q_scaled.device).contiguous()
    #         print(f"[A100 Trace] ✅ CPU GEMM成功！attn_reshaped.shape={attn_reshaped.shape}, 类型={attn_reshaped.dtype}")

    #     # ========== 步骤3：恢复原始维度（逆重塑） ==========
    #     attn = attn_reshaped.reshape(B, self.num_heads, N_ori, N_ori).float().contiguous()
    #     print(f"[A100 Trace] 恢复维度后：attn.shape={attn.shape}, 连续={attn.is_contiguous()}, 类型={attn.dtype}")

    #     # ========== 步骤4：原有逻辑（相对位置偏置 + mask + softmax） ==========
    #     # 相对位置偏置：强制FP32
    #     relative_position_bias = self.relative_position_bias_table[
    #         self.relative_position_index.view(-1)].view(
    #             self.window_size * self.window_size,
    #             self.window_size * self.window_size, -1)
    #     relative_position_bias = relative_position_bias.permute(2, 0, 1).float().contiguous()
    #     attn = attn + relative_position_bias.unsqueeze(0)

    #     # Mask处理：强制FP32
    #     if mask is not None:
    #         mask = mask.float().contiguous()  # 强制FP32
    #         nW = mask.shape[0]
    #         attn = attn.view(B // nW, nW, self.num_heads, N_ori, N_ori) + mask.unsqueeze(1).unsqueeze(0)
    #         attn = attn.view(-1, self.num_heads, N_ori, N_ori).float().contiguous()
        
    #     # Softmax + Dropout：强制FP32
    #     attn = self.softmax(attn)
    #     attn = self.attn_drop(attn).float().contiguous()

    #     # ========== 步骤5：最终输出计算（确保FP32+连续） ==========
    #     attn_reshaped = attn.reshape(batch_num_heads, N_ori, N_ori).float().contiguous()
    #     x_reshaped = torch.matmul(attn_reshaped, v_reshaped).float().contiguous()  # FP32计算
        
    #     # 恢复原始维度 + 转回原始类型（兼容后续流程）
    #     x = x_reshaped.reshape(B, self.num_heads, N_ori, C_head).transpose(1, 2).reshape(B, N_ori, C).float().contiguous()
    #     x = self.proj(x).float().contiguous()
    #     x = self.proj_drop(x).float().contiguous()

    #     print(f"[A100 Trace] ✅ 最终输出：x.shape={x.shape}, 连续={x.is_contiguous()}, 类型={x.dtype}, device={x.device}")
    #     return x


    # def forward(self, x, mask=None):
    #     """
    #     Args:
    #         x (tensor): input features with shape of (num_windows*B, N, C)
    #         mask (tensor | None, Optional): mask with shape of (num_windows,
    #             Wh*Ww, Wh*Ww), value should be between (-inf, 0].
    #     """
    #     # ========== 基础信息 + 分布式Trace ==========
    #     print(f"🔵[xmy][A100 Trace]>>> window_size: {self.window_size}, 类型: {type(self.window_size)}")
    #     import torch.distributed as dist
    #     try:
    #         rank = dist.get_rank()
    #         world_size = dist.get_world_size()
    #     except:
    #         rank = -1
    #         world_size = -1
    #     print(f"\n===== [A100 Trace] 分布式环境：rank={rank}, world_size={world_size} =====")
        
    #     B, N_ori, C = x.shape  # N_ori=49（window_size=7）
    #     # 关键修复1：强制所有张量转为FP32，避免Half/FP16导致的兼容问题
    #     x = x.float().contiguous()  # 从Half→float32
    #     print(f"[A100 Trace] 输入x: shape={x.shape}, 连续={x.is_contiguous()}, 类型={x.dtype}, device={x.device}")

    #     # ========== 步骤1：QKV计算 + 维度重塑（绕开Strided Batched） ==========
    #     qkv = self.qkv(x).reshape(B, N_ori, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    #     q, k, v = qkv[0].float().contiguous(), qkv[1].float().contiguous(), qkv[2].float().contiguous()  # 强制FP32
        
    #     # 重塑维度：将 (B, num_heads, N, C_head) → (B*num_heads, N, C_head)
    #     batch_num_heads = B * self.num_heads
    #     C_head = C // self.num_heads
    #     q_reshaped = q.reshape(batch_num_heads, N_ori, C_head).float().contiguous()  # 强制FP32
    #     k_reshaped = k.reshape(batch_num_heads, N_ori, C_head).float().contiguous()
    #     v_reshaped = v.reshape(batch_num_heads, N_ori, C_head).float().contiguous()
    #     print(f"[A100 Trace] 重塑后：q_reshaped.shape={q_reshaped.shape}, 连续={q_reshaped.is_contiguous()}, 类型={q_reshaped.dtype}")

    #     # ========== 步骤2：矩阵乘法（强制FP32，绕开Strided Batched接口） ==========
    #     q_scaled = q_reshaped * self.scale
    #     k_t_reshaped = k_reshaped.transpose(-2, -1).float().contiguous()  # 强制FP32
        
    #     try:
    #         # 普通GEMM：FP32计算，避免A100 FP16的接口缺陷
    #         attn_reshaped = torch.matmul(q_scaled, k_t_reshaped).float().contiguous()
    #         print(f"[A100 Trace] ✅ CUDA普通GEMM成功！attn_reshaped.shape={attn_reshaped.shape}, 类型={attn_reshaped.dtype}")
    #     except Exception as e:
    #         print(f"[A100 Trace] ❌ CUDA GEMM失败，降级CPU：{str(e)[:200]}")
    #         # 关键修复2：CPU计算时强制FP32，避免Half不兼容
    #         q_scaled_cpu = q_scaled.cpu().float().contiguous()
    #         k_t_cpu = k_t_reshaped.cpu().float().contiguous()
    #         attn_reshaped = torch.matmul(q_scaled_cpu, k_t_cpu).float().to(q_scaled.device).contiguous()
    #         print(f"[A100 Trace] ✅ CPU GEMM成功！attn_reshaped.shape={attn_reshaped.shape}, 类型={attn_reshaped.dtype}")

    #     # ========== 步骤3：恢复原始维度（逆重塑） ==========
    #     attn = attn_reshaped.reshape(B, self.num_heads, N_ori, N_ori).float().contiguous()
    #     print(f"[A100 Trace] 恢复维度后：attn.shape={attn.shape}, 连续={attn.is_contiguous()}, 类型={attn.dtype}")

    #     # ========== 步骤4：原有逻辑（相对位置偏置 + mask + softmax） ==========
    #     # 相对位置偏置：强制FP32 + 修复tuple乘法错误
    #     relative_position_bias = self.relative_position_bias_table[
    #         self.relative_position_index.view(-1)].view(
    #             self.window_size[0] * self.window_size[1],  # 核心修复：tuple→int乘法
    #             self.window_size[0] * self.window_size[1],
    #             -1)
    #     relative_position_bias = relative_position_bias.permute(2, 0, 1).float().contiguous()
    #     attn = attn + relative_position_bias.unsqueeze(0)

    #     # Mask处理：强制FP32
    #     if mask is not None:
    #         mask = mask.float().contiguous()  # 强制FP32
    #         nW = mask.shape[0]
    #         attn = attn.view(B // nW, nW, self.num_heads, N_ori, N_ori) + mask.unsqueeze(1).unsqueeze(0)
    #         attn = attn.view(-1, self.num_heads, N_ori, N_ori).float().contiguous()
        
    #     # Softmax + Dropout：强制FP32
    #     attn = self.softmax(attn)
    #     attn = self.attn_drop(attn).float().contiguous()

    #     # ========== 步骤5：最终输出计算（确保FP32+连续） ==========
    #     attn_reshaped = attn.reshape(batch_num_heads, N_ori, N_ori).float().contiguous()
    #     x_reshaped = torch.matmul(attn_reshaped, v_reshaped).float().contiguous()  # FP32计算
        
    #     # 恢复原始维度 + 转回原始类型（兼容后续流程）
    #     x = x_reshaped.reshape(B, self.num_heads, N_ori, C_head).transpose(1, 2).reshape(B, N_ori, C).float().contiguous()
    #     x = self.proj(x).float().contiguous()
    #     x = self.proj_drop(x).float().contiguous()

    #     print(f"[A100 Trace] ✅ 最终输出：x.shape={x.shape}, 连续={x.is_contiguous()}, 类型={x.dtype}, device={x.device}")
    #     return x


    def forward(self, x, mask=None):
        """
        Args:
            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        # ========== 基础信息 + 分布式Trace ==========
        import torch.distributed as dist
        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            rank = -1
            world_size = -1
        print(f"\n===== [A100 Trace] 分布式环境：rank={rank}, world_size={world_size} =====")
        
        B, N_ori, C = x.shape  # N_ori=49（window_size=7）
        # 关键1：保存qkv层原本的设备（cuda:0），临时将权重转到CPU
        orig_device = self.qkv.weight.device
        self.qkv.weight = torch.nn.Parameter(self.qkv.weight.cpu().float())
        if self.qkv.bias is not None:
            self.qkv.bias = torch.nn.Parameter(self.qkv.bias.cpu().float())
        
        # 关键2：输入x转到CPU（和权重设备对齐）+ FP32
        x = x.cpu().float().contiguous()  
        print(f"[A100 Trace] 输入x: shape={x.shape}, 连续={x.is_contiguous()}, 类型={x.dtype}, device={x.device}")

        # ========== 步骤1：QKV计算 + 维度重塑（CPU，设备对齐） ==========
        qkv = self.qkv(x).reshape(B, N_ori, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0].cpu().float().contiguous(), qkv[1].cpu().float().contiguous(), qkv[2].cpu().float().contiguous()
        
        # 重塑维度：全程CPU
        batch_num_heads = B * self.num_heads
        C_head = C // self.num_heads
        q_reshaped = q.reshape(batch_num_heads, N_ori, C_head).cpu().float().contiguous()
        k_reshaped = k.reshape(batch_num_heads, N_ori, C_head).cpu().float().contiguous()
        v_reshaped = v.reshape(batch_num_heads, N_ori, C_head).cpu().float().contiguous()
        print(f"[A100 Trace] 重塑后：q_reshaped.shape={q_reshaped.shape}, 连续={q_reshaped.is_contiguous()}, 类型={q_reshaped.dtype}, device={q_reshaped.device}")

        # ========== 步骤2：矩阵乘法（全程CPU，禁用CUDA） ==========
        q_scaled = q_reshaped * self.scale
        k_t_reshaped = k_reshaped.transpose(-2, -1).cpu().float().contiguous()
        
        # 直接CPU计算，不尝试CUDA，彻底绕开A100接口缺陷
        attn_reshaped = torch.matmul(q_scaled, k_t_reshaped).cpu().float().contiguous()
        print(f"[A100 Trace] ✅ CPU GEMM成功！attn_reshaped.shape={attn_reshaped.shape}, 类型={attn_reshaped.dtype}, device={attn_reshaped.device}")

        # ========== 步骤3：恢复原始维度（全程CPU） ==========
        attn = attn_reshaped.reshape(B, self.num_heads, N_ori, N_ori).cpu().float().contiguous()
        print(f"[A100 Trace] 恢复维度后：attn.shape={attn.shape}, 连续={attn.is_contiguous()}, 类型={attn.dtype}, device={attn.device}")

        # ========== 步骤4：相对位置偏置 + mask + softmax（全程CPU） ==========
        # 修复tuple乘法 + 强制CPU+FP32（先转CPU，和当前张量对齐）
        relative_position_bias_table = self.relative_position_bias_table.cpu().float().contiguous()
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).cpu().float().contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        # Mask处理：强制CPU+FP32（若有mask，先转CPU）
        if mask is not None:
            mask = mask.cpu().float().contiguous() if mask is not None else None
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N_ori, N_ori) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N_ori, N_ori).cpu().float().contiguous()

        # Softmax + Dropout：全程CPU（修复Dropout设备处理逻辑）
        attn = self.softmax(attn)
        # 核心修复：Dropout的p是浮点数，无device属性，直接将Dropout层转到CPU计算
        orig_attn_drop_device = next(self.attn_drop.parameters()).device if len(list(self.attn_drop.parameters())) > 0 else self.qkv.weight.device
        self.attn_drop = self.attn_drop.cpu()  # 转CPU
        attn = self.attn_drop(attn).cpu().float().contiguous()
        self.attn_drop = self.attn_drop.to(orig_attn_drop_device)  # 恢复设备

        # ========== 步骤5：最终输出计算（CPU→CUDA，适配后续流程） ==========
        attn_reshaped = attn.reshape(batch_num_heads, N_ori, N_ori).cpu().float().contiguous()
        x_reshaped = torch.matmul(attn_reshaped, v_reshaped).cpu().float().contiguous()  # CPU计算

        # 恢复维度 + 转回CUDA（适配后续模型流程）
        x = x_reshaped.reshape(B, self.num_heads, N_ori, C_head).transpose(1, 2).reshape(B, N_ori, C).cpu().float().contiguous()

        # 临时将proj层转到CPU计算，再转回CUDA
        orig_proj_device = self.proj.weight.device
        self.proj = self.proj.cpu()  # 直接转层到CPU，而非单独转权重
        x = self.proj(x).cpu().float().contiguous()
        self.proj = self.proj.to(orig_proj_device)  # 恢复层设备

        # 修复proj_drop的设备处理（同理，不访问p.device）
        orig_proj_drop_device = next(self.proj_drop.parameters()).device if len(list(self.proj_drop.parameters())) > 0 else orig_proj_device
        self.proj_drop = self.proj_drop.cpu()
        x = self.proj_drop(x).cpu().float().contiguous()
        self.proj_drop = self.proj_drop.to(orig_proj_drop_device)

        # 关键3：恢复qkv层权重到原设备（cuda:0），避免影响后续训练
        self.qkv.weight = torch.nn.Parameter(self.qkv.weight.to(orig_device))
        if self.qkv.bias is not None:
            self.qkv.bias = torch.nn.Parameter(self.qkv.bias.to(orig_device))

        # 最终转回CUDA，确保后续流程在GPU运行
        x = x.to(orig_device).float().contiguous()
        print(f"[A100 Trace] ✅ 最终输出：x.shape={x.shape}, 连续={x.is_contiguous()}, 类型={x.dtype}, device={x.device}")
        return x        




    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super(SwinBlock, self).__init__()

        self.init_cfg = init_cfg
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class SwinBlockSequence(BaseModule):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


@BACKBONES.register_module()
class SwinTransformer(BaseModule):
    """ Swin Transformer
    A PyTorch implement of : `Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows`  -
        https://arxiv.org/abs/2103.14030

    Inspiration from
    https://github.com/microsoft/Swin-Transformer

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dims (int): The feature dimension. Default: 96.
        patch_size (int | tuple[int]): Patch size. Default: 4.
        window_size (int): Window size. Default: 7.
        mlp_ratio (int | float): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin
            Transformer stage. Default: (3, 6, 12, 24).
        strides (tuple[int]): The patch merging or patch embedding stride of
            each Swin Transformer stage. (In swin, we set kernel size equal to
            stride.) Default: (4, 2, 2, 2).
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch
            merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: False.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer at
            output of backone. Defaults: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            Default: -1 (-1 means not freezing any parameters).
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 init_cfg=None):
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super(SwinTransformer, self).__init__(init_cfg=init_cfg)

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, embed_dims, patch_row, patch_col)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=int(mlp_ratio * in_channels),
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            if self.convert_weights:
                # supported loading weight from original repo,
                _state_dict = swin_converter(_state_dict)

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, False)

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            h, w = self.absolute_pos_embed.shape[1:3]
            if hw_shape[0] != h or hw_shape[1] != w:
                absolute_pos_embed = F.interpolate(
                    self.absolute_pos_embed,
                    size=hw_shape,
                    mode='bicubic',
                    align_corners=False).flatten(2).transpose(1, 2)
            else:
                absolute_pos_embed = self.absolute_pos_embed.flatten(
                    2).transpose(1, 2)
            x = x + absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)

        return outs