#!/usr/bin/env python3
"""
离线扫描 TYJT 数据集，找出可能导致训练 NaN 的异常样本。
使用方法：修改脚本中的 CONFIG_FILE 和 SPLIT 后直接运行。
"""

import os
import sys
import torch
import numpy as np
from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets.pipelines.transforms_3d import GTDepth
from mmdet3d.models import build_model
from mmdet3d.models.heads.bbox.centerpoint import CenterHead
from mmdet3d.utils import recursive_eval

# ==================== 用户配置区域 ====================
CONFIG_FILE = "/data2/xmy/01_project/bevfusion_mit_xmy_1205/xmy_tools/tyjt2pkl_v2/0311/configs/tyjt_2d_CenterheadLSSfpn_PlanB_V1_2_i7_0507_epoch02_XmyCenterHeadPerCls_TYJTDatasetV2_360x640_Car_LRcircle_BEVDepth.yaml"   # 你的训练配置文件路径
SPLIT = "train"        # 可选 "train", "val", "test"
MAX_SAMPLES = None     # 限制扫描样本数，None 表示全部
# ====================================================

def check_bbox(bboxes, token):
    """检查 3D 框基本属性"""
    issues = []
    dims = bboxes.dims
    if (dims <= 0).any():
        issues.append(f"non-positive dims: {dims.tolist()}")
    centers = bboxes.gravity_center
    if not torch.isfinite(centers).all():
        issues.append(f"infinite center: {centers.tolist()}")
    yaws = bboxes.yaw
    if not torch.isfinite(yaws).all():
        issues.append(f"infinite yaw: {yaws.tolist()}")
    return issues

def simulate_heatmap_target(gt_boxes, gt_labels, task_id, head):
    """模拟 get_targets 生成热图，检查是否有 NaN"""
    gt_boxes_list = [gt_boxes]
    gt_labels_list = [gt_labels]
    try:
        heatmaps, _, _, _ = head.get_targets(gt_boxes_list, gt_labels_list)
        heatmap = heatmaps[task_id][0]  # 取出对应任务的热图（单样本）
        if torch.isnan(heatmap).any() or torch.isinf(heatmap).any():
            return True, heatmap
        else:
            return False, heatmap
    except Exception as e:
        return True, str(e)

def main():
    # 加载配置（与官方训练脚本一致）
    configs.load(CONFIG_FILE, recursive=True)
    cfg = Config(recursive_eval(configs), filename=CONFIG_FILE)

    # 获取对应划分的数据集配置
    if SPLIT == "train":
        dataset_cfg = cfg.data.train.dataset
    elif SPLIT == "val":
        dataset_cfg = cfg.data.val
    elif SPLIT == "test":
        dataset_cfg = cfg.data.test
    else:
        raise ValueError(f"Unknown split: {SPLIT}")

    # 强制使用 test_pipeline（确定性，无随机增强）
    if hasattr(cfg, "test_pipeline"):
        dataset_cfg.pipeline = cfg.test_pipeline
    else:
        dataset_cfg.pipeline = cfg.data.val.pipeline

    # 构建数据集
    print("Building dataset...")
    dataset = build_dataset(dataset_cfg)
    print(f"Dataset size: {len(dataset)}")

    # # 构建检测头（用于 get_targets）
    # head_cfg = cfg.model.heads.object
    # # 使用基类 CenterHead（XmyCenterHeadPerCls 未重写 get_targets）
    # head = CenterHead(
    #     tasks=head_cfg.tasks,
    #     common_heads=head_cfg.common_heads,
    #     share_conv_channel=head_cfg.share_conv_channel,
    #     bbox_coder=head_cfg.bbox_coder,
    #     train_cfg=head_cfg.train_cfg,
    #     test_cfg=head_cfg.test_cfg,
    #     loss_cls=head_cfg.loss_cls,
    #     loss_bbox=head_cfg.loss_bbox,
    #     norm_bbox=head_cfg.norm_bbox,
    # )
    # if torch.cuda.is_available():
    #     head = head.cuda()
    # head.eval()

    # ========== 替换原来的 head 构建部分 ==========
    # 构建完整模型（不加载预训练权重，仅用于 get_targets）
    cfg.model.train_cfg = None   # 训练配置不需要
    model = build_model(cfg.model)
    model.eval()
    # 获取检测头（你的自定义 head 类名可能是 XmyCenterHeadPerCls）
    head = model.heads['object']
    # 移到 CPU 以节省显存（如果不需要 GPU）
    head = head.cpu()

    # 初始化 GTDepth
    gtdepth = GTDepth(keyframe_only=True)



    anomalies = []

    total = len(dataset) if MAX_SAMPLES is None else min(MAX_SAMPLES, len(dataset))
    for idx in range(total):
        if idx % 1000 == 0:
            print(f"Processing {idx}/{total}")
        try:
            data = dataset[idx]
        except Exception as e:
            print(f"Error loading sample index {idx}: {e}")
            continue

        # 获取 token
        token = data.get("metas", {}).get("token", f"sample_{idx}")
        if token == "unknown":
            token = data.get("sample_idx", data.get("token", f"sample_{idx}"))

        # ---- 1. 检查 3D 框 ----
        gt_bboxes = data.get("gt_bboxes_3d")
        gt_labels = data.get("gt_labels_3d")
        if gt_bboxes is not None and len(gt_bboxes) > 0:
            issues = check_bbox(gt_bboxes, token)
            if issues:
                anomalies.append((token, "BBox", issues))
                continue  # 已发现异常，不再检查热图

        # ---- 2. 模拟热图目标生成 ----
        task_id = 0  # 只训练 car，只有一个任务
        if gt_bboxes is not None and len(gt_bboxes) > 0:
            has_nan, result = simulate_heatmap_target(gt_bboxes, gt_labels, task_id, head)
            if has_nan:
                anomalies.append((token, "Heatmap", f"NaN/Inf in heatmap target: {result}"))
                continue

        # ---- 3. 检查深度图 ----
        temp_data = {}
        required_keys = ["camera2ego", "camera_intrinsics", "img_aug_matrix", "lidar_aug_matrix",
                         "lidar2ego", "camera2lidar", "lidar2image", "points", "img"]
        for k in required_keys:
            if k in data:
                temp_data[k] = data[k]
        try:
            depth_out = gtdepth(temp_data)
            depth = depth_out["depths"]
            if torch.isnan(depth).any() or torch.isinf(depth).any():
                anomalies.append((token, "Depth", "NaN/Inf in depth tensor"))
        except Exception as e:
            anomalies.append((token, "Depth", f"GTDepth error: {e}"))

    # 输出结果
    print("\n" + "="*80)
    print(f"Scan completed. Total samples: {total}")
    print(f"Anomalies found: {len(anomalies)}")
    if anomalies:
        print("\nList of anomalies (token, type, reason):")
        for token, typ, reason in anomalies:
            # 将 reason 转为字符串（可能为列表）
            reason_str = str(reason)
            print(f"  {token} : [{typ}] {reason_str}")
        # 保存到文件
        with open("anomaly_samples.txt", "w") as f:
            for token, typ, reason in anomalies:
                f.write(f"{token} : [{typ}] {str(reason)}\n")
        print("\nAnomaly list saved to anomaly_samples.txt")
    else:
        print("No anomalies found.")

if __name__ == "__main__":
    main()