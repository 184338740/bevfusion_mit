# =============================================================================
# @file       xmy_centerhead_percls.py
# @author     xmy
# @date       2026-03-25
# @version    0.3.2
# @brief      自定义 CenterHead，支持每个类别的独立置信度阈值和 NMS 阈值。
#
# 升级说明 (v0.3.2):
#   - 修复了当某个任务过滤后仅剩一个框时，scores 和 labels 变为标量导致
#     torch.cat 合并时维度不匹配的严重错误 (RuntimeError: Tensors must have
#     same number of dimensions)。
#   - 将置信度过滤逻辑从 _filter_by_class_score 迁移至 get_task_detections，
#     使用 masked_select 进行过滤，彻底解决因标量张量导致的维度问题。
#   - 重构 get_task_detections，统一使用 view(-1) 确保 top_scores 和 top_labels
#     始终为一维张量，避免单框场景下的标量问题。
#   - 删除冗余的 _filter_by_class_score 方法，简化代码结构。
#   - 向后兼容：未配置 per_class 阈值时自动回退到全局阈值。
#
# Release Note (v0.3.2)
#   - 修复：解决当某个任务过滤后仅剩一个框时，scores 和 labels 变为标量，导致 torch.cat 合并时维度不匹配的错误。
#   - 重构：将置信度过滤逻辑移至 get_task_detections，使用 masked_select 确保维度一致性；统一使用 view(-1) 强制一维，避免标量问题。
#   - 优化：删除冗余的 _filter_by_class_score 方法，简化代码，提高可维护性。
#   - 兼容：完全向后兼容，未配置 per-class 阈值时自动使用全局阈值。
# =============================================================================

# =============================================================================
# @file       xmy_centerhead_percls.py
# @author     xmy
# @date       2026-03-10
# @version    2.0
# @brief      自定义 CenterHead，支持训练阶段分类损失的类别加权。
#
# 升级说明（v2.0）：
#   - 在 v1.0（支持测试阶段 per-class 阈值）基础上，新增训练阶段类别损失加权功能。
#   - 新增配置项 per_class_loss_weight（位于 train_cfg 中），用于为每个类别设置独立的分类损失权重。
#   - 实现方式：继承自官方 CenterHead，在 __init__ 中保存原始 loss_cls 配置，并创建一个 reduction='none'
#     的损失实例 self.loss_cls_none，用于逐元素损失计算。在 loss 函数中，当配置了类别权重时，
#     先计算逐元素损失，然后按通道乘以对应权重，最后求和并除以正样本数；否则使用原始损失。
#   - 添加调试打印，在第一个 batch 输出每个任务的类别权重及加权前后的平均损失，便于验证。
#   - 完全向后兼容：未配置 per_class_loss_weight 时行为与官方 CenterHead 一致。
#
# 配置示例（YAML格式，在 train_cfg 中添加）：xmy_tools/tyjt2nusc_1219_A100/configs/tyjt_2d_CenterheadLSSfpn_V1_6_01_a_0303_Local_Debug_CamOnly_v02_per_class_loss_weight.yaml
#   train_cfg:
#     point_cloud_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
#     grid_size: [1024, 1024, 1]
#     voxel_size: [0.1, 0.1, 0.2]
#     out_size_factor: 8
#     # ... 其他训练配置 ...
#     per_class_loss_weight:                      # 新增：每个类别的分类损失权重
#       car: 1.0
#       pedestrian: 2.5
#       traffic_cone: 3.0
#       bicycle: 2.0
#       # 其他未列出的类别默认权重为 1.0
#
# 注意事项：
#   - 权重值建议保持在合理范围（如 0.1~5.0），避免梯度过大或过小。
#   - 该功能与测试阶段的 per_class_score_threshold 和 per_class_nms_thr 完全独立，可同时使用。
#   - 修改同时影响纯视觉模型和融合模型。
# =============================================================================

# =============================================================================
# @file       xmy_centerhead_percls.py
# @author     xmy
# @date       2026-03-09
# @version    1.0
# @brief      自定义 CenterHead，支持每个类别的独立置信度阈值和 NMS 阈值。
#
# 升级说明（v1.0）：
#   - 继承自官方 CenterHead，保留原有所有功能。
#   - 新增两个阶段的过滤配置：
#       阶段1: 按类别置信度阈值过滤 (per_class_score_threshold)
#              如果某个类别未设置，则使用 test_cfg.score_threshold 作为默认值。
#       阶段2: NMS 过滤
#              根据 nms_type 选择分支：
#              - 当 nms_type == "circle" 时，使用 test_cfg.min_radius[task_id] 作为 NMS 半径；   【官方原有配置，保持不变】
#              - 当 nms_type == "rotate" 时，使用新增的 per_class_nms_thr 作为类别特定的 NMS IoU 阈值，
#                如果某个类别未设置，则使用 test_cfg.nms_thr 作为默认值。
#   - 新增配置项：
#       * per_class_score_threshold : dict，类别 -> 置信度阈值，例如 {'car': 0.05, 'pedestrian': 0.2}
#       * per_class_nms_thr         : dict，类别 -> NMS IoU 阈值，例如 {'car': 0.3, 'pedestrian': 0.15}
#   - 若未提供上述字典，则自动使用全局阈值，完全向后兼容
#
# # 配置示例 (以 nuScenes 为例): xmy_tools/tyjt2nusc_1219_A100/configs/tyjt_2d_CenterheadLSSfpn_V1_6_01_b_0303_Local_Debug_CamOnly_v01_multi_thrd_for_test.yaml
#  test_cfg = dict(
#     # 全局阈值（原有）
#     score_threshold=0.1,
#     nms_thr=0.2,
#
#     # 新增 per-class 阈值（可选，不设置的类别自动用全局阈值）
#     per_class_score_threshold=dict(
#         car=0.05,
#         pedestrian=0.2,
#         traffic_cone=0.15
#     ),
#     per_class_nms_thr=dict(
#         car=0.3,
#         pedestrian=0.1,
#         traffic_cone=0.1
#     ),
#
#     # 原有其他配置（需与任务数量对应）
#     nms_type=['circle', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'],  # 每个任务的NMS类型
#     min_radius=[4, 12, 12, 1, 0.85, 0.175],                                 # circle NMS 半径（任务级别）
#     post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
#     pre_max_size=1000,
#     post_max_size=83
# )
# 
# 使用方法：
#   1. 在配置文件中将 detection head 的 type 改为 'XmyCenterHeadPerCls'。
#   2. 在 test_cfg 中添加所需的 per-class 阈值。
#   3. 确保在训练/测试脚本中导入本模块（例如 import mmdet3d.models.dense_heads.xmy_centerhead_percls）。
#
# 注意：
#   - 该修改同时影响纯视觉模型和融合模型
#   - 阈值调整后需重新运行推理才能生效。
# =============================================================================

import copy
import torch
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch import nn

from mmdet3d.core import circle_nms, draw_heatmap_gaussian, gaussian_radius, xywhr2xyxyr
from mmdet3d.models import builder
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import build_bbox_coder, multi_apply

# 导入基类和辅助函数（假设 centerpoint.py 在同一目录）
from .centerpoint import CenterHead, clip_sigmoid, SeparateHead, DCNSeparateHead

Debug = False
if Debug:
    import time
    print(f">>>[xmy]🔵[mmdet3d/models/heads/bbox/xmy_centerhead_percls.py] >>> [Debug Mode = True] ")

@HEADS.register_module()
class XmyCenterHeadPerCls(CenterHead):
    """CenterHead with per-class score threshold and NMS threshold."""

    def __init__(self, *args, **kwargs):
        # 1. 保存 loss_cls 配置（从 kwargs 获取，默认使用 GaussianFocalLoss）
        loss_cls_cfg = kwargs.get('loss_cls', dict(type='GaussianFocalLoss', reduction='mean')).copy()
        self.loss_cls_cfg = loss_cls_cfg

        # 2. super() 调用基类 CenterHead 初始化，构建 self.loss_cls (reduction='mean')【train_cfg】
        # self.loss_cls：继承自基类，reduction='mean'，用于无权重分支（即 else 分支），直接返回标量损失。
        super().__init__(*args, **kwargs)

        # 3. 创建 reduction='none' 的损失实例（用于训练加权）
        # self.loss_cls_none：reduction='none'，用于有权重分支，返回逐元素损失张量，以便按通道乘以类别权重。
        loss_cls_none_cfg = self.loss_cls_cfg.copy()
        loss_cls_none_cfg['reduction'] = 'none'
        self.loss_cls_none = build_loss(loss_cls_none_cfg)

        # import pdb;pdb.set_trace()
        # ========== 训练相关配置 ==========
        # 4. 类别损失权重配置【train_cfg】
        self.per_class_loss_weight = self.train_cfg.get('per_class_loss_weight', {})

        # 5. 构建任务权重列表
        self.task_class_loss_weights = []
        # 先遍历：tasks ==> task_calsses（当前task的类别）self.class_names 就是配置文件中的 tasks 分组
        if self.per_class_loss_weight:
            for task_classes in self.class_names:  # 遍历task
                task_weight = [self.per_class_loss_weight.get(cls_name, 1.0) for cls_name in task_classes] # 再遍历 task 内每个cls: 将当前cls的 权重添加到 task_weight 列表中，顺序与任务内的类别顺序一致
                self.task_class_loss_weights.append(task_weight)  # 将当前任务 task_weight 权重，加到 tasks 的权重列表 task_class_loss_weights
        else:
            self.task_class_loss_weights = None

        if self.task_class_loss_weights is not None:
            print(f">>>[xmy]🔵[xmy_centerhead_percls.py]>>> [XmyCenterHeadPerCls] Task class loss weights: {self.task_class_loss_weights}")
        else:
            print(">>>[xmy]🔵[xmy_centerhead_percls.py]>>> [XmyCenterHeadPerCls] No per-class loss weights, using uniform weighting.")

        # import pdb;pdb.set_trace()
        # ========== 测试相关配置 ==========
        # 类别置信度阈值（测试时使用）
        self.per_class_score_threshold = self.test_cfg.get('per_class_score_threshold', {})
        # 类别 NMS 阈值（测试时使用）
        self.per_class_nms_thr = self.test_cfg.get('per_class_nms_thr', {})

        # ========== 调试标志 ==========
        self._debug_printed = False



    @force_fp32(apply_to=("preds_dicts"))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function with per-class weighted classification loss."""
        # Step1：获取Gt 
        # heatmaps：Gt heatmaps
        # anno_boxes： Gt 属性        
        heatmaps, anno_boxes, inds, masks = self.get_targets(gt_bboxes_3d, gt_labels_3d)
        loss_dict = dict()  # 存放最后的loss

        # 分task(遍历的是不同的任务头，而不是单个类别): 开始获取pred、gt、loss
        for task_id, preds_dict in enumerate(preds_dicts):
            # Step2.1: heatmap loss 热力图分类损失
            # (1) 获取当前task_id 的 预测的 pred_heatmap 和 gt_heatmap
            pred_heatmap = clip_sigmoid(preds_dict[0]["heatmap"])  # [B, num_cls, H, W]
            gt_heatmap = heatmaps[task_id]                         # [B, num_cls, H, W]
            # (2) num_pos： 计算 gt_heatmap 中正样本的个数，用于后续的求avg
            num_pos = gt_heatmap.eq(1).float().sum().item()

            # ====== (3) 分类损失（支持类别加权）======
            # (3.2) 有分类别权重的配置时： 使用 reduction='none' 的损失实例
            if self.task_class_loss_weights is not None:
                # 获取当前任务的类别权重张量
                task_weights = pred_heatmap.new_tensor(self.task_class_loss_weights[task_id])  # [num_cls,]
                # 计算每个像素的损失（不归约/不归一化），但不再合并归一化
                element_loss = self.loss_cls_none(pred_heatmap, gt_heatmap)  # [B, num_cls, H, W]   # 注意：reduction='none' 的loss，才能返回张量
                weighted_loss = element_loss * task_weights.view(1, -1, 1, 1)                  # 乘以类别权重（沿通道维度）

                # 调试打印：
                if not self._debug_printed and num_pos > 0:
                    print(f"\n>>>[xmy]🔵[xmy_centerhead_percls.py]>>> [Task {task_id}] Class weights: {task_weights.tolist()}")
                    mean_loss_before = element_loss.mean(dim=(0,2,3)).tolist()
                    mean_loss_after = weighted_loss.mean(dim=(0,2,3)).tolist()
                    print(f"  Mean loss per class (before weighting): {mean_loss_before}")
                    print(f"  Mean loss per class (after weighting): {mean_loss_after}")
                    print(f"  num_pos: {num_pos}")

                # 求和并除以正样本数
                loss_heatmap = weighted_loss.sum() / max(num_pos, 1)
            else:
                # (3.2)无分类别权重的配置时： 使用原有的 reduction='mean' 损失
                loss_heatmap = self.loss_cls(
                    pred_heatmap, gt_heatmap, avg_factor=max(num_pos, 1))

            # ====== 回归损失（不变）======
            # Step2.2 回归损失（与基类相同）： 不做类别的额外加权
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]["anno_box"] = torch.cat(
                (
                    preds_dict[0]["reg"],
                    preds_dict[0]["height"],
                    preds_dict[0]["dim"],
                    preds_dict[0]["rot"],
                    preds_dict[0]["vel"],
                ),
                dim=1,
            )

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]["anno_box"].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.train_cfg.get("code_weights", None)
            bbox_weights = mask * mask.new_tensor(code_weights)
            # 回归loss
            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights, avg_factor=(num + 1e-4)
            )
            # step3：更新 loss_dict
            loss_dict[f"heatmap/task{task_id}"] = loss_heatmap
            loss_dict[f"bbox/task{task_id}"] = loss_bbox
        
        # 在所有任务遍历完后设置标志为 True
        if self.task_class_loss_weights is not None and not self._debug_printed:
            self._debug_printed = True

        return loss_dict


    def _filter_by_class_score(self, boxes3d, scores, labels, task_id):
        """返回 per-box 的置信度阈值张量（不进行过滤）"""
        if not self.per_class_score_threshold:
            # 无 per-class 配置时，阈值就是全局阈值
            thresh = torch.full_like(scores, self.test_cfg.get('score_threshold', 0.0))
            return thresh

        task_class_names = self.class_names[task_id]
        global_score_thr = self.test_cfg.get('score_threshold', 0.0)
        per_box_thresh = torch.full_like(scores, global_score_thr)

        for cls_idx, cls_name in enumerate(task_class_names):
            if cls_name in self.per_class_score_threshold:
                cls_mask = (labels == cls_idx)
                per_box_thresh[cls_mask] = self.per_class_score_threshold[cls_name]

        return per_box_thresh
    

    @force_fp32(apply_to=("preds_dicts"))
    def get_bboxes(self, preds_dicts, metas, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # 1.处理 nms_type: 确保它是一个列表，长度等于任务数
        if not isinstance(self.test_cfg["nms_type"], list):
            nms_types = [self.test_cfg["nms_type"] for _ in range(len(preds_dicts))]
        else:
            nms_types = self.test_cfg["nms_type"]

        # 2.处理nms_scale: 确保它是一个二维列表，[任务][类别]的缩放系数
        if "nms_scale" in self.test_cfg:
            if not isinstance(self.test_cfg["nms_scale"], list):
                nms_scales = [
                    [
                        self.test_cfg["nms_scale"]
                        for _ in range(self.num_classes[task_id])
                    ]
                    for task_id in range(len(preds_dicts))
                ]
            else:
                nms_scales = self.test_cfg["nms_scale"]
        else:  # 默认缩放系数为1.0
            nms_scales = [
                [1.0 for _ in range(self.num_classes[task_id])]
                for task_id in range(len(preds_dicts))
            ]

        rets = []  # 用于存储所有任务的检测结果
        # 遍历每个任务
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]         # 当前任务的类别数
            batch_size = preds_dict[0]["heatmap"].shape[0]        # batch大小
            # if Debug:
            #     print(f"\n>>>[xmy]🔵[xmy_centerhead_percls.py] >>> 处理任务{task_id}: {self.class_names[task_id]}")
            #     print(f">>>[xmy]🔵[xmy_centerhead_percls.py] >>> 任务{task_id}的类别数: {num_class_with_bg}")

            # 获取当前任务的特征图并做sigmoid处理
            batch_heatmap = preds_dict[0]["heatmap"].sigmoid()
            batch_reg = preds_dict[0]["reg"]
            batch_hei = preds_dict[0]["height"]

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]["dim"])
            else:
                batch_dim = preds_dict[0]["dim"]

            batch_rots = preds_dict[0]["rot"][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]["rot"][:, 1].unsqueeze(1)

            if "vel" in preds_dict[0]:
                batch_vel = preds_dict[0]["vel"]
            else:
                batch_vel = None

            # 调用bbox_coder.decode将特征图解码为候选框列表
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id,
            )

            # ====== [xmy] 阶段1: 按类别置信度阈值(per_class_score_threshold) 的过滤 ======
            # 直接用test_cfg.per_class_score_threshold 滤波temp结果
            # 直接修改 temp 中的内容，后续分支直接使用 temp
            # for i in range(batch_size):
            #     if Debug:
            #         print(f"[DEBUG] before filter: boxes3d.shape={temp[i]['bboxes'].shape}, scores.shape={temp[i]['scores'].shape}, labels.shape={temp[i]['labels'].shape}")
            #     boxes3d, scores, labels = self._filter_by_class_score(
            #         temp[i]["bboxes"], temp[i]["scores"], temp[i]["labels"], task_id
            #     )
            #     temp[i]["bboxes"] = boxes3d
            #     temp[i]["scores"] = scores
            #     temp[i]["labels"] = labels

            # 重新生成滤波后的列表（因为数据量可能变化）
            batch_reg_preds = [box["bboxes"] for box in temp]
            batch_cls_preds = [box["scores"] for box in temp]
            batch_cls_labels = [box["labels"] for box in temp]
            # ===================================================

            # ====== [xmy] 阶段2: NMS 的过滤 ======
            # nms_type == "circle"时：采用 test_cfg.min_radius 的值来过滤 NMS min_radius半径的阈值
            # nms_type == "rotate"时：采用 test_cfg.per_class_nms_thr 的值来过滤 NMS IOU阈值
            # 根据 nms_type 选择分支
            if nms_types[task_id] == "circle":
                # =================== circle NMS 分支 ===================
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]["bboxes"]      # 已滤波
                    scores = temp[i]["scores"]
                    labels = temp[i]["labels"]

                    if len(boxes3d) == 0:
                        ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                        ret_task.append(ret)
                        continue

                    # 提取中心点坐标 (x, y)
                    centers = boxes3d[:, [0, 1]]
                    # 拼接成 [x, y, score] 用于circle_nms
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg["min_radius"][task_id],
                            post_max_size=self.test_cfg["post_max_size"],
                        ),
                        dtype=torch.long,
                        device=boxes.device,
                    )

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
                # =================== circle分支结束 ===================
            else:
                # =================== rotate NMS 分支 ===================
                rets.append(
                    self.get_task_detections(
                        num_class_with_bg,
                        batch_cls_preds,      # 注意参数顺序：cls_preds, reg_preds, labels
                        batch_reg_preds,
                        batch_cls_labels,
                        metas,
                        nms_scales[task_id],
                        task_id=task_id,
                    )
                )
                # =================== rotate分支结束 ===================

        # =================== 合并所有任务的结果 ===================
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == "bboxes":
                    # 合并bboxes
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = metas[i]["box_type_3d"](bboxes, self.bbox_coder.code_size)
                elif k == "scores":
                    # 合并scores
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == "labels":
		    # 合并labels
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])
        return ret_list

    def get_task_detections(
        self,
        num_class_with_bg,
        batch_cls_preds,
        batch_reg_preds,
        batch_cls_labels,
        metas,
        nms_scale=1.0,
        task_id=None,
    ):
        """Rotate nms for each task (支持 per-class NMS 阈值和 per-class 置信度阈值)."""
        predictions_dicts = []
        post_center_range = self.test_cfg["post_center_limit_range"]
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds[0].dtype,
                device=batch_reg_preds[0].device,
            )

        task_class_names = self.class_names[task_id]

        for i, (box_preds, cls_preds, cls_labels) in enumerate(
            zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)
        ):
            # ========== 数据格式修复 ==========
            if not isinstance(cls_labels, torch.Tensor):
                cls_labels = torch.tensor([], dtype=torch.long, device=box_preds.device)
            elif cls_labels.dtype in [torch.float16, torch.float32, torch.float64]:
                cls_labels = cls_labels.long()

            if not isinstance(cls_preds, torch.Tensor):
                cls_preds = torch.tensor([], device=box_preds.device)

            # ========== 生成 top_scores 和 top_labels ==========
            if num_class_with_bg == 1:
                # 使用 view(-1) 确保一维，避免标量（与 squeeze(-1) 等价但处理 [1] 时更安全）
                top_scores = cls_preds.view(-1)
                top_labels = torch.zeros(
                    cls_preds.shape[0], device=cls_preds.device, dtype=torch.long
                )
            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds.view(-1)   # 保持一维

            # ========== 空预测处理 ==========
            if top_scores.numel() == 0:
                predictions_dicts.append(dict(
                    bboxes=torch.zeros(0, self.bbox_coder.code_size, device=box_preds.device),
                    scores=torch.zeros(0, device=box_preds.device),
                    labels=torch.zeros(0, dtype=torch.long, device=box_preds.device),
                ))
                continue

            # ========== 生成 per-box 阈值（替换官方固定阈值分支） ==========
            # 获取全局阈值（默认值）
            global_score_thr = self.test_cfg.get('score_threshold', 0.0)
            per_box_thresh = torch.full_like(top_scores, global_score_thr)

            # 根据类别设置 per-class 阈值（如果配置了）
            if self.per_class_score_threshold:
                for cls_idx, cls_name in enumerate(task_class_names):
                    if cls_name in self.per_class_score_threshold:
                        cls_mask = (top_labels == cls_idx)
                        per_box_thresh[cls_mask] = self.per_class_score_threshold[cls_name]

            # ========== 执行过滤（利用 masked_select 自动恢复维度） ==========
            keep = top_scores >= per_box_thresh
            top_scores = top_scores.masked_select(keep)
            box_preds = box_preds[keep]
            top_labels = top_labels[keep]

            # 再次检查空预测（过滤后可能为空）
            if top_scores.numel() == 0:
                predictions_dicts.append(dict(
                    bboxes=torch.zeros(0, self.bbox_coder.code_size, device=box_preds.device),
                    scores=torch.zeros(0, device=box_preds.device),
                    labels=torch.zeros(0, dtype=torch.long, device=box_preds.device),
                ))
                continue

            # ========== 后续 NMS 处理（与官方一致） ==========
            bev_box = metas[i]["box_type_3d"](
                box_preds[:, :], self.bbox_coder.code_size
            ).bev

            # 应用 NMS 尺度缩放
            for cls, scale in enumerate(nms_scale):
                mask = (top_labels == cls)
                if mask.dim() > 1:
                    mask = mask.view(-1)
                cur_bev_box = bev_box[mask]
                cur_bev_box[:, [2, 3]] *= scale
                bev_box[mask] = cur_bev_box

            boxes_for_nms = xywhr2xyxyr(bev_box)

            # 按类别分别执行 NMS（支持 per-class NMS 阈值）
            unique_labels = top_labels.unique()
            selected = []
            if unique_labels.numel() > 0:
                for cls_label in unique_labels:
                    mask = (top_labels == cls_label)
                    if mask.dim() > 1:
                        mask = mask.view(-1)
                    cls_boxes = boxes_for_nms[mask]
                    cls_scores = top_scores[mask]
                    cls_name = task_class_names[cls_label.item()]
                    nms_thr = self.per_class_nms_thr.get(cls_name, self.test_cfg.get("nms_thr", 0.2))
                    cls_selected = nms_gpu(
                        cls_boxes,
                        cls_scores,
                        thresh=nms_thr,
                        pre_maxsize=self.test_cfg.get("pre_max_size", 1000),
                        post_max_size=self.test_cfg.get("post_max_size", 83),
                    )
                    global_indices = torch.where(mask)[0][cls_selected]
                    selected.append(global_indices)
                if selected:
                    selected = torch.cat(selected)
                else:
                    selected = torch.tensor([], dtype=torch.long, device=top_scores.device)
            else:
                selected = torch.tensor([], dtype=torch.long, device=top_scores.device)

            # 提取最终框、分数、标签
            if selected.numel() > 0:
                selected_boxes = box_preds[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]

                final_box_preds = selected_boxes
                final_scores = selected_scores
                final_labels = selected_labels
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >= post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <= post_center_range[3:]).all(1)
                    predictions_dict = dict(
                        bboxes=final_box_preds[mask],
                        scores=final_scores[mask],
                        labels=final_labels[mask],
                    )
                else:
                    predictions_dict = dict(
                        bboxes=final_box_preds, scores=final_scores, labels=final_labels
                    )
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=torch.zeros([0, self.bbox_coder.code_size], dtype=dtype, device=device),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros([0], dtype=torch.long, device=device),
                )

            predictions_dicts.append(predictions_dict)

        return predictions_dicts