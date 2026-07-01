#!/usr/bin/env python3
"""
精确复现 XmyCenterHeadPerCls 中维度错误的测试脚本
包含两种输入场景：
1. 输入二维 [1,1]（当前测试中未触发错误）
2. 输入一维 [1]（模拟真实问题来源）
"""

import torch

def simulate_filter(scores, labels, keep_mask):
    """模拟 _filter_by_class_score 的原始实现（不处理维度）"""
    scores_filtered = scores[keep_mask]
    labels_filtered = labels[keep_mask]
    return scores_filtered, labels_filtered

def simulate_task_detections(num_class_with_bg, cls_preds, cls_labels, has_unsqueeze_bug=True):
    """模拟 get_task_detections 中有 bug 的部分"""
    print(f"\n--- 进入 simulate_task_detections ---")
    print(f"输入 cls_preds: {cls_preds}, shape: {cls_preds.shape}")
    print(f"输入 cls_labels: {cls_labels}, shape: {cls_labels.shape}")
    print(f"num_class_with_bg: {num_class_with_bg}")

    if num_class_with_bg == 1:
        top_scores = cls_preds.squeeze(-1)
        print(f"cls_preds.squeeze(-1) 后 top_scores: {top_scores}, shape: {top_scores.shape}")
        top_labels = torch.zeros(cls_preds.shape[0], dtype=torch.long)
        print(f"创建的 top_labels: {top_labels}, shape: {top_labels.shape}")
    else:
        top_labels = cls_labels.long()
        top_scores = cls_preds.squeeze(-1)
        print(f"多类: top_scores shape: {top_scores.shape}, top_labels shape: {top_labels.shape}")

    print(f"检查 top_scores.dim(): {top_scores.dim()}")
    if top_scores.dim() == 0:
        print("top_scores 是标量，进入 unsqueeze 分支")
        top_scores = top_scores.unsqueeze(0)
        print(f"unsqueeze 后 top_scores: {top_scores}, shape: {top_scores.shape}")
        if has_unsqueeze_bug:
            print("执行错误扩展：top_labels = top_labels.unsqueeze(0)")
            top_labels = top_labels.unsqueeze(0)
            print(f"unsqueeze 后 top_labels: {top_labels}, shape: {top_labels.shape}")
    else:
        print("top_scores 不是标量，不执行 unsqueeze")

    return top_scores, top_labels

# ==================== 场景1：输入二维 [1,1]（不触发错误） ====================
print("\n" + "="*60)
print("场景1：过滤后得到 [1,1] 形状")
print("="*60)
scores_one = torch.tensor([[0.9]])   # 形状 [1,1]
labels_one = torch.tensor([[0]])      # 形状 [1,1]
keep_one = torch.tensor([True])
scores_filtered, labels_filtered = simulate_filter(scores_one, labels_one, keep_one)
print(f"过滤后 scores_filtered: {scores_filtered}, shape: {scores_filtered.shape}")
print(f"过滤后 labels_filtered: {labels_filtered}, shape: {labels_filtered.shape}")

cls_preds = scores_filtered
cls_labels = labels_filtered
num_class_with_bg = 1

# 有 bug 的版本
top_scores_bug, top_labels_bug = simulate_task_detections(num_class_with_bg, cls_preds, cls_labels, has_unsqueeze_bug=True)
print("\n--- 有 bug 的代码结果 ---")
print(f"top_scores: {top_scores_bug}, shape: {top_scores_bug.shape}")
print(f"top_labels: {top_labels_bug}, shape: {top_labels_bug.shape}")
if top_labels_bug.dim() == 2:
    print("❌ 错误：top_labels 变成了二维 [1,1]")
else:
    print("✅ 正常：top_labels 保持一维")

# ==================== 场景2：输入一维 [1]（模拟真实问题） ====================
print("\n" + "="*60)
print("场景2：过滤后得到 [1] 形状（模拟真实问题）")
print("="*60)
scores_one_1d = torch.tensor([0.9])   # 形状 [1]
labels_one_1d = torch.tensor([0])      # 形状 [1]
keep_one = torch.tensor([True])
scores_filtered_1d, labels_filtered_1d = simulate_filter(scores_one_1d, labels_one_1d, keep_one)
print(f"过滤后 scores_filtered: {scores_filtered_1d}, shape: {scores_filtered_1d.shape}")
print(f"过滤后 labels_filtered: {labels_filtered_1d}, shape: {labels_filtered_1d.shape}")

cls_preds = scores_filtered_1d
cls_labels = labels_filtered_1d
num_class_with_bg = 1

# 有 bug 的版本
top_scores_bug, top_labels_bug = simulate_task_detections(num_class_with_bg, cls_preds, cls_labels, has_unsqueeze_bug=True)
print("\n--- 有 bug 的代码结果 ---")
print(f"top_scores: {top_scores_bug}, shape: {top_scores_bug.shape}")
print(f"top_labels: {top_labels_bug}, shape: {top_labels_bug.shape}")
if top_labels_bug.dim() == 2:
    print("❌ 错误：top_labels 变成了二维 [1,1]，后续合并时会报错")
else:
    print("✅ 正常：top_labels 保持一维")