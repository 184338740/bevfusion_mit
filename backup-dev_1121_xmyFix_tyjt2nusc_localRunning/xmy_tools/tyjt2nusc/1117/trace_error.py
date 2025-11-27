import traceback

# 模拟错误调用栈
print("=== 错误调用栈分析 ===")
print("文件: /opt/venv/lib/python3.8/site-packages/nuscenes/eval/detection/evaluate.py")
print("行号: 85")
print("代码: assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens)")
print("")
print("调用路径:")
print("1. tools/train.py")
print("2. mmdet3d/apis/train.py:130 → train_model()")
print("3. mmcv/runner/epoch_based_runner.py:148 → run()") 
print("4. mmdet3d/runner/epoch_based_runner.py:14 → train()")
print("5. mmcv/runner/hooks/evaluation.py:267 → after_train_epoch()")
print("6. mmdet/core/evaluation/eval_hooks.py:123 → _do_evaluate()")
print("7. mmcv/runner/hooks/evaluation.py:361 → evaluate()")
print("8. mmdet3d/datasets/nuscenes_dataset.py:570 → evaluate()")
print("9. mmdet3d/datasets/nuscenes_dataset.py:442 → _evaluate_single()")
print("10. nuscenes/eval/detection/evaluate.py:85 → __init__() ← 错误发生!")
