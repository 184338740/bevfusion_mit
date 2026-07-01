#!/usr/bin/env python3
"""
TYJT 数据集训练脚本，基于 BEVFusion 框架。
用法: python train_tyjt.py <config_file> [--run-dir RUN_DIR]
"""

import argparse
import os
import random
import time

import numpy as np
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval


# torch.autograd.set_detect_anomaly(True)  # detect_anomaly() 会启用异常检测模式，一旦发现异常，立即抛出 RuntimeError，并附带堆栈跟踪
# ========== 梯度监控开关（直接写在代码中，不依赖配置文件） ==========
DEBUG_GRADIENT = False   # 需要监控时改为 True，不需要时改为 False


def main():
    dist.init()

    parser = argparse.ArgumentParser(description="TYJT training script")
    parser.add_argument("config", metavar="FILE", help="path to config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    args = parser.parse_args()

    # 加载配置文件
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    # 设置cudnn
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # 设置运行目录
    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    # 保存配置副本
    cfg.dump(os.path.join(cfg.run_dir, "config.yaml"))
    cfg.dump("cur_config.yaml")

    # 初始化日志
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    logger.info(f"Config:\n{cfg.pretty_text}")

    # 设置随机种子
    if cfg.seed is not None:
        logger.info(f"Set random seed to {cfg.seed}, deterministic: {cfg.deterministic}")
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # 构建数据集
    datasets = [build_dataset(cfg.data.train)]

    # 构建模型
    model = build_model(cfg.model)
    model.init_weights()

    # ========== 新增：梯度监控 Hook（可选，通过配置开关） ==========
    if DEBUG_GRADIENT:
        # import torch.distributed
        def register_gradient_hooks(model):
            def make_hook(param_name):
                def hook(grad):
                    if torch.isinf(grad).any() or torch.isnan(grad).any():
                        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
                            return
                        print(f"\n🚨 Gradient exploded at parameter: {param_name}")
                        print(f"   grad min: {grad.min().item():.6f}, max: {grad.max().item():.6f}")
                        print(f"   has_inf: {torch.isinf(grad).any()}, has_nan: {torch.isnan(grad).any()}")
                        # 可选：保存模型状态或抛出异常
                return hook
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.register_hook(make_hook(name))
        register_gradient_hooks(model)
        logger.info("Gradient monitoring hooks registered (DEBUG_GRADIENT=True).")
    # ============================================================

    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    logger.info(f"Model:\n{model}")

    # 开始训练
    train_model(
        model,
        datasets,
        cfg,
        distributed=True,
        validate=True,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()