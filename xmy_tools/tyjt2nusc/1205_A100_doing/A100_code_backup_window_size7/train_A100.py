import debug_init_v2_xmy  # 必须最先导入！xmy

import argparse
import copy
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


def main():
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    # dump config
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    # log some basic info
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    print(f"🔵[xmy]>>> tools/train.py::  ==>  datasets = [build_dataset(cfg.data.train)]")
    datasets = [build_dataset(cfg.data.train)]

    print(f"🔵[xmy]>>> tools/train.py::  ==>  model = build_model(cfg.model) ")
    model = build_model(cfg.model)

    print(f"🔵[xmy]>>> tools/train.py::  ==>  model.init_weights() ")
    model.init_weights()
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    logger.info(f"Model:\n{model}")
    print(f"🔵[xmy]>>> tools/train.py::  ==>  train_model(model, datasets, cfg, distributed=True, validate=True,  timestamp=timestamp,) ")
    
    # ========== 新增：强制禁用FP16（关键！） ==========
    # 1. 禁用模型的FP16配置
    if hasattr(cfg.model, 'fp16'):
        delattr(cfg.model, 'fp16')
    cfg.model.pop('fp16', None)
    
    # 2. 禁用优化器的FP16配置
    if 'optimizer_config' in cfg and hasattr(cfg.optimizer_config, 'fp16'):
        delattr(cfg.optimizer_config, 'fp16')
    cfg.optimizer_config.pop('fp16', None)
    
    # 3. 全局禁用FP16
    if 'fp16' in cfg:
        del cfg['fp16']
    
    # 4. 强制模型所有参数为FP32
    model = model.float()  # 关键：把模型权重全转FP32
    
    train_model(
        model,
        datasets,
        cfg,
        distributed=True,
        validate=True,
        timestamp=timestamp,
    )
    print(f"🔵[xmy]>>> tools/train.py::  train_model() End =============================================== ")



if __name__ == "__main__":
    main()
