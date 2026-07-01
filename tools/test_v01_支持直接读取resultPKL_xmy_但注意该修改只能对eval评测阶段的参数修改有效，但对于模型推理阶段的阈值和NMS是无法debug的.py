# -*- coding: utf-8 -*-
# @Version: v1.0
# @Date:   2026-03-06
# @Author:   xmy
# @Description: 修改支持 --result-pkl 直接加载已有结果进行评估，跳过推理。
# @使用命令：torchpack dist-run -np 1 python tools/test_v01_支持直接读取result_xmy.py   ./xmy_tools/tyjt2nusc/1219_A100/configs/tyjt_2d_CenterheadLSSfpn_V1_5_10_a_0113_Local_Debug_CamOnly.yaml   --result-pkl output/tmp/cam_only/tyjt_2d_CenterheadLSSfpn_V1_5_10_a_0113_Local_Debug_CamOnly/cameraDet_results.pkl   --eval bbox
# @注意：test_v01_支持直接读取resultPKL_xmy_但注意该修改只能对eval评测阶段的参数修改有效，但对于模型推理阶段的阈值和NMS是无法debug的.py
# @评价：无用，因为耗时主要在推理阶段，不在评测阶段
import argparse
import copy
import os
import warnings

import mmcv
import torch
from torchpack.utils.config import configs
from torchpack import distributed as dist
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import recursive_eval

Debug = True
if Debug:
    print(f"\n>>>[xmy]🔵[tools/test.py] >>> [Debug Mode = True] ")

def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", nargs='?', help="checkpoint file")  # v01：不允许缺省; v02（xmy）允许缺省
    parser.add_argument("--out", help="output result file in pickle format: xxx.pkl")
    parser.add_argument("--result-pkl", help="xmy: path to an existing result pkl file to evaluate directly, skipping inference")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where results will be saved")
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options

    # 当提供 --result-pkl 时，checkpoint 变为可选（但仍需 config）
    if args.result_pkl and args.checkpoint is None:
        # 允许不传 checkpoint，但为了保持参数顺序，可传入占位符或直接省略
        # 这里只是
        pass
    return args


def main():
    args = parse_args()
    dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    # assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
    #     "Please specify at least one operation (save/eval/format/show the "
    #     'results / save the results) with the argument "--out", "--eval"'
    #     ', "--format-only", "--show" or "--show-dir"'
    # )
    assert args.out or args.eval or args.format_only or args.show or args.show_dir or args.result_pkl, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir" or provide --result-pkl for evaluation.'
    )

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    # 加载配置（始终需要，因为数据集依赖它）
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    print(f">>>[xmy]🔵 tools/test.py: Before merge_from_dict: cfg = {cfg} ")
    if Debug:
        import pdb; pdb.set_trace()

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    print(f">>>[xmy]🔵 tools/test.py: After merge_from_dict: cfg = {cfg} ")
    if Debug:
        import pdb; pdb.set_trace()

    cfg.dump(os.path.join('cur_configs.yaml'))
    cfg.dump(os.path.join('cur_configs.json'))


    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    # 处理测试集配置
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    distributed = True

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # 构建test数据集
    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )
    # 根据是否提供 result_pkl 决定是否执行推理
    if args.result_pkl:
        # 如果用户已经提供 result_pkl， 则不再构建model、不再加载checkpoint模型参数、不再推理
        # 直接load 历史的模型推理结果
        if not os.path.exists(args.result_pkl):
            raise FileNotFoundError(f">>>[xmy]🔵 test.py: Result pkl file not found: {args.result_pkl}")
        print(f">>>[xmy]🔵 test.py: Loading existing results from {args.result_pkl} ")
        outputs = mmcv.load(args.result_pkl)

        # 如果同时指定了 --out，给出警告并忽略（避免覆盖）
        if args.out:
            warnings.warn(">>>[xmy]🟡 test.py: --out is ignored when --result-pkl is provided, as results are already saved.")
    else:
        # 官方正规流程：构建model --> 加载checkpoint模型参数 --> 推理 --> 评测
        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
        if args.fuse_conv_bn:
            model = fuse_conv_bn(model)
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            model.CLASSES = dataset.CLASSES

        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
            )
            outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}")
            import pdb; pdb.set_trace()
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule",
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == "__main__":
    main()