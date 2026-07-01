"""
功能:
    - 实现TYJTDatasetV2的可视化
    - 支持可视化gt mode = gt
    - 支持可视化推理 mode = pred
    
# 版本说明


版本v1.2
    - 新增可视化, 5个字图 合并显示的功能
        - 新函数: make_composite_image(); 
        - 新参数: --make_composite_image, 默认是 True
        - 新参数: --keep-individual, 默认是 False

版本v1.1
    - 增加step可视化参数。控制可视化步长(vis_step)、最大数目(vis_max)
    - 增加5个字图 合并显示的功能
        - 新函数: make_composite_image(); 
        - 新参数: --make_composite_image 默认是
        - 新参数: --keep-individual

版本v1.0
    - 针对TYJTDatasetV2类，实现了自己的可视化脚本


# 使用说明
v1.1
torchpack dist-run -np 1 python tools/visualize_tyjt_pkl.py \
    ./xmy_tools/tyjt2pkl_v2/0311/configs/tyjt_2d_CenterheadLSSfpn_PlanB_V1_7_f_0408_XmyCenterHeadPerCls_TYJTDatasetV2_360x640_del_lidar_OnlyCar.yaml  \
    --mode gt \
    --out-dir runs/Vis_Gt/tyjt_2d_CenterheadLSSfpn_PlanB_V1_7_f_0408_XmyCenterHeadPerCls_TYJTDatasetV2_360x640_del_lidar_OnlyCar-pretrain \
    --vis-step 10 \
    --vis-max 20
v1.0
torchpack dist-run -np 1 python tools/visualize_tyjt_pkl.py \
    ./xmy_tools/tyjt2pkl_v2/0311/configs/tyjt_2d_CenterheadLSSfpn_PlanB_V1_7_f_0408_XmyCenterHeadPerCls_TYJTDatasetV2_360x640_del_lidar_OnlyCar.yaml  \
    --mode gt \
    --out-dir runs/Vis_Gt/tyjt_2d_CenterheadLSSfpn_PlanB_V1_7_f_0408_XmyCenterHeadPerCls_TYJTDatasetV2_360x640_del_lidar_OnlyCar-pretrain
"""
import argparse
import copy
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
# fix(visualized-pred Bug): TypeError: get_cam_feats() takes 2 positional arguments but 3 were given 或   takes 3 positional arguments but 4 were given
# 日期：2025-10-11，xmy
# https://github.com/mit-han-lab/bevfusion/issues/519
# https://blog.csdn.net/Chenqinghe528/article/details/150921963
# 修改： 仿照 bevfusion_mit/tools/test.py 进行，代码修改
# 修改1： 添加import wrap_fp16_model
from mmcv.runner import load_checkpoint, wrap_fp16_model
from torchpack import distributed as dist
from torchpack.utils.config import configs
# fix(visualize Bug， tqdm): ModuleNotFoundError: No module named 'torchpack.utils.tqdm'
# 日期：2025-10-10，xmy
# https://blog.csdn.net/Chenqinghe528/article/details/150921963
# 修改：将 from torchpack.utils.tqdm import tqdm 改为 from tqdm import tqdm
from tqdm import tqdm


from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def make_composite_image(out_dir, name, num_cameras, composite_path):
    """读取已保存的相机和LiDAR图像，合成为3x2宫格图，缺失位置用灰色填充"""
    from PIL import Image, ImageDraw, ImageFont
    cell_width, cell_height = 640, 360  # 每个子图尺寸
    cols, rows = 2, 3  # 3行2列
    canvas = Image.new('RGB', (cols * cell_width, rows * cell_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # 相机图像（最多4个）放置在第0行和第1行
    for k in range(4):
        if k < num_cameras:
            img_path = os.path.join(out_dir, f"camera-{k}", f"{name}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path).resize((cell_width, cell_height), Image.Resampling.LANCZOS)
            else:
                img = None
        else:
            img = None
        if img is None:
            img = Image.new('RGB', (cell_width, cell_height), (128, 128, 128))
            d = ImageDraw.Draw(img)
            d.text((10, 10), f"CAM_{k} (no image)", fill=(255,255,255), font=font)
        # 计算行列：前4个相机占第0行和第1行，每行2个
        row = k // cols
        col = k % cols
        # 确保 row 不超过1（相机只占前两行）
        if row > 1:
            row = 1
        canvas.paste(img, (col * cell_width, row * cell_height))
        draw.text((col * cell_width + 10, row * cell_height + 10), f"CAM_{k}", fill=(255,255,0), font=font)
    
    # LiDAR 图像（放在第2行第0列）
    lidar_path = os.path.join(out_dir, "lidar", f"{name}.png")
    if os.path.exists(lidar_path):
        lidar_img = Image.open(lidar_path).resize((cell_width, cell_height), Image.Resampling.LANCZOS)
    else:
        lidar_img = Image.new('RGB', (cell_width, cell_height), (128, 128, 128))
        d = ImageDraw.Draw(lidar_img)
        d.text((10, 10), "LIDAR (no data)", fill=(255,255,255), font=font)
    row_lidar, col_lidar = 2, 0
    canvas.paste(lidar_img, (col_lidar * cell_width, row_lidar * cell_height))
    draw.text((col_lidar * cell_width + 10, row_lidar * cell_height + 10), "LIDAR BEV", fill=(255,255,0), font=font)
    
    # 可选：地图（如果有）放在第2行第1列（预留位置）
    # 这里可以根据需要扩展，比如检测是否存在 map 图像
    # 简单起见，留空或放灰色占位
    placeholder = Image.new('RGB', (cell_width, cell_height), (128, 128, 128))
    d = ImageDraw.Draw(placeholder)
    d.text((10, 10), "No Map", fill=(255,255,255), font=font)
    canvas.paste(placeholder, (1 * cell_width, 2 * cell_height))
    draw.text((1 * cell_width + 10, 2 * cell_height + 10), "MAP", fill=(255,255,0), font=font)
    
    canvas.save(composite_path)
    

def main() -> None:
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    parser.add_argument("--vis-step", type=int, default=1,
                        help="Sample every N samples (e.g., 10 means take one every 10 samples).")
    parser.add_argument("--vis-max", type=int, default=None,
                        help="Maximum number of samples to visualize. If None, visualize all sampled ones.")
    parser.add_argument("--composite", action="store_true", default=True,
                        help="Save composite 2x3 grid image (default: True).")
    parser.add_argument("--keep-individual", action="store_true", default=False,
                        help="Keep individual camera/lidar images when composite is enabled.")

    args, opts = parser.parse_known_args()
    print(f"\n>>>[xmy]🔵[tools/visualize_tyjt_pkl.py]>>> \n 被识别的参数 parser.args = {args}; \n parser.opts = {opts}")  # 查看未被识别的参数

    # print(f"\n>>>[xmy]🔵[tools/visualize_tyjt_pkl.py]>>> 将被识别的参数 传递给 args.configs ")  # 查看未被识别的参数
    configs.load(args.config, recursive=True)   # Step1：加载 YAML
    # print(f"\n>>>[xmy]🔵🔵🔵[tools/visualize_tyjt_pkl.py]>>> Before update ops: configs = \n {configs}")
    configs.update(opts)   # Step2：将命令行 `--` 后的参数覆盖进去
    # print(f"\n>>>[xmy]🔵🔵🔵[tools/visualize_tyjt_pkl.py]>>> After update: \n {configs}")
    # exit()

    cfg = Config(recursive_eval(configs), filename=args.config)  # 最终俄cfg
    # print(f"data.val.ann_file = {cfg.data.val.ann_file}")
    # print(f"data.train.ann_file = {cfg.data.train.ann_file}")

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])

    print("="*50)
    print("Dataset pipeline (Collect3D meta_keys):")
    # 兼容 CBGSDataset 包装
    if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'pipeline'):
        pipeline = dataset.dataset.pipeline
    elif hasattr(dataset, 'pipeline'):
        pipeline = dataset.pipeline
    else:
        pipeline = None

    if pipeline is not None:
        for idx, t in enumerate(pipeline.transforms):
            if t.__class__.__name__ == 'Collect3D':
                print(f"  meta_keys: {t.meta_keys}")
                break
    else:
        print("  Cannot retrieve pipeline (dataset may be wrapped)")
    print("="*50)

    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    # build the model and load checkpoint
    print(f"\n>>>[xmy]🔵[tools/visualize_tyjt_pkl.py]>>> args.mode = {args.mode}")
    if args.mode == "pred":
        model = build_model(cfg.model)

        # fix(visualized-pred Bug): TypeError: get_cam_feats() takes 2 positional arguments but 3 were given 或   takes 3 positional arguments but 4 were given
        # 日期：2025-10-11，xmy
        # https://github.com/mit-han-lab/bevfusion/issues/519
        # https://blog.csdn.net/Chenqinghe528/article/details/150921963
        # 修改： 仿照 bevfusion_mit/tools/test.py 进行，代码修改
        # 修改2： 添加 fp16_cfg
        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)

        print(f"\n>>>[xmy]🔵[tools/visualize_tyjt_pkl.py]>>> Loading checkpoint from: {args.checkpoint}")
        checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

        # 打印加载状态（可选）
        if hasattr(checkpoint, 'keys'):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            if 'meta' in checkpoint:
                print(f"Meta info: {checkpoint['meta']}")
            if 'epoch' in checkpoint:
                print(f"Epoch: {checkpoint['epoch']}")
        print(">>>[xmy]🔵[tools/visualize_tyjt_pkl.py]>>>  Model loaded successfully.\n")

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval()

    sample_idx = 0          # 全局样本索引（从0开始）
    vis_count = 0           # 已可视化的样本数量
    for data in tqdm(dataflow):
        # 等距采样：只处理索引满足 vis_step 倍数的样本
        if sample_idx % args.vis_step != 0:
            sample_idx += 1
            continue
        # 如果设置了最大可视化数量，且已达到，则停止
        if args.vis_max is not None and vis_count >= args.vis_max:
            break

        sample_idx += 1
        vis_count += 1

        metas = data["metas"].data[0][0]
        # nusc
        # name = "{}-{}".format(metas["timestamp"], metas["token"])      
        # tyjt
        name = "{}-{}".format(metas["timestamp"], os.path.basename(metas["lidar_path"]).split('.')[0])

        if args.mode == "pred":
            with torch.inference_mode():
                outputs = model(**data)

        # extract bboxes and labels
        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            # adjust box center: from bottom center to gravity center for visualization
            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        else:
            bboxes = None
            labels = None

        # extract map masks (if any)
        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(np.bool)
        elif args.mode == "pred" and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
        else:
            masks = None

        # visualize camera images
        if "img" in data:
            for k, image_path in enumerate(metas["filename"]):
                image = mmcv.imread(image_path)
                visualize_camera(
                    os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    transform=metas["lidar2image"][k],
                    classes=cfg.object_classes,
                )

        # visualize lidar point cloud
        if "points" in data:
            lidar = data["points"].data[0][0].numpy()
            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
            )


        # 新增：合成宫格图
        if args.composite:
            composite_dir = os.path.join(args.out_dir, "composite")
            os.makedirs(composite_dir, exist_ok=True)
            composite_path = os.path.join(composite_dir, f"{name}.png")
            make_composite_image(args.out_dir, name, num_cameras=len(metas["filename"]), composite_path=composite_path)
            if not args.keep_individual:
                # 删除独立图片（可选）
                for k in range(len(metas["filename"])):
                    cam_path = os.path.join(args.out_dir, f"camera-{k}", f"{name}.png")
                    if os.path.exists(cam_path):
                        os.remove(cam_path)
                lidar_path = os.path.join(args.out_dir, "lidar", f"{name}.png")
                if os.path.exists(lidar_path):
                    os.remove(lidar_path)


        # visualize map (if available)
        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )


if __name__ == "__main__":
    main()
