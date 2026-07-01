#!/usr/bin/env python3
"""
功能:
    - 实现 TYJTDatasetV2 的可视化
    - 支持可视化 gt mode = gt
    - 支持可视化推理 mode = pred
    - 支持等距采样 (--vis-step) 和最大数量 (--vis-max)
    - 支持合成宫格图 (--composite) 和保留单独图片 (--keep-individual)

版本说明:
v1.3
    - 完全保留 v1.2 的显示风格（通过后处理添加分数和箭头，不修改原始绘图）
    - 相机图像：调用 visualize_camera 后，叠加分数和箭头（OpenCV）
    - LiDAR 图像：调用 visualize_lidar 后，根据物理坐标映射叠加分数和箭头
    - 扩大 LiDAR 显示范围（外扩 15 米），画出原始 ROI 绿色框，添加比例尺
    - 所有图像内存处理，无临时文件残留
v1.2
    - 新增宫格图合成功能
v1.1
    - 增加 step 可视化参数
v1.0
    - 初始版本

使用示例:
    # GT 模式
    torchpack dist-run -np 1 python tools/visualize_tyjt_pkl.py \
        config.yaml --mode gt --out-dir ./viz_gt --vis-step 10 --vis-max 20

    # 预测模式（带分数和箭头）
    torchpack dist-run -np 1 python tools/visualize_tyjt_pkl.py \
        config.yaml --mode pred --checkpoint epoch.pth --out-dir ./viz_pred \
        --vis-step 10 --vis-max 20 --split val
"""

import argparse
import copy
import os
import tempfile
from io import BytesIO

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from PIL import Image, ImageDraw, ImageFont
from torchpack import distributed as dist
from torchpack.utils.config import configs
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


# ---------- 辅助函数：3D 点投影（相机） ----------
def project_points(points_3d, lidar2image):
    N = points_3d.shape[0]
    pts_homo = np.hstack([points_3d, np.ones((N, 1))])
    pts_cam = pts_homo @ lidar2image.T
    pts_cam = pts_cam / (pts_cam[:, 2:3] + 1e-8)
    return pts_cam[:, :2].astype(np.int32)


def box_to_corners_3d(box):
    x, y, z, l, w, h, yaw = box[:7]
    half_l, half_w, half_h = l/2, w/2, h/2
    local = np.array([
        [ half_l,  half_w, -half_h], [ half_l, -half_w, -half_h],
        [-half_l, -half_w, -half_h], [-half_l,  half_w, -half_h],
        [ half_l,  half_w,  half_h], [ half_l, -half_w,  half_h],
        [-half_l, -half_w,  half_h], [-half_l,  half_w,  half_h],
    ])
    cos, sin = np.cos(yaw), np.sin(yaw)
    R = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    corners = (local @ R.T) + np.array([x, y, z])
    return corners


def draw_scores_and_arrows_on_camera(image, bboxes, scores, transform):
    """在相机图像上叠加分数和箭头（原地修改 image，BGR 格式）"""
    if bboxes is None or scores is None:
        return
    # 转换 bboxes 到 numpy
    if hasattr(bboxes, 'tensor'):
        bboxes = bboxes.tensor.cpu().numpy()
    elif isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if bboxes.ndim == 1:
        bboxes = bboxes.reshape(1, -1)

    for i in range(len(bboxes)):
        box = bboxes[i]
        corners_2d = project_points(box_to_corners_3d(box), transform)

        # 朝向箭头（蓝色）
        x, y, z, l, w, h, yaw = box[:7]
        center_3d = np.array([x, y, z])
        front_3d = np.array([x + l/2*np.cos(yaw), y + l/2*np.sin(yaw), z])
        center_2d = project_points(center_3d.reshape(1,3), transform).flatten()
        front_2d = project_points(front_3d.reshape(1,3), transform).flatten()
        cv2.arrowedLine(image, tuple(center_2d), tuple(front_2d), (255, 0, 0), 2, tipLength=0.3)

        # 分数（红底白字）
        min_x = np.min(corners_2d[:,0])
        min_y = np.min(corners_2d[:,1])
        text = f"{scores[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (min_x, min_y - th - 5), (min_x + tw, min_y), (0, 0, 255), -1)
        cv2.putText(image, text, (min_x, min_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def draw_scores_and_arrows_on_lidar(image_pil, bboxes, scores, vis_xlim, vis_ylim, orig_xlim, orig_ylim):
    """
    在 BEV 图像上叠加分数、箭头，并绘制原始 ROI 绿色框和比例尺。
    vis_xlim, vis_ylim: 当前可视化的物理坐标范围（扩大后的）
    orig_xlim, orig_ylim: 原始模型训练范围（用于绘制绿色框）
    """
    if bboxes is None or scores is None:
        return image_pil
    # 转换 bboxes 到 numpy
    if hasattr(bboxes, 'tensor'):
        bboxes = bboxes.tensor.cpu().numpy()
    elif isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if bboxes.ndim == 1:
        bboxes = bboxes.reshape(1, -1)

    # 将 PIL 图像转为 OpenCV BGR
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    x_min, x_max = vis_xlim
    y_min, y_max = vis_ylim

    def phys_to_pixel(x, y):
        px = (x - x_min) / (x_max - x_min) * w
        py = (y_max - y) / (y_max - y_min) * h
        return int(px), int(py)

    # 绘制原始 ROI 区域（绿色框）
    p1 = phys_to_pixel(orig_xlim[0], orig_ylim[0])
    p2 = phys_to_pixel(orig_xlim[1], orig_ylim[1])
    cv2.rectangle(img_cv, p1, p2, (0, 255, 0), 2)

    # 绘制箭头和分数
    for i in range(len(bboxes)):
        x, y, z, l, w_box, h_box, yaw = bboxes[i][:7]
        half_l, half_w = l/2, w_box/2
        cx, cy = phys_to_pixel(x, y)
        fx, fy = phys_to_pixel(x + half_l * np.cos(yaw), y + half_l * np.sin(yaw))
        cv2.arrowedLine(img_cv, (cx, cy), (fx, fy), (255, 0, 0), 2, tipLength=0.3)

        text = f"{scores[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_cv, (cx - tw//2 - 2, cy - th - 2), (cx + tw//2 + 2, cy), (0, 0, 255), -1)
        cv2.putText(img_cv, text, (cx - tw//2, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 添加比例尺（例如 20 米）
    scale_m = 20
    scale_px = int(scale_m / (x_max - x_min) * w)
    start_x = w - scale_px - 20
    start_y = h - 20
    end_x = w - 20
    cv2.line(img_cv, (start_x, start_y), (end_x, start_y), (255, 255, 255), 3)
    cv2.putText(img_cv, f"{scale_m}m", (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def make_composite_image_from_memory(images, labels, composite_path, cell_size=(640, 360), cols=2, rows=3):
    canvas = Image.new('RGB', (cols * cell_size[0], rows * cell_size[1]), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()

    for idx, (img, label) in enumerate(zip(images, labels)):
        if idx >= cols * rows:
            break
        row = idx // cols
        col = idx % cols
        x = col * cell_size[0]
        y = row * cell_size[1]
        if img is None:
            placeholder = Image.new('RGB', cell_size, (128, 128, 128))
            d = ImageDraw.Draw(placeholder)
            d.text((10, 10), f"{label} (no image)", fill=(255, 255, 255), font=font)
            canvas.paste(placeholder, (x, y))
        else:
            img_resized = img.resize(cell_size, Image.Resampling.LANCZOS)
            canvas.paste(img_resized, (x, y))
        draw.text((x + 10, y + 10), label, fill=(255, 255, 0), font=font)

    os.makedirs(os.path.dirname(composite_path), exist_ok=True)
    canvas.save(composite_path)


def main():
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
    print(f"\n>>>[xmy]🔵[tools/visualize_tyjt_pkl.py]>>> args = {args}; opts = {opts}")

    configs.load(args.config, recursive=True)
    # 启用配置覆盖
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    dataset = build_dataset(cfg.data[args.split])

    print("="*50)
    print("Dataset pipeline (Collect3D meta_keys):")
    if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'pipeline'):
        pipeline = dataset.dataset.pipeline
    elif hasattr(dataset, 'pipeline'):
        pipeline = dataset.pipeline
    else:
        pipeline = None
    if pipeline is not None:
        for t in pipeline.transforms:
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

    if args.mode == "pred":
        model = build_model(cfg.model)
        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        print(f"\n>>> Loading checkpoint from: {args.checkpoint}")
        load_checkpoint(model, args.checkpoint, map_location="cpu")
        print(">>> Model loaded successfully.\n")
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval()

    # 原始训练范围（从配置文件读取）
    orig_xlim = [cfg.point_cloud_range[0], cfg.point_cloud_range[3]]
    orig_ylim = [cfg.point_cloud_range[1], cfg.point_cloud_range[4]]
    # 可视化扩大范围（外扩 15 米）
    expand = 15.0
    vis_xlim = [orig_xlim[0] - expand, orig_xlim[1] + expand]
    vis_ylim = [orig_ylim[0] - expand, orig_ylim[1] + expand]

    sample_idx = 0
    vis_count = 0
    for data in tqdm(dataflow):
        if sample_idx % args.vis_step != 0:
            sample_idx += 1
            continue
        if args.vis_max is not None and vis_count >= args.vis_max:
            break
        sample_idx += 1
        vis_count += 1

        metas = data["metas"].data[0][0]
        name = "{}-{}".format(metas["timestamp"], os.path.basename(metas["lidar_path"]).split('.')[0])

        if args.mode == "pred":
            with torch.inference_mode():
                outputs = model(**data)

        # 提取 bboxes 和 labels
        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()
            scores = None
            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]
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
            scores = None

        # 地图掩码
        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy().astype(np.bool)
        elif args.mode == "pred" and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy() >= args.map_score
        else:
            masks = None

        # 收集相机图像（内存模式）
        cam_images = []
        if "img" in data:
            for k, image_path in enumerate(metas["filename"]):
                image = mmcv.imread(image_path)  # BGR
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name
                visualize_camera(
                    tmp_path,
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    transform=metas["lidar2image"][k],
                    classes=cfg.object_classes
                )
                img_cv = cv2.imread(tmp_path)
                os.unlink(tmp_path)
                if args.mode == "pred" and scores is not None:
                    draw_scores_and_arrows_on_camera(img_cv, bboxes, scores, metas["lidar2image"][k])
                pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                cam_images.append(pil_img)

        # LiDAR 图像
        lidar_img = None
        if "points" in data:
            points = data["points"].data[0][0].numpy()
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
            # 使用扩大后的范围生成原始 BEV 图
            visualize_lidar(
                tmp_path,
                points,
                bboxes=bboxes,
                labels=labels,
                classes=cfg.object_classes,
                xlim=vis_xlim,
                ylim=vis_ylim,
            )
            lidar_img = Image.open(tmp_path)
            os.unlink(tmp_path)
            if args.mode == "pred" and scores is not None:
                lidar_img = draw_scores_and_arrows_on_lidar(
                    lidar_img, bboxes, scores,
                    vis_xlim=vis_xlim, vis_ylim=vis_ylim,
                    orig_xlim=orig_xlim, orig_ylim=orig_ylim
                )

        # 保存单独图片（如果需要）
        if args.keep_individual:
            for k, pil_img in enumerate(cam_images):
                cam_path = os.path.join(args.out_dir, f"camera-{k}", f"{name}.png")
                os.makedirs(os.path.dirname(cam_path), exist_ok=True)
                pil_img.save(cam_path)
            if lidar_img is not None:
                lidar_path = os.path.join(args.out_dir, "lidar", f"{name}.png")
                os.makedirs(os.path.dirname(lidar_path), exist_ok=True)
                lidar_img.save(lidar_path)

        # 合成宫格图
        if args.composite:
            all_images = cam_images + [lidar_img] + [None]  # 4相机 + 1 lidar + 1 map占位
            all_labels = [f"CAM_{i}" for i in range(4)] + ["LIDAR BEV", "MAP"]
            composite_path = os.path.join(args.out_dir, "composite", f"{name}.png")
            make_composite_image_from_memory(all_images, all_labels, composite_path)

        # 保存地图（如果需要）
        if masks is not None and args.keep_individual:
            map_path = os.path.join(args.out_dir, "map", f"{name}.png")
            os.makedirs(os.path.dirname(map_path), exist_ok=True)
            visualize_map(map_path, masks, classes=cfg.map_classes)


if __name__ == "__main__":
    main()