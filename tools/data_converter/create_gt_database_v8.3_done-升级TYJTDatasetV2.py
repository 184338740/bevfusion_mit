# ============================================================================
# create_gt_database.py - 多进程加速版本 v8.3,  for v03数据集
# 路径: tools/data_converter/create_gt_database.py
# 修改说明：
# 1. 添加新分支: 支持 TYJTDatasetV2 类数据的 create_gt_database 生成
# 修改日期：2026-03
# 修改者：xmy
# 产出数据：03
# ============================================================================
# ============================================================================
# create_gt_database.py - 多进程加速版本 v8.2,  for v02数据集
# 路径: tools/data_converter/create_gt_database.py
# 修改说明：
# 1. 增加多进程支持，通过 workers 参数控制并行数（默认1，即原单进程行为）
# 2. 修复保存点云文件的模式：'w' -> 'wb'（二进制模式）
# 3. 重构代码，将样本处理逻辑抽取为 _process_sample 函数，供单进程/多进程共用
# 4. 增加 _init_worker 和 _worker_func 以支持多进程池
# 5. 保留所有原有调试输出（🔵[xmy] 标记、pdb 断点等），并在多进程模式下合理触发
# 6. 若 with_mask=True 且 workers>1，自动回退到单进程（避免复杂依赖）
# 修改日期：2026-02-13
# 修改者：xmy
# 产出数据：v2.0-tyjt
# ============================================================================
# create_gt_database.py - A100适配 v8.1
# 路径: tools/data_converter/create_gt_database.py
# 修改说明：
# 1. 适配tyjt的数据集
# 2. 适配A100
# 修改日期：2026-02-13
# 修改者：xmy
# 产出数据：v1.0-tyjt
# dataset_root: /data2/xmy/01_project/bevfusion_mit_xmy_1205/xmy_tools/tyjt2nusc/1219_A100/output-1204-sub/step1/nuscenes_tyjt
# dataset_root: /data2/xmy/01_project/bevfusion_mit_xmy_1205/xmy_tools/tyjt2nusc/1219_A100/output-1226-v9.2.3-all/step1/nuscenes_tyjt
# ============================================================================

import pickle
from os import path as osp
import multiprocessing as mp
from functools import partial

import mmcv
import numpy as np
from mmcv import track_iter_progress
from mmcv.ops import roi_align
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

from mmdet3d.core.bbox import box_np_ops as box_np_ops
from mmdet3d.datasets import build_dataset
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

Debug = True
if Debug:
    import time
    print(f">>>[xmy]🔵[tools/data_converter/create_gt_database.py] >>> [Debug Mode = True] ")


def _poly2mask(mask_ann, img_h, img_w):
    if isinstance(mask_ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann["counts"], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask


def _parse_coco_ann_info(ann_info):
    gt_bboxes = []
    gt_labels = []
    gt_bboxes_ignore = []
    gt_masks_ann = []

    for i, ann in enumerate(ann_info):
        if ann.get("ignore", False):
            continue
        x1, y1, w, h = ann["bbox"]
        if ann["area"] <= 0:
            continue
        bbox = [x1, y1, x1 + w, y1 + h]
        if ann.get("iscrowd", False):
            gt_bboxes_ignore.append(bbox)
        else:
            gt_bboxes.append(bbox)
            gt_masks_ann.append(ann["segmentation"])

    if gt_bboxes:
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
    else:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([], dtype=np.int64)

    if gt_bboxes_ignore:
        gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
    else:
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

    ann = dict(bboxes=gt_bboxes, bboxes_ignore=gt_bboxes_ignore, masks=gt_masks_ann)

    return ann


def crop_image_patch_v2(pos_proposals, pos_assigned_gt_inds, gt_masks):
    import torch
    from torch.nn.modules.utils import _pair

    device = pos_proposals.device
    num_pos = pos_proposals.size(0)
    fake_inds = torch.arange(num_pos, device=device).to(dtype=pos_proposals.dtype)[
        :, None
    ]
    rois = torch.cat([fake_inds, pos_proposals], dim=1)  # Nx5
    mask_size = _pair(28)
    rois = rois.to(device=device)
    gt_masks_th = (
        torch.from_numpy(gt_masks)
        .to(device)
        .index_select(0, pos_assigned_gt_inds)
        .to(dtype=rois.dtype)
    )
    # Use RoIAlign could apparently accelerate the training (~0.1s/iter)
    targets = roi_align(gt_masks_th, rois, mask_size[::-1], 1.0, 0, True).squeeze(1)
    return targets


def crop_image_patch(pos_proposals, gt_masks, pos_assigned_gt_inds, org_img):
    num_pos = pos_proposals.shape[0]
    masks = []
    img_patches = []
    for i in range(num_pos):
        gt_mask = gt_masks[pos_assigned_gt_inds[i]]
        bbox = pos_proposals[i, :].astype(np.int32)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)

        mask_patch = gt_mask[y1 : y1 + h, x1 : x1 + w]
        masked_img = gt_mask[..., None] * org_img
        img_patch = masked_img[y1 : y1 + h, x1 : x1 + w]

        img_patches.append(img_patch)
        masks.append(mask_patch)
    return img_patches, masks


# ==================== 新增：多进程辅助函数 ====================
def _process_sample(args, dataset_cfg, database_save_path, info_prefix, with_mask,
                    mask_anno_path, data_path, used_classes, relative_path):
    """处理单个样本（可被单进程或多进程调用）"""
    idx, dataset = args
    if Debug:
        print(f">>>[xmy]🔵[tools/data_converter/create_gt_database.py] >>> [DEBUG] _process_sample: dataset id={id(dataset)}, dataset.modality={dataset.modality}")

    # 保留原调试断点：处理第一个样本时触发（可根据需要注释）
    if idx == 0:
        import pdb; pdb.set_trace()
        print(">>>[xmy]🔵[tools/data_converter/create_gt_database.py] >>> 进入 _process_sample 处理第一个样本")

    print("DEBUG: calling dataset.get_data_info")
    input_dict = dataset.get_data_info(idx)
    print("DEBUG: dataset.get_data_info returned")

    print("DEBUG: calling dataset.pre_pipeline")
    dataset.pre_pipeline(input_dict)
    print("DEBUG: dataset.pre_pipeline returned")
    print("DEBUG: calling dataset.pipeline")
    example = dataset.pipeline(input_dict)
    print("DEBUG: dataset.pipeline returned")

    annos = example["ann_info"]
    image_idx = example["sample_idx"]
    points = example["points"].tensor.numpy()
    gt_boxes_3d = annos["gt_bboxes_3d"].tensor.numpy()
    names = annos["gt_names"]
    if "group_ids" in annos:
        group_ids = annos["group_ids"]
    else:
        group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
    difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
    if "difficulty" in annos:
        difficulty = annos["difficulty"]

    num_obj = gt_boxes_3d.shape[0]
    point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

    local_db_infos = {}
    group_counter_local = 0
    group_dict_local = {}

    # ---------- 2D mask 处理（与您的原代码完全相同）----------
    if with_mask:
        # 多进程模式下，每个子进程独立加载 COCO 标注（仅当 with_mask=True 时）
        # 但本整合版本已做降级处理：with_mask=True 且 workers>1 时自动回退单进程
        # 此处保留完整代码以保证单进程时行为一致
        coco = COCO(osp.join(data_path, mask_anno_path))
        imgIds = coco.getImgIds()
        file2id = {coco.loadImgs([i])[0]["file_name"]: i for i in imgIds}

        gt_boxes = annos["gt_bboxes"]
        img_path = osp.split(example["img_info"]["filename"])[-1]
        if img_path not in file2id.keys():
            print(f"skip image {img_path} for empty mask")
            return local_db_infos, group_counter_local

        img_id = file2id[img_path]
        kins_annIds = coco.getAnnIds(imgIds=img_id)
        kins_raw_info = coco.loadAnns(kins_annIds)
        kins_ann_info = _parse_coco_ann_info(kins_raw_info)

        h, w = annos["img_shape"][:2]
        gt_masks = [_poly2mask(mask, h, w) for mask in kins_ann_info["masks"]]

        bbox_iou = bbox_overlaps(kins_ann_info["bboxes"], gt_boxes)
        mask_inds = bbox_iou.argmax(axis=0)
        valid_inds = bbox_iou.max(axis=0) > 0.5

        object_img_patches, object_masks = crop_image_patch(
            gt_boxes, gt_masks, mask_inds, annos["img"]
        )

    # ---------- 遍历该样本的所有目标 ----------
    for i in range(num_obj):
        filename = f"{image_idx}_{names[i]}_{i}.bin"
        abs_filepath = osp.join(database_save_path, filename)
        rel_filepath = osp.join(f"{info_prefix}_gt_database", filename)

        gt_points = points[point_indices[:, i]]
        gt_points[:, :3] -= gt_boxes_3d[i, :3]

        # 2D mask 相关保存（若启用）
        if with_mask:
            if object_masks[i].sum() == 0 or not valid_inds[i]:
                continue
            img_patch_path = abs_filepath + ".png"
            mask_patch_path = abs_filepath + ".mask.png"
            mmcv.imwrite(object_img_patches[i], img_patch_path)
            mmcv.imwrite(object_masks[i], mask_patch_path)

        # 保存点云（已修正为二进制模式）
        with open(abs_filepath, "wb") as f:
            gt_points.tofile(f)

        if (used_classes is None) or names[i] in used_classes:
            db_info = {
                "name": names[i],
                "path": rel_filepath,
                "image_idx": image_idx,
                "gt_idx": i,
                "box3d_lidar": gt_boxes_3d[i],
                "num_points_in_gt": gt_points.shape[0],
                "difficulty": difficulty[i],
            }
            local_group_id = group_ids[i]
            if local_group_id not in group_dict_local:
                group_dict_local[local_group_id] = group_counter_local
                group_counter_local += 1
            db_info["group_id"] = group_dict_local[local_group_id]

            if "score" in annos:
                db_info["score"] = annos["score"][i]
            if with_mask:
                db_info.update({"box2d_camera": gt_boxes[i]})

            if names[i] in local_db_infos:
                local_db_infos[names[i]].append(db_info)
            else:
                local_db_infos[names[i]] = [db_info]

    return local_db_infos, group_counter_local


def _init_worker(dataset_cfg):
    """每个子进程初始化自己的 dataset 对象"""
    global worker_dataset
    worker_dataset = build_dataset(dataset_cfg)


def _worker_func(indices, dataset_cfg, database_save_path, info_prefix, with_mask,
                 mask_anno_path, data_path, used_classes, relative_path):
    """子进程入口：处理一批样本索引"""
    global worker_dataset
    if worker_dataset is None:
        worker_dataset = build_dataset(dataset_cfg)
    dataset = worker_dataset

    all_local_infos = {}
    total_group_cnt = 0
    for idx in indices:
        local_infos, group_cnt = _process_sample(
            (idx, dataset), dataset_cfg, database_save_path, info_prefix, with_mask,
            mask_anno_path, data_path, used_classes, relative_path
        )
        for cat, infos in local_infos.items():
            if cat not in all_local_infos:
                all_local_infos[cat] = []
            all_local_infos[cat].extend(infos)
        total_group_cnt += group_cnt
    return all_local_infos, total_group_cnt


# ==================== 主函数（已增加多进程支持）====================
def create_groundtruth_database(
    dataset_class_name,
    data_path,
    info_prefix,
    info_path=None,
    mask_anno_path=None,
    used_classes=None,
    database_save_path=None,
    db_info_save_path=None,
    relative_path=True,
    add_rgb=False,
    lidar_only=False,
    bev_only=False,
    coors_range=None,
    with_mask=False,
    load_augmented=None,
    workers=1,   # <--- 新增：并行进程数，默认1（单进程）
):
    """Given the raw data, generate the ground truth database (支持多进程加速).

    Args:
        ... (原有参数保持不变)
        workers (int): 并行进程数。若 <=1 则单进程；若 >1 则启用多进程。
                      当 with_mask=True 时强制为单进程。
    """
    print(f"Create GT Database of {dataset_class_name} (workers={workers})")
    
    # 若启用了 2D mask 且要求多进程，则降级为单进程（避免复杂依赖）
    if with_mask and workers > 1:
        print("🔵[xmy] 警告: with_mask=True 时不支持多进程，已自动切换为单进程模式")
        workers = 1

    dataset_cfg = dict(
        type=dataset_class_name, dataset_root=data_path, ann_file=info_path
    )

    # ---------- 数据集配置（与原代码完全相同）----------
    if dataset_class_name == "KittiDataset":
        dataset_cfg.update(
            test_mode=False,
            split="training",
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=with_mask,
            ),
            pipeline=[
                dict(
                    type="LoadPointsFromFile",
                    coord_type="LIDAR",
                    load_dim=4,
                    use_dim=4,
                ),
                dict(
                    type="LoadAnnotations3D",
                    with_bbox_3d=True,
                    with_label_3d=True,
                ),
            ],
        )

    elif dataset_class_name == "NuScenesDataset":
        if not load_augmented:
            dataset_cfg.update(
                use_valid_flag=True,
                pipeline=[
                    dict(
                        type="LoadPointsFromFile",
                        coord_type="LIDAR",
                        load_dim=5,
                        use_dim=5,
                    ),
                    dict(
                        type="LoadPointsFromMultiSweeps",
                        sweeps_num=10,
                        use_dim=[0, 1, 2, 3, 4],
                        pad_empty_sweeps=True,
                        remove_close=True,
                    ),
                    dict(
                        type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True
                    ),
                ],
            )
        else:
            dataset_cfg.update(
                use_valid_flag=True,
                pipeline=[
                    dict(
                        type="LoadPointsFromFile",
                        coord_type="LIDAR",
                        load_dim=16,
                        use_dim=list(range(16)),
                        load_augmented=load_augmented,
                    ),
                    dict(
                        type="LoadPointsFromMultiSweeps",
                        sweeps_num=10,
                        load_dim=16,
                        use_dim=list(range(16)),
                        pad_empty_sweeps=True,
                        remove_close=True,
                        load_augmented=load_augmented,
                    ),
                    dict(
                        type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True
                    ),
                ],
            )

    elif dataset_class_name == "WaymoDataset":
        dataset_cfg.update(
            test_mode=False,
            split="training",
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=False,
            ),
            pipeline=[
                dict(
                    type="LoadPointsFromFile",
                    coord_type="LIDAR",
                    load_dim=6,
                    use_dim=5,
                ),
                dict(
                    type="LoadAnnotations3D",
                    with_bbox_3d=True,
                    with_label_3d=True,
                ),
            ],
        )
    
    # ============ TYJTDataset 配置 ============
    elif dataset_class_name == "TYJTDataset" or dataset_class_name == "TyjtDataset":
        print(f"🔵 使用TYJT数据集配置")
        if not load_augmented:
            dataset_cfg.update(
                test_mode=False,
                use_valid_flag=True,
                modality=dict(
                    use_lidar=True,
                    use_camera=False,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                ),
                pipeline=[
                    dict(
                        type="LoadPointsFromFile",
                        coord_type="LIDAR",
                        load_dim=5,
                        use_dim=5,
                    ),
                    dict(
                        type="LoadPointsFromMultiSweeps",
                        sweeps_num=10,
                        use_dim=[0, 1, 2, 3, 4],
                        pad_empty_sweeps=True,
                        remove_close=True,
                    ),
                    dict(
                        type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True
                    ),
                ],
            )
        else:
            dataset_cfg.update(
                test_mode=False,
                use_valid_flag=True,
                modality=dict(
                    use_lidar=True,
                    use_camera=False,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                ),
                pipeline=[
                    dict(
                        type="LoadPointsFromFile",
                        coord_type="LIDAR",
                        load_dim=16,
                        use_dim=list(range(16)),
                        load_augmented=load_augmented,
                    ),
                    dict(
                        type="LoadPointsFromMultiSweeps",
                        sweeps_num=10,
                        load_dim=16,
                        use_dim=list(range(16)),
                        pad_empty_sweeps=True,
                        remove_close=True,
                        load_augmented=load_augmented,
                    ),
                    dict(
                        type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True
                    ),
                ],
            )
    
    # ============ TYJTDatasetV2 配置 ============
    elif dataset_class_name == "TYJTDatasetV2":
        print(f"🔵 使用TYJTDatasetV2配置")
        dataset_cfg.update(
            use_valid_flag=True,
            modality=dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            ),
            pipeline=[
                dict(
                    type="LoadPointsFromFile",
                    coord_type="LIDAR",
                    load_dim=6,                      # 根据实际点云维度设置（6维）
                    use_dim=6,                       # 和原始的主点云一致，生成的数据库点云也是6维度
                    # use_dim=[0,1,2,3,4],           # 使用前5维（或使用全部6维：use_dim=6）
                ),
                dict(
                    type="LoadPointsFromMultiSweeps",
                    sweeps_num=10,
                    load_dim=6,
                    use_dim=6,                       # 和原始的主点云一致，生成的数据库点云也是6维度
                    # use_dim=[0,1,2,3,4],
                    pad_empty_sweeps=True,
                    remove_close=True,
                ),
                dict(
                    type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True
                ),
            ],
        )


    # ---------- 保留原调试打印和断点 ----------
    print(f"🔵[xmy]>>> bevfusion_mit_xmy/tools/data_converter/create_gt_database.py::create_groundtruth_database()  workers={workers}")
    import pdb; pdb.set_trace()

    # ---------- 输出目录初始化 ----------
    if database_save_path is None:
        database_save_path = osp.join(data_path, f"{info_prefix}_gt_database")
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path, f"{info_prefix}_dbinfos_train.pkl")
    mmcv.mkdir_or_exist(database_save_path)

    # ---------- 单进程模式（兼容原逻辑）----------
    if workers <= 1:
        print("🔵[xmy] 运行单进程模式")
        dataset = build_dataset(dataset_cfg)
        all_db_infos = dict()
        group_counter = 0

        for ii in track_iter_progress(list(range(len(dataset)))):
            # 调用 _process_sample 处理每个样本
            local_infos, group_cnt = _process_sample(
                (ii, dataset), dataset_cfg, database_save_path, info_prefix, with_mask,
                mask_anno_path, data_path, used_classes, relative_path
            )
            # 合并结果
            for cat, infos in local_infos.items():
                if cat not in all_db_infos:
                    all_db_infos[cat] = []
                all_db_infos[cat].extend(infos)
            group_counter += group_cnt

    # ---------- 多进程模式 ----------
    else:
        print(f"🔵[xmy] 运行多进程模式，workers={workers}")
        # 构建 dataset 仅用于获取总样本数（不实际加载数据）
        tmp_dataset = build_dataset(dataset_cfg)
        total_samples = len(tmp_dataset)
        del tmp_dataset  # 释放

        # 划分索引块
        indices = list(range(total_samples))
        chunk_size = (total_samples + workers - 1) // workers
        chunks = [indices[i:i + chunk_size] for i in range(0, total_samples, chunk_size)]

        # 创建进程池，每个子进程独立初始化 dataset
        pool = mp.Pool(processes=workers, initializer=_init_worker, initargs=(dataset_cfg,))
        worker_func = partial(
            _worker_func,
            dataset_cfg=dataset_cfg,
            database_save_path=database_save_path,
            info_prefix=info_prefix,
            with_mask=with_mask,
            mask_anno_path=mask_anno_path,
            data_path=data_path,
            used_classes=used_classes,
            relative_path=relative_path,
        )

        results = pool.map(worker_func, chunks)
        pool.close()
        pool.join()

        # 合并所有子进程的结果
        all_db_infos = {}
        total_group_counter = 0
        for db_infos_part, group_cnt_part in results:
            for cat, infos in db_infos_part.items():
                if cat not in all_db_infos:
                    all_db_infos[cat] = []
                all_db_infos[cat].extend(infos)
            total_group_counter += group_cnt_part
        group_counter = total_group_counter

    # ---------- 打印统计并保存 ----------
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, "wb") as f:
        pickle.dump(all_db_infos, f)

    print(f"🔵[xmy] GT database saved to {database_save_path}")
    print(f"🔵[xmy] DB infos saved to {db_info_save_path}")