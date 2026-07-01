import pickle
from os import path as osp

import mmcv
import numpy as np
from mmcv import track_iter_progress
from mmcv.ops import roi_align
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

from mmdet3d.core.bbox import box_np_ops as box_np_ops
from mmdet3d.datasets import build_dataset
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


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
):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name （str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str): Path of the info file.
            Default: None.
        mask_anno_path (str): Path of the mask_anno.
            Default: None.
        used_classes (list[str]): Classes have been used.
            Default: None.
        database_save_path (str): Path to save database.
            Default: None.
        db_info_save_path (str): Path to save db_info.
            Default: None.
        relative_path (bool): Whether to use relative path.
            Default: True.
        with_mask (bool): Whether to use mask.
            Default: False.
    """
    print(f"Create GT Database of {dataset_class_name}")
    dataset_cfg = dict(
        type=dataset_class_name, dataset_root=data_path, ann_file=info_path
    )
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
    
    # ============ 添加TYJTDataset配置 ============
    elif dataset_class_name == "TYJTDataset" or dataset_class_name == "TyjtDataset":
        # TYJT数据集配置（与NuScenesDataset类似）
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
    
    print(f"🔵[xmy]>>> bevfusion_mit_xmy/tools/data_converter/create_gt_database.py::251:: create_groundtruth_database() ")
    import pdb; pdb.set_trace()
    # 1. 数据加载与预处理
    dataset = build_dataset(dataset_cfg)  # 1.1 数据加载。构建数据集

    if database_save_path is None:
        database_save_path = osp.join(data_path, f"{info_prefix}_gt_database")
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path, f"{info_prefix}_dbinfos_train.pkl")
    mmcv.mkdir_or_exist(database_save_path)
    all_db_infos = dict()
    if with_mask:
        coco = COCO(osp.join(data_path, mask_anno_path))
        imgIds = coco.getImgIds()
        file2id = dict()
        for i in imgIds:
            info = coco.loadImgs([i])[0]
            file2id.update({info["file_name"]: i})

    group_counter = 0
    for ii, j in enumerate(track_iter_progress(list(range(len(dataset))))):
        input_dict = dataset.get_data_info(j)  # 1.2.1 获取原始数据信息, 目前只有lidar\radar的路径
        dataset.pre_pipeline(input_dict)       # 1.2.2 数据预处理准备
        # import pdb; pdb.set_trace()
        # 真正的下载数据
        # （1）<mmdet3d.datasets.pipelines.loading.LoadPointsFromFile>      # 1. 加载点云文件
        # （2）LoadPointsFromMultiSweeps(sweeps_num=10)                     # 2. 多帧点云融合
        # （3）LoadAnnotations3D（）                                         # 3. 加载标注（但关闭了所有标注）
        example = dataset.pipeline(input_dict)  # 1.2.3 数据预处理。流水线pipeline

        # 1.3 处理label标注信息
        annos = example["ann_info"]  # 获取 标注 annos
        image_idx = example["sample_idx"]  # 获取 idx
        points = example["points"].tensor.numpy()
        gt_boxes_3d = annos["gt_bboxes_3d"].tensor.numpy()  # 从标注annos中获取 gt_boxes_3d
        names = annos["gt_names"]  # 从标注annos中获取 gt_names 类别
        group_dict = dict()
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        num_obj = gt_boxes_3d.shape[0]
        # 2. 物体Lidar点云提取
        # 判断每个点云点位于哪些3D边界框内部。index
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)  # 2.1 提取属于该物体的点云

        if ii == 0: import pdb; pdb.set_trace()
        # BEVFusion中 with_mask = False
        if with_mask:
            # 这部分代码将3D检测与2D图像信息关联，为每个3D物体提取对应的2D图像区域和掩码。
            # prepare masks, 2D信息
            gt_boxes = annos["gt_bboxes"]  # 2D边界框 [x1, y1, x2, y2]
            img_path = osp.split(example["img_info"]["filename"])[-1]
            if img_path not in file2id.keys():
                print(f"skip image {img_path} for empty mask")
                continue
            
            # 加载COCO格式的掩码标注
            img_id = file2id[img_path]  # 通过文件名找到COCO图像ID
            kins_annIds = coco.getAnnIds(imgIds=img_id)  # 获取该图像的所有标注ID
            kins_raw_info = coco.loadAnns(kins_annIds)   # 加载标注详细信息
            kins_ann_info = _parse_coco_ann_info(kins_raw_info)  # 解析为标准格式
            
            # 3. 生成2D掩码图像掩码处理
            # 3.1 处理2D检测框 和 分割掩码
            h, w = annos["img_shape"][:2]
            gt_masks = [_poly2mask(mask, h, w) for mask in kins_ann_info["masks"]]

            # 4. 3D-2D关联匹配，基于IOU
            # get mask inds based on iou mapping
            bbox_iou = bbox_overlaps(kins_ann_info["bboxes"], gt_boxes)  # 计算2D检测框与掩码边界框的IoU
            # 找到每个3D物体对应的最佳掩码
            mask_inds = bbox_iou.argmax(axis=0)    # 最大IoU的掩码索引
            valid_inds = bbox_iou.max(axis=0) > 0.5  # IoU大于0.5才认为有效

            # mask the image
            # use more precise crop when it is ready
            # object_img_patches = np.ascontiguousarray(
            #     np.stack(object_img_patches, axis=0).transpose(0, 3, 1, 2))
            # crop image patches using roi_align
            # object_img_patches = crop_image_patch_v2(
            #     torch.Tensor(gt_boxes),
            #     torch.Tensor(mask_inds).long(), object_img_patches)

            #  3.2 裁剪图像块 
            object_img_patches, object_masks = crop_image_patch(
                gt_boxes, gt_masks, mask_inds, annos["img"]
            )

        # 4.保存数据库信息
        if ii == 0: import pdb; pdb.set_trace()
        for i in range(num_obj):  # 遍历每个3D物体
            # 4.1 文件名：图像索引_类别_物体索引.bin
            filename = f"{image_idx}_{names[i]}_{i}.bin"
            abs_filepath = osp.join(database_save_path, filename)  # 绝对路径
            rel_filepath = osp.join(f"{info_prefix}_gt_database", filename)  # 相对路径
            
            # 4.2 提取并处理物体点云
            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]   # 提取属于该物体的点云
            gt_points[:, :3] -= gt_boxes_3d[i, :3]     # 中心化：将点云坐标转换到物体局部坐标系

            # 4.3. 图像掩码处理（BEVFusion不需要的部分）
            if with_mask:  # BEVfsuon不需要
                if object_masks[i].sum() == 0 or not valid_inds[i]:
                    # Skip object for empty or invalid mask
                    continue
                # 保存图像patch和掩码（BEVFusion用不到）
                img_patch_path = abs_filepath + ".png"
                mask_patch_path = abs_filepath + ".mask.png"
                mmcv.imwrite(object_img_patches[i], img_patch_path)
                mmcv.imwrite(object_masks[i], mask_patch_path)

            # 4.4 保存点云文件
            with open(abs_filepath, "w") as f:
                gt_points.tofile(f)  # 以二进制格式保存点云
            
            # 4.5 记录物体元信息
            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    "name": names[i],        # 物体类别（如'car', 'pedestrian'）
                    "path": rel_filepath,    # 点云文件相对路径  
                    "image_idx": image_idx,  # 所属图像的索引
                    "gt_idx": i,             # 在当前图像中的物体索引
                    "box3d_lidar": gt_boxes_3d[i],  # 3D边界框 [x,y,z,dx,dy,dz,yaw]
                    "num_points_in_gt": gt_points.shape[0],  # 该物体的点云数量
                    "difficulty": difficulty[i],   # 检测难度等级
                }
                 # 4.6. 分组信息（用于数据增强时保持物体间关系）
                local_group_id = group_ids[i]  # 同一组物体（如同一辆车的不同部分）
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]  # 分配组ID

                # 4.7 可选字段
                if "score" in annos:  # 如果有置信度分数
                    db_info["score"] = annos["score"][i]
                if with_mask:  # 如果有掩码，记录2D框（BEVFusion不需要）
                    db_info.update({"box2d_camera": gt_boxes[i]})

                # import pdb; pdb.set_trace()
                # 4.8. 按类别组织存储
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)  # 添加到已有类别列表
                else:
                    all_db_infos[names[i]] = [db_info]      # 创建新的类别列表

    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, "wb") as f:
        pickle.dump(all_db_infos, f)
