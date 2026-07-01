"""
功能:
    - 实现 TYJT 数据集的官方 nuScenes 评估指标（mAP、NDS 等）计算。
    - 直接使用 TYJT 的 data_infos 作为 GT，无需模拟 NuScenes 对象或依赖 split 文件。
    - 通过继承 DetectionEval 并重写 GT 加载逻辑，复用官方评估核心算法。
    - 预测结果保持自车坐标系，避免全局坐标转换导致的范围过滤失效。

版本v2.3.3
    - 修复TYJTDatasetV2 类内 self.CLASSES 的顺序错位问题,会导致训练类型完全错误
        - 原因:TYJTDatasetV2.CLASSES 的顺序与配置文件中的 object_classes 顺序不一致，导致 get_ann_info 生成的 gt_labels_3d 索引与模型检测头的输出通道顺序错位。
        - self.CLASSES 改为和官方配置(mmdet3d/datasets/nuscenes_dataset.py)一致的CLASSES顺序,防止后续的 loss\get_ann_info\TYJTEval 受到影响
    
版本v2.3.2
    - 修复test是报错：KeyError: 'box_type_3d' 。原因： metas 缺少了 box_type_3d 字段
        - 在 TYJTDatasetV2.get_data_info 中返回 box_type_3d
        - 在配置文件的 test_pipeline 的 Collect3D 中，将 box_type_3d 添加到 meta_lis_keys

版本v2.3.1
    - 修复验证集（val）运行时报错 KeyError: 'ann_info' 的问题
        - val/test 模式时, class TYJTDatasetV2 的 def get_data_info() 也生成 'ann_info'

版本v2.3.0
    - 新增 TYJTEval 类（继承 DetectionEval）：
        -_load_tyjt_gt()：从 data_infos 加载 GT，转换为 DetectionBox，并计算 center_dist、num_pts。
        - _add_center_dist()：为预测框计算 center_dist。
        - _filter_boxes_by_range()：手动实现基于 class_range 的框过滤，替代官方 filter_eval_boxes（避免 nusc 对象依赖）。
    - 修改 _format_bbox()：移除 lidar_nusc_box_to_global 调用，保持预测框在自车坐标系。
    - 添加类别映射（CATEGORY_MAPPING_EVAL），将原始 TYJT 类别名转换为 nuScenes 标准名称。
    - 在 _evaluate_single 中实例化 TYJTEval 并调用其 main() 方法获取评估结果。

兼容性:
    - 完全向后兼容，不影响原有训练、验证流程。
    - 评估结果与 nuScenes 官方指标对齐，可直接用于模型对比。
训练命令：
    torchpack dist-run -np 1 python tools/train_tyjt_pkl.py \
  ./xmy_tools/tyjt2pkl_v2/0311/configs/tyjt_2d_CenterheadLSSfpn_vA010_a_0316_XmyCenterHeadPerCls_TYJTDatasetV2_360x640.yaml \
  --run-dir runs/tyjt_2d_CenterheadLSSfpn_vA010_a_0316_XmyCenterHeadPerCls_TYJTDatasetV2_360x640
"""

import numpy as np
import pickle
import json
import mmcv
import tempfile
import os
import warnings
from os import path as osp
from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets import Custom3DDataset
from pyquaternion import Quaternion

# ========== [xmy][v2.3]新增导入 ==========
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.constants import DETECTION_NAMES
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from mmdet3d.datasets.nuscenes_dataset import output_to_nusc_box, lidar_nusc_box_to_global

# ========== [xmy][v2.3.5]新增导入，kitti评测 ==========
# import sys
# KITTI_EVAL_PATH = '/mnt/bevfusion_mit_xmy/kitti-object-eval-python'
# if KITTI_EVAL_PATH not in sys.path:
#     sys.path.insert(0, KITTI_EVAL_PATH)
# from eval import get_official_eval_result   # 根据实际函数名调整
import time          # 添加这一行


Debug = True

print(f"\n>>>[xmy]🟡[TYJTDatasetV2]🟡[mmdet3d/datasets/tyjt_dataset_v2.py] >>> [Debug Mode = {Debug}] ")


class TYJTEval(DetectionEval):
    """TYJT 数据集专用评估器，直接使用 TYJT 的 data_infos 作为 GT，无需 NuScenes 对象"""
    def __init__(self,
                 config,
                 result_path: str,
                 gt_infos: list,
                 output_dir: str,
                 verbose: bool = True):
        self.cfg = config
        self.result_path = result_path
        self.output_dir = output_dir
        self.verbose = verbose

        # 加载预测框
        self.pred_boxes, self.meta = load_prediction(
            self.result_path,
            self.cfg.max_boxes_per_sample,
            DetectionBox,
            verbose=verbose
        )

        # 为预测框添加 center_dist 属性
        self.pred_boxes = self._add_center_dist(self.pred_boxes)

        # 加载 GT 框
        self.gt_boxes = self._load_tyjt_gt(gt_infos, verbose)

        # 手动过滤预测框（基于 class_range）
        print(f"\n>>>[xmy]🟡[TYJTDatasetV2]🟡[mmdet3d/datasets/tyjt_dataset_v2.py] >>> [nusc/TYJTEval-评测器 对Gt/pred 的范围滤波]: \n    self.cfg.class_range =  {self.cfg.class_range}] ")

        print(f"\n>>>[xmy]🟡[TYJTDatasetV2]🟡[tyjt_dataset_v2.py] >>> [Running: 对预测框(pred_boxes), 进行范围滤波]; self.cfg.class_range = {self.cfg.class_range}")
        self.pred_boxes = self._filter_boxes_by_range(self.pred_boxes, self.cfg.class_range, verbose)

        # 手动过滤 GT 框（基于 class_range）
        print(f"\n>>>[xmy]🟡[TYJTDatasetV2]🟡[tyjt_dataset_v2.py] >>> [Running: 对Gt框(gt_boxes), 进行范围滤波]; self.cfg.class_range = {self.cfg.class_range}")
        self.gt_boxes = self._filter_boxes_by_range(self.gt_boxes, self.cfg.class_range, verbose)

        # 样本 token 列表（从 GT 获取）
        self.sample_tokens = self.gt_boxes.sample_tokens

        if verbose:
            print(f"✅ TYJTEval 初始化完成: 预测框 {len(self.pred_boxes.all)} 个, "
                  f"GT框 {len(self.gt_boxes.all)} 个, 样本数 {len(self.sample_tokens)}")

    def _add_center_dist(self, eval_boxes: EvalBoxes) -> EvalBoxes:
        """为每个预测框添加 center_dist 属性"""
        for sample_token, boxes in eval_boxes.boxes.items():
            for box in boxes:
                # 直接使用 translation（自车坐标）
                box.center_dist = np.linalg.norm(box.translation[:2])
        return eval_boxes

    def _filter_boxes_by_range(self, eval_boxes: EvalBoxes, class_range: dict, verbose: bool) -> EvalBoxes:
        """过滤超出指定范围的框"""
        filtered = EvalBoxes()
        total_before = 0
        total_after = 0
        for sample_token, boxes in eval_boxes.boxes.items():
            kept = []
            for box in boxes:
                max_dist = class_range.get(box.detection_name, 50.0)
                if box.center_dist <= max_dist:
                    kept.append(box)
            filtered.add_boxes(sample_token, kept)
            total_before += len(boxes)
            total_after += len(kept)
        if verbose:
            print(f"\n>>>[xmy]🟡[TYJTDatasetV2]🟡[tyjt_dataset_v2.py] >>> 过滤范围后: 框数从 {total_before} 减少到 {total_after} ")
        return filtered

    def _load_tyjt_gt(self, infos: list, verbose: bool) -> EvalBoxes:
        """从 TYJT 的 data_infos 加载 GT，转换为 DetectionBox 列表（手动添加 num_pts 和 center_dist）"""
        # 类别映射（与 TYJTDatasetV2.CATEGORY_MAPPING 保持一致）
        CATEGORY_MAPPING_EVAL = {
            'car': 'car',
            'truck': 'truck',
            'construction_truck': 'construction_vehicle',
            'van': 'car',
            'bus': 'bus',
            'robot': 'construction_vehicle',
            'pedestrian': 'pedestrian',
            'cyclist': 'bicycle',
            'tricyclist': 'motorcycle',
            'trolley': 'trailer',
            'cone': 'traffic_cone',
            'barrier': 'barrier',
        }

        eval_boxes = EvalBoxes()
        total_boxes = 0
        for info in infos:
            token = info['token']
            boxes = []
            gt_boxes = info['gt_boxes']
            gt_names = info['gt_names']
            num_pts_list = info.get('num_lidar_pts', np.ones(len(gt_boxes), dtype=np.int32))

            for i, (box, name) in enumerate(zip(gt_boxes, gt_names)):
                # 映射类别名
                name = CATEGORY_MAPPING_EVAL.get(name, name)
                if name not in DETECTION_NAMES:
                    if verbose:
                        print(f"警告: 未知类别 '{name}' (原始: {gt_names[i]})，跳过该框")
                    continue

                # 速度
                if box.shape[0] == 7:
                    velocity = (0.0, 0.0)
                else:
                    velocity = tuple(box[7:9].tolist())

                # 中心距离
                center_dist = np.linalg.norm(box[:2])

                # yaw -> 四元数
                quat = Quaternion(axis=[0, 0, 1], radians=box[6])
                rotation = quat.elements.tolist()

                det_box = DetectionBox(
                    sample_token=token,
                    translation=tuple(box[:3].tolist()),
                    size=tuple(box[3:6].tolist()),
                    rotation=tuple(rotation),
                    velocity=velocity,
                    detection_name=name,
                    detection_score=1.0,
                    attribute_name='',
                    num_pts=int(num_pts_list[i] if isinstance(num_pts_list, np.ndarray) else 1),
                )
                det_box.center_dist = center_dist
                boxes.append(det_box)
            eval_boxes.add_boxes(token, boxes)
            total_boxes += len(boxes)

        if verbose:
            print(f"加载 GT: {len(infos)} 个样本, {total_boxes} 个标注")
        return eval_boxes


@DATASETS.register_module()
class TYJTDatasetV2(Custom3DDataset):
    """TYJT Dataset V2 for BEVFusion (适配新版 info 格式,类别映射为10类).

    This dataset reads info pkl files generated by tyjt_converter_v9.4.2_xxx.py.
    The pkl file should be a dict with 'infos' and 'metadata' keys.
    It maps original TYJT categories (15 classes) to 10 classes consistent with nuScenes.
    """

    # 1. 标注规则
    """
    TYJT的标注规则(15类型):
    (1)car:双轴小型车辆,包括 7 座及以下的轿车、SUV、皮卡、出租车、无人驾驶车辆、网约车
    (2)truck: 包括厢式货车、栅式货车、半挂车、平板运输货车、牵引车头、油罐车、混凝土搅拌车、消防车、环卫洒水车、泥头车、渣土车
    (3)工程车辆 construction_truck: 包括叉车、铲车、起重机、压路机、挖土机、推土机、吊车、矿车等工程车辆,多数拥有异形、可变形的特点。注意:只标注车主体部分,延伸出车体的部分不标注
    (4)van 厢式面包车 : 包括中型客运或货运面包车、救护车、MPV 等。注意:MPV 和 SUV 的区别在于 MPV 是 7 座车,SUV 是 5 座车,MPV较长,归类到面包车类型,若 MPV 和 SUV 从图像和点云上显示不全无法分辨是,优先给小型车。
    (5)bus 巴士:包括大型客运车辆,包括校巴、旅游巴、客运班车、城市公交、无人驾驶巴士等
    (6)robot 机器人:括智能物流小车、智能清扫机器人等,特点是小型、低速、多数情况下无人驾驶。当有人驾驶时应该把人也框进去
    (7)pedestrian 行人:包含各种姿态的行人,站立、蹲坐、躺着、静止与动态等,包含人的所有部位,手持任何其他东西不框进(如:伞、行李箱等不框)。注意:背在背上的背包和斜挎包等贴合人体躯干的物品需整体标注,行人手提或拿着的物品不标注,贴合人体四肢即可
    (8)cyclist 两轮车骑行者:有人骑着的两轮车,包含两轮摩托车、两轮电动车、两轮自行车、多人两轮车等
    (9)bicycle 两轮车:没有人骑着的两轮车,包含两轮摩托车、两轮电动车、两轮自行车、多人两轮车等
    (10)tricycle 无人三轮车: 没有人骑着的三轮车,包含三轮摩托车、三轮电动车、三轮自行车、三轮或四轮的老年代步车等
    (11)tricyclist 三轮车骑行者: 有人骑着的三轮车,需要把人也框进去,包含三轮摩托车、三轮电动车、三轮自行车、三轮或四轮的老年代步车等。注意:老年代步车或封闭式三轮车停在可行驶区域(十字路口、车道等)时默认为三轮车骑行人,若停在人行道或停车位时可以给无人三轮车
    (12)trolley手推车:包含手推车、婴儿车、轮椅。手推车不包含推车的人。婴儿车和轮椅如果有人坐着,应该标为一个整体
    (13)cone 锥桶:施工、交通意外中用于隔离的锥形桶、柱形防撞桶等
    (14)barrier水马、栅栏:施工、交通意外中用于隔离的水马、栅栏等
    (15)other 其他:车道上影响车辆正常驾驶的其他物体, 例如路面碎片或落在车道上的落石、树木、纸箱等。不包括路上不影响驾驶的小物体,比如一小块垃圾、小石子
    """
    # 2. 标注类to训练类的mappingg
    # Mapping from original TYJT category names (without prefix) to target classes
    CATEGORY_MAPPING = {
        # 车辆类
        'car': 'car',
        'truck': 'truck',
        'construction_truck': 'construction_vehicle',
        'van': 'car',
        'bus': 'bus',
        'robot': 'construction_vehicle',
        # 行人
        'pedestrian': 'pedestrian',
        # 二轮车(含有人和无人)
        'cyclist': 'bicycle',
        # 三轮车(含有人和无人)
        'tricyclist': 'motorcycle',
        # 其他
        'trolley': 'trailer',
        'cone': 'traffic_cone',
        'barrier': 'barrier',
        # 'other' 忽略
    }

    # 3. 训练类名
    # Final training classes (10 classes, same as nuScenes)
    # 官方配置：mmdet3d/datasets/nuscenes_dataset.py
    CLASSES = (
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    )
    # 自己的历史错误,错位的版本, 仅作警戒!!!
    # CLASSES = (
    #     "car",
    #     "truck",
    #     "bus",
    #     "pedestrian",
    #     "motorcycle",
    #     "bicycle",
    #     "construction_vehicle",
    #     "trailer",
    #     "traffic_cone",
    #     "barrier",
    # )

    # Fixed camera order (must match the keys in info['cams'])
    CAM_ORDER = ['CAM_A', 'CAM_B', 'CAM_C', 'CAM_D']

    # Placeholders for future compatibility (unused, print warning if accessed)
    NameMapping = {}
    DefaultAttribute = {}
    AttrMapping = {}
    AttrMapping_rev = []
    ErrNameMapping = {}

    def __init__(
            self,
            ann_file,
            pipeline=None,
            dataset_root=None,
            object_classes=None,
            map_classes=None,
            load_interval=1,
            with_velocity=True,
            modality=None,
            box_type_3d='LiDAR',
            filter_empty_gt=True,
            test_mode=False,
            eval_version='detection_cvpr_2019',
            use_valid_flag=False,
            **kwargs):
        # 将 object_classes 转换为父类需要的 classes
        classes = object_classes if object_classes is not None else self.CLASSES

        # 存储未使用的参数(仅用于接口兼容)
        self.load_interval = load_interval
        self.with_velocity = with_velocity
        self.eval_version = eval_version
        self.use_valid_flag = use_valid_flag
        self.map_classes = map_classes

        # 设置默认 modality(避免 None)
        if modality is None:
            modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        self.modality = modality

        # 加载 info pkl,兼容新格式(dict 包含 'infos')和旧格式(直接列表)
        with open(ann_file, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict) and 'infos' in data:
            self.data_infos = data['infos']
            self.metadata = data.get('metadata', {'version': 'unknown'})
        else:
            self.data_infos = data
            self.metadata = {'version': 'old'}

        # Set dataset_root if provided
        if dataset_root is not None:
            self.dataset_root = dataset_root

        super().__init__(
            dataset_root=dataset_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

        self.version = self.metadata.get('version', 'tyjt_v1')

        # 评估配置初始化
        self.eval_detection_configs = config_factory(self.eval_version)

    def load_annotations(self, ann_file):
        return self.data_infos

    def get_data_info(self, index):
        info = self.data_infos[index]

        data = {
            'lidar_path': info['lidar_path'],
            'sweeps': info.get('sweeps', []),
            'timestamp': info['timestamp'],
            'ego2global': np.array(info['ego2global'], dtype=np.float32),
            'lidar2ego': np.eye(4, dtype=np.float32),
            'sample_idx': info.get('token', str(index)),   # 用于数据库生成的文件名
        }
            
        # 相机信息(仅在启用时添加)
        if self.modality.get('use_camera', False):
            img_paths = []
            lidar2camera = []
            camera_intrinsics = []
            lidar2image = []
            camera2ego = []
            camera2lidar = []

            for cam in self.CAM_ORDER:
                cam_info = info['cams'].get(cam)
                if cam_info is None:
                    continue

                # 图像路径
                if 'data_path' in cam_info:
                    img_paths.append(cam_info['data_path'])
                else:
                    img_paths.append(cam_info['img_path'])

                # 相机到 ego 变换
                if 'sensor2ego_translation' in cam_info:
                    trans = cam_info['sensor2ego_translation']
                    rot = cam_info['sensor2ego_rotation']
                else:
                    trans = cam_info['cam2ego_translation']
                    rot = cam_info['cam2ego_rotation']

                cam2ego = np.eye(4, dtype=np.float32)
                cam2ego[:3, :3] = Quaternion(rot).rotation_matrix
                cam2ego[:3, 3] = trans
                camera2ego.append(cam2ego)

                lidar2camera_mat = np.linalg.inv(cam2ego)
                lidar2camera.append(lidar2camera_mat)
                camera2lidar.append(cam2ego.copy())

                intrinsic_3x3 = np.array(cam_info['cam_intrinsic'], dtype=np.float32)
                intrinsic_4x4 = np.eye(4, dtype=np.float32)
                intrinsic_4x4[:3, :3] = intrinsic_3x3
                camera_intrinsics.append(intrinsic_4x4)

                lidar2image_mat = intrinsic_4x4 @ lidar2camera_mat
                lidar2image.append(lidar2image_mat)

            # 在循环结束后，将列表转换为 numpy 数组
            camera2ego = np.stack(camera2ego, axis=0)        # (N, 4, 4)
            camera2lidar = np.stack(camera2lidar, axis=0)    # (N, 4, 4)
            camera_intrinsics = np.stack(camera_intrinsics, axis=0)  # (N, 4, 4)
            lidar2camera = np.stack(lidar2camera, axis=0)    # (N, 4, 4)
            lidar2image = np.stack(lidar2image, axis=0)      # (N, 4, 4)

            data['image_paths'] = img_paths
            data['filename'] = img_paths   # 添加这一行，与官方键名对齐
            data['lidar2camera'] = lidar2camera
            data['camera_intrinsics'] = camera_intrinsics
            data['lidar2image'] = lidar2image
            data['camera2ego'] = camera2ego
            data['camera2lidar'] = camera2lidar
            data['box_type_3d'] = self.box_type_3d   # 或者直接写 'LiDAR'
            # 纯视觉模式下提供占位的 lidar_aug_matrix
            if not self.modality.get('use_lidar', True):
                data['lidar_aug_matrix'] = np.eye(4, dtype=np.float32)

        # 标注信息(非测试模式)
        # if not self.test_mode:
        #     data['ann_info'] = self.get_ann_info(index)
        data['ann_info'] = self.get_ann_info(index)
        return data

    def get_ann_info(self, index):
        info = self.data_infos[index]

        # 从 info 直接读取基础框和名称
        gt_bboxes_3d = info['gt_boxes'].copy()          # (N,7)  
        gt_names = info['gt_names'].copy()              # (N,)

        # 如果需要速度,拼接速度字段
        if self.with_velocity:
            if 'gt_velocity' in info:
                gt_velocity = info['gt_velocity']       # (N,2)
            else:
                gt_velocity = np.zeros((len(gt_bboxes_3d), 2), dtype=np.float32)
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)   # (N,9)

        # 类别映射
        mapped_names = []
        gt_labels_3d = []
        for name in gt_names:
            target = self.CATEGORY_MAPPING.get(name)
            if target is None or target not in self.CLASSES:
                # 未映射的类别(如 other)设为 -1,但保留名称用于数据库生成(可设为 'unknown')
                mapped_names.append('unknown')
                gt_labels_3d.append(-1)  # 不在目标class内的目标，index会被设置为-1,代表unknown。   ObjectNameFilter 会给过滤掉
            else:
                mapped_names.append(target)
                gt_labels_3d.append(self.CLASSES.index(target))
        mapped_names = np.array(mapped_names, dtype=str)
        gt_labels_3d = np.array(gt_labels_3d, dtype=np.int64)

        # 转换为 LiDARInstance3DBoxes(底部中心)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        return dict(
            gt_bboxes_3d=gt_bboxes_3d,          # 源数据Gt是全部加载：List：3D边界框，形状 (N, 7) 或 (N, 9) (含速度)，N 为原始样本中的目标数量（未经过滤）
            gt_labels_3d=gt_labels_3d,          # 源数据Gt是全部加载：List：类别标签索引，形状 (N,)，取值范围 -1 或 (0 ~ C-1)，    -1 表示无效/忽略的目标（原始数据未裁剪）
                                                # 但：-1代表 unknown类型； 可以被 ObjectNameFilter 默认mask掉；其他的类型则需要配置
            gt_names=mapped_names,              # 源数据Gt是全部加载：List：类别名称字符串列表，长度 N，用于数据库生成等辅助功能（原始长度）
            gt_names_3d=mapped_names,           # 源数据Gt是全部加载：List：类别名称字符串列表，长度 N，供 DefaultFormatBundle3D 动态生成标签索引（原始长度）
        )
    
    def get_cat_ids(self, idx):
        info = self.data_infos[idx]
        if self.use_valid_flag and 'valid_flag' in info:
            mask = info['valid_flag']
            raw_names = set(info['gt_names'][mask])
        else:
            raw_names = set(info['gt_names'])

        cat_ids = []
        for name in raw_names:
            target = self.CATEGORY_MAPPING.get(name)
            if target in self.CLASSES:
                cat_ids.append(self.CLASSES.index(target))
        return cat_ids

    # [xmy][v2.3] nusc评测，使用 TYJTEval 替代适配器
    def _format_bbox(self, results, jsonfile_prefix):
        """将模型输出转换为 nuScenes 评估所需的 JSON 文件"""
        nusc_annos = {}
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det)   # 转换检测结果为 NuScenesBox（LiDAR坐标系，自车）
            sample_token = self.data_infos[sample_id]['token']
            # 注释掉转换到全局坐标系的调用，保持自车坐标系
            # boxes = lidar_nusc_box_to_global(
            #     self.data_infos[sample_id],
            #     boxes,
            #     self.CLASSES,
            #     self.eval_detection_configs,
            #     self.eval_version
            # )
            for box in boxes:
                name = self.CLASSES[box.label]
                if np.linalg.norm(box.velocity[:2]) > 0.2:
                    attr = 'vehicle.moving'
                else:
                    attr = 'vehicle.parked'
                nusc_anno = {
                    'sample_token': sample_token,
                    'translation': box.center.tolist(),
                    'size': box.wlh.tolist(),
                    'rotation': box.orientation.elements.tolist(),
                    'velocity': box.velocity[:2].tolist(),
                    'detection_name': name,
                    'detection_score': box.score,
                    'attribute_name': attr,
                }
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos

        submission = {'meta': self.modality, 'results': nusc_annos}
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        mmcv.dump(submission, res_path)
        return res_path

    def _evaluate_single(self, result_path, logger=None, metric='bbox', result_name='pts_bbox'):
        """使用 TYJTEval 进行评估"""
        config = self.eval_detection_configs
        output_dir = osp.join(*osp.split(result_path)[:-1])
        tyjt_eval = TYJTEval(
            config=config,
            result_path=result_path,
            gt_infos=self.data_infos,
            output_dir=output_dir,
            verbose=True,
        )
        # # 打印 nuScenes 原始框（仅第一个样本）
        # print("\n========== 所有样本的 GT 和预测框（用于检查潜在匹配） ==========")
        # for sample_token in tyjt_eval.gt_boxes.boxes.keys():
        #     gt_boxes = tyjt_eval.gt_boxes.boxes[sample_token]
        #     pred_boxes = tyjt_eval.pred_boxes.boxes.get(sample_token, [])
        #     print(f"Sample {sample_token}: GT 框数={len(gt_boxes)}, Pred 框数={len(pred_boxes)}")
        #     if len(gt_boxes) > 0:
        #         print("  GT 框:")
        #         for gt in gt_boxes:
        #             print(f"    {gt.detection_name}: center={gt.translation}, size={gt.size}, yaw={Quaternion(gt.rotation).yaw_pitch_roll[0]:.3f}")
        #     if len(pred_boxes) > 0:
        #         print("  Pred 框（前5个）:")
        #         for pred in pred_boxes[:5]:
        #             print(f"    {pred.detection_name}: center={pred.translation}, size={pred.size}, yaw={Quaternion(pred.rotation).yaw_pitch_roll[0]:.3f}, score={pred.detection_score:.4f}")
        # print("====================================================\n")

        metrics_summary = tyjt_eval.main(render_curves=False)   # 返回 metrics.serialize()

        # 整理为字典（格式与 NuScenesDataset 一致）
        detail = {}
        from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
        nusc_classes = NuScenesDataset.CLASSES
        for name in nusc_classes:
            for k, v in metrics_summary['label_aps'][name].items():
                detail[f'object/{name}_ap_dist_{k}'] = v
            for k, v in metrics_summary['label_tp_errors'][name].items():
                detail[f'object/{name}_{k}'] = v
        for k, v in metrics_summary['tp_errors'].items():
            detail[f'object/{NuScenesDataset.ErrNameMapping[k]}'] = v
        detail['object/nds'] = metrics_summary['nd_score']
        detail['object/map'] = metrics_summary['mean_ap']
        return detail

    def format_results(self, results, jsonfile_prefix=None):
        """将结果保存为 JSON 文件，并返回文件路径和临时目录"""
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(self), "results length mismatch"

        if jsonfile_prefix is None:
            base_tmp = os.path.join(os.getcwd(), 'tmp')
            os.makedirs(base_tmp, exist_ok=True)
            tmp_dir = tempfile.TemporaryDirectory(dir=base_tmp)
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        result_files = self._format_bbox(results, jsonfile_prefix)
        return result_files, tmp_dir

    # ========== [v2.3.5][适配kitti评测] 新增 KITTI 评测分支  ==========
    def evaluate(self, results, metric='bbox', jsonfile_prefix=None, result_names=['pts_bbox'], **kwargs):
        # v2.3.3 中 metric 是没有用的，因为只支持 nusc-bbox 评测
        # v2.3.5 中 metric 是可用的，支持nusc-bbox 和 kitti
        # if Debug:
        #     print("\n=== 模型预测结果格式验证 ===")
        #     print(f"results 长度: {len(results)}")
        #     first = results[0]
        #     print(f"第一个样本的键: {first.keys()}")
        #     if 'boxes_3d' in first:
        #         box = first['boxes_3d']
        #         print(f"boxes_3d 类型: {type(box)}, shape: {box.shape if hasattr(box, 'shape') else 'N/A'}")
        #         print(f"第一个框的数值 (前7维): {box[0][:7].cpu().numpy() if hasattr(box, 'cpu') else box[0][:7]}")
        #     if 'scores_3d' in first:
        #         print(f"scores_3d 前5个: {first['scores_3d'][:5].cpu().numpy() if hasattr(first['scores_3d'], 'cpu') else first['scores_3d'][:5]}")
        #     if 'labels_3d' in first:
        #         print(f"labels_3d 前5个: {first['labels_3d'][:5].cpu().numpy() if hasattr(first['labels_3d'], 'cpu') else first['labels_3d'][:5]}")
        #     # exit()
        metrics = {}
        if isinstance(metric, str):
            metric = [metric]
        # 原有的 nuScenes 评估（bbox）
        if ('bbox' in metric) or ('nuscenes' in metric)  or ('nusc' in metric):
            # ... 保持原有代码不变 ...
            if "masks_bev" in results[0]:
                metrics.update(self.evaluate_map(results))
            if "boxes_3d" in results[0]:
                result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
                if isinstance(result_files, dict):
                    for name in result_names:
                        ret_dict = self._evaluate_single(result_files[name])
                        metrics.update(ret_dict)
                elif isinstance(result_files, str):
                    metrics.update(self._evaluate_single(result_files))
                if tmp_dir is not None:
                    pass
        # 新增 KITTI 评估分支
        if 'kitti' in metric:
            if "boxes_3d" in results[0]:
                kitti_metrics = self._evaluate_kitti(results)
                metrics.update(kitti_metrics)
        return metrics
    
    def evaluate_map(self, results):
        warnings.warn('evaluate_map not implemented.')
        return {}


    # ========== [v2.3.5][适配kitti评测] 新增 KITTI 评估相关方法 ==========
    def _map_to_kitti_class(self, tyjt_class):
        """将 TYJT 训练类别（10类）或原始类别映射到 KITTI 评估类别： 官方只支持 Car\Pedestrian\Cyclist 三类 """
        mapping = {
            'car': 'Car',
            'truck': 'DontCare',
            'van': 'DontCare',
            'bus': 'DontCare',
            'construction_vehicle': 'DontCare',
            'trailer': 'DontCare',
            'pedestrian': 'DontCare',
            'cyclist': 'DontCare',
            'bicycle': 'DontCare',
            'motorcycle': 'DontCare',
            'tricyclist': 'DontCare',
            # 以下为原始类别（如果直接使用原始 name）
            'construction_truck': 'DontCare',
            'robot': 'DontCare',
            'trolley': 'DontCare',
            'cone': 'DontCare',
            'barrier': 'DontCare',
            'other': 'DontCare',
        }
        # 注意：self.CLASSES 是训练用 10 类，输入 tyjt_class 可能是原始名称或训练类别名
        if tyjt_class in self.CLASSES:
            # 训练类别已映射，直接查找
            target = tyjt_class
        else:
            # 可能是原始类别，先通过 CATEGORY_MAPPING 映射到训练类别
            target = self.CATEGORY_MAPPING.get(tyjt_class, tyjt_class)
        return mapping.get(target, 'DontCare')
    
    def _format_kitti_gt(self):
        """
        将 TYJT 数据集的原始 GT 标注转换为 KITTI 评估工具所需的格式。

        输入来源: 
            - 由 tools/tyjt_converter.py 中的 build_sample_info 函数 进行组包（版本 v9.4.8 及以后，尺寸顺序已修正）
            - 由 TYJTDatasetV2 的 __init__() 通过 pickle.load(f) 加载 self.data_infos
            - 每个 info 包含：
                - gt_boxes: numpy.ndarray, shape (N, 7)
                    含义：[cx, cy, cz, w, l, h, yaw]   ✅ 已修正：宽度(w)、长度(l)、高度(h)
                    - cx, cy, cz: 3D 框的几何中心（米），坐标系：路口/自车系，X前 Y左 Z上
                    - w: 宽度（Y轴方向，左右），l: 长度（X轴方向，前后），h: 高度（Z轴方向，上下）
                    - yaw: 绕 Z 轴的偏航角（弧度），逆时针为正，0 表示朝 X 正向
                - gt_names: numpy.ndarray, shape (N,)，原始类别字符串（如 'car', 'truck'）

        输出格式：列表 gt_annos，每个元素对应一个样本，是一个字典，包含以下字段（与 KITTI 官方一致）：
            {
                'name': np.ndarray(str),          # 映射后的 KITTI 类别：'Car', 'Pedestrian', 'Cyclist' 或 'DontCare'（跳过）
                'truncated': np.ndarray(float32), # 截断程度，固定 0.0
                'occluded': np.ndarray(int32),    # 遮挡程度，固定 0
                'alpha': np.ndarray(float32),     # 观察角，简化为 rotation_y
                'bbox': np.ndarray(float32),      # 2D 边界框，占位 [0,0,0,0]
                'dimensions': np.ndarray(float32), # [h, w, l] 高度、宽度、长度（米）
                'location': np.ndarray(float32),   # [cx, cy, cz] 几何中心（米），与 KITTI 的底部中心要求不同，
                                                # 但与预测框保持一致即可保证 IoU 正确。
                'rotation_y': np.ndarray(float32), # 绕 Y 轴的朝向角（弧度），范围 [-π, π]
            }

        转换细节：
            1. 类别映射：通过 _map_to_kitti_class 将原始类别转为 KITTI 三类之一，无效类别返回 'DontCare' 并跳过。
            2. 尺寸顺序：原始 [w, l, h] -> KITTI [h, w, l]。
            3. 位置：直接保留几何中心 (cx, cy, cz)，不做底部中心转换（与预测框一致）。
            4. 旋转角：将原始 yaw（绕 Z 轴）转换为 KITTI 的 rotation_y（绕 Y 轴）。
            公式: ry = - (yaw + π/2) ，然后归一化到 [-π, π)。
            5. 其他字段（truncated, occluded, alpha, bbox）按 KITTI 要求填充占位值。
        """
        gt_annos = []
        for info in self.data_infos:
            gt_boxes = info['gt_boxes']      # (N,7) [cx,cy,cz,w,l,h,yaw]  ✅ 修正后顺序
            gt_names = info['gt_names']      # (N,)
            names, truncated, occluded, alpha = [], [], [], []
            bbox, dimensions, location, rotation_y = [], [], [], []
            for i, box in enumerate(gt_boxes):
                kitti_cls = self._map_to_kitti_class(gt_names[i])
                if kitti_cls == 'DontCare':
                    continue
                # ===== 支持 tyjt_converter.py 版本 v9.4.8 (0424) 及以后 =====
                # 解析顺序：[cx, cy, cz, w, l, h, yaw]
                x, y, z, w, l, h, yaw = box[:7]
                # ===== 兼容旧版本 v9.4.7 及以前（已废弃，保留注释）=====
                # x, y, z, l, w, h, yaw = box[:7]  # 老版本错误顺序

                # KITTI 尺寸顺序 [h, w, l]
                dimensions.append([h, w, l])

                # 位置：保持几何中心
                location.append([x, y, z])
                # 旋转角转换
                ry = - (yaw + np.pi / 2.0)
                ry = (ry + np.pi) % (2 * np.pi) - np.pi
                rotation_y.append(ry)
                alpha.append(ry)
                names.append(kitti_cls)
                truncated.append(0.0)
                occluded.append(0)
                bbox.append([0.0, 0.0, 0.0, 0.0])
            gt_annos.append({
                'name': np.array(names, dtype=str),
                'truncated': np.array(truncated, dtype=np.float32),
                'occluded': np.array(occluded, dtype=np.int32),
                'alpha': np.array(alpha, dtype=np.float32),
                'bbox': np.array(bbox, dtype=np.float32).reshape(-1,4) if bbox else np.zeros((0,4), dtype=np.float32),
                'dimensions': np.array(dimensions, dtype=np.float32).reshape(-1,3) if dimensions else np.zeros((0,3), dtype=np.float32),
                'location': np.array(location, dtype=np.float32).reshape(-1,3) if location else np.zeros((0,3), dtype=np.float32),
                'rotation_y': np.array(rotation_y, dtype=np.float32),
            })
        return gt_annos

    def _format_kitti_pred(self, results):
        """生成按样本分组的预测标注列表（数组形式，含 score），适配 KITTI 格式。
        
        输入：results 列表，每个元素包含：
            - boxes_3d: LiDARInstance3DBoxes，内部 tensor 格式为 [cx, cy, cz, w, l, h, yaw]（底部中心）
            - scores_3d: torch.Tensor
            - labels_3d: torch.Tensor
        输出：与 _format_kitti_gt 相同格式的字典，增加 'score' 字段。
        """
        pred_annos = []
        for det in results:
            boxes = det['boxes_3d'].tensor.cpu().numpy()   # (N,7) [cx, cy, cz, w, l, h, yaw]（底部中心）
            scores = det['scores_3d'].cpu().numpy()
            labels = det['labels_3d'].cpu().numpy()
            names, truncated, occluded, alpha = [], [], [], []
            bbox, dimensions, location, rotation_y, score_list = [], [], [], [], []
            for i in range(len(boxes)):
                kitti_cls = self._map_to_kitti_class(self.CLASSES[labels[i]])
                if kitti_cls == 'DontCare':
                    continue
                box = boxes[i]
                x, y, z, w, l, h, yaw = box[:7]   # z 是底部中心
                # ===== 支持 tyjt_converter.py 版本 v9.4.8 (0424) 及以后 =====
                # 尺寸转换：原始 [w, l, h] -> KITTI [h, w, l]
                dimensions.append([h, w, l])
                # ===== 兼容旧版本 v9.4.7 及以前（已废弃，保留注释）=====
                # dimensions.append([h, l, w])    # 老版本临时交换

                # 位置：将底部中心转换为几何中心（与 GT 处理对齐）
                cz = z + h / 2.0
                location.append([x, y, cz])
                # 旋转角转换
                ry = - (yaw + np.pi / 2.0)
                ry = (ry + np.pi) % (2 * np.pi) - np.pi
                rotation_y.append(ry)
                alpha.append(ry)
                names.append(kitti_cls)
                truncated.append(0.0)
                occluded.append(0)
                bbox.append([0.0, 0.0, 0.0, 0.0])
                score_list.append(float(scores[i]))
            pred_annos.append({
                'name': np.array(names, dtype=str),
                'truncated': np.array(truncated, dtype=np.float32),
                'occluded': np.array(occluded, dtype=np.int32),
                'alpha': np.array(alpha, dtype=np.float32),
                'bbox': np.array(bbox, dtype=np.float32).reshape(-1,4) if bbox else np.zeros((0,4), dtype=np.float32),
                'dimensions': np.array(dimensions, dtype=np.float32).reshape(-1,3) if dimensions else np.zeros((0,3), dtype=np.float32),
                'location': np.array(location, dtype=np.float32).reshape(-1,3) if location else np.zeros((0,3), dtype=np.float32),
                'rotation_y': np.array(rotation_y, dtype=np.float32),
                'score': np.array(score_list, dtype=np.float32),
            })
        return pred_annos

    def _filter_kitti_annos_by_range(self, annos_list, class_range, default_dist=50.0):
        """过滤 KITTI 格式的标注列表，移除超出 class_range 的框"""
        filtered = []
        for idx, anno in enumerate(annos_list):
            # 防御：确保 anno 是字典且包含 'name' 键
            if not isinstance(anno, dict) or 'name' not in anno:
                print(f"⚠️ 警告：第 {idx} 个元素不是有效字典，已跳过。类型: {type(anno)}")
                continue  # 跳过该样本（或根据需求保留）

            names = anno['name']
            locs = anno['location']
            if len(names) == 0:
                filtered.append(anno)
                continue

            # 计算 xy 平面中心距离
            dists = np.linalg.norm(locs[:, :2], axis=1)
            keep = [i for i, name in enumerate(names) if dists[i] <= class_range.get(name, default_dist)]

            if not keep:
                # 无保留框：创建空数组（注意：v[[]] 对于字符串数组安全）
                filtered.append({k: v[[]] for k, v in anno.items()})
            else:
                keep_arr = np.array(keep)
                filtered.append({k: v[keep_arr] for k, v in anno.items()})
        return filtered
        
    def _evaluate_kitti(self, results):
        import sys
        sys.path.insert(0, '/mnt/bevfusion_mit_xmy/kitti-object-eval-python')
        from eval import get_official_eval_result

        gt_annos = self._format_kitti_gt()
        pred_annos = self._format_kitti_pred(results)
        ## kitti开启 范围过滤
        nusc_class_range = self.eval_detection_configs.class_range  
        kitti_class_range = {k.capitalize(): v for k, v in nusc_class_range.items()}
        # class_range = {'Car': 50.0, 'Pedestrian': 30.0, 'Cyclist': 30.0}  # 与 nuScenes cfg 中的 class_range 一致
        print(f"\n>>>[xmy]🟡[TYJTDatasetV2]🟡[mmdet3d/datasets/tyjt_dataset_v2.py] >>> [Kitti-评测器 对Gt/pred 的范围滤波]: \n    kitti_class_range =  {kitti_class_range}] ")

        # gt_annos = self._filter_kitti_annos_by_range(gt_annos, kitti_class_range)
        # pred_annos = self._filter_kitti_annos_by_range(pred_annos, kitti_class_range)
        # # ========== 验证数据格式 ==========
        # print("\n=== KITTI 数据格式验证 ===")
        # print(f"样本数 (gt_annos): {len(gt_annos)}")
        # if len(gt_annos) > 0:
        #     first = gt_annos[0]
        #     print(f"GT 第一个样本的键: {first.keys()}")
        #     for key in ['name', 'alpha', 'bbox', 'dimensions', 'location', 'rotation_y']:
        #         if key in first:
        #             print(f"  {key}: type={type(first[key])}, shape={getattr(first[key], 'shape', 'N/A')}")
        #         else:
        #             print(f"  {key}: 缺失")
        # print(f"预测样本数: {len(pred_annos)}")
        # if len(pred_annos) > 0:
        #     print(f"预测第一个样本的键: {pred_annos[0].keys()}")
        #     if 'score' in pred_annos[0]:
        #         print(f"  score: shape={pred_annos[0]['score'].shape}")

        # ========== 验证返回值类型 ==========
        try:
            # result = get_official_eval_result(gt_annos, pred_annos, ['Car', 'Pedestrian', 'Cyclist'])
            result = get_official_eval_result(
                gt_annos, pred_annos, ['Car', 'Pedestrian', 'Cyclist'],
                z_axis=2,       # 因为 location 顺序是 [x,y,z]，高度索引 2
                z_center=0.5    # 因为我们已经将坐标转换为几何中心（cz = z + h/2）
            )
            time.sleep(2)          # 等待 2 秒后打印
            print(f"\n返回值类型: {type(result)}")
            if isinstance(result, str):
                print("返回值是字符串（符合预期）")
                # 打印前 500 字符
                print("内容预览:\n", result[:500])
            else:
                print("返回值不是字符串，需检查评估函数")
        except Exception as e:
            print(f"\n评估过程出错: {e}")
            import traceback
            traceback.print_exc()
        return {}

