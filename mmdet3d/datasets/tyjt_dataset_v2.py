"""
功能:
    - 实现 TYJT 数据集的官方 nuScenes 评估指标（mAP、NDS 等）计算。
    - 直接使用 TYJT 的 data_infos 作为 GT，无需模拟 NuScenes 对象或依赖 split 文件。
    - 通过继承 DetectionEval 并重写 GT 加载逻辑，复用官方评估核心算法。
    - 预测结果保持自车坐标系，避免全局坐标转换导致的范围过滤失效。

版本v2.3.0
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
    torchpack dist-run -np 1 python tools/train_tyjt_pkl.py ./xmy_tools/tyjt2pkl_v2/0311/configs/tyjt_2d_CenterheadLSSfpn_vA010_a_0316_XmyCenterHeadPerCls_TYJTDatasetV2_360x640.yaml --run-dir runs/tyjt_2d_CenterheadLSSfpn_vA010_a_0316_XmyCenterHeadPerCls_TYJTDatasetV2_360x640
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

Debug = True

print(f"\n>>>[xmy]🔵[TYJTDatasetV2]🔵[mmdet3d/datasets/tyjt_dataset_v2.py] >>> [Debug Mode = {Debug}] ")


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
        self.pred_boxes = self._filter_boxes_by_range(self.pred_boxes, self.cfg.class_range, verbose)

        # 手动过滤 GT 框（基于 class_range）
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
            print(f"过滤范围后: 框数从 {total_before} 减少到 {total_after}")
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
    CLASSES = (
        "car",
        "truck",
        "bus",
        "pedestrian",
        "motorcycle",
        "bicycle",
        "construction_vehicle",
        "trailer",
        "traffic_cone",
        "barrier",
    )

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

            data['image_paths'] = img_paths
            data['lidar2camera'] = lidar2camera
            data['camera_intrinsics'] = camera_intrinsics
            data['lidar2image'] = lidar2image
            data['camera2ego'] = camera2ego
            data['camera2lidar'] = camera2lidar

        # # 标注信息(非测试模式)
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
                gt_labels_3d.append(-1)
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
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=mapped_names,   # 添加 gt_names,供数据库生成使用
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

    # [xmy][v2.3] 使用 TYJTEval 替代适配器
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

    def evaluate(self, results, metric='bbox', jsonfile_prefix=None, result_names=['pts_bbox'], **kwargs):
        """评估入口"""
        metrics = {}
        if "masks_bev" in results[0]:
            metrics.update(self.evaluate_map(results))
        if "boxes_3d" in results[0]:
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
            if isinstance(result_files, dict):
                for name in result_names:
                    print(f"Evaluating bboxes of {name}")
                    ret_dict = self._evaluate_single(result_files[name])
                metrics.update(ret_dict)
            elif isinstance(result_files, str):
                metrics.update(self._evaluate_single(result_files))
            if tmp_dir is not None:
                pass
        return metrics

    def evaluate_map(self, results):
        warnings.warn('evaluate_map not implemented.')
        return {}
