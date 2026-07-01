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

from pyquaternion import Quaternion

# ========== [xmy][v2.2]新增导入 ==========
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.detection.config import config_factory
from mmdet3d.datasets.nuscenes_dataset import output_to_nusc_box, lidar_nusc_box_to_global

Debug = True

print(f"\n>>>[xmy]🔵[TYJTDatasetV2]🔵[mmdet3d/datasets/tyjt_dataset_v2.py] >>> [Debug Mode = {Debug}] ")


class NuScenesAdapter:
    """模拟 NuScenes 对象的最小接口,供 DetectionEval 使用"""
    # 标准属性列表（从 nuScenes 复制，用于属性映射）
    ATTRIBUTES = [
        {'token': 'cycle.with_rider', 'name': 'cycle.with_rider'},
        {'token': 'cycle.without_rider', 'name': 'cycle.without_rider'},
        {'token': 'pedestrian.moving', 'name': 'pedestrian.moving'},
        {'token': 'pedestrian.standing', 'name': 'pedestrian.standing'},
        {'token': 'pedestrian.sitting_lying_down', 'name': 'pedestrian.sitting_lying_down'},
        {'token': 'vehicle.moving', 'name': 'vehicle.moving'},
        {'token': 'vehicle.parked', 'name': 'vehicle.parked'},
        {'token': 'vehicle.stopped', 'name': 'vehicle.stopped'},
    ]

    def __init__(self, infos, classes, version='v1.0-tyjt'):
        """
        Args:
            infos (list[dict]): TYJT 数据集的 info 列表。
            classes (tuple): TYJT 使用的类别列表（顺序与训练时一致）。
            version (str): 版本号，用于标识。
        """
        self.version = version
        self.dataroot = '/dummy'                     # 评估器需要此属性
        self._infos = infos
        self._classes = classes
        # 构建属性列表（作为属性）
        self.attribute = self.ATTRIBUTES
        # 构建类别列表
        self.category = [{'token': f'category_{cls}', 'name': cls} for cls in classes]
        if Debug:
            print(f">>>[xmy]🔵[tyjt_dataset_v2.py]>>> class NuScenesAdapter: self._classes = {self._classes}; ")
            print(f">>>[xmy]🔵[tyjt_dataset_v2.py]>>> class NuScenesAdapter: self.category = {self.category}; ")

        self._build_annotations()
        # 在 __init__ 中添加
        self.scene = [{'token': 'scene_dummy', 'name': 'dummy_scene'}]
        self.log = [{'token': 'log_dummy', 'name': 'dummy_log'}]

    def _build_annotations(self):
        """从 TYJT info 构建 sample 和 sample_annotation 数据"""
        self.sample = []
        self.sample_annotation = []

        for info in self._infos:
            token = info['token']
            # 为 sample 添加评估器所需的字段
            self.sample.append({
                'token': token,
                'scene_token': 'scene_dummy',               # 虚拟场景 token
                'log_token': 'log_dummy',                   # 虚拟日志 token
                'timestamp': info.get('timestamp', 0),      # 使用已有时间戳，或默认0
            })

            gt_boxes = info['gt_boxes']
            gt_names = info['gt_names']

            for i, (box, name) in enumerate(zip(gt_boxes, gt_names)):
                # box 前7维: x, y, z, l, w, h, yaw (弧度,绕 z 轴,右手系)
                quat = Quaternion(axis=[0, 0, 1], radians=box[6])
                # 构建标注字典（包含评估所需字段）
                self.sample_annotation.append({
                    'token': f'{token}_{i}',
                    'sample_token': token,
                    'category_token': f'category_{name}',          # 对应类别表中的 token
                    'translation': box[:3].tolist(),
                    'size': box[3:6].tolist(),
                    'rotation': quat.elements.tolist(),
                    'num_lidar_pts': 1,                            # 假设都有点云点，避免被过滤
                    'attribute_tokens': [],                        # 无属性标注，留空
                })

    def get(self, table_name, token):
        """
        模拟 nusc.get() 方法，返回指定表名和 token 对应的数据。
        """
        if table_name == 'sample':
            # 返回 sample 对象（已在 _build_annotations 中构建）
            for s in self.sample:
                if s['token'] == token:
                    return s
            raise KeyError(f"Sample token {token} not found")
        elif table_name == 'sample_annotation':
            for ann in self.sample_annotation:
                if ann['token'] == token:
                    return ann
        elif table_name == 'category':
            for cat in self.category:
                if cat['token'] == token:
                    return cat
        elif table_name == 'attribute':
            for attr in self.attribute:
                if attr['token'] == token:
                    return attr
        elif table_name == 'scene':
            for sc in self.scene:
                if sc['token'] == token:
                    return sc
        elif table_name == 'log':
            for lg in self.log:
                if lg['token'] == token:
                    return lg
        raise KeyError(f"Token {token} not found in table {table_name}")


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

        # # 评估配置初始化(占位)
        # self.eval_detection_configs = None
        self.eval_detection_configs = config_factory(self.eval_version)  # [xmy][v2.2]新增: 添加评估配置初始化, 为四'detection_cvpr_2019'

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

        # 标注信息(非测试模式)
        if not self.test_mode:
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

    # [xmy][v2.2]
    def _format_bbox(self, results, jsonfile_prefix):
        """将模型输出转换为 nuScenes 评估所需的 JSON 文件"""
        nusc_annos = {}
        # 注意：使用输出转框函数，返回的 NuScenesBox 的 label 是 TYJT 的标签索引（0-9）
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det)   # 转换检测结果为 NuScenesBox（LiDAR坐标系）
            sample_token = self.data_infos[sample_id]['token']
            # 转换到全局坐标系（同时应用类别范围过滤）
            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                self.CLASSES,                     # 传入 TYJT 类别顺序（名称是 nuScenes 标准名）
                self.eval_detection_configs,
                self.eval_version
            )
            for box in boxes:
                # 根据 TYJT 的标签索引获取 nuScenes 标准类别名（因为 self.CLASSES 中的名称已是标准名）
                name = self.CLASSES[box.label]
                # 属性：根据速度简单判断，可自行调整
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

    # [xmy][v2.2]
    def _evaluate_single(self, result_path, logger=None, metric='bbox', result_name='pts_bbox'):
        # 创建适配器，
        adapter = NuScenesAdapter(self.data_infos, self.CLASSES, version='v1.0-trainval')  # [临时修改] 写死 'v1.0-trainval' 绕过版本检查，实际不是这个版本名
        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc_eval = DetectionEval(
            nusc=adapter,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set='val',               # 适配器包含所有样本，此处使用 'val' 占位
            output_dir=output_dir,
            verbose=True,
        )
        nusc_eval.main(render_curves=False)
        # 读取指标
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        # 整理为字典（格式与 NuScenesDataset 一致）
        detail = {}
        from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
        nusc_classes = NuScenesDataset.CLASSES
        for name in nusc_classes:
            for k, v in metrics['label_aps'][name].items():
                detail[f'object/{name}_ap_dist_{k}'] = v
            for k, v in metrics['label_tp_errors'][name].items():
                detail[f'object/{name}_{k}'] = v
        for k, v in metrics['tp_errors'].items():
            detail[f'object/{NuScenesDataset.ErrNameMapping[k]}'] = v
        detail['object/nds'] = metrics['nd_score']
        detail['object/map'] = metrics['mean_ap']
        return detail

    # [xmy][v2.2]
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

    # [xmy][v2.2]
    def evaluate(self, results, metric='bbox', jsonfile_prefix=None, result_names=['pts_bbox'], **kwargs):
        """评估入口"""
        metrics = {}
        # 如果有地图分割结果（当前未使用）
        if "masks_bev" in results[0]:
            metrics.update(self.evaluate_map(results))
        # 3D 检测结果
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
                # 可清理临时目录，但有时需要保留以查看中间文件，此处暂不清理
                pass
        return metrics

    def evaluate_map(self, results):
        warnings.warn('evaluate_map not implemented.')
        return {}