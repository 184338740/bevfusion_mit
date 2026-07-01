import mmcv
import numpy as np
import os
import random
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union

from mmdet3d.core.bbox.box_np_ops import points_cam2img
from mmdet3d.datasets import NuScenesDataset, TYJTDataset

# 从Dataset类获取配置（避免硬编码）
NUS_CATEGORIES = NuScenesDataset.CLASSES
TYJT_CATEGORIES = TYJTDataset.CLASSES
NUS_NAME_MAPPING = NuScenesDataset.NameMapping
TYJT_CATEGORY_MAPPING = TYJTDataset.TYJT_CATEGORY_MAPPING  # 导入tyjt类别映射

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')



def create_nuscenes_infos(
    root_path,
    info_prefix,
    version='v1.0-trainval',
    dataset='tyjt',
    max_sweeps=10,
    max_radar_sweeps=10,
    random_seed=42,
    train_val_ratio=0.5  # 训练验证比例，默认5:5
):
    """创建TYJT的info文件（pkl）
    
    仿照nusc_converter实现，支持trainval和test两种模式
    """
    # 初始化NuScenes对象
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    
    # 获取可用场景
    available_scenes = get_available_scenes(nusc)
    print(f"🔵 TYJT转换器: 找到 {len(available_scenes)} 个可用场景")
    
    # 检测是否为test模式
    test = 'test' in version.lower()
    
    if test:
        # 测试模式：所有样本作为测试集
        print(f"🔵 测试模式: 生成测试集")
        
        # 获取所有样本的token
        all_samples = nusc.sample
        all_samples.sort(key=lambda x: x['timestamp'])
        all_sample_tokens = [sample['token'] for sample in all_samples]
        
        # 生成测试集info
        train_infos, _ = _fill_trainval_infos(
            nusc,
            train_scenes=set(all_sample_tokens),
            val_scenes=set(),
            test=True,
            max_sweeps=max_sweeps,
            max_radar_sweeps=0,
            is_sample_level=True
        )
        
        # 保存测试集
        metadata = dict(version=version, dataset='tyjt')
        data = dict(infos=train_infos, metadata=metadata)
        info_path = osp.join(root_path, '{}_infos_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        print(f"✅ TYJT测试集已保存: {info_path}")
        
    else:
        # 训练验证模式
        print(f"🔵 训练验证模式: 比例 {train_val_ratio}:{1-train_val_ratio}")
        
        if len(available_scenes) == 1:
            # 单场景数据集：按样本划分
            print(f"🔵 单场景数据集，使用样本级划分")
            
            all_samples = nusc.sample
            all_samples.sort(key=lambda x: x['timestamp'])
            all_sample_tokens = [sample['token'] for sample in all_samples]
            
            # 按比例划分样本
            split_idx = int(train_val_ratio * len(all_sample_tokens))
            train_sample_tokens = set(all_sample_tokens[:split_idx])
            val_sample_tokens = set(all_sample_tokens[split_idx:])
            
            print(f"  训练样本: {len(train_sample_tokens)}")
            print(f"  验证样本: {len(val_sample_tokens)}")
            
            train_infos, val_infos = _fill_trainval_infos(
                nusc,
                train_scenes=train_sample_tokens,
                val_scenes=val_sample_tokens,
                test=False,
                max_sweeps=max_sweeps,
                max_radar_sweeps=0,
                is_sample_level=True
            )
            
        else:
            # 多场景数据集：按场景划分
            print(f"🔵 多场景数据集，使用场景级划分")
            
            random.seed(random_seed)
            total_scenes = available_scenes.copy()
            random.shuffle(total_scenes)
            
            # 按比例划分场景
            split_idx = int(train_val_ratio * len(total_scenes))
            train_scene_list = total_scenes[:split_idx]
            val_scene_list = total_scenes[split_idx:]
            
            train_scene_tokens = set([s['token'] for s in train_scene_list])
            val_scene_tokens = set([s['token'] for s in val_scene_list])
            
            print(f"  训练场景: {len(train_scene_tokens)}")
            print(f"  验证场景: {len(val_scene_tokens)}")
            
            train_infos, val_infos = _fill_trainval_infos(
                nusc,
                train_scenes=train_scene_tokens,
                val_scenes=val_scene_tokens,
                test=False,
                max_sweeps=max_sweeps,
                max_radar_sweeps=0,
                is_sample_level=False
            )
        
        # 保存训练集和验证集
        metadata = dict(version=version, dataset='tyjt')
        
        # 保存训练集
        data = dict(infos=train_infos, metadata=metadata)
        mmcv.dump(data, osp.join(root_path, f"{info_prefix}_infos_train.pkl"))
        print(f"✅ TYJT训练集已保存: {len(train_infos)} 个样本")
        
        # 保存验证集  
        data['infos'] = val_infos
        info_val_path = osp.join(root_path, f"{info_prefix}_infos_val.pkl")
        mmcv.dump(data, info_val_path)
        print(f"✅ TYJT验证集已保存: {len(val_infos)} 个样本")




def get_available_scenes(nusc):
    """复用原逻辑：获取所有存在数据的场景"""
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes



def _fill_trainval_infos(
    nusc,
    train_scenes,
    val_scenes,
    test=False,
    max_sweeps=10,
    max_radar_sweeps=0,
    is_sample_level=False
):
    """生成train/val的info数据（仿照nusc_converter实现）
    
    Args:
        nusc: NuScenes对象
        train_scenes: 训练场景集合
        val_scenes: 验证场景集合
        test: 是否为测试模式
        max_sweeps: 最大sweeps数
        max_radar_sweeps: 最大雷达sweeps数
        is_sample_level: 是否为样本级划分
    """
    train_nusc_infos = []
    val_nusc_infos = []
    token2idx = {}

    print(f"🔵 TYJT数据划分模式: {'样本级' if is_sample_level else '场景级'}")
    print(f"🔵 训练集大小: {len(train_scenes)}, 验证集大小: {len(val_scenes)}")

    # 获取类别映射
    from mmdet3d.datasets import TYJTDataset
    TYJT_CATEGORY_MAPPING = TYJTDataset.TYJT_CATEGORY_MAPPING

    for sample in mmcv.track_iter_progress(nusc.sample):
        try:
            # 1. 获取基础信息
            lidar_token = sample['data']['LIDAR_TOP']
            sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
            lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

            mmcv.check_file_exist(lidar_path)

            # 2. 创建基础info字典
            info = {
                'lidar_path': lidar_path,
                'token': sample['token'],
                'sweeps': [],
                'cams': dict(),
                'radars': dict(), 
                'lidar2ego_translation': cs_record['translation'],
                'lidar2ego_rotation': cs_record['rotation'],
                'ego2global_translation': pose_record['translation'],
                'ego2global_rotation': pose_record['rotation'],
                'timestamp': sample['timestamp'],
                'prev_token': sample['prev']
            }

            # 3. 计算转换矩阵
            l2e_r = info['lidar2ego_rotation']
            l2e_t = info['lidar2ego_translation']
            e2g_r = info['ego2global_rotation']
            e2g_t = info['ego2global_translation']
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            # 4. 相机信息
            camera_types = [
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
            ]
            
            for cam in camera_types:
                if cam not in sample['data']:
                    continue
                cam_token = sample['data'][cam]
                cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
                cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                           e2g_t, e2g_r_mat, cam)
                cam_info.update(cam_intrinsic=cam_intrinsic)
                info['cams'].update({cam: cam_info})

            # 5. LiDAR sweeps信息
            sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            sweeps = []
            while len(sweeps) < max_sweeps:
                if not sd_rec['prev'] == '':
                    sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                            l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                    sweeps.append(sweep)
                    sd_rec = nusc.get('sample_data', sd_rec['prev'])
                else:
                    break
            info['sweeps'] = sweeps

            # 6. 获取标注信息
            if not test:
                annotations = [
                    nusc.get('sample_annotation', token)
                    for token in sample['anns']
                ]

                locs = np.array([b.center for b in boxes]).reshape(-1, 3)
                dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
                rots = np.array([b.orientation.yaw_pitch_roll[0]
                                for b in boxes]).reshape(-1, 1)
                
                # 处理速度
                velocity = np.array([nusc.box_velocity(token)[:2] for token in sample['anns']])
                valid_flag = np.array([(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                                    for anno in annotations], dtype=bool).reshape(-1)
                
                # 速度转换
                for i in range(len(boxes)):
                    velo = np.array([*velocity[i], 0.0])
                    velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                    velocity[i] = velo[:2]

                # 应用TYJT类别映射（15→10）
                names = [b.name for b in boxes]
                mapped_names = []
                for name in names:
                    if name in TYJT_CATEGORY_MAPPING:
                        mapped_names.append(TYJT_CATEGORY_MAPPING[name])
                    else:
                        mapped_names.append('barrier')
                        if not hasattr(_fill_trainval_infos, '_warned_missing'):
                            print(f"⚠️ 警告: 类别 '{name}' 不在TYJT映射中，使用'barrier'代替")
                            _fill_trainval_infos._warned_missing = True
                
                names = np.array(mapped_names)
                
                # 转换box格式
                gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
                
                info['gt_boxes'] = gt_boxes
                info['gt_names'] = names
                info['gt_velocity'] = velocity.reshape(-1, 2)
                info['num_lidar_pts'] = np.array([a['num_lidar_pts'] for a in annotations])
                info['num_radar_pts'] = np.array([a['num_radar_pts'] for a in annotations])
                info['valid_flag'] = valid_flag

            # 7. 样本划分
            if is_sample_level:
                # 样本级划分：直接检查样本token
                if sample['token'] in train_scenes:
                    train_nusc_infos.append(info)
                    token2idx[info['token']] = ('train', len(train_nusc_infos) - 1)
                if sample['token'] in val_scenes:
                    val_nusc_infos.append(info)
                    token2idx[info['token']] = ('val', len(val_nusc_infos) - 1)
            else:
                # 场景级划分：检查样本所属场景
                if sample['scene_token'] in train_scenes:
                    train_nusc_infos.append(info)
                    token2idx[info['token']] = ('train', len(train_nusc_infos) - 1)
                elif sample['scene_token'] in val_scenes:
                    val_nusc_infos.append(info)
                    token2idx[info['token']] = ('val', len(val_nusc_infos) - 1)

        except Exception as e:
            print(f"❌ 处理样本 {sample['token']} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 8. 处理prev_token链接
    for info in train_nusc_infos:
        prev_token = info['prev_token']
        if prev_token == '':
            info['prev'] = -1
        else:
            if prev_token in token2idx:
                prev_set, prev_idx = token2idx[prev_token]
                if prev_set == 'train':
                    info['prev'] = prev_idx
                else:
                    info['prev'] = -1
            else:
                info['prev'] = -1

    for info in val_nusc_infos:
        prev_token = info['prev_token']
        if prev_token == '':
            info['prev'] = -1
        else:
            if prev_token in token2idx:
                prev_set, prev_idx = token2idx[prev_token]
                if prev_set == 'val':
                    info['prev'] = prev_idx
                else:
                    info['prev'] = -1
            else:
                info['prev'] = -1

    # 9. 最终统计信息
    if not test:
        train_categories = {}
        val_categories = {}
        
        for info in train_nusc_infos:
            for name in info.get('gt_names', []):
                train_categories[name] = train_categories.get(name, 0) + 1
                
        for info in val_nusc_infos:
            for name in info.get('gt_names', []):
                val_categories[name] = val_categories.get(name, 0) + 1
        
        print(f"🔵 训练集类别分布: {dict(sorted(train_categories.items()))}")
        print(f"🔵 验证集类别分布: {dict(sorted(val_categories.items()))}")

    print(f"🔵 最终结果: 训练样本 {len(train_nusc_infos)}, 验证样本 {len(val_nusc_infos)}")
    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type='lidar'):
    """复用原逻辑：获取传感器到LiDAR的转换矩阵"""
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:
        data_path = data_path.split(f'{os.getcwd()}/')[-1]
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T
    sweep['sensor2lidar_translation'] = T
    return sweep


# 以下函数（export_2d_annotation、get_2d_boxes等）复用原逻辑，无需修改
def export_2d_annotation(root_path, info_path, version, mono3d=True):
    camera_types = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]
    nusc_infos = mmcv.load(info_path)['infos']
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    cat2Ids = [dict(id=NUS_CATEGORIES.index(cat_name), name=cat_name)
               for cat_name in NUS_CATEGORIES]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(nusc_infos):
        for cam in camera_types:
            if cam not in info['cams']:
                continue
            cam_info = info['cams'][cam]
            coco_infos = get_2d_boxes(
                nusc, cam_info['sample_data_token'],
                visibilities=['', '1', '2', '3', '4'], mono3d=mono3d)
            (height, width, _) = mmcv.imread(cam_info['data_path']).shape
            coco_2d_dict['images'].append(
                dict(
                    file_name=cam_info['data_path'].split('data/nuscenes/')[-1],
                    id=cam_info['sample_data_token'],
                    token=info['token'],
                    cam2ego_rotation=cam_info['sensor2ego_rotation'],
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    ego2global_rotation=info['ego2global_rotation'],
                    ego2global_translation=info['ego2global_translation'],
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    width=width,
                    height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(nusc, sample_data_token: str, visibilities: List[str], mono3d=True):
    sd_rec = nusc.get('sample_data', sample_data_token)
    assert sd_rec['sensor_modality'] == 'camera', 'Error: get_2d_boxes only works for camera!'
    if not sd_rec['is_key_frame']:
        raise ValueError('The 2D re-projections are available only for keyframes.')
    s_rec = nusc.get('sample', sd_rec['sample_token'])
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    ann_recs = [nusc.get('sample_annotation', token) for token in s_rec['anns']]
    ann_recs = [ann_rec for ann_rec in ann_recs
                if (ann_rec['visibility_token'] in visibilities)]

    repro_recs = []
    for ann_rec in ann_recs:
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token
        box = nusc.get_box(ann_rec['token'])

        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()
        final_coords = post_process_coords(corner_coords)

        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token, sd_rec['filename'])

        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()
            dim = box.wlh.tolist()
            rot = [box.orientation.yaw_pitch_roll[0]]

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            if repro_rec['center2d'][2] <= 0:
                continue

            ann_token = nusc.get('sample_annotation', box.token)['attribute_tokens']
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = nusc.get('attribute', ann_token[0])['name']
            attr_id = nus_attributes.index(attr_name)
            repro_rec['attribute_name'] = attr_name
            repro_rec['attribute_id'] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(corner_coords: List, imsize: Tuple[int, int] = (1600, 900)):
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    repro_rec = OrderedDict()
    relevant_keys = [
        'attribute_tokens', 'category_name', 'instance_token', 'next',
        'num_lidar_pts', 'num_radar_pts', 'prev', 'sample_annotation_token',
        'sample_data_token', 'visibility_token'
    ]
    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec = dict()
    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in NUS_NAME_MAPPING:
        return None
    cat_name = NUS_NAME_MAPPING[repro_rec['category_name']]
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = NUS_CATEGORIES.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec


if __name__ == '__main__':
    # 测试tyjt转换（示例命令）
    create_nuscenes_infos(
        root_path='./data/tyjt',
        info_prefix='tyjt',
        version='v1.0-tyjt',
        dataset='tyjt',
        max_sweeps=10,
        max_radar_sweeps=0  # tyjt无雷达
    )