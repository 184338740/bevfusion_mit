# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.

from typing import List, Optional


def category_to_detection_name(category_name: str) -> Optional[str]:
    """
    Default label mapping from nuScenes to nuScenes detection classes.
    Note that pedestrian does not include personal_mobility, stroller and wheelchair.
    :param category_name: Generic nuScenes class.
    :return: nuScenes detection class.
    """
    # 🔵[xmy-修复]>>> 添加TYJT数据集类别映射
    tyjt_category_mapping = {
        # 标准类别映射
        'car': 'car',
        'truck': 'truck', 
        'trailer': 'trailer',
        'bus': 'bus',
        'construction_vehicle': 'construction_vehicle',
        'pedestrian': 'pedestrian',
        'motorcycle': 'motorcycle',
        'bicycle': 'bicycle',
        'traffic_cone': 'traffic_cone',
        'barrier': 'barrier',
        
        # 添加TYJT特有的类别映射（带tyjt.前缀）
        'tyjt.car': 'car',
        'tyjt.van': 'car',
        'tyjt.truck': 'truck',
        'tyjt.construction_truck': 'construction_vehicle',
        'tyjt.bus': 'bus',
        'tyjt.robot': 'construction_vehicle',
        'tyjt.pedestrian': 'pedestrian',
        'tyjt.cyclist': 'pedestrian',
        'tyjt.tricyclist': 'motorcycle',
        'tyjt.bicycle': 'bicycle',
        'tyjt.tricycle': 'motorcycle',
        'tyjt.trolley': 'trailer',
        'tyjt.cone': 'traffic_cone',
        'tyjt.barrier': 'barrier',
        'tyjt.other': 'barrier',
        
        # 原有的其他映射
        'vehicle.car': 'car',
        'vehicle.truck': 'truck',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.van': 'car',
        'vehicle.bus': 'bus',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.cyclist': 'motorcycle',
        'vehicle.bicycle': 'bicycle',
        'vehicle.tricycle': 'motorcycle',
        'movable_object.trafficcone': 'traffic_cone',
        'movable_object.barrier': 'barrier'
    }
    
    # 首先检查TYJT映射
    if category_name in tyjt_category_mapping:
        mapped_name = tyjt_category_mapping[category_name]
        print(f'🔵[TYJT映射]>>> {category_name} -> {mapped_name}')  # 调试信息
        return mapped_name

    detection_mapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }

    if category_name in detection_mapping:
        return detection_mapping[category_name]
    else:
        print(f'🔵[TYJT警告]>>> 未知类别: {category_name}')  # 调试信息
        return None


def detection_name_to_rel_attributes(detection_name: str) -> List[str]:
    """
    Returns a list of relevant attributes for a given detection class.
    :param detection_name: The detection class.
    :return: List of relevant attributes.
    """
    if detection_name in ['pedestrian']:
        rel_attributes = ['pedestrian.moving', 'pedestrian.sitting_lying_down', 'pedestrian.standing']
    elif detection_name in ['bicycle', 'motorcycle']:
        rel_attributes = ['cycle.with_rider', 'cycle.without_rider']
    elif detection_name in ['car', 'bus', 'construction_vehicle', 'trailer', 'truck']:
        rel_attributes = ['vehicle.moving', 'vehicle.parked', 'vehicle.stopped']
    elif detection_name in ['barrier', 'traffic_cone']:
        # Classes without attributes: barrier, traffic_cone.
        rel_attributes = []
    else:
        raise ValueError('Error: %s is not a valid detection class.' % detection_name)

    return rel_attributes

