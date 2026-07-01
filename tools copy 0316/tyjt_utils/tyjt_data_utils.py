import numpy as np
from pathlib import Path

def get_base_name(info):
    lidar_path = Path(info['lidar_path'])
    sub_packet = lidar_path.parent.parent.parent.name
    timestamp = lidar_path.stem
    pkg_name = info.get('package_name', 'unknown_pkg')
    road_id = info.get('road_id', 'unknown_road')
    return f"{pkg_name}_{road_id}_{sub_packet}_{timestamp}"

def load_points(lidar_path):
    points = np.load(lidar_path).astype(np.float32)
    if points.shape[1] >= 3:
        return points[:, :3]
    else:
        return points

def is_obj_in_range(obj, x_range, y_range):
    box3d = obj.get('box3d')
    if not box3d or len(box3d) < 3:
        return False
    cx, cy = box3d[0], box3d[1]
    return (x_range[0] <= cx <= x_range[1]) and (y_range[0] <= cy <= y_range[1])