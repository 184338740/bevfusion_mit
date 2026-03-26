#!/usr/bin/env python3
"""
# v9.3
TYJT 数据集预处理脚本，生成 BEVFusion 所需的 info pkl 文件。
基于 tyjt_calib_utils.py 中的 CalibrationProcessor 进行标定解析。
"""

"""
# V02
# Local
python tyjt_converter_v9.2_0312.py --data-root /mnt/dataset/tyjt_RawData_all --train-split tyjt_data_infos/tyjt_train.txt --val-split tyjt_data_infos/tyjt_val.txt --out-dir ./tyjt_data_infos_v02 --max-sweeps 10


# A100
python tyjt_converter_v9.2_0312_A100.py --data-root /cephfsdata/users/lishan/00_Data/00_RawData --train-split tyjt_data_infos/tyjt_train.txt --val-split tyjt_data_infos/tyjt_val.txt --out-dir ./tyjt_data_infos_v02 --max-sweeps 10

python tyjt_converter_v9.2_0312_A100.py --data-root /cephfsdata/users/lishan/00_Data/00_RawData --train-split tyjt_data_infos/tyjt_train.txt --val-split tyjt_data_infos/tyjt_val.txt --out-dir ./tyjt_data_infos_v03 --max-sweeps 10
# tag
python tyjt_converter_v9.3_0312_A100.py --detect --info-pkl ./tyjt_data_infos_v02/tyjt_infos_train.pkl --detect-out ./detect_samples_Error_train --thresh-ratio 0.2 --thresh-abs 10


# V03
# create
# Local
python tyjt_converter_v9.3_0312_A100.py --data-root /mnt/dataset/tyjt_RawData_all --train-split tyjt_data_infos/tyjt_train.txt --val-split tyjt_data_infos/tyjt_val.txt --out-dir ./tyjt_data_infos_v03 --max-sweeps 10
# A100
python tyjt_converter_v9.3_0312_A100.py --data-root /cephfsdata/users/lishan/00_Data/00_RawData --train-split tyjt_data_infos/tyjt_train.txt --val-split tyjt_data_infos/tyjt_val.txt --out-dir ./tyjt_data_infos_v03 --max-sweeps 10

# tag-val
python tyjt_converter_v9.3_0312_A100.py \
    --detect \
    --info-pkl ./tyjt_data_infos_v03/tyjt_infos_val.pkl \
    --detect-out ./detect_samples_Error_val_v03 \
    --thresh-ratio 0.2 \
    --thresh-abs 10 \
    --x-range -60 60 \
    --y-range -60 60
# tag-train
python tyjt_converter_v9.3_0312_A100.py \
    --detect \
    --info-pkl ./tyjt_data_infos_v03/tyjt_infos_train.pkl \
    --detect-out ./detect_samples_Error_train_v03 \
    --thresh-ratio 0.2 \
    --thresh-abs 10 \
    --x-range -60 60 \
    --y-range -60 60

    
    
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
import time
from collections import defaultdict
import datetime

# 导入自定义标定工具
from tyjt_utils.tyjt_calib_utils import CalibrationProcessor

# 可视化
import matplotlib.pyplot as plt
import cv2
# 导入可视化工具
# 导入自定义可视化工具（需提前创建 tyjt_vis_utils.py）
from tyjt_utils.tyjt_vis_utils import (
    project_box_to_image,
    draw_projected_box,
    generate_bev_image,
    quaternion_to_rotation_matrix
)

BEV_POINT_CLOUD_DOWNSAMPLE = 30000  # BEV 图中最多显示的点数
BEV_POINT_SIZE = 0.5
BEV_POINT_ALPHA = 0.6

# ==================== 全局配置 ====================
# 数据包配置（可单独放在 config.json 中，这里为简便直接写为字典）

Mode = "A100_all"

if Mode == "Local":
    PACKAGE_CONFIG = {
        "2d3d_20250114": {
            "type": "hikvision",
            "calib_root": "calib/2d3d_20250114",
            "group2map_file": "group2map_calib.json",
            "camera2map_file": "camera2map_calib.json",
            "intersections": {
                "R9": {
                    "group_key": "G32050700009M00",
                    "cameras": [
                        ("R9_Aw_CamS", "A"),
                        ("R9_Bn_CamW", "B"),
                        ("R9_Ce_CamN", "C"),
                        ("R9_Ds_CamE", "D")
                    ]
                }
            }
        },
        "2d3d4d_20250728_weiyuan": {
            "type": "all_in_one",
            "calib_root": "calib/G51102400001M00",
            "sensor2map_file": "sensor2map_calib.json",
            "intersections": {
                "R01": {
                    "group_key": "group2map",
                    "cameras": [
                        ("SC_1A_CamR", "SC_1A_CamR_new", "A"),
                        ("SC_1B_CamR", "SC_1B_CamR_new", "B"),
                        ("SC_1C_CamR", "SC_1C_CamR_new", "C"),
                        ("SC_1D_CamR", "SC_1D_CamR_new", "D")
                    ]
                }
            }
        }
        # 可根据需要继续添加
    }
elif Mode == "A100_sub":
    PACKAGE_CONFIG = {
        "2d3d_20250114": {
            "type": "hikvision",
            "calib_root": "calib/2d3d_20250114",
            "group2map_file": "group2map_calib.json",
            "camera2map_file": "camera2map_calib.json",
            "intersections": {
                "R9": {
                    "group_key": "G32050700009M00",
                    "cameras": [
                        ("R9_Aw_CamS", "A"),
                        ("R9_Bn_CamW", "B"),
                        ("R9_Ce_CamN", "C"),
                        ("R9_Ds_CamE", "D")
                    ]
                }
            }
        },
        "2d3d4d_20250728_weiyuan": {
            "type": "all_in_one",
            "calib_root": "calib/G51102400001M00",
            "sensor2map_file": "sensor2map_calib.json",
            "intersections": {
                "R01": {
                    "group_key": "group2map",
                    "cameras": [
                        ("SC_1A_CamR", "SC_1A_CamR_new", "A"),
                        ("SC_1B_CamR", "SC_1B_CamR_new", "B"),
                        ("SC_1C_CamR", "SC_1C_CamR_new", "C"),
                        ("SC_1D_CamR", "SC_1D_CamR_new", "D")
                    ]
                }
            }
        }
    }
elif Mode == "A100_all":
    PACKAGE_CONFIG ={
        # 海康相机产品（9个）
       # 海康相机产品（9个）
        "2d3d_20250114": {
            "type": "hikvision",
            "calib_root": "calib/2d3d_20250114",
            "group2map_file": "group2map_calib.json",
            "camera2map_file": "camera2map_calib.json",
            "intersections": {
                "R9": {
                    "group_key": "G32050700009M00",
                    "cameras": [  # (相机文件夹名, 相机顺序A/B/C/D)
                        ("R9_Aw_CamS", "A"),
                        ("R9_Bn_CamW", "B"),
                        ("R9_Ce_CamN", "C"),
                        ("R9_Ds_CamE", "D")
                    ]
                }
            }
        },
        "2d3d_20250218": {
            "type": "hikvision",
            "calib_root": "calib/2d3d_20250218",
            "group2map_file": "group2map_calib.json",
            "camera2map_file": "camera2map_calib.json",
            "intersections": {
                "R03": {
                    "group_key": "G32050700003M00",
                    "cameras": [  # C相机缺失
                        ("R3_Aw_CamS", "A"),
                        ("R3_Bn_CamW", "B"),
                        ("R3_Ds_CamE", "D")
                    ]
                },
                "R04": {
                    "group_key": "G32050700004M00",
                    "cameras": [  # C相机缺失
                        ("R4_Aw_CamS", "A"),
                        ("R4_Bn_CamW", "B"),
                        ("R4_Ds_CamE", "D")
                    ]
                },
                "R10": {
                    "group_key": "G32050700010M00",
                    "cameras": [
                        ("R10_Aw_CamS", "A"),
                        ("R10_Bn_CamW", "B"),
                        ("R10_Ce_CamN", "C"),
                        ("R10_Ds_CamE", "D")
                    ]
                }
            }
        },
        "2d3d_20250221": {
            "type": "hikvision",
            "calib_root": "calib/2d3d_20250221",
            "group2map_file": "group2map_calib.json",
            "camera2map_file": "camera2map_calib.json",
            "intersections": {
                "R03": {
                    "group_key": "G32050700003M00",
                    "cameras": [  # C相机缺失
                        ("R3_Aw_CamS", "A"),
                        ("R3_Bn_CamW", "B"),
                        ("R3_Ds_CamE", "D")
                    ]
                },
                "R04": {
                    "group_key": "G32050700004M00",
                    "cameras": [  # C相机缺失
                        ("R4_Aw_CamS", "A"),
                        ("R4_Bn_CamW", "B"),
                        ("R4_Ds_CamE", "D")
                    ]
                },
                "R10": {
                    "group_key": "G32050700010M00",
                    "cameras": [
                        ("R10_Aw_CamS", "A"),
                        ("R10_Bn_CamW", "B"),
                        ("R10_Ce_CamN", "C"),
                        ("R10_Ds_CamE", "D")
                    ]
                }
            }
        },
        "2d3d4d_20241122_wuxi": {
            "type": "hikvision",
            "calib_root": "calib/2d3d4d_20241122",
            "group2map_file": "group2map_calib.json",
            "camera2map_file": "camera2map_calib.json",
            "intersections": {
                "R01": {"group_key": "G32020500001M00", "cameras": []}  # 无有效相机
            }
        },
        "2d3d4d_20241218": {
            "type": "hikvision",
            "calib_root": "calib/2d3d4d_20241218",
            "group2map_file": "group2map_calib.json",
            "camera2map_file": "camera2map_calib.json",
            "intersections": {
                "R02": {
                    "group_key": "G32050700002M00",
                    "cameras": [
                        ("R2_Aw_CamS", "A"),
                        ("R2_Bn_CamW", "B"),
                        ("R2_Ce_CamN", "C"),
                        ("R2_Ds_CamE", "D")
                    ]
                }
            }
        },
        "2d3d4d_20250117": {
            "type": "hikvision",
            "calib_root": "calib/2d3d4d_20250117",
            "group2map_file": "group2map_calib.json",
            "camera2map_file": "camera2map_calib.json",
            "intersections": {
                "R26": {
                    "group_key": "G32050700026M00",
                    "cameras": [
                        ("R26_Aw_CamS", "A"),
                        ("R26_Bn_CamW", "B"),
                        ("R26_Ce_CamN", "C"),
                        ("R26_Ds_CamE", "D")
                    ]
                }
            }
        },
        "2d3d4d_20250213": {
            "type": "hikvision",
            "calib_root": "calib/2d3d4d_20250213",
            "group2map_file": "group2map_calib.json",
            "camera2map_file": "camera2map_calib.json",
            "intersections": {
                "R26": {
                    "group_key": "G32050700026M00",
                    "cameras": [
                        ("R26_Aw_CamS", "A"),
                        ("R26_Bn_CamW", "B"),
                        ("R26_Ce_CamN", "C"),
                        ("R26_Ds_CamE", "D")
                    ]
                }
            }
        },
        "2d3d4d_20250218": {
            "type": "hikvision",
            "calib_root": "calib/2d3d4d_20250218",
            "group2map_file": "group2map_calib.json",
            "camera2map_file": "camera2map_calib.json",
            "intersections": {
                "R26": {
                    "group_key": "G32050700026M00",
                    "cameras": [
                        ("R26_Aw_CamS", "A"),
                        ("R26_Bn_CamW", "B"),
                        ("R26_Ce_CamN", "C"),
                        ("R26_Ds_CamE", "D")
                    ]
                }
            }
        },
        "2d3d4d_20250403": {
            "type": "hikvision",
            "calib_root": "calib/2d3d4d_20250403",
            "group2map_file": "group2map_calib.json",
            "camera2map_file": "camera2map_calib.json",
            "intersections": {
                "R26": {
                    "group_key": "G32050700026M00",
                    "cameras": [
                        ("R26_Aw_CamS", "A"),
                        ("R26_Bn_CamW", "B"),
                        ("R26_Ce_CamN", "C"),
                        ("R26_Ds_CamE", "D")
                    ]
                }
            }
        },
        "2d3d4d_20250408": {
            "type": "hikvision",
            "calib_root": "calib/2d3d4d_20250408",
            "group2map_file": "group2map_calib.json",
            "camera2map_file": "camera2map_calib.json",
            "intersections": {
                "R26": {
                    "group_key": "G32050700026M00",
                    "cameras": [
                        ("R26_Aw_CamS", "A"),
                        ("R26_Bn_CamW", "B"),
                        ("R26_Ce_CamN", "C"),
                        ("R26_Ds_CamE", "D")
                    ]
                }
            }
        },
        # 一体机产品（3个）
        "2d3d4d_20250618_weiyuan": {
            "type": "all_in_one",
            "calib_root": "info/G51102400002M00",
            "sensor2map_file": "sensor2map_calib.json",
            "intersections": {
                "R02": {
                    "group_key": "group2map",
                    "cameras": [  # (相机文件夹名, 标定键, 相机顺序A/B/C/D)
                        ("SC_2A_CamR", "SC_2A_CamR_new", "A"),
                        ("SC_2B_CamR", "SC_2B_CamR_new", "B"),
                        ("SC_2C_CamR", "SC_2C_CamR_new", "C"),
                        ("SC_2D_CamR", "SC_2D_CamR_new", "D")
                    ]
                }
            }
        },
        "2d3d4d_20250708_weiyuan": {
            "type": "all_in_one",
            "calib_root": "calib/G51102400002M00",
            "sensor2map_file": "sensor2map_calib.json",
            "intersections": {
                "R02": {
                    "group_key": "group2map",
                    "cameras": [
                        ("SC_2A_CamR", "SC_2A_CamR_new", "A"),
                        ("SC_2B_CamR", "SC_2B_CamR_new", "B"),
                        ("SC_2C_CamR", "SC_2C_CamR_new", "C"),
                        ("SC_2D_CamR", "SC_2D_CamR_new", "D")
                    ]
                }
            }
        },
        "2d3d4d_20250728_weiyuan": {
            "type": "all_in_one",
            "calib_root": "calib/G51102400001M00",
            "sensor2map_file": "sensor2map_calib.json",
            "intersections": {
                "R01": {
                    "group_key": "group2map",
                    "cameras": [
                        ("SC_1A_CamR", "SC_1A_CamR_new", "A"),
                        ("SC_1B_CamR", "SC_1B_CamR_new", "B"),
                        ("SC_1C_CamR", "SC_1C_CamR_new", "C"),
                        ("SC_1D_CamR", "SC_1D_CamR_new", "D")
                    ]
                }
            }
        }
    }
else:
    print(f"❌【Error】: TYJT_ROOT & DATA_PACKAGE_CONFIG: None")
    sys.exit(1)




# 相机顺序映射，用于统一命名
CAM_ORDER_TO_NAME = {
    "A": "CAM_A",
    "B": "CAM_B",
    "C": "CAM_C",
    "D": "CAM_D"
}

# ==================== 工具函数 ====================
def load_split(split_file: str) -> List[str]:
    """读取划分文件，每行一个数据包名（或子包路径），返回列表"""
    with open(split_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def load_calibration(package_path: Path, config: Dict) -> Dict:
    """加载数据包的标定文件，返回包含所有标定参数的字典"""
    calib_data = {}
    calib_root = package_path / config["calib_root"]
    if config["type"] == "hikvision":
        # 加载 group2map.json
        g2m_file = calib_root / config["group2map_file"]
        if g2m_file.exists():
            with open(g2m_file, 'r') as f:
                calib_data.update(json.load(f))
        # 加载 camera2map.json
        c2m_file = calib_root / config["camera2map_file"]
        if c2m_file.exists():
            with open(c2m_file, 'r') as f:
                calib_data.update(json.load(f))
    elif config["type"] == "all_in_one":
        # 加载 sensor2map.json
        s2m_file = calib_root / config["sensor2map_file"]
        if s2m_file.exists():
            with open(s2m_file, 'r') as f:
                calib_data = json.load(f)  # 直接覆盖，因为此文件包含所有标定
    return calib_data

def scan_package_files(package_path: Path, config: Dict) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """
    扫描数据包内所有子包，返回时间戳到文件路径的映射，以及时间戳到子包名的映射。
    返回格式: (file_map, sub_packet_map)
        file_map: { timestamp: { 'lidar': path, 'label': path, 'cam_A': path, ... } }
        sub_packet_map: { timestamp: sub_packet_name }
    """
    file_map = defaultdict(dict)
    sub_packet_map = {}  # 新增：记录时间戳所属的子包名
    datasets_dir = package_path / "datasets"
    if not datasets_dir.exists():
        return {}, {}

    # 遍历所有子包
    for sub_packet in datasets_dir.iterdir():
        if not sub_packet.is_dir():
            continue

        # --- 扫描 lidar 点云 ---
        lidar_dir = sub_packet / "lidar" / "pcd"
        if lidar_dir.exists():
            for npy_file in lidar_dir.glob("*.npy"):
                ts = npy_file.stem
                if ts.isdigit():
                    file_map[ts]['lidar'] = str(npy_file)
                    sub_packet_map[ts] = sub_packet.name  # 记录子包名

        # --- 扫描 label 文件 ---
        label_dir = sub_packet / "lidar" / "label"
        if label_dir.exists():
            for json_file in label_dir.glob("*.json"):
                ts = json_file.stem
                if ts.isdigit():
                    file_map[ts]['label'] = str(json_file)
                    # 如果之前已记录子包名，可不再重复，但为了安全也可更新
                    sub_packet_map.setdefault(ts, sub_packet.name)

        # --- 扫描相机图像 ---
        for road_id, road_cfg in config["intersections"].items():
            for cam_item in road_cfg["cameras"]:
                if config["type"] == "hikvision":
                    folder, cam_order = cam_item[0], cam_item[1]
                else:
                    folder, _, cam_order = cam_item
                cam_dir = sub_packet / folder / "image_dc"
                if cam_dir.exists():
                    for jpg_file in cam_dir.glob("*.jpg"):
                        ts = jpg_file.stem
                        if ts.isdigit():
                            file_map[ts][f'cam_{cam_order}'] = str(jpg_file)
                            sub_packet_map.setdefault(ts, sub_packet.name)

    # 过滤出至少包含 lidar 和 label 的样本，并同时过滤 sub_packet_map
    valid_file_map = {}
    valid_sub_packet_map = {}
    for ts, files in file_map.items():
        if 'lidar' in files and 'label' in files:
            cam_keys = [k for k in files if k.startswith('cam_')]
            if len(cam_keys) >= 2:
                valid_file_map[ts] = files
                valid_sub_packet_map[ts] = sub_packet_map.get(ts, 'unknown')  # 保底值
    return valid_file_map, valid_sub_packet_map

def compute_cam2ego(cam_calib: Dict, group2map_params: Any) -> Tuple[List[float], List[float]]:
    """
    计算相机到 ego 的变换。
    输入：
        cam_calib: 相机标定参数（原始格式）
        group2map_params: group2map 标定参数
    返回：
        (translation, rotation) 四元数 [w,x,y,z]
    """
    # 解析 cam2map 矩阵
    cam2map = CalibrationProcessor.parse_transform(cam_calib)
    # 解析 group2map 矩阵
    g2m = CalibrationProcessor.parse_transform(group2map_params)
    # 计算 map2group = inv(group2map)
    map2group = np.linalg.inv(g2m)
    # cam2ego = map2group @ cam2map
    cam2ego = map2group @ cam2map
    translation = cam2ego[:3, 3].tolist()
    rotation = CalibrationProcessor.rotation_matrix_to_quaternion(cam2ego[:3, :3])
    return translation, rotation



def build_sample_info(ts: str, files: Dict, calib_data: Dict,
                       config: Dict, road_id: str, package_name) -> Optional[Dict]:
    """
    为单个时间戳构建 info 字典。
    """
    # 获取道路配置
    road_cfg = config["intersections"][road_id]
    group_key = road_cfg["group_key"]
    group2map_params = calib_data.get(group_key)
    if group2map_params is None:
        print(f"警告：缺失 group_key {group_key}")
        return None

    # 计算 ego2global（即 group2map 矩阵），并确保为列表
    ego2global_matrix = CalibrationProcessor.parse_transform(group2map_params)
    if isinstance(ego2global_matrix, np.ndarray):
        ego2global_matrix = ego2global_matrix.tolist()

    # 处理相机
    cams_info = {}
    for cam_item in road_cfg["cameras"]:
        if config["type"] == "hikvision":
            folder, cam_order = cam_item[0], cam_item[1]
            calib_key = folder
        else:
            folder, calib_key, cam_order = cam_item[0], cam_item[1], cam_item[2]

        img_path = files.get(f'cam_{cam_order}')
        if not img_path:
            continue

        cam_calib = calib_data.get(calib_key)
        if cam_calib is None:
            print(f"警告：缺失相机标定 {calib_key}")
            continue

        translation, rotation = compute_cam2ego(cam_calib, group2map_params)

        if isinstance(cam_calib, dict):
            fx = cam_calib.get('fx', 1680.0)
            fy = cam_calib.get('fy', 1851.0)
            cx = cam_calib.get('cx', 960.0)
            cy = cam_calib.get('cy', 540.0)
        else:
            fx, fy, cx, cy = 1680.0, 1851.0, 960.0, 540.0
        intrinsic = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

        cam_name = CAM_ORDER_TO_NAME.get(cam_order, f"CAM_{cam_order}")
        cams_info[cam_name] = {
            "img_path": img_path,
            "cam2ego_translation": translation,
            "cam2ego_rotation": rotation,
            "cam_intrinsic": intrinsic
        }

    if not cams_info:
        return None

    # 构造相机映射（可选）
    cameras_mapping = []
    for cam_item in config['intersections'][road_id]['cameras']:
        if config['type'] == 'hikvision':
            folder, order = cam_item[0], cam_item[1]
            calib_key = folder
        else:
            folder, calib_key, order = cam_item[0], cam_item[1], cam_item[2]
        cameras_mapping.append({
            'folder': folder,
            'calib_key': calib_key,
            'order': order
        })

    # 人类可读时间
    try:
        dt = datetime.datetime.utcfromtimestamp(int(ts) / 1e9)
        human_time = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    except:
        human_time = str(ts)

    # ========== 新增：统计标注信息 ==========
    total_objects = 0
    obj_stats = {}
    label_path = files.get('label')
    if label_path and Path(label_path).exists():
        try:
            with open(label_path, 'r') as f:
                label_data = json.load(f)
            objects = label_data.get('objects', []) if isinstance(label_data, dict) else label_data
            total_objects = len(objects)
            for obj in objects:
                cat = obj.get('type', 'unknown')
                obj_stats[cat] = obj_stats.get(cat, 0) + 1
        except Exception as e:
            print(f"警告：无法解析标注文件 {label_path}: {e}")
    # =====================================

    info = {
        "lidar_path": files['lidar'],
        "label_path": files['label'],
        "timestamp": int(ts),
        "human_time": human_time,
        "cams": cams_info,
        "ego2global": ego2global_matrix,
        "lidar2ego": np.eye(4).tolist(),
        "package_name": package_name,
        "road_id": road_id,
        "group_key": config['intersections'][road_id]['group_key'],
        "calib_root": config['calib_root'],
        "cameras_mapping": cameras_mapping,
        "package_type": config['type'],
        # 新增字段
        "total_objects": total_objects,
        "obj_stats": obj_stats,
    }

    return info



def process_package(pkg_name, pkg_config, data_root, out_infos, max_sweeps=10):
    """
    处理单个数据包，生成带时序信息的 info，并追加到 out_infos 列表。
    参数：
        pkg_name: 数据包名
        pkg_config: 该数据包的配置字典
        data_root: 数据集根目录
        out_infos: 全局 info 列表（会被修改）
        max_sweeps: 最多收集的历史帧数
    """
    pkg_path = data_root / pkg_name
    if not pkg_path.exists():
        print(f"警告：数据包路径不存在 {pkg_path}")
        return

    print(f"正在处理数据包: {pkg_name}")
    calib_data = load_calibration(pkg_path, pkg_config)
    if not calib_data:
        print(f"  标定加载失败，跳过")
        return

    # 扫描文件，得到时间戳到文件路径的映射以及时间戳到子包名的映射
    file_map, sub_packet_map = scan_package_files(pkg_path, pkg_config)
    if not file_map:
        return

    # 按子包分组
    sub_packet_samples = defaultdict(list)
    for ts, files in file_map.items():
        sub_pkt = sub_packet_map.get(ts)
        if sub_pkt:
            sub_packet_samples[sub_pkt].append((int(ts), files))

    # 当前数据包只有一个路口，取第一个（如有多个路口需循环）
    road_id = next(iter(pkg_config["intersections"]))

    # 处理每个子包
    for sub_pkt, samples in sub_packet_samples.items():
        # 按时间戳排序
        samples.sort(key=lambda x: x[0])

        # 为子包内每个样本构建 info（暂存，还未设置时序索引）
        temp_infos = []
        for ts, files in samples:
            info = build_sample_info(str(ts), files, calib_data, pkg_config, road_id, pkg_name)
            if info:
                temp_infos.append(info)

        if not temp_infos:
            continue

        # 记录当前子包在全局列表中的起始索引
        start_idx = len(out_infos)

        # 先设置子包内的相对索引，并追加到全局列表
        for i, info in enumerate(temp_infos):
            info['prev'] = i - 1 if i > 0 else -1
            info['next'] = i + 1 if i < len(temp_infos) - 1 else -1
            out_infos.append(info)

        # 修正为全局索引
        for i, info in enumerate(temp_infos):
            if info['prev'] != -1:
                info['prev'] = start_idx + info['prev']
            if info['next'] != -1:
                info['next'] = start_idx + info['next']

        # 为该子包的每个样本生成 sweeps（历史帧信息）
        for i in range(len(temp_infos)):
            idx = start_idx + i
            info = out_infos[idx]
            sweeps = []
            prev_idx = info['prev']
            count = 0
            while prev_idx != -1 and count < max_sweeps:
                prev_info = out_infos[prev_idx]

                # 提取当前帧和历史帧的 ego2global 矩阵（需从列表转为 numpy 数组）
                ego2global_cur = np.array(info['ego2global'])
                ego2global_prev = np.array(prev_info['ego2global'])

                # 计算将历史点云变换到当前帧坐标系的矩阵： T_prev_to_cur = inv(ego2global_cur) @ ego2global_prev
                T_prev_to_cur = np.linalg.inv(ego2global_cur) @ ego2global_prev

                sweep = {
                    'data_path': prev_info['lidar_path'],
                    'timestamp': prev_info['timestamp'],
                    'human_time': prev_info['human_time'], 
                    'transform': T_prev_to_cur.tolist()   # 存储为列表，便于序列化
                }
                sweeps.append(sweep)

                # 继续向前追溯
                prev_idx = prev_info['prev']
                count += 1

            info['sweeps'] = sweeps


def get_base_name(info):
    lidar_path = Path(info['lidar_path'])
    sub_packet = lidar_path.parent.parent.parent.name
    timestamp = lidar_path.stem
    pkg_name = info.get('package_name', 'unknown_pkg')
    road_id = info.get('road_id', 'unknown_road')
    return f"{pkg_name}_{road_id}_{sub_packet}_{timestamp}"


def load_points(lidar_path):
    """加载点云，返回前三维坐标数组"""
    points = np.load(lidar_path).astype(np.float32)
    if points.shape[1] >= 3:
        return points[:, :3]
    else:
        return points

def detect_missing_labels(info_pkl, out_dir, thresh_ratio, thresh_abs,
                          x_range=None, y_range=None):
    """
    检测标注缺失的帧对（objects 数量突变），并生成可视化文件。
    现在还会输出类别统计信息。
    """
    import pickle
    import json
    import numpy as np
    import shutil
    from pathlib import Path
    import cv2
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # 加载 info 列表
    with open(info_pkl, 'rb') as f:
        infos = pickle.load(f)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    problem_pairs = []
    total_samples = len(infos)

    # 统计有效 prev 的数量
    valid_prev = sum(1 for info in infos if info.get('prev', -1) != -1)
    print(f"总样本数: {total_samples}, 具有有效 prev 的样本数: {valid_prev}")
    print(f"目标过滤范围: x∈{x_range}, y∈{y_range}")

    # 遍历所有样本
    for idx, info in enumerate(infos):
        if idx % 1000 == 0:
            print(f"已处理 {idx}/{total_samples} 个样本，当前发现 {len(problem_pairs)} 个异常帧对")

        prev_idx = info.get('prev', -1)
        if prev_idx == -1:
            continue
        prev_info = infos[prev_idx]

        # 使用预存的统计字段，若无则回退到解析标注文件（兼容旧版）
        prev_stats = prev_info.get('obj_stats', {})
        curr_stats = info.get('obj_stats', {})
        prev_num = prev_info.get('total_objects', sum(prev_stats.values()))
        curr_num = info.get('total_objects', sum(curr_stats.values()))

        # 应用范围过滤后重新计算数量（如果需要过滤）
        if x_range is not None and y_range is not None:
            # 注意：如果启用了范围过滤，不能直接使用预存统计，必须重新加载标注并过滤
            # 为了简化，这里假设我们不需要再次过滤（或预存统计已经是过滤后的）
            # 如果需要精确过滤，请保持原有读取标注文件的逻辑
            pass  # 可根据实际需求决定是否重新过滤

        # 检测变化（双向）
        if prev_num > 0 or curr_num > 0:
            max_num = max(prev_num, curr_num)
            diff = abs(prev_num - curr_num)
            ratio = diff / max_num if max_num > 0 else 0
            if ratio >= thresh_ratio and diff >= thresh_abs:
                direction = 'increase' if curr_num > prev_num else 'decrease'
                # 保存完整的 objects 列表和统计信息
                problem_pairs.append((prev_idx, idx, prev_num, curr_num, direction,
                                      prev_info, info))  # 直接保存整个 info，以便获取所有字段

    print(f"发现 {len(problem_pairs)} 个异常帧对，开始生成可视化文件...")

    # 处理每个异常帧对
    for i, (prev_idx, curr_idx, prev_num, curr_num, direction,
            prev_info, curr_info) in enumerate(problem_pairs):
        pair_dir = out_root / f"pair_{i:04d}_prev{prev_idx}_curr{curr_idx}"
        pair_dir.mkdir(exist_ok=True)

        prev_base = get_base_name(prev_info)
        curr_base = get_base_name(curr_info)

        # 为前一帧和当前帧生成可视化文件
        for role, info, base in [('prev', prev_info, prev_base), ('curr', curr_info, curr_base)]:
            role_dir = pair_dir / role
            role_dir.mkdir(exist_ok=True)

            # 加载点云（用于 BEV 图）
            points = load_points(info['lidar_path'])

            # 获取该帧的所有 objects（原始标注数据）
            try:
                with open(info['label_path'], 'r') as f:
                    label_data = json.load(f)
                objs_all = label_data.get('objects', []) if isinstance(label_data, dict) else label_data
            except Exception as e:
                print(f"警告：无法读取标注文件 {info['label_path']}，跳过可视化")
                objs_all = []

            # 为每个相机生成带框图像（并绘制目标 ID）
            for cam in ['CAM_A', 'CAM_B', 'CAM_C', 'CAM_D']:
                if cam in info['cams']:
                    src_img = info['cams'][cam]['img_path']
                    if Path(src_img).exists():
                        img = cv2.imread(src_img)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # 投影每个框
                        for obj in objs_all:
                            box = {
                                'center': obj['box3d'][:3],
                                'size': obj['box3d'][3:6],
                                'yaw': obj['rotation'][0],
                                'type': obj.get('type', 'unknown')
                            }
                            uv_box, valid_box = project_box_to_image(box, info['cams'][cam])
                            img = draw_projected_box(img, uv_box, valid_box, color=(255,0,0), thickness=2)

                            # 绘制目标 ID
                            center_ego = np.array([box['center']])
                            t_cam2ego = np.array(info['cams'][cam]['cam2ego_translation'], dtype=np.float32)
                            q = np.array(info['cams'][cam]['cam2ego_rotation'], dtype=np.float32)
                            R_cam2ego = quaternion_to_rotation_matrix(q)
                            R_ego2cam = R_cam2ego.T
                            t_ego2cam = -R_ego2cam @ t_cam2ego
                            center_cam = (R_ego2cam @ center_ego.T).T + t_ego2cam
                            z = center_cam[0, 2]
                            if z > 0:
                                intrinsic = np.array(info['cams'][cam]['cam_intrinsic'], dtype=np.float32)
                                u = intrinsic[0, 0] * center_cam[0, 0] / z + intrinsic[0, 2]
                                v = intrinsic[1, 1] * center_cam[0, 1] / z + intrinsic[1, 2]
                                h_img, w_img = img.shape[:2]
                                if 0 <= u < w_img and 0 <= v < h_img:
                                    cv2.putText(img, str(obj.get('id', -1)), (int(u), int(v)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        out_img = role_dir / f"{base}_{cam}.jpg"
                        cv2.imwrite(str(out_img), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # 生成 BEV 图（带目标 ID）
            boxes_bev = []
            ids_list = []
            for obj in objs_all:
                boxes_bev.append({
                    'center': obj['box3d'][:3],
                    'size': obj['box3d'][3:6],
                    'yaw': obj['rotation'][0]
                })
                ids_list.append(obj.get('id', -1))
            generate_bev_image(points, boxes_bev, base, role_dir,
                               x_range=(-100,100), y_range=(-100,100),
                               point_size=0.5, alpha=0.6, ids=ids_list)

        # 保存 info.txt，增加类别统计
        prev_stats = prev_info.get('obj_stats', {})
        curr_stats = curr_info.get('obj_stats', {})

        with open(pair_dir / "info.txt", 'w') as f:
            f.write(f"异常类型: {direction}\n")
            f.write(f"prev_idx: {prev_idx}, curr_idx: {curr_idx}\n")
            f.write(f"prev_objects: {prev_num}, curr_objects: {curr_num}\n")
            f.write(f"变化比例: {abs(prev_num-curr_num)/max(prev_num, curr_num) if max(prev_num, curr_num)>0 else 0:.2f}, 变化绝对值: {abs(prev_num-curr_num)}\n\n")

            # 输出类别统计对比
            f.write("【类别统计对比】\n")
            all_cats = set(prev_stats.keys()) | set(curr_stats.keys())
            for cat in sorted(all_cats):
                prev_cnt = prev_stats.get(cat, 0)
                curr_cnt = curr_stats.get(cat, 0)
                if prev_cnt != curr_cnt:
                    f.write(f"  {cat}: {prev_cnt} → {curr_cnt} (变化 {curr_cnt - prev_cnt:+d})\n")
                else:
                    f.write(f"  {cat}: {prev_cnt} (不变)\n")
            f.write("\n")

            f.write("【前一帧】\n")
            f.write(f"  base_name: {prev_base}\n")
            f.write(f"  human_time: {prev_info.get('human_time', 'N/A')}\n")
            f.write(f"  label_path: {prev_info['label_path']}\n")
            for cam in ['CAM_A', 'CAM_B', 'CAM_C', 'CAM_D']:
                f.write(f"  {cam}_path: {prev_info['cams'].get(cam, {}).get('img_path', 'N/A')}\n")
            f.write("\n【当前帧】\n")
            f.write(f"  base_name: {curr_base}\n")
            f.write(f"  human_time: {curr_info.get('human_time', 'N/A')}\n")
            f.write(f"  label_path: {curr_info['label_path']}\n")
            for cam in ['CAM_A', 'CAM_B', 'CAM_C', 'CAM_D']:
                f.write(f"  {cam}_path: {curr_info['cams'].get(cam, {}).get('img_path', 'N/A')}\n")

    return problem_pairs

def is_obj_in_range(obj, x_range, y_range):
    """判断目标中心点是否在指定范围内"""
    box3d = obj.get('box3d')
    if not box3d or len(box3d) < 3:
        return False
    cx, cy = box3d[0], box3d[1]
    return (x_range[0] <= cx <= x_range[1]) and (y_range[0] <= cy <= y_range[1])


def main():
    parser = argparse.ArgumentParser(description="TYJT 数据集 info 生成工具")
    parser.add_argument("--data-root", type=str, default="/mnt/dataset/tyjt_RawData_all",
                        help="TYJT 数据集根目录")
    parser.add_argument("--train-split", type=str, default="tyjt_data_infos/tyjt_train.txt",
                        help="训练集划分文件")
    parser.add_argument("--val-split", type=str, default="tyjt_data_infos/tyjt_val.txt",
                        help="验证集划分文件")
    parser.add_argument("--out-dir", type=str, default="./tyjt_data_infos_v02",
                        help="输出 pkl 文件的目录")
    parser.add_argument('--max-sweeps', type=int, default=10,
                    help='Number of sweeps (previous frames) to include for each sample')

    # ==================== 额外功能：数据质量检测（前后帧是否有object总数的突变） ====================
    parser.add_argument('--detect', action='store_true',
                        help='启用标注缺失检测模式')
    parser.add_argument('--info-pkl', type=str,
                        help='待检测的 info pkl 文件路径（启用 detect 时必须提供）')
    parser.add_argument('--detect-out', type=str, default='./detect_results',
                        help='异常帧对输出根目录')
    parser.add_argument('--thresh-ratio', type=float, default=0.5,
                        help='objects 数量减少比例阈值，默认0.5（减少50%）')
    parser.add_argument('--thresh-abs', type=int, default=3,
                        help='objects 数量减少绝对值阈值，默认3')
    parser.add_argument('--x-range', nargs=2, type=float, default=[-60, 60],
                        help='X range for filtering objects (min max)')
    parser.add_argument('--y-range', nargs=2, type=float, default=[-60, 60],
                        help='Y range for filtering objects (min max)')

    args = parser.parse_args()


    if args.detect:
        # 检测模式
        if not args.info_pkl:
            print("错误：检测模式必须提供 --info-pkl")
            return
        detect_missing_labels(args.info_pkl, args.detect_out, args.thresh_ratio, args.thresh_abs, x_range=args.x_range, y_range=args.y_range,)
        return

    # 创建输出目录
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载划分
    train_pkgs = load_split(args.train_split)
    val_pkgs = load_split(args.val_split)
    print(f"训练集数据包数: {len(train_pkgs)}，验证集数据包数: {len(val_pkgs)}")

    data_root = Path(args.data_root)

    # 分别收集训练和验证 info
    train_infos = []
    val_infos = []

    start_time = time.time()

    # 处理训练集
    for pkg_name in train_pkgs:
        if pkg_name not in PACKAGE_CONFIG:
            print(f">>>[xmy]🟣[tools/tyjt_converter_v9.1_0311.py]>>> Warining: 跳过未配置的数据包: {pkg_name}")
            continue
        process_package(pkg_name, PACKAGE_CONFIG[pkg_name], data_root, train_infos, max_sweeps=args.max_sweeps)

    # 处理验证集
    for pkg_name in val_pkgs:
        if pkg_name not in PACKAGE_CONFIG:
            print(f"跳过未配置的数据包: {pkg_name}")
            continue
        process_package(pkg_name, PACKAGE_CONFIG[pkg_name], data_root, val_infos, max_sweeps=args.max_sweeps)

    elapsed = time.time() - start_time
    print(f"处理完成，耗时 {elapsed:.2f} 秒")
    print(f"训练集样本数: {len(train_infos)}")
    print(f"验证集样本数: {len(val_infos)}")

    # 保存 pkl
    train_pkl = out_dir / "tyjt_infos_train.pkl"
    val_pkl = out_dir / "tyjt_infos_val.pkl"
    with open(train_pkl, 'wb') as f:
        pickle.dump(train_infos, f)
    with open(val_pkl, 'wb') as f:
        pickle.dump(val_infos, f)

    # 同步保存 JSON（用于调试和人工查看）
    train_json = out_dir / "tyjt_infos_train.json"
    val_json = out_dir / "tyjt_infos_val.json"
    with open(train_json, 'w', encoding='utf-8') as f:
        json.dump(train_infos, f, indent=2, ensure_ascii=False)
    with open(val_json, 'w', encoding='utf-8') as f:
        json.dump(val_infos, f, indent=2, ensure_ascii=False)

    print(f"训练集 info 已保存至 {train_pkl} 和 {train_json}")
    print(f"验证集 info 已保存至 {val_pkl} 和 {val_json}")

    def extract_simple_info(infos):
        """提取每个样本的关键时序字段"""
        simple_list = []
        for info in infos:
            simple = {
                'timestamp': info['timestamp'],
                'human_time': info.get('human_time', ''),
                'prev': info.get('prev', -1),
                'next': info.get('next', -1),
                'sweeps_timestamps': [s['timestamp'] for s in info.get('sweeps', [])],
                'sweeps_human_time': [s['human_time'] for s in info.get('sweeps', [])],
            }
            simple_list.append(simple)
        return simple_list

    # 生成简化版样本文件
    train_simple = extract_simple_info(train_infos)
    val_simple = extract_simple_info(val_infos)

    train_simple_path = out_dir / "sample_train.json"
    val_simple_path = out_dir / "sample_val.json"
    with open(train_simple_path, 'w', encoding='utf-8') as f:
        json.dump(train_simple, f, indent=2, ensure_ascii=False)
    with open(val_simple_path, 'w', encoding='utf-8') as f:
        json.dump(val_simple, f, indent=2, ensure_ascii=False)

    print(f"简化版样本信息已保存至 {train_simple_path} 和 {val_simple_path}")


if __name__ == "__main__":
    main()