#!/usr/bin/env python3
"""
TYJT 数据集预处理脚本，生成 BEVFusion 所需的 info pkl 文件。
基于 tyjt_calib_utils.py 中的 CalibrationProcessor 进行标定解析

版本 v9.4.5 (0402)
    - 更新 tyjt场景包的解析配置: 相机映射、路径
    - 升级 打印: 每个包处理时，打印样本数目
    - 升级 多线程调用: def create_tyjt_gt_database() ==> def create_groundtruth_database()【注意：目前能运行，但训练会有bug】

版本 v9.4.4: (0327)
    - 新增命令行选项，允许用户选择性生成 info 文件和 GT 数据库，避免重复处理数据包：
        * --generate-info：生成 tyjt_infos_train.pkl 和 tyjt_infos_val.pkl（默认不生成）
        * --generate-database：生成 tyjt_dbinfos_train.pkl 和 tyjt_gt_database/（默认不生成）
    - 新增 --version 参数，用于指定数据集版本号；若未提供，自动生成格式为 "noVersion_YYYY_MM_DD_HH_MM" 的时间戳版本。
    - 优化执行逻辑：当跳过 info 生成时，不再执行耗时的数据包扫描和处理，大幅提升脚本执行效率。
    - 保留原有功能，向后兼容（未使用新参数时，脚本行为与 v9.4.3 相同，即同时生成 info 和数据库）。

版本 v9.4.3:
    - 关联修改:
        - 新文件: mmdet3d/datasets/tyjt_dataset_v2.py: 构建 class TYJTDatasetV2 类
        - 文件修改: mmdet3d/datasets/__init__.py 引入 class TYJTDatasetV2
        - 文件修改: tools/data_converter/create_gt_database.py 升级v8.3。 添加新分支: 支持 TYJTDatasetV2 类数据的 create_gt_database 生成
    - 升级: 基于 TYJTDatasetV2 生成  用于CopyPaste 的 GT 数据库(单目标的点云)
        - 引用 class TYJTDatasetV2
        - 新函数: def create_tyjt_gt_database
        - 生成: tyjt_dbinfos_train.pkl 和 tyjt_gt_database文件夹
    - 升级: tyjt_infos_{train/val}.pkl 中添加字段Vx\Vy=0; 类似于nusc的速度值, 适配训练接口
    - 输出：
        tyjt_infos_train.pkl
        tyjt_infos_val.pkl
        tyjt_dbinfos_train.pkl
        tyjt_gt_database/ 文件夹
版本 v9.4.2: 
    - 输出格式与 NuScenesDataset 完全兼容。(meta\infos)
    - 生成:
        tyjt_infos_train.pkl
        tyjt_infos_val.pkl

版本 v9.4.1:  
    - 移除 检测detect异常不连续tag的相关功能,  仅保留纯粹的数据转换info.pkl。
    - 生成:
        tyjt_infos_train.pkl
        tyjt_infos_val.pkl
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

# ==================== 全局配置 ====================
DatasetInfos = "A100_all"

if DatasetInfos == "Local":
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
elif DatasetInfos == "A100_sub":
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
elif DatasetInfos == "A100_all":
    PACKAGE_CONFIG = {
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
                    "cameras": [  # C相机缺失，T形路口
                        ("R3_Aw_CamS", "A"),
                        ("R3_Bn_CamW", "B"),
                        ("R3_Ds_CamE", "D")
                    ]
                },
                "R04": {
                    "group_key": "G32050700004M00",
                    "cameras": [  # C相机缺失，T形路口
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
                    "cameras": [  # C相机缺失，T形路口
                        ("R3_Aw_CamS", "A"),
                        ("R3_Bn_CamW", "B"),
                        ("R3_Ds_CamE", "D")
                    ]
                },
                "R04": {
                    "group_key": "G32050700004M00",
                    "cameras": [  # C相机缺失,T形路口
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
                "R01": {
                    "group_key": "G32020500001M00", 
                    "cameras": [
                        ("R2_Ae_CamS", "A"),
                        ("R2_Bs_CamW", "B"),
                        ("R2_Cw_CamN", "C"),
                        ("R2_Dn_CamE", "D")
                    ]}
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
                "R26": {  # 发现图像没有做去畸变, 没有image_dc数据,导致为空
                    "group_key": "G32050700026M00",
                    # "cameras": [
                    #     ("R26_Aw_CamS", "A"),
                    #     ("R26_Bn_CamW", "B"),
                    #     ("R26_Ce_CamN", "C"),
                    #     ("R26_Ds_CamE", "D")
                    # ]
                    "cameras": [
                        ("26A_CamR", "A"),
                        ("26B_CamR", "B"),
                        ("26C_CamR", "C"),
                        ("26D_CamR", "D")
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
elif DatasetInfos == "A100_V031_sub":
    PACKAGE_CONFIG = {
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
        }
    }
else:
    print(f"❌【Error】: PACKAGE_CONFIG not defined; DatasetVersion = {DatasetVersion}")
    import sys
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
        g2m_file = calib_root / config["group2map_file"]
        if g2m_file.exists():
            with open(g2m_file, 'r') as f:
                calib_data.update(json.load(f))
        c2m_file = calib_root / config["camera2map_file"]
        if c2m_file.exists():
            with open(c2m_file, 'r') as f:
                calib_data.update(json.load(f))
    elif config["type"] == "all_in_one":
        s2m_file = calib_root / config["sensor2map_file"]
        if s2m_file.exists():
            with open(s2m_file, 'r') as f:
                calib_data = json.load(f)
    return calib_data

def scan_package_files(package_path: Path, config: Dict) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """
    扫描数据包内所有子包，返回时间戳到文件路径的映射，以及时间戳到子包名的映射。
    返回格式: (file_map, sub_packet_map)
        file_map: { timestamp: { 'lidar': path, 'label': path, 'cam_A': path, ... } }
        sub_packet_map: { timestamp: sub_packet_name }
    """
    file_map = defaultdict(dict)
    sub_packet_map = {}
    datasets_dir = package_path / "datasets"
    if not datasets_dir.exists():
        return {}, {}

    for sub_packet in datasets_dir.iterdir():
        if not sub_packet.is_dir():
            continue

        lidar_dir = sub_packet / "lidar" / "pcd"
        if lidar_dir.exists():
            for npy_file in lidar_dir.glob("*.npy"):
                ts = npy_file.stem
                if ts.isdigit():
                    file_map[ts]['lidar'] = str(npy_file)
                    sub_packet_map[ts] = sub_packet.name

        label_dir = sub_packet / "lidar" / "label"
        if label_dir.exists():
            for json_file in label_dir.glob("*.json"):
                ts = json_file.stem
                if ts.isdigit():
                    file_map[ts]['label'] = str(json_file)
                    sub_packet_map.setdefault(ts, sub_packet.name)

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

    # 过滤有效样本
    valid_file_map = {}
    valid_sub_packet_map = {}
    for ts, files in file_map.items():
        if 'lidar' in files and 'label' in files:
            cam_keys = [k for k in files if k.startswith('cam_')]
            if len(cam_keys) >= 2:
                valid_file_map[ts] = files
                valid_sub_packet_map[ts] = sub_packet_map.get(ts, 'unknown')
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
    cam2map = CalibrationProcessor.parse_transform(cam_calib)
    g2m = CalibrationProcessor.parse_transform(group2map_params)
    map2group = np.linalg.inv(g2m)
    cam2ego = map2group @ cam2map
    translation = cam2ego[:3, 3].tolist()
    rotation = CalibrationProcessor.rotation_matrix_to_quaternion(cam2ego[:3, :3])
    return translation, rotation


def build_sample_info(ts: str, files: Dict, calib_data: Dict,
                       config: Dict, road_id: str, package_name) -> Optional[Dict]:
    """
    为单个时间戳构建符合 NuScenesDataset 格式的 info 字典。
    采用通用逻辑：lidar 和 ego 可能不同，通过 lidar2ego 矩阵计算 sensor2lidar。
    若有效相机数少于2个，则跳过该样本。
    """
    road_cfg = config["intersections"][road_id]
    group_key = road_cfg["group_key"]
    group2map_params = calib_data.get(group_key)
    if group2map_params is None:
        print(f"警告：缺失 group_key {group_key}")
        return None

    # 计算 ego2global 矩阵，并分解为旋转和平移
    ego2global_matrix = CalibrationProcessor.parse_transform(group2map_params)
    if isinstance(ego2global_matrix, np.ndarray):
        ego2global_matrix = ego2global_matrix.tolist()
    ego2global_matrix_np = np.array(ego2global_matrix)
    ego2global_rotation = CalibrationProcessor.rotation_matrix_to_quaternion(ego2global_matrix_np[:3, :3])
    ego2global_translation = ego2global_matrix_np[:3, 3].tolist()

    # lidar2ego 矩阵（目前为单位矩阵，若未来 lidar 与 ego 不重合，可从此处修改）
    lidar2ego_mat = np.eye(4, dtype=np.float32)          # 从 lidar 到 ego 的变换
    lidar2ego_rotation = [1.0, 0.0, 0.0, 0.0]            # 对应四元数 [w,x,y,z]
    lidar2ego_translation = [0.0, 0.0, 0.0]

    # 生成唯一 token
    sub_pkt = Path(files['lidar']).parent.parent.parent.name
    token = f"{package_name}_{road_id}_{sub_pkt}_{ts}"

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

        # ---------- 新增：检查相机内参完整性 ----------
        # 必须包含 fx, fy, cx, cy，否则跳过该相机
        if not all(k in cam_calib for k in ['fx', 'fy', 'cx', 'cy']):
            print(f"警告：相机 {calib_key} 内参缺失，跳过该相机。样本 {token}")
            continue
        # -----------------------------------------

        # 计算 cam2ego (sensor2ego)
        # translation：从相机坐标系到 ego 坐标系的平移向量（3 个元素）。
        # rotation：从相机坐标系到 ego 坐标系的旋转四元数（4 个元素，格式为 [w, x, y, z]）
        translation, rotation = compute_cam2ego(cam_calib, group2map_params)

        # 构建 sensor2ego 矩阵 (齐次变换矩阵)
        sensor2ego_mat = np.eye(4)
        sensor2ego_mat[:3, :3] = CalibrationProcessor.quaternion_to_rotation_matrix(rotation)
        sensor2ego_mat[:3, 3] = translation

        # 计算 sensor2lidar = inv(lidar2ego) @ sensor2ego
        sensor2lidar_mat = np.linalg.inv(lidar2ego_mat) @ sensor2ego_mat
        sensor2lidar_rotation = CalibrationProcessor.rotation_matrix_to_quaternion(sensor2lidar_mat[:3, :3])
        sensor2lidar_translation = sensor2lidar_mat[:3, 3].tolist()

        # 内参（此时已确保存在）
        fx = cam_calib['fx']
        fy = cam_calib['fy']
        cx = cam_calib['cx']
        cy = cam_calib['cy']
        intrinsic = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

        cam_name = CAM_ORDER_TO_NAME.get(cam_order, f"CAM_{cam_order}")
        cams_info[cam_name] = {
            "data_path": img_path,
            "sensor2ego_rotation": rotation,                      # 从相机到 ego 的旋转
            "sensor2ego_translation": translation,                # 从相机到 ego 的平移
            "sensor2lidar_rotation": sensor2lidar_rotation,      # 从相机到 lidar 的旋转
            "sensor2lidar_translation": sensor2lidar_translation, # 从相机到 lidar 的平移
            "cam_intrinsic": intrinsic
        }

    # ---------- 新增：检查有效相机数量 ----------
    if len(cams_info) < 2:
        print(f"警告：样本 {token} 有效相机数不足2个（仅 {len(cams_info)}），跳过该样本")
        return None
    # -----------------------------------------

    # 读取标注，生成 gt_boxes, gt_names, num_lidar_pts, valid_flag
    gt_boxes = []
    gt_names = []
    num_lidar_pts = []
    gt_velocity = []   # v9.4.3: 新增速度列表
    valid_flag = []
    label_path = files.get('label')
    if label_path and Path(label_path).exists():
        try:
            with open(label_path, 'r') as f:
                label_data = json.load(f)
            objects = label_data.get('objects', []) if isinstance(label_data, dict) else label_data
            for obj in objects:
                box3d = obj['box3d']  # [cx, cy, cz, l, w, h]
                yaw = obj['rotation'][0]
                gt_boxes.append(box3d + [yaw])  # 7维
                gt_names.append(obj['type'])
                num_lidar_pts.append(obj.get('pointNum', -1))
                valid_flag.append(True)
                gt_velocity.append([0.0, 0.0])   # v9.4.3: 速度设为0

        except Exception as e:
            print(f"警告：无法解析标注文件 {label_path}: {e}")

    if not gt_boxes:
        gt_boxes = np.zeros((0, 7), dtype=np.float32)
        gt_names = np.array([], dtype=str)
        num_lidar_pts = np.zeros((0,), dtype=np.int32)
        gt_velocity = np.zeros((0, 2), dtype=np.float32)   #  v9.4.3 空速度数组
        valid_flag = np.zeros((0,), dtype=bool)

    else:
        gt_boxes = np.array(gt_boxes, dtype=np.float32)
        gt_names = np.array(gt_names, dtype=str)
        num_lidar_pts = np.array(num_lidar_pts, dtype=np.int32)
        gt_velocity = np.array(gt_velocity, dtype=np.float32)   #  v9.4.3 (N, 2)
        valid_flag = np.array(valid_flag, dtype=bool)

    # 人类可读时间
    try:
        dt = datetime.datetime.utcfromtimestamp(int(ts) / 1e9)
        human_time = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    except:
        human_time = str(ts)

    # 构造相机映射（可选，用于调试）
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

    # 统计类别
    obj_stats = {}
    for name in gt_names:
        obj_stats[name] = obj_stats.get(name, 0) + 1

    info = {
        "token": token,
        "lidar_path": files['lidar'],
        "label_path": files['label'],
        "timestamp": int(ts),
        "human_time": human_time,
        "cams": cams_info,
        "ego2global": ego2global_matrix,                 # 保留原始矩阵（4x4列表）
        "ego2global_rotation": ego2global_rotation,
        "ego2global_translation": ego2global_translation,
        "lidar2ego_rotation": lidar2ego_rotation,
        "lidar2ego_translation": lidar2ego_translation,
        "gt_velocity": gt_velocity,   # v9.4.3 新增
        "gt_boxes": gt_boxes,
        "gt_names": gt_names,
        "num_lidar_pts": num_lidar_pts,
        "valid_flag": valid_flag,
        "package_name": package_name,
        "road_id": road_id,
        "group_key": config['intersections'][road_id]['group_key'],
        "calib_root": config['calib_root'],
        "cameras_mapping": cameras_mapping,
        "package_type": config['type'],
        "total_objects": len(gt_boxes),
        "obj_stats": obj_stats,
    }

    return info


def process_package(pkg_name, pkg_config, data_root, out_infos, max_sweeps=10):
    """
    处理单个数据包，生成带时序信息的 info，并追加到 out_infos 列表
    返回生成的样本数量
    """
    pkg_path = data_root / pkg_name
    if not pkg_path.exists():
        print(f"警告：数据包路径不存在 {pkg_path}")
        return 0

    print(f"正在处理数据包: {pkg_name}")
    calib_data = load_calibration(pkg_path, pkg_config)
    if not calib_data:
        print(f"  标定加载失败，跳过")
        return 0

    file_map, sub_packet_map = scan_package_files(pkg_path, pkg_config)
    if not file_map:
        print(f"  未找到任何有效样本（无点云/标注/相机配对）")
        return 0
    
    # 记录处理前的样本数
    before_count = len(out_infos)

    sub_packet_samples = defaultdict(list)
    for ts, files in file_map.items():
        sub_pkt = sub_packet_map.get(ts)
        if sub_pkt:
            sub_packet_samples[sub_pkt].append((int(ts), files))

    road_id = next(iter(pkg_config["intersections"]))

    for sub_pkt, samples in sub_packet_samples.items():
        samples.sort(key=lambda x: x[0])
        temp_infos = []
        for ts, files in samples:
            info = build_sample_info(str(ts), files, calib_data, pkg_config, road_id, pkg_name)
            if info:
                temp_infos.append(info)

        if not temp_infos:
            continue

        start_idx = len(out_infos)
        for i, info in enumerate(temp_infos):
            info['prev'] = i - 1 if i > 0 else -1
            info['next'] = i + 1 if i < len(temp_infos) - 1 else -1
            out_infos.append(info)

        for i, info in enumerate(temp_infos):
            if info['prev'] != -1:
                info['prev'] = start_idx + info['prev']
            if info['next'] != -1:
                info['next'] = start_idx + info['next']

        for i in range(len(temp_infos)):
            idx = start_idx + i
            info = out_infos[idx]
            sweeps = []
            prev_idx = info['prev']
            count = 0
            while prev_idx != -1 and count < max_sweeps:
                prev_info = out_infos[prev_idx]

                ego2global_cur = np.array(info['ego2global'])
                ego2global_prev = np.array(prev_info['ego2global'])
                T_prev_to_cur = np.linalg.inv(ego2global_cur) @ ego2global_prev

                # 分解变换矩阵为旋转矩阵和平移向量
                R = T_prev_to_cur[:3, :3]
                t = T_prev_to_cur[:3, 3]

                # 将旋转矩阵转换为四元数（[w,x,y,z]）
                # quat = CalibrationProcessor.rotation_matrix_to_quaternion(R)

                sweep = {
                    'data_path': prev_info['lidar_path'],
                    'timestamp': prev_info['timestamp'],
                    'human_time': prev_info['human_time'],
                    'sensor2lidar_rotation': R,                 # 直接存储旋转矩阵（numpy 数组）
                    'sensor2lidar_translation': t,              # 平移向量
                    'transform': T_prev_to_cur.tolist()
                }
                sweeps.append(sweep)
                prev_idx = prev_info['prev']
                count += 1

            info['sweeps'] = sweeps
    after_count = len(out_infos)
    sample_count = after_count - before_count
    print(f"  数据包 {pkg_name} 生成 {sample_count} 个样本")
    return sample_count    

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

class NumpyEncoder(json.JSONEncoder):
    """用于将 numpy 数组转换为列表的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def create_tyjt_gt_database(info_pkl, out_dir, data_root, workers=1):
    """基于 TYJTDatasetV2 生成 GT 数据库"""
    from tools.data_converter.create_gt_database import create_groundtruth_database

    # 注意：create_groundtruth_database 内部会调用 build_dataset，要求数据集类已注册。
    # 确保 TYJTDatasetV2 已在 mmdet3d/datasets/__init__.py 中注册。
    create_groundtruth_database(
        dataset_class_name='TYJTDatasetV2',   # 数据集类名
        data_path=data_root,                 # 数据集根目录
        info_prefix='tyjt',                  # 前缀
        info_path=info_pkl,                  # info pkl 路径
        database_save_path=str(Path(out_dir) / 'tyjt_gt_database'),
        db_info_save_path=str(Path(out_dir) / 'tyjt_dbinfos_train.pkl'),
        relative_path=True,                  # 使用相对路径
        # 如果不需要加载增强数据，可以设置 load_augmented=None
        load_augmented=None,
        workers=workers   # 添加多线程
    )

def main():
    parser = argparse.ArgumentParser(description="TYJT 数据集 info 生成工具 v9.4.4 (NuScenes格式)")
    parser.add_argument("--data-root", type=str, default="/mnt/dataset/tyjt_RawData_all",
                        help="TYJT 数据集根目录")
    parser.add_argument("--train-split", type=str, default="../datasets/tyjt2pkl/Local_V031/tyjt_train.txt",
                        help="训练集划分文件")
    parser.add_argument("--val-split", type=str, default="../datasets/tyjt2pkl/Local_V031/tyjt_val.txt",
                        help="验证集划分文件")
    parser.add_argument("--out-dir", type=str, default="../datasets/tyjt2pkl/Local_V031/",
                        help="输出 pkl 文件的目录")
    parser.add_argument('--max-sweeps', type=int, default=10,
                        help='Number of sweeps (previous frames) to include for each sample')
    # 新增参数（字符串类型，支持 true/false）
    parser.add_argument('--generate-info', type=str, default='true',
                        help='Generate tyjt_infos_train.pkl and tyjt_infos_val.pkl (true/false, default: true)')
    parser.add_argument('--generate-database', type=str, default='true',
                        help='Generate GT database (tyjt_dbinfos_train.pkl and tyjt_gt_database/) (true/false, default: true)')
    parser.add_argument('--version', type=str, default=None,
                        help='Dataset version string (default: auto-generated timestamp)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of worker processes for GT database generation (default: 1)')
    args = parser.parse_args()

    # 解析布尔值
    generate_info = args.generate_info.lower() in ('true', '1', 'yes')
    generate_database = args.generate_database.lower() in ('true', '1', 'yes')

    # 生成版本号（如果未提供）
    if args.version is None:
        version_str = datetime.datetime.now().strftime('noVersion_%Y_%m_%d_%H_%M')
    else:
        version_str = args.version

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_pkgs = load_split(args.train_split)
    val_pkgs = load_split(args.val_split)
    print(f"训练集数据包数: {len(train_pkgs)}，验证集数据包数: {len(val_pkgs)}")

    data_root = Path(args.data_root)

    # 根据 --generate-info 决定是否处理数据包并生成 info 文件
    if generate_info:
        print(" ========== 开始生成 info ... ========== ")
        train_infos = []
        val_infos = []
        start_time = time.time()
        total_train_samples = 0
        total_val_samples = 0
        for pkg_name in train_pkgs:
            if pkg_name not in PACKAGE_CONFIG:
                print(f"警告：跳过未配置的数据包: {pkg_name}")
                continue
            cnt = process_package(pkg_name, PACKAGE_CONFIG[pkg_name], data_root, train_infos, max_sweeps=args.max_sweeps)
            total_train_samples += cnt
        for pkg_name in val_pkgs:
            if pkg_name not in PACKAGE_CONFIG:
                print(f"警告：跳过未配置的数据包: {pkg_name}")
                continue
            cnt = process_package(pkg_name, PACKAGE_CONFIG[pkg_name], data_root, val_infos, max_sweeps=args.max_sweeps)
            total_val_samples += cnt

        elapsed = time.time() - start_time
        print(f"处理完成，耗时 {elapsed:.2f} 秒")
        print(f"训练集总样本数: {len(train_infos)} (各包累计: {total_train_samples})")
        print(f"验证集总样本数: {len(val_infos)} (各包累计: {total_val_samples})")

        # 构建带 metadata 的完整结构
        metadata = {
            "version": version_str,
            "description": "TYJT dataset converted for BEVFusion",
            "date_created": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        train_data = {"infos": train_infos, "metadata": metadata}
        val_data = {"infos": val_infos, "metadata": metadata}

        # 保存 pkl
        train_pkl = out_dir / "tyjt_infos_train.pkl"
        val_pkl = out_dir / "tyjt_infos_val.pkl"
        with open(train_pkl, 'wb') as f:
            pickle.dump(train_data, f)
        with open(val_pkl, 'wb') as f:
            pickle.dump(val_data, f)

        # 可选保存完整 JSON（用于调试），需处理 numpy 数组
        train_json = out_dir / "tyjt_infos_train.json"
        val_json = out_dir / "tyjt_infos_val.json"
        with open(train_json, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
        with open(val_json, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)

        print(f"训练集 info 已保存至 {train_pkl} 和 {train_json}")
        print(f"验证集 info 已保存至 {val_pkl} 和 {val_json}")

        # 生成简化版样本文件（不包含 numpy 数组）
        train_simple = extract_simple_info(train_infos)
        val_simple = extract_simple_info(val_infos)

        train_simple_path = out_dir / "sample_train.json"
        val_simple_path = out_dir / "sample_val.json"
        with open(train_simple_path, 'w', encoding='utf-8') as f:
            json.dump(train_simple, f, indent=2, ensure_ascii=False)
        with open(val_simple_path, 'w', encoding='utf-8') as f:
            json.dump(val_simple, f, indent=2, ensure_ascii=False)

        print(f"简化版样本信息已保存至 {train_simple_path} 和 {val_simple_path}")
        print(" ========== info 生成完成。========== ")
    else:
        print(" ========== 跳过生成 info 文件 (--generate-info false) ==========")
        # 定义路径变量供后续数据库生成使用（即使跳过 info，也需要知道路径）
        train_pkl = out_dir / "tyjt_infos_train.pkl"
        val_pkl = out_dir / "tyjt_infos_val.pkl"

    # 根据 --generate-database 决定是否生成 GT 数据库
    if generate_database:
        if not train_pkl.exists():
            print(f"错误：info 文件 {train_pkl} 不存在，无法生成数据库。请先使用 --generate-info true 生成 info 文件。")
        else:
            print(" ========== 开始生成 GT 数据库... ========== ")
            create_tyjt_gt_database(str(train_pkl), out_dir, str(data_root), workers=args.workers)
            print(" ========== GT 数据库生成完成。========== ")
    else:
        print("跳过生成 GT 数据库 (--generate-database false)")


if __name__ == "__main__":
    main()