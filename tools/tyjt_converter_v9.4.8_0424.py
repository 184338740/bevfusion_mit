#!/usr/bin/env python3
"""
TYJT 数据集预处理脚本，生成 BEVFusion 所需的 info pkl 文件。
基于 tyjt_calib_utils.py 中的 CalibrationProcessor 进行标定解析

- 版本 v9.4.8 (0424)
    - 重要修复：修正 TYJT 数据转换中 gt_boxes 尺寸顺序错误。
    - 原顺序 [cx, cy, cz, l, w, h] (长、宽、高) 与 nuScenes 官方格式 [cx, cy, cz, w, l, h] (宽、长、高) 不一致，导致模型学习到颠倒的尺寸预测。
    - 现已更正转换逻辑，在 build_sample_info 中将 box3d 的 [l, w, h] 交换为 [w, l, h]，输出 gt_boxes = [cx, cy, cz, w, l, h, yaw]。
    - 影响：修正后需重新生成数据集（.pkl 文件）并重新训练/微调模型，否则 KITTI 评估仍为 0。
    - 同步更新 tyjt_converter.py 及数据层相关注释，确保与 MMDetection3D / nuScenes 标准严格对齐。

版本 v9.4.7 (0421)
    - 标注目录自动探测：支持 ["label","labels","perc"] 候选列表，解决 2d3d4d_20241122_wuxi 使用 perc 的问题
    - 图像目录可配置：支持数据包级 image_subdir（默认 image_dc），适配非标准图像目录
    - 图像格式扩展：支持 .png 图像，解决 2d3d4d_20250618_weiyuan 图像为 png 的问题
    - 修复标注解析中 pointNum 字段为 None 导致的 TypeError
    - 智能路口匹配：自动识别子包所属路口，消除无关警告
    - 修正 process_package 只处理第一个路口的缺陷，确保十字路口样本被正确生成
    - 添加详细分层树状统计，清晰展示每级过滤规则及数量
    - 增加 ENABLE_LIDAR_COUNT 开关（默认 False），控制总点云数统计（耗时操作）

测试结果（正确配置下）：
    - 2d3d_20250218: 生成 7916 个四相机样本
    - 2d3d_20250221: 生成 5725 个四相机样本
    - 2d3d4d_20250117: 生成 7090 个样本（验证集）
    - 2d3d4d_20241122_wuxi、2d3d4d_20250618_weiyuan 正常生成样本
    - 2d3d4d_20250403 因相机目录与标定键名不匹配，需手动创建软链接（见下方说明）

对于 2d3d4d_20250403 数据包的预处理建议：
    由于该数据包的相机文件夹名（26A_CAMR 等）与标定文件中的键名（R26_Aw_CamS 等）不一致，且无法修改标定文件或目录结构，可通过创建软链接解决：
    ```bash
    cd /cephfsdata/users/lishan/00_Data/00_RawData/2d3d4d_20250403/datasets
    for sub in */; do
        cd "$sub"
        ln -sf ../26A_CAMR/image_dc R26_Aw_CamS/image_dc
        ln -sf ../26B_CAMR/image_dc R26_Bn_CamW/image_dc
        ln -sf ../26C_CAMR/image_dc R26_Ce_CamN/image_dc
        ln -sf ../26D_CAMR/image_dc R26_Ds_CamE/image_dc
        cd ..
    done

版本 v9.4.6 (0417)
    - 严格过滤：仅保留同时包含 CAM_A、CAM_B、CAM_C、CAM_D 四路相机的样本
    - 实现方式：在 build_sample_info() 中检查 cams_info 的键集合是否完全等于 {'CAM_A','CAM_B','CAM_C','CAM_D'}
    - 影响：T 型路口（3 相机）和多相机路口（≥5 相机）的样本将被丢弃，确保训练数据统一为四相机格式
    - 理由：BEVFusion 训练 pipeline 固定需要 A/B/C/D 四路相机

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

Debug = False
VERBOSE = False   # 设为 True 时打印详细相机状态
ENABLE_LIDAR_FILE_COUNT = True   # 耗时统计开关

print(f">>>[xmy]🔵[TYJTDatasetV2]🔵[tools/tyjt_converter_A100.py] >>> Debug Mode = {Debug}")

from collections import defaultdict
skip_detail = defaultdict(lambda: defaultdict(int))
skip_stats = defaultdict(lambda: defaultdict(int))   # skip_stats[package_name][reason] += 1
early_filter_detail = defaultdict(lambda: defaultdict(int))

# ==================== 全局配置 ====================
# label_subdir： 标注文件路径为 <packet>/datasets/<sub_packet>/lidar/<label_subdir>/, 默认为 label, 允许缺省，如果不存在则跳过，给出warning
# image_subdir： 去畸变后图像的路径为 <packet>/datasets/<sub_packet>/<camera_folder:ABCD等文件夹>/<image_subdir>/, 默认为 image_dc, 允许缺省，如果不存在则跳过，给出warning

# LABEL_DIR_CANDIDATES 标注文件的目录候选（按优先级排序，标准优先），如果不用自动化标签，就把"perc"删除
# LABEL_DIR_CANDIDATES = ["label", "labels", "perc"]
LABEL_DIR_CANDIDATES = ["label", "labels"]

print(f">>>[xmy]🔵[TYJTDatasetV2]🔵[tools/tyjt_converter_A100.py] >>> config: LABEL_DIR_CANDIDATES = {LABEL_DIR_CANDIDATES}")

DatasetInfos = "A100_all"
if DatasetInfos == "A100_all":
    PACKAGE_CONFIG = {
        # 海康相机产品（9个）
        "2d3d_20250114": {
            "type": "hikvision",
            "calib_root": "calib/2d3d_20250114",
            "group2map_file": "group2map_calib.json",
            "camera2map_file": "camera2map_calib.json",
            "label_subdir": "label",       # 可选，默认 "label"
            "image_subdir": "image_dc",    # 可选，默认 "image_dc"
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
            "label_subdir": "label",       # 可选，默认 "label"
            "image_subdir": "image_dc",    # 可选，默认 "image_dc"
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
            "label_subdir": "label",       # 可选，默认 "label"
            "image_subdir": "image_dc",    # 可选，默认 "image_dc"
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
            "label_subdir": "perc",       # 可选，默认 "label"
            "image_subdir": "image_dc",    # 可选，默认 "image_dc"
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
            "label_subdir": "label",       # 可选，默认 "label"
            "image_subdir": "image_dc",    # 可选，默认 "image_dc"
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
            "label_subdir": "label",       # 可选，默认 "label"
            "image_subdir": "image_dc",    # 可选，默认 "image_dc"
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
            "label_subdir": "label",       # 可选，默认 "label"
            "image_subdir": "image_dc",    # 可选，默认 "image_dc"
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
            "label_subdir": "label",       # 可选，默认 "label"
            "image_subdir": "image_dc",    # 可选，默认 "image_dc"
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
            "label_subdir": "label",       # 可选，默认 "label"
            "image_subdir": "image_dc",    # 可选，默认 "image_dc"
            "intersections": {
                "R26": {  # 发现图像没有做去畸变, 没有image_dc数据,导致为空
                    "group_key": "G32050700026M00",
                    "cameras": [
                        ("R26_Aw_CamS", "A"),
                        ("R26_Bn_CamW", "B"),
                        ("R26_Ce_CamN", "C"),
                        ("R26_Ds_CamE", "D")
                    ]
                    # "cameras": [
                    #     ("26A_CAMR", "A"),
                    #     ("26B_CAMR", "B"),
                    #     ("26C_CAMR", "C"),
                    #     ("26D_CAMR", "D")
                    # ]
                }
            }
        },
        "2d3d4d_20250408": {
            "type": "hikvision",
            "calib_root": "calib/2d3d4d_20250408",
            "group2map_file": "group2map_calib.json",
            "camera2map_file": "camera2map_calib.json",
            "label_subdir": "label",       # 可选，默认 "label"
            "image_subdir": "image_dc",    # 可选，默认 "image_dc"
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
            "calib_root": "info/G51102400002M00",   # xmy注意: 标定文件路径是info， 不是calib
            "sensor2map_file": "sensor2map_calib.json",
            "label_subdir": "label",       # 可选，默认 "label"
            "image_subdir": "image_dc",    # 可选，默认 "image_dc"
            "intersections": {
                "R02": {
                    "group_key": "group2map",
                    "cameras": [  # (相机文件夹名, 标定键, 相机顺序A/B/C/D)
                        ("SC_2A_CamR", "SC_2A_CamR_new", "A"),   # xmy注意: 该图像格式是.png 
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
            "label_subdir": "label",       # 可选，默认 "label"
            "image_subdir": "image_dc",    # 可选，默认 "image_dc"
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
            "label_subdir": "label",       # 可选，默认 "label"
            "image_subdir": "image_dc",    # 可选，默认 "image_dc"
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
    1. 扫描数据包内所有子包，返回时间戳到文件路径的映射，以及时间戳到子包名的映射。
    2. 自动识别每个子包所属的路口，只扫描匹配路口的相机。
    3. 自动探测标注目录，支持多种图像格式。
    返回格式: (file_map, sub_packet_map)
        file_map: { timestamp: { 'lidar': path, 'label': path, 'cam_A': path, ... } }
        sub_packet_map: { timestamp: sub_packet_name }
    """
    # 获取子目录配置（提供默认值）
    image_subdir = config.get("image_subdir", "image_dc")  # 去畸变图像文件夹：默认为 image_dc

    # 标注目录候选列表（全局，按优先级）
    label_candidates = LABEL_DIR_CANDIDATES
    # 去重并保持顺序（避免重复尝试相同目录）
    seen = set()
    label_candidates = [c for c in label_candidates if not (c in seen or seen.add(c))]

    file_map = defaultdict(dict)
    sub_packet_map = {}
    datasets_dir = package_path / "datasets"
    if not datasets_dir.exists():
        return {}, {}
    
    # 预构建每个路口需要的相机文件夹集合（用于匹配）
    road_cam_folders = {}
    for road_id, road_cfg in config["intersections"].items():
        folders = set()
        for cam_item in road_cfg["cameras"]:
            if config["type"] == "hikvision":
                folder, _ = cam_item[0], cam_item[1]
            else:
                folder, _, _ = cam_item
            folders.add(folder)
        road_cam_folders[road_id] = folders

    for sub_packet in datasets_dir.iterdir():
        if not sub_packet.is_dir():
            continue

        # 获取子包下存在的所有文件夹名
        existing_folders = {entry.name for entry in sub_packet.iterdir() if entry.is_dir()}
        
        # 匹配路口：找到第一个满足其所需相机文件夹全部存在的路口
        matched_road = None
        for road_id, required_folders in road_cam_folders.items():
            if required_folders.issubset(existing_folders):
                matched_road = road_id
                break
        if matched_road is None:
            if Debug:
                print(f"  调试：子包 {sub_packet.name} 未匹配任何路口配置，跳过")
            continue
        road_cfg = config["intersections"][matched_road]

        # 扫描点云
        lidar_dir = sub_packet / "lidar" / "pcd"
        if lidar_dir.exists():
            for npy_file in lidar_dir.glob("*.npy"):
                ts = npy_file.stem
                if ts.isdigit():
                    file_map[ts]['lidar'] = str(npy_file)
                    file_map[ts]['road_id'] = matched_road
                    sub_packet_map[ts] = sub_packet.name

        # 扫描标注：尝试候选目录列表
        label_path = None
        for cand in label_candidates:
            test_dir = sub_packet / "lidar" / cand
            if test_dir.exists():
                label_path = test_dir
                if VERBOSE and cand not in ["label", "labels"]:
                    print(f">>>[xmy]🔵[TYJTDatasetV2]🔵[tyjt_converter_A100.py] >>>  调试：子包 {sub_packet.name} 使用标注目录 '{cand}'（非默认）")
                break
        if label_path:
            for json_file in label_path.glob("*.json"):
                ts = json_file.stem
                if ts.isdigit():
                    file_map[ts]['label'] = str(json_file)
                    file_map[ts]['road_id'] = matched_road
                    sub_packet_map.setdefault(ts, sub_packet.name)
        else:
            print(f">>>[xmy]🔵[TYJTDatasetV2]🔵[tyjt_converter_A100.py] >>>  警告：未找到标注目录（尝试过 {label_candidates}; {label_path}")


        # 扫描相机（使用配置的 image_subdir）（仅扫描匹配路口的相机）
        for cam_item in road_cfg["cameras"]:
            if config["type"] == "hikvision":
                folder, cam_order = cam_item[0], cam_item[1]
            else:
                folder, _, cam_order = cam_item
            cam_dir = sub_packet / folder / image_subdir
            if cam_dir.exists():
                for img_file in cam_dir.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        ts = img_file.stem
                        if ts.isdigit():
                            file_map[ts][f'cam_{cam_order}'] = str(img_file)
                            file_map[ts]['road_id'] = matched_road
                            sub_packet_map.setdefault(ts, sub_packet.name)
            else:
                print(f">>>[xmy]🔵[TYJTDatasetV2]🔵[tyjt_converter_A100.py] >>>   警告：未找到相机目录 {cam_dir}")

    # ========== 新增：统计早期过滤原因 ==========
    early_stats = defaultdict(int)
    for ts, files in file_map.items():
        has_lidar = 'lidar' in files
        has_label = 'label' in files
        cam_keys = [k for k in files if k.startswith('cam_')]
        cam_count = len(cam_keys)
        
        if not has_lidar:
            early_stats['missing_lidar'] += 1
        elif not has_label:
            early_stats['missing_label'] += 1
        elif cam_count < 2:
            early_stats[f'cam_count_{cam_count}'] += 1
        else:
            early_stats['valid'] += 1

    # 将非 valid 的原因记录到全局字典（以数据包名称为键）
    pkg_key = package_path.name
    for reason, cnt in early_stats.items():
        if reason != 'valid':
            early_filter_detail[pkg_key][reason] += cnt
    # ============================================

    # 过滤有效样本（必须同时有点云、标注、至少2个相机）
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
    # 记录失败原因（用于详细统计）
    skip_reason = None

    road_cfg = config["intersections"][road_id]
    group_key = road_cfg["group_key"]
    group2map_params = calib_data.get(group_key)
    if group2map_params is None:
        print(f"警告：缺失 group_key {group_key}")
        skip_reason = f"missing_group_key_{group_key}"
        skip_detail[package_name][skip_reason] += 1
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
    # if Debug:
    #     print(f"\n>>>[xmy]🔵[TYJTDatasetV2]🔵[tyjt_dataset_v2.py] >>> [DEBUG] 处理样本 {package_name}_{road_id}_{ts}")
    #     print(f">>>[xmy]🔵[TYJTDatasetV2]🔵[tyjt_dataset_v2.py] >>> [DEBUG] 可用相机文件: {[k for k in files.keys() if k.startswith('cam_')]}")

    for cam_item in road_cfg["cameras"]:
        if config["type"] == "hikvision":
            folder, cam_order = cam_item[0], cam_item[1]
            calib_key = folder
        else:
            folder, calib_key, cam_order = cam_item[0], cam_item[1], cam_item[2]

        img_path = files.get(f'cam_{cam_order}')
        if not img_path:
            if VERBOSE:
                print(f"\n>>>[xmy]🔵[TYJTDatasetV2]🔵[tyjt_dataset_v2.py] >>> [DEBUG] 相机 {cam_order} 跳过：无图像路径")
            skip_reason = f"cam_{cam_order}_no_image"
            skip_detail[package_name][skip_reason] += 1
            return None

        cam_calib = calib_data.get(calib_key)
        if cam_calib is None:
            if VERBOSE:
                print(f"\n>>>[xmy]🔵[TYJTDatasetV2]🔵[tyjt_dataset_v2.py] >>> [DEBUG] 相机 {cam_order} 跳过：标定缺失")
            skip_reason = f"cam_{cam_order}_no_calib"
            skip_detail[package_name][skip_reason] += 1
            return None

        # ---------- 新增：检查相机内参完整性 ----------
        # 必须包含 fx, fy, cx, cy，否则跳过该相机
        if not all(k in cam_calib for k in ['fx', 'fy', 'cx', 'cy']):
            if VERBOSE:
                print(f"\n>>>[xmy]🔵[TYJTDatasetV2]🔵[tyjt_dataset_v2.py] >>> [DEBUG] 相机 {cam_order} 内参缺失: {cam_calib.keys()}")
            skip_reason = f"cam_{cam_order}_no_intrinsic"
            skip_detail[package_name][skip_reason] += 1
            return None
        # -----------------------------------------

        # 计算 cam2ego (sensor2ego)
        # translation：从相机坐标系到 ego 坐标系的平移向量（3 个元素）。
        # rotation：从相机坐标系到 ego 坐标系的旋转四元数（4 个元素，格式为 [w, x, y, z]）
        try:
            translation, rotation = compute_cam2ego(cam_calib, group2map_params)
        except Exception as e:
            if VERBOSE:
                print(f"\n>>>[xmy]🔵[TYJTDatasetV2]🔵[tyjt_dataset_v2.py] >>> [DEBUG] 相机 {cam_order} 变换计算失败: {e}")
            skip_reason = f"cam_{cam_order}_transform_error"
            skip_detail[package_name][skip_reason] += 1
            return None

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

    # ---------- 检查有效相机数量 ----------
    required_cams = {'CAM_A', 'CAM_B', 'CAM_C', 'CAM_D'}
    actual_cams = set(cams_info.keys())
    if actual_cams != required_cams:
        missing = required_cams - actual_cams
        extra = actual_cams - required_cams
        
        # 始终打印一行简要信息（日志量可控）
        if VERBOSE: print(f"⚠️ 跳过样本 {token}，缺少相机: {sorted(missing) if missing else '无'}, 多余: {sorted(extra) if extra else '无'}")
        
        # 统计跳过原因（兼容原有 skip_stats）
        if missing:
            reason = f"missing_{'_'.join(sorted(missing))}"
        else:
            reason = "extra_cams"
        skip_stats[package_name][reason] += 1
        # 同时记录到详细统计
        skip_detail[package_name][f"final_{reason}"] += 1
        
        # 可选：如果需要深度排查，临时将下面的 VERBOSE 设为 True
        # VERBOSE = False   # 调试时改为 True 可打印每个相机的详细状态
        if VERBOSE:
            print(f"   详细相机状态:")
            for cam_item in road_cfg["cameras"]:
                if config["type"] == "hikvision":
                    folder, cam_order = cam_item[0], cam_item[1]
                    calib_key = folder
                else:
                    folder, calib_key, cam_order = cam_item[0], cam_item[1], cam_item[2]
                img_path = files.get(f'cam_{cam_order}')
                cam_calib = calib_data.get(calib_key)
                has_img = img_path is not None
                has_calib = cam_calib is not None
                has_intr = has_calib and all(k in cam_calib for k in ['fx', 'fy', 'cx', 'cy'])
                print(f"      {cam_order}: image={has_img}, calib={has_calib}, intrinsic={has_intr}")
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
                # [bugfix][xmy] converter时，尺寸的长宽顺序错位了
                # 1. tyjt的标注规则：https://tyjt.yuque.com/lz6a2x/perc/ptkaar
                # "type": "car",
                # "box3d": [x, y, z, l, w, h],
                # "rotation": [yaw, roll, pitch],
                # "pointNum": n,
                # "id": "abcde01234",
                # "door_open": 0,
                # "trunk_open": 0,
                # "hood_open": 0
                # 2. 因此此处： box3d = obj['box3d'] 获取的顺序为： [cx, cy, cz, l, w, h]
                # 3. nusc官方需求为： [cx, cy, cz, w, l, h]
                # ==> 需要修复bug ==> 重新生成 数据集 ==> 重新训练

                # box3d = obj['box3d']  # [cx, cy, cz, l, w, h]
                # yaw = obj['rotation'][0]
                # gt_boxes.append(box3d + [yaw])  # 7维
                cx, cy, cz, l, w, h = obj['box3d']  # 解析原始顺序
                yaw = obj['rotation'][0]
                gt_boxes.append([cx, cy, cz, w, l, h, yaw])   # 交换 w 和 l，并添加 yaw   ==》 nusc 官方 匹配代码

                yaw = obj['rotation'][0]
                gt_boxes.append(box3d + [yaw])  # 7维
                gt_names.append(obj['type'])
                point_num = obj.get('pointNum')
                if point_num is None:
                    point_num = -1
                num_lidar_pts.append(point_num)
                valid_flag.append(True)
                gt_velocity.append([0.0, 0.0])   # v9.4.3: 速度设为0

        except Exception as e:
            print(f"警告：无法解析标注文件 {label_path}: {e}")
            skip_reason = "label_parse_error"
            skip_detail[package_name][skip_reason] += 1
            return None
    else:
        # 没有标注文件
        skip_reason = "missing_label_file"
        skip_detail[package_name][skip_reason] += 1
        return None

    if not gt_boxes:
        # 没有有效标注（可能是空标注文件）
        skip_reason = "empty_gt_boxes"
        skip_detail[package_name][skip_reason] += 1
        return None

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

def print_package_stats(pkg_name, total_lidar, total_scanned, early_detail,
                        filtered_cam, success):
    """
    打印数据包统计树状图
    total_lidar: 总点云数（若为 None 则不显示第一层）
    total_scanned: 扫描阶段有效样本数
    early_detail: 早期过滤原因字典，如 {'missing_label':12, 'no_matching_road':600}
    filtered_cam: 因缺少 CAM_C 被过滤的数量
    success: 成功生成的样本数
    """
    indent = "  "
    if total_lidar is not None:
        # 详细模式：显示总点云数及第一次过滤
        print(f"{indent}总点云数: {total_lidar}")
        print(f"{indent}   第一次过滤（规则：点云时间戳纯数字、子包匹配路口、同时存在点云/标注/≥2相机）：")
        silent = total_lidar - total_scanned
        print(f"{indent}   ├─ 过滤掉: {silent}")
        print(f"{indent}   └─ 过滤后保留: {total_scanned}  (扫描阶段有效样本)")
        inner_indent = f"{indent}       "
    else:
        # 简洁模式：直接从扫描阶段有效样本开始
        print(f"{indent}扫描阶段有效样本: {total_scanned}")
        inner_indent = f"{indent}   "

    # 第二次过滤（早期过滤）
    early_total = sum(early_detail.values())
    print(f"{inner_indent}├─ 第二次过滤（规则：标注目录存在、相机数≥2）：")
    print(f"{inner_indent}│   ├─ 过滤掉: {early_total}")
    if early_detail:
        # 打印具体原因
        items = list(early_detail.items())
        for i, (reason, cnt) in enumerate(items):
            if i == len(items) - 1:
                print(f"{inner_indent}│   │   └─ {reason}: {cnt}")
            else:
                print(f"{inner_indent}│   │   ├─ {reason}: {cnt}")
    print(f"{inner_indent}│   └─ 过滤后保留: {total_scanned}  (进入 build_sample_info)")

    # 第三次过滤（相机数量）
    print(f"{inner_indent}└─ 第三次过滤（规则：必须包含四路相机）：")
    print(f"{inner_indent}    ├─ 过滤掉: {filtered_cam}  (缺少 CAM_C)")
    print(f"{inner_indent}    └─ 过滤后保留: {success}  (成功生成)")


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

    # 扫描阶段有效样本数
    total_scanned = len(file_map)

    before_count = len(out_infos)

    # 按子包分组
    sub_packet_samples = defaultdict(list)
    for ts, files in file_map.items():
        sub_pkt = sub_packet_map.get(ts)
        if sub_pkt:
            sub_packet_samples[sub_pkt].append((int(ts), files))

    success_count = 0
    for sub_pkt, samples in sub_packet_samples.items():
        samples.sort(key=lambda x: x[0])
        temp_infos = []
        for ts, files in samples:
            road_id = files.get('road_id')
            if road_id is None:
                skip_detail[pkg_name]["missing_road_id"] += 1
                continue
            info = build_sample_info(str(ts), files, calib_data, pkg_config, road_id, pkg_name)
            if info:
                temp_infos.append(info)
                success_count += 1

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
                R = T_prev_to_cur[:3, :3]
                t = T_prev_to_cur[:3, 3]
                sweep = {
                    'data_path': prev_info['lidar_path'],
                    'timestamp': prev_info['timestamp'],
                    'human_time': prev_info['human_time'],
                    'sensor2lidar_rotation': R,
                    'sensor2lidar_translation': t,
                    'transform': T_prev_to_cur.tolist()
                }
                sweeps.append(sweep)
                prev_idx = prev_info['prev']
                count += 1
            info['sweeps'] = sweeps

    after_count = len(out_infos)
    sample_count = after_count - before_count

    # ========== 统计总点云数（仅在 VERBOSE 时执行，耗时） ==========
    if ENABLE_LIDAR_FILE_COUNT:
        all_lidar_count = 0
        for sub_pkt in (pkg_path / "datasets").iterdir():
            if sub_pkt.is_dir():
                lidar_dir = sub_pkt / "lidar" / "pcd"
                if lidar_dir.exists():
                    all_lidar_count += len(list(lidar_dir.glob("*.npy")))
    else:
        all_lidar_count = None

    # 打印树状统计图
    print_package_stats(
        pkg_name,
        all_lidar_count,
        total_scanned,
        early_filter_detail.get(pkg_name, {}),
        skip_detail.get(pkg_name, {}).get('final_missing_CAM_C', 0),
        sample_count
    )

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