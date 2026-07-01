#!/usr/bin/env python3
"""
TYJT 数据集预处理脚本，生成 BEVFusion 所需的 info pkl 文件。
基于 tyjt_calib_utils.py 中的 CalibrationProcessor 进行标定解析。
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

# 导入自定义标定工具
from tyjt_utils.tyjt_calib_utils import CalibrationProcessor

# ==================== 全局配置 ====================
# 数据包配置（可单独放在 config.json 中，这里为简便直接写为字典）
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

def scan_package_files(package_path: Path, config: Dict) -> Dict[str, Dict[str, Any]]:
    """
    扫描数据包内所有子包，返回时间戳到文件路径的映射。
    返回格式: { timestamp: { 'lidar': path, 'label': path, 'cam_A': path, ... } }
    """
    file_map = defaultdict(dict)
    datasets_dir = package_path / "datasets"
    if not datasets_dir.exists():
        return {}

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

        # --- 扫描 label 文件 ---
        label_dir = sub_packet / "lidar" / "label"
        if label_dir.exists():
            for json_file in label_dir.glob("*.json"):
                ts = json_file.stem
                if ts.isdigit():
                    file_map[ts]['label'] = str(json_file)

        # --- 扫描相机图像 ---
        # 从配置中获取相机文件夹名与相机顺序的对应关系
        # 注意：不同路口可能有不同的相机列表，这里简单处理：遍历所有可能的相机文件夹
        # 更精确的做法是结合 config 中的 intersections 来查找
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

    # 过滤出至少包含 lidar 和 label 的样本
    valid_map = {}
    for ts, files in file_map.items():
        if 'lidar' in files and 'label' in files:
            # 检查相机数量（至少 2 个）
            cam_keys = [k for k in files if k.startswith('cam_')]
            if len(cam_keys) >= 2:
                valid_map[ts] = files
    return valid_map

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

    # 计算 ego2global（即 group2map 矩阵）
    # 计算 ego2global（即 group2map 矩阵），并确保为列表
    ego2global_matrix = CalibrationProcessor.parse_transform(group2map_params)
    if isinstance(ego2global_matrix, np.ndarray):
        ego2global_matrix = ego2global_matrix.tolist()  # 将 numpy 数组转为列表

    # 处理相机
    cams_info = {}
    for cam_item in road_cfg["cameras"]:
        if config["type"] == "hikvision":
            folder, cam_order = cam_item[0], cam_item[1]
            calib_key = folder
        else:
            folder, calib_key, cam_order = cam_item[0], cam_item[1], cam_item[2]

        # 图像路径
        img_path = files.get(f'cam_{cam_order}')
        if not img_path:
            continue  # 该相机缺失

        # 相机标定参数
        cam_calib = calib_data.get(calib_key)
        if cam_calib is None:
            print(f"警告：缺失相机标定 {calib_key}")
            continue

        # 计算 cam2ego
        translation, rotation = compute_cam2ego(cam_calib, group2map_params)

        # 提取内参（假设标定文件中包含 fx, fy, cx, cy）
        # 注意：对于 all_in_one 格式，内参可能在 cam_calib 中直接提供；对于 hikvision，可能需要从嵌套中提取
        # 这里简化：从 cam_calib 中直接取，若缺失则用默认值
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
            "cam2ego_rotation": rotation,      # [w,x,y,z]
            "cam_intrinsic": intrinsic
        }

    if not cams_info:
        return None  # 无有效相机

    # 构造相机映射（可选，如果不需要可以省略）
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

    info = {
        "lidar_path": files['lidar'],
        "label_path": files['label'],
        "timestamp": int(ts),
        "cams": cams_info,
        "ego2global": ego2global_matrix,
        "lidar2ego": np.eye(4).tolist(), 
        # 新增总包信息
        "package_name": package_name,
        "road_id": road_id,
        "group_key": config['intersections'][road_id]['group_key'],
        "calib_root": config['calib_root'],
        "cameras_mapping": cameras_mapping,   # 可选
        "package_type": config['type'],
    }

    return info

def process_package(pkg_name: str, pkg_config: Dict, data_root: Path,
                    out_infos: List[Dict]):
    """处理单个数据包，将生成的 info 添加到 out_infos 列表中"""
    pkg_path = data_root / pkg_name
    if not pkg_path.exists():
        print(f"警告：数据包路径不存在 {pkg_path}")
        return

    print(f"正在处理数据包: {pkg_name}")
    # 加载标定
    calib_data = load_calibration(pkg_path, pkg_config)
    if not calib_data:
        print(f"  标定加载失败，跳过")
        return

    # 扫描文件
    file_map = scan_package_files(pkg_path, pkg_config)
    print(f"  扫描到 {len(file_map)} 个有效样本")

    # 对于每个路口，构建样本（假设每个数据包只有一个路口，若多个路口需循环）
    # 这里简化，取第一个路口
    road_id = next(iter(pkg_config["intersections"]))
    for ts, files in file_map.items():
        info = build_sample_info(ts, files, calib_data, pkg_config, road_id, pkg_name)
        if info:
            out_infos.append(info)

def main():
    parser = argparse.ArgumentParser(description="TYJT 数据集 info 生成工具")
    parser.add_argument("--data-root", type=str, default="/mnt/dataset/tyjt_RawData_all",
                        help="TYJT 数据集根目录")
    parser.add_argument("--train-split", type=str, default="tyjt_data_infos/tyjt_train.txt",
                        help="训练集划分文件")
    parser.add_argument("--val-split", type=str, default="tyjt_data_infos/tyjt_val.txt",
                        help="验证集划分文件")
    parser.add_argument("--out-dir", type=str, default="./tyjt_data_infos",
                        help="输出 pkl 文件的目录")
    args = parser.parse_args()

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
        process_package(pkg_name, PACKAGE_CONFIG[pkg_name], data_root, train_infos)

    # 处理验证集
    for pkg_name in val_pkgs:
        if pkg_name not in PACKAGE_CONFIG:
            print(f"跳过未配置的数据包: {pkg_name}")
            continue
        process_package(pkg_name, PACKAGE_CONFIG[pkg_name], data_root, val_infos)

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

if __name__ == "__main__":
    main()