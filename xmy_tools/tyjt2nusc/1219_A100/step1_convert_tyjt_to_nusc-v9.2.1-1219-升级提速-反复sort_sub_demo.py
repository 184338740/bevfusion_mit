#!/usr/bin/env python3
"""
tyjt数据集转nuscenes格式转换工具 - 分阶段优化版
版本: v9.2 (分阶段+缓存优化)
修改日期: 2024-12-19
核心升级:
1. 分阶段处理：--scan-only只扫描，--use-cache跳过检查
2. 缓存优化：预扫描结果保存到文件
3. 性能提升：预计10-15样本/秒
"""

import os
import json
import numpy as np
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union
import hashlib
import math
from collections import defaultdict
import re
import logging
import threading
import time
import concurrent.futures
from datetime import datetime
import sys
import argparse

# ==================== 命令行参数 ====================
def parse_args():
    parser = argparse.ArgumentParser(description='TYJT转NuScenes转换工具')
    parser.add_argument('--scan-only', action='store_true',
                       help='只扫描验证文件存在性，不处理数据')
    parser.add_argument('--use-cache', action='store_true',
                       help='使用缓存文件跳过文件检查')
    parser.add_argument('--cache-file', type=str, default='match_list.txt',
                       help='缓存文件路径（默认: match_list.txt）')
    parser.add_argument('--threads', type=int, default=4,
                       help='处理线程数（默认: 4）')
    parser.add_argument('--skip-existing', action='store_true',
                       help='跳过已处理的样本（基于输出文件）')
    return parser.parse_args()

args = parse_args()

# ==================== 全局标志 ====================
SCAN_ONLY_MODE = args.scan_only
USE_CACHE_MODE = args.use_cache
SKIP_FILE_CHECKS = args.use_cache or args.skip_existing
PROCESS_THREADS = args.threads
CACHE_FILE = Path(args.cache_file)

# ==================== 全局配置 ====================
Mode = "A100"
if Mode == "Local":
    TYJT_ROOT = "/mnt/dataset/tyjt_RawData_all/"
    OUTPUT_ROOT = "./output-1204/step1/nuscenes_tyjt"
    NUSC_VERSION = "v1.0-tyjt"
    DATA_PACKAGE_CONFIG = {
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
        }
    }
elif Mode == "A100":
    TYJT_ROOT = "/cephfsdata/users/lishan/00_Data/00_RawData"
    OUTPUT_ROOT = "./output-1204-sub-v9.2.1/step1/nuscenes_tyjt"
    NUSC_VERSION = "v1.0-tyjt"
    DATA_PACKAGE_CONFIG = {
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
else:
    print(f"❌【Error】: TYJT_ROOT & DATA_PACKAGES_TO_PROCESS: None")
    sys.exit(1)



# ==================== 分布式存储优化组件 ====================
class FastFileChecker:
    """快速文件检查器，减少分布式存储访问"""
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache = {}
        self.cache_expiry = {}
        self.ttl = ttl_seconds
        self.lock = threading.RLock()
        
    def exists(self, path: Path) -> bool:
        """带缓存的exists检查"""
        path_str = str(path)
        now = time.time()
        
        with self.lock:
            if path_str in self.cache:
                expiry_time = self.cache_expiry.get(path_str, 0)
                if now < expiry_time:
                    return self.cache[path_str]
        
        # 实际检查
        result = path.exists()
        
        with self.lock:
            self.cache[path_str] = result
            self.cache_expiry[path_str] = now + self.ttl
        
        return result
    
    def batch_exists(self, paths: List[Path]) -> Dict[Path, bool]:
        """批量检查文件存在性"""
        results = {}
        now = time.time()
        
        with self.lock:
            for path in paths:
                path_str = str(path)
                if path_str in self.cache:
                    expiry_time = self.cache_expiry.get(path_str, 0)
                    if now < expiry_time:
                        results[path] = self.cache[path_str]
        
        remaining_paths = [p for p in paths if p not in results]
        for path in remaining_paths:
            result = path.exists()
            results[path] = result
            
            path_str = str(path)
            with self.lock:
                self.cache[path_str] = result
                self.cache_expiry[path_str] = now + self.ttl
        
        return results
    
    def clear_cache(self):
        """清理缓存"""
        with self.lock:
            self.cache.clear()
            self.cache_expiry.clear()

fast_checker = FastFileChecker(ttl_seconds=300)


# ==================== Log系统初始化 ====================
def setup_logging(output_root: str):
    """设置日志系统：主日志+各数据包独立日志"""
    output_path = Path(output_root)
    log_dir = output_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 主日志
    main_log_path = log_dir / "step_1_main.log"
    main_logger = logging.getLogger("main")
    main_logger.setLevel(logging.INFO)
    
    if not main_logger.handlers:
        file_handler = logging.FileHandler(main_log_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        main_logger.addHandler(file_handler)
        main_logger.addHandler(console_handler)
    
    return main_logger, log_dir

def get_package_logger(package_name: str, log_dir: Path):
    """获取数据包特定的日志器"""
    logger = logging.getLogger(f"package_{package_name}")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        log_file = log_dir / f"{package_name}.log"
        handler = logging.FileHandler(log_file, encoding='utf-8')
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    
    return logger

# 全局日志器引用
main_logger = None
log_dir = None

# 核心修正：A/B/C/D相机→NuScenes相机固定映射
CAM_ORDER_TO_NUSC = {
    "A": "CAM_FRONT",
    "B": "CAM_FRONT_RIGHT", 
    "C": "CAM_BACK",
    "D": "CAM_FRONT_LEFT"
}

# Step1类别映射
CATEGORY_MAPPING = {
    "car": "tyjt.car",
    "truck": "tyjt.truck", 
    "construction_truck": "tyjt.construction_truck",
    "van": "tyjt.van",
    "bus": "tyjt.bus",
    "robot": "tyjt.robot",
    "pedestrian": "tyjt.pedestrian",
    "cyclist": "tyjt.cyclist",
    "bicycle": "tyjt.bicycle",
    "tricycle": "tyjt.tricycle",
    "tricyclist": "tyjt.tricyclist",
    "trolley": "tyjt.trolley",
    "cone": "tyjt.cone",
    "barrier": "tyjt.barrier",
    "other": "tyjt.other"
}

LIDAR_NUSC_NAME = "LIDAR_TOP"

# ==================== 快速扫描模式函数 ====================
def run_scan_only_mode():
    """只扫描模式：快速验证文件存在性"""
    print("=" * 60)
    print("🔍 快速扫描模式（--scan-only）")
    print("=" * 60)
    
    start_time = time.time()
    
    # 查找数据包
    tyjt_root_path = Path(TYJT_ROOT)
    packages = []
    for item in tyjt_root_path.iterdir():
        if item.is_dir() and item.name in DATA_PACKAGE_CONFIG:
            packages.append(item)
    
    print(f"找到 {len(packages)} 个数据包")
    
    all_valid_samples = []
    
    for package in packages:
        package_name = package.name
        print(f"扫描数据包: {package_name}")
        
        if package_name not in DATA_PACKAGE_CONFIG:
            continue
            
        package_config = DATA_PACKAGE_CONFIG[package_name]
        
        # 扫描每个路口
        for intersection_key in package_config["intersections"]:
            valid_samples = fast_scan_intersection(
                package, package_name, intersection_key, package_config
            )
            all_valid_samples.extend(valid_samples)
    
    # 保存结果
    with open(CACHE_FILE, 'w') as f:
        for sample in all_valid_samples:
            f.write(f"{sample}\n")
    
    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"✅ 扫描完成！")
    print(f"📊 有效样本数: {len(all_valid_samples)}")
    print(f"⏱️  耗时: {elapsed:.1f}秒")
    print(f"📄 缓存文件: {CACHE_FILE}")
    print("=" * 60)
    
    if all_valid_samples:
        print("📋 前10个样本示例:")
        for sample in all_valid_samples[:10]:
            print(f"  {sample}")
    
    sys.exit(0)

def fast_scan_intersection(package_dir, package_name, intersection_key, package_config):
    """快速扫描单个路口"""
    valid_samples = []
    datasets_dir = package_dir / "datasets"
    
    if not datasets_dir.exists():
        return []
    
    # 获取所有子包
    sub_packets = list(datasets_dir.iterdir())
    
    # 并行扫描
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for sub_packet in sub_packets:
            future = executor.submit(
                fast_scan_subpacket,
                sub_packet, package_name, intersection_key, package_config
            )
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            valid_samples.extend(future.result())
    
    return valid_samples

def fast_scan_subpacket(sub_packet, package_name, intersection_key, package_config):
    """快速扫描单个子包"""
    samples = []
    
    # 检查lidar目录
    lidar_dir = sub_packet / "lidar" / "pcd"
    if not lidar_dir.exists():
        return []
    
    try:
        # 一次性获取所有npy文件
        npy_files = list(lidar_dir.glob("*.npy"))
        
        for npy_file in npy_files:
            timestamp = npy_file.stem
            if not timestamp.isdigit():
                continue
            
            # 快速检查必需文件
            if check_required_files_fast(sub_packet, timestamp, package_config, intersection_key):
                sample_key = f"{package_name}|{intersection_key}|{sub_packet.name}|{timestamp}"
                samples.append(sample_key)
                
    except Exception as e:
        print(f"扫描{sub_packet.name}失败: {e}")
    
    return samples

def check_required_files_fast(sub_packet, timestamp, package_config, intersection_key):
    """快速检查必需文件"""
    # 检查label文件
    label_path = sub_packet / "lidar" / "label" / f"{timestamp}.json"
    if not label_path.exists():
        return False
    
    # 检查相机文件
    intersection_config = package_config["intersections"][intersection_key]
    
    for cam_info in intersection_config["cameras"]:
        if package_config["type"] == "hikvision":
            cam_folder, _ = cam_info
        else:
            cam_folder, _, _ = cam_info
            
        img_path = sub_packet / cam_folder / "image_dc" / f"{timestamp}.jpg"
        if not img_path.exists():
            return False
    
    return True

def load_cache_file():
    """加载缓存文件"""
    if not CACHE_FILE.exists():
        main_logger.error(f"缓存文件不存在: {CACHE_FILE}")
        return []
    
    with open(CACHE_FILE, 'r') as f:
        samples = [line.strip() for line in f if line.strip()]
    
    main_logger.info(f"从缓存加载 {len(samples)} 个样本")
    return samples

# ==================== Token管理器 ====================
class EnhancedIDGenerator:
    def __init__(self):
        self.uuid_map = {}
        self.used_tokens: Set[str] = set()
        self.entity_counters = defaultdict(int)
        self.lock = threading.RLock()
    
    def get_scoped_token(self, entity_type: str, unique_key: str) -> str:
        with self.lock:
            scope_key = f"{entity_type}::{unique_key}"
            if scope_key not in self.uuid_map:
                hash_input = f"{entity_type}_{unique_key}_{self.entity_counters[entity_type]}"
                hash_obj = hashlib.sha256(hash_input.encode())
                hash_hex = hash_obj.hexdigest()[:32]
                debug_uuid = f"{hash_hex[:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:32]}"
                counter = 0
                original_uuid = debug_uuid
                while debug_uuid in self.used_tokens:
                    counter += 1
                    debug_uuid = f"{original_uuid}_{counter}"
                self.uuid_map[scope_key] = debug_uuid
                self.used_tokens.add(debug_uuid)
                self.entity_counters[entity_type] += 1
            return self.uuid_map[scope_key]
    
    def validate_token_uniqueness(self) -> Tuple[bool, Dict]:
        with self.lock:
            tokens = list(self.uuid_map.values())
            unique_tokens = set(tokens)
            stats = {
                'total_tokens': len(tokens),
                'unique_tokens': len(unique_tokens),
                'entity_counts': dict(self.entity_counters)
            }
            return len(tokens) == len(unique_tokens), stats

id_gen = EnhancedIDGenerator()

# ==================== 数学工具函数 ====================
def quaternion_from_euler(yaw: float, pitch: float = 0, roll: float = 0) -> List[float]:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y, z]

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])

def rotation_matrix_to_quaternion(matrix: np.ndarray) -> List[float]:
    m = matrix
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return [w, x, y, z]

def parse_tyjt_transform(calib_params: Union[Dict, List], package_logger=None) -> np.ndarray:
    """解析tyjt标定参数为4x4变换矩阵"""
    try:
        if isinstance(calib_params, Dict):
            if 'translation' in calib_params and 'rotation' in calib_params:
                translation = np.array(calib_params['translation'])
                rotation_quat = calib_params['rotation']
                
                if len(rotation_quat) == 4:
                    x, y, z, w = rotation_quat
                    w, x, y, z = w, x, y, z
                else:
                    return np.eye(4)
                    
            elif all(key in calib_params for key in ['tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'rw']):
                translation = np.array([
                    calib_params['tx'],
                    calib_params['ty'], 
                    calib_params['tz']
                ])
                x, y, z, w = (
                    calib_params['rx'],
                    calib_params['ry'],
                    calib_params['rz'], 
                    calib_params['rw']
                )
            else:
                return np.eye(4)
                
        elif isinstance(calib_params, List):
            if len(calib_params) == 7:
                tx, ty, tz, rx, ry, rz, rw = calib_params
                translation = np.array([tx, ty, tz])
                x, y, z, w = rx, ry, rz, rw
            else:
                return np.eye(4)
        else:
            return np.eye(4)
        
        rotation_matrix = quaternion_to_rotation_matrix([w, x, y, z])
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation
        
        return transform
        
    except Exception as e:
        if package_logger:
            package_logger.error(f"解析标定参数失败: {e}")
        return np.eye(4)

def get_sensor2ego_transform_fixed(calib_data: Dict, sensor_calib_key: str, group2map_key: str, package_logger=None) -> Tuple[List[float], List[float]]:
    """修复版标定转换：sensor→map→group(ego)"""
    try:
        if sensor_calib_key not in calib_data:
            return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        
        sensor_calib_params = calib_data[sensor_calib_key]
        sensor2map = parse_tyjt_transform(sensor_calib_params, package_logger)
        
        if group2map_key in calib_data:
            group2map_params = calib_data[group2map_key]
            
            if isinstance(group2map_params, dict) and 'transform' in group2map_params:
                group2map_params = group2map_params['transform']
            
            group2map = parse_tyjt_transform(group2map_params, package_logger)
            map2group = np.linalg.inv(group2map)
        else:
            return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        
        sensor2ego = map2group @ sensor2map
        
        if not np.isfinite(sensor2ego).all():
            return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        
        translation = sensor2ego[:3, 3].tolist()
        rotation_matrix = sensor2ego[:3, :3]
        rotation = rotation_matrix_to_quaternion(rotation_matrix)
        
        return translation, rotation
        
    except Exception as e:
        if package_logger:
            package_logger.error(f"计算sensor2ego变换时出错: {e}")
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]

# ==================== 目录结构与数据加载 ====================
def create_directory_structure(output_root: str, version: str):
    """创建输出目录结构"""
    base_path = Path(output_root)
    base_path.mkdir(parents=True, exist_ok=True)
    
    version_dir = base_path / version
    version_dir.mkdir(parents=True, exist_ok=True)
    
    for data_type in ["samples", "sweeps"]:
        root_dir = base_path / data_type
        root_dir.mkdir(parents=True, exist_ok=True)
        for nusc_cam in CAM_ORDER_TO_NUSC.values():
            (root_dir / nusc_cam).mkdir(parents=True, exist_ok=True)
        (root_dir / LIDAR_NUSC_NAME).mkdir(parents=True, exist_ok=True)
    
    maps_dir = base_path / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    
    return base_path

def find_data_packages(tyjt_root: str) -> List[Path]:
    """查找数据包"""
    root_path = Path(tyjt_root)
    packages = []
    configured_package_names = set(DATA_PACKAGE_CONFIG.keys())
    
    main_logger.info(f"在目录 {tyjt_root} 中查找数据包...")
    
    found_packages = []
    for item in root_path.iterdir():
        if item.is_dir() and item.name in configured_package_names:
            datasets_dir = item / "datasets"
            if fast_checker.exists(datasets_dir):
                packages.append(item)
                found_packages.append(item.name)
                main_logger.info(f"✅ 找到有效数据包: {item.name}")
    
    missing = configured_package_names - set(found_packages)
    if missing:
        main_logger.warning(f"配置中存在但未找到的数据包: {missing}")
    
    main_logger.info(f"📦 总共找到 {len(packages)} 个有效数据包: {found_packages}")
    return packages

def load_calibration(package_path: Path, package_config: Dict, package_logger) -> Dict:
    """加载标定数据"""
    calib_root = package_path / package_config["calib_root"]
    if not fast_checker.exists(calib_root):
        package_logger.error(f"标定目录不存在: {calib_root}")
        return {}
    
    calib_data = {}
    try:
        if package_config["type"] == "hikvision":
            package_logger.info(f"加载海康相机标定数据...")
            group2map_path = calib_root / package_config["group2map_file"]
            if fast_checker.exists(group2map_path):
                with open(group2map_path, 'r') as f:
                    calib_data.update(json.load(f))
            
            camera2map_path = calib_root / package_config["camera2map_file"]
            if fast_checker.exists(camera2map_path):
                with open(camera2map_path, 'r') as f:
                    calib_data.update(json.load(f))
        
        elif package_config["type"] == "all_in_one":
            package_logger.info(f"加载一体机标定数据...")
            sensor2map_path = calib_root / package_config["sensor2map_file"]
            if fast_checker.exists(sensor2map_path):
                with open(sensor2map_path, 'r') as f:
                    calib_data.update(json.load(f))
        
        package_logger.info(f"✅ 加载标定数据成功，共 {len(calib_data)} 个键")
    except Exception as e:
        package_logger.error(f"❌ 加载标定数据失败: {e}")
    return calib_data

# ==================== 数据表初始化 ====================
def initialize_nusc_data() -> Dict:
    return {
        "category": [], "attribute": [], "sensor": [], "calibrated_sensor": [],
        "ego_pose": [], "log": [], "scene": [], "sample": [], "sample_data": [],
        "sample_annotation": [], "instance": [], "visibility": [], "map": []
    }


def generate_basic_tables(nusc_data: Dict, scene_name: str, calib_data: Dict, 
                         package_config: Dict, intersection_key: str, 
                         package_logger, global_nusc_data_lock=None) -> Dict:
    """生成基础数据表 - 线程安全版本"""
    token_map = {}
    intersection_config = package_config["intersections"][intersection_key]
    
    # 1. log表（不涉及共享数据，放在锁外）
    log_unique_key = f"{package_config['type']}_{scene_name}_{intersection_key}"
    log_token = id_gen.get_scoped_token("log", log_unique_key)
    token_map['log_token'] = log_token
    nusc_data["log"].append({
        "token": log_token,
        "logfile": f"{log_unique_key}.log",
        "vehicle": "tyjt_system",
        "date_captured": scene_name.split('_')[1] if '_' in scene_name else "2024-01-01",
        "location": scene_name.split('_')[-1] if scene_name.endswith(('wuxi', 'weiyuan')) else "default"
    })
    
    # 2. scene表（不涉及共享数据，放在锁外）
    scene_unique_key = f"{scene_name}_{intersection_key}"
    scene_token = id_gen.get_scoped_token("scene", scene_unique_key)
    token_map['scene_token'] = scene_token
    nusc_data["scene"].append({
        "token": scene_token,
        "name": scene_unique_key,
        "description": f"TYJT: {scene_name} - {intersection_key}",
        "log_token": log_token,
        "nbr_samples": 0,
        "first_sample_token": "",
        "last_sample_token": ""
    })
    
    # ========== 关键修改：对共享数据的操作全部放入锁内 ==========
    if global_nusc_data_lock:
        with global_nusc_data_lock:
            # 3. category表 - 检查并添加
            existing_categories = {cat['name'] for cat in nusc_data['category']}
            for tyjt_cat, nusc_cat in CATEGORY_MAPPING.items():
                if nusc_cat not in existing_categories:
                    nusc_data["category"].append({
                        "token": id_gen.get_scoped_token("category", nusc_cat),
                        "name": nusc_cat,
                        "description": f"From tyjt: {tyjt_cat}"
                    })
                    existing_categories.add(nusc_cat)
            
            # 4. attribute表 - 检查并添加
            existing_attributes = {attr['name'] for attr in nusc_data['attribute']}
            for attr_name in ["vehicle.moving", "pedestrian.moving", "vehicle.stopped"]:
                if attr_name not in existing_attributes:
                    nusc_data["attribute"].append({
                        "token": id_gen.get_scoped_token("attribute", attr_name),
                        "name": attr_name,
                        "description": f"{attr_name.replace('.', ' is ')}"
                    })
                    existing_attributes.add(attr_name)
            
            # 5. visibility表 - 只需添加一次
            if not nusc_data["visibility"]:
                nusc_data["visibility"].append({
                    "token": id_gen.get_scoped_token("visibility", "full"),
                    "level": "full",
                    "description": "Fully visible"
                })
            
            # 6. sensor表 - 核心修复部分（防止KeyError）
            existing_sensors = {sensor['channel'] for sensor in nusc_data['sensor']}
            sensor_tokens = {}
            
            # 处理相机传感器
            for cam_info in intersection_config["cameras"]:
                if package_config["type"] == "hikvision":
                    cam_folder, cam_order = cam_info
                else:
                    cam_folder, _, cam_order = cam_info
                
                nusc_cam_name = CAM_ORDER_TO_NUSC.get(cam_order)
                if not nusc_cam_name:
                    continue
                
                if nusc_cam_name not in existing_sensors:
                    sensor_token = id_gen.get_scoped_token("sensor", nusc_cam_name)
                    sensor_tokens[nusc_cam_name] = sensor_token
                    nusc_data["sensor"].append({
                        "token": sensor_token,
                        "channel": nusc_cam_name,
                        "modality": "camera"
                    })
                    existing_sensors.add(nusc_cam_name)  # 及时更新集合
                else:
                    # 从现有传感器中查找token
                    found = False
                    for sensor in nusc_data['sensor']:
                        if sensor['channel'] == nusc_cam_name:
                            sensor_tokens[nusc_cam_name] = sensor['token']
                            found = True
                            break
                    if not found:
                        # 理论上不应该发生，但添加容错
                        sensor_token = id_gen.get_scoped_token("sensor", nusc_cam_name)
                        sensor_tokens[nusc_cam_name] = sensor_token
                        nusc_data["sensor"].append({
                            "token": sensor_token,
                            "channel": nusc_cam_name,
                            "modality": "camera"
                        })
            
            # 处理激光雷达传感器（修复KeyError的关键）
            if LIDAR_NUSC_NAME not in existing_sensors:
                lidar_sensor_token = id_gen.get_scoped_token("sensor", LIDAR_NUSC_NAME)
                sensor_tokens[LIDAR_NUSC_NAME] = lidar_sensor_token
                nusc_data["sensor"].append({
                    "token": lidar_sensor_token,
                    "channel": LIDAR_NUSC_NAME,
                    "modality": "lidar"
                })
            else:
                found = False
                for sensor in nusc_data['sensor']:
                    if sensor['channel'] == LIDAR_NUSC_NAME:
                        sensor_tokens[LIDAR_NUSC_NAME] = sensor['token']
                        found = True
                        break
                if not found:
                    lidar_sensor_token = id_gen.get_scoped_token("sensor", LIDAR_NUSC_NAME)
                    sensor_tokens[LIDAR_NUSC_NAME] = lidar_sensor_token
                    nusc_data["sensor"].append({
                        "token": lidar_sensor_token,
                        "channel": LIDAR_NUSC_NAME,
                        "modality": "lidar"
                    })
            
            token_map['sensor_tokens'] = sensor_tokens
    else:
        # 单线程模式（原逻辑）
        # category表
        existing_categories = {cat['name'] for cat in nusc_data['category']}
        for tyjt_cat, nusc_cat in CATEGORY_MAPPING.items():
            if nusc_cat not in existing_categories:
                nusc_data["category"].append({
                    "token": id_gen.get_scoped_token("category", nusc_cat),
                    "name": nusc_cat,
                    "description": f"From tyjt: {tyjt_cat}"
                })
        
        # attribute表
        existing_attributes = {attr['name'] for attr in nusc_data['attribute']}
        for attr_name in ["vehicle.moving", "pedestrian.moving", "vehicle.stopped"]:
            if attr_name not in existing_attributes:
                nusc_data["attribute"].append({
                    "token": id_gen.get_scoped_token("attribute", attr_name),
                    "name": attr_name,
                    "description": f"{attr_name.replace('.', ' is ')}"
                })
        
        # visibility表
        if not nusc_data["visibility"]:
            nusc_data["visibility"].append({
                "token": id_gen.get_scoped_token("visibility", "full"),
                "level": "full",
                "description": "Fully visible"
            })
        
        # sensor表
        existing_sensors = {sensor['channel'] for sensor in nusc_data['sensor']}
        sensor_tokens = {}
        
        for cam_info in intersection_config["cameras"]:
            if package_config["type"] == "hikvision":
                cam_folder, cam_order = cam_info
            else:
                cam_folder, _, cam_order = cam_info
            
            nusc_cam_name = CAM_ORDER_TO_NUSC.get(cam_order)
            if not nusc_cam_name:
                continue
            
            if nusc_cam_name not in existing_sensors:
                sensor_token = id_gen.get_scoped_token("sensor", nusc_cam_name)
                sensor_tokens[nusc_cam_name] = sensor_token
                nusc_data["sensor"].append({
                    "token": sensor_token,
                    "channel": nusc_cam_name,
                    "modality": "camera"
                })
            else:
                for sensor in nusc_data['sensor']:
                    if sensor['channel'] == nusc_cam_name:
                        sensor_tokens[nusc_cam_name] = sensor['token']
                        break
        
        # 激光雷达传感器
        if LIDAR_NUSC_NAME not in existing_sensors:
            lidar_sensor_token = id_gen.get_scoped_token("sensor", LIDAR_NUSC_NAME)
            sensor_tokens[LIDAR_NUSC_NAME] = lidar_sensor_token
            nusc_data["sensor"].append({
                "token": lidar_sensor_token,
                "channel": LIDAR_NUSC_NAME,
                "modality": "lidar"
            })
        else:
            for sensor in nusc_data['sensor']:
                if sensor['channel'] == LIDAR_NUSC_NAME:
                    sensor_tokens[LIDAR_NUSC_NAME] = sensor['token']
                    break
        
        token_map['sensor_tokens'] = sensor_tokens
    
    # 7. calibrated_sensor表（不涉及共享数据竞争，放在锁外）
    calib_sensor_tokens = {}
    group2map_key = intersection_config["group_key"]
    
    package_logger.info(f"生成 {intersection_key} 路口标定数据")
    
    for cam_info in intersection_config["cameras"]:
        if package_config["type"] == "hikvision":
            cam_folder, cam_order = cam_info
            sensor_calib_key = cam_folder
        else:
            cam_folder, sensor_calib_key, cam_order = cam_info
        
        nusc_cam_name = CAM_ORDER_TO_NUSC.get(cam_order)
        if not nusc_cam_name or nusc_cam_name not in token_map['sensor_tokens']:
            continue
        
        sensor_token = token_map['sensor_tokens'][nusc_cam_name]
        
        translation, rotation = get_sensor2ego_transform_fixed(
            calib_data, sensor_calib_key, group2map_key, package_logger
        )
        
        calib_unique_key = f"{scene_unique_key}_{nusc_cam_name}"
        calib_token = id_gen.get_scoped_token("calibrated_sensor", calib_unique_key)
        calib_sensor_tokens[nusc_cam_name] = calib_token
        
        cam_intrinsic = [
            [1680.0, 0, 960.0],
            [0, 1851.0, 540.0],
            [0, 0, 1]
        ]
        
        if sensor_calib_key in calib_data and isinstance(calib_data[sensor_calib_key], Dict):
            cam_calib = calib_data[sensor_calib_key]
            if 'fx' in cam_calib and 'fy' in cam_calib and 'cx' in cam_calib and 'cy' in cam_calib:
                cam_intrinsic = [
                    [cam_calib['fx'], 0, cam_calib['cx']],
                    [0, cam_calib['fy'], cam_calib['cy']],
                    [0, 0, 1]
                ]
        
        nusc_data["calibrated_sensor"].append({
            "token": calib_token,
            "sensor_token": sensor_token,
            "translation": translation,
            "rotation": rotation,
            "camera_intrinsic": cam_intrinsic
        })
    
    # 激光雷达标定
    lidar_calib_unique_key = f"{scene_unique_key}_{LIDAR_NUSC_NAME}"
    lidar_calib_token = id_gen.get_scoped_token("calibrated_sensor", lidar_calib_unique_key)
    calib_sensor_tokens[LIDAR_NUSC_NAME] = lidar_calib_token
    nusc_data["calibrated_sensor"].append({
        "token": lidar_calib_token,
        "sensor_token": token_map['sensor_tokens'][LIDAR_NUSC_NAME],
        "translation": [0.0, 0.0, 0.0],
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "camera_intrinsic": []
    })
    
    token_map['calib_sensor_tokens'] = calib_sensor_tokens
    
    # 8. ego_pose表（放在锁外）
    ego_pose_unique_key = f"{scene_unique_key}_ego"
    ego_pose_token = id_gen.get_scoped_token("ego_pose", ego_pose_unique_key)
    token_map['ego_pose_token'] = ego_pose_token
    nusc_data["ego_pose"].append({
        "token": ego_pose_token,
        "translation": [0.0, 0.0, 0.0],
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "timestamp": 0
    })
    
    # 9. map表（涉及共享数据，但只需在锁内添加一次）
    if global_nusc_data_lock:
        with global_nusc_data_lock:
            if not nusc_data["map"]:
                map_token = id_gen.get_scoped_token("map", "tyjt_empty")
                nusc_data["map"].append({
                    "token": map_token,
                    "log_tokens": [log_token],
                    "category": "city",
                    "filename": "maps/empty_map.png",
                    "map_name": "tyjt_empty"
                })
                token_map['map_token'] = map_token
            else:
                map_record = nusc_data["map"][0]
                current_log_tokens = set(map_record.get('log_tokens', []))
                current_log_tokens.add(log_token)
                map_record['log_tokens'] = list(current_log_tokens)
                token_map['map_token'] = map_record['token']
    else:
        if not nusc_data["map"]:
            map_token = id_gen.get_scoped_token("map", "tyjt_empty")
            nusc_data["map"].append({
                "token": map_token,
                "log_tokens": [log_token],
                "category": "city",
                "filename": "maps/empty_map.png",
                "map_name": "tyjt_empty"
            })
            token_map['map_token'] = map_token
        else:
            map_record = nusc_data["map"][0]
            current_log_tokens = set(map_record.get('log_tokens', []))
            current_log_tokens.add(log_token)
            map_record['log_tokens'] = list(current_log_tokens)
            token_map['map_token'] = map_record['token']
    
    package_logger.info(f"基础数据表生成完成，场景: {scene_unique_key}")
    return token_map


# ==================== 样本处理（优化版） ====================
def npy_to_bin(npy_path: Path, bin_path: Path, package_logger) -> bool:
    """npy→bin转换"""
    try:
        if SKIP_FILE_CHECKS and bin_path.exists():
            return True
            
        if not fast_checker.exists(npy_path):
            return False
        
        point_cloud = np.load(npy_path, allow_pickle=False)
        
        if point_cloud.shape[1] >= 5:
            points_5d = point_cloud[:, :5].astype(np.float32)
        elif point_cloud.shape[1] >= 3:
            xyz = point_cloud[:, :3].astype(np.float32)
            n_points = xyz.shape[0]
            points_5d = np.zeros((n_points, 5), dtype=np.float32)
            points_5d[:, :3] = xyz
            points_5d[:, 3] = 1.0
            points_5d[:, 4] = 0.0
        else:
            return False
        
        points_5d.tofile(bin_path)
        return True
        
    except Exception as e:
        package_logger.error(f"点云转换失败 {npy_path}: {e}")
        return False

def process_single_sample_fast(
    nusc_data: Dict,
    sub_packet: Path,
    timestamp: str,
    output_root: Path,
    package_name: str,
    intersection_key: str,
    sub_packet_name: str,
    prev_sample_token: str,
    sample_index: int,
    token_map: Dict,
    package_config: Dict,
    package_logger
) -> str:
    """快速处理单个样本（跳过文件检查）"""
    unique_key = f"{package_name}_{intersection_key}_{sub_packet_name}_{timestamp}"
    sample_token = id_gen.get_scoped_token("sample", unique_key)
    
    sample_data = {
        "token": sample_token,
        "timestamp": int(timestamp) if timestamp.isdigit() else sample_index * 1000000,
        "scene_token": token_map['scene_token'],
        "prev": prev_sample_token,
        "next": ""
    }
    
    if prev_sample_token:
        for sample in nusc_data["sample"]:
            if sample["token"] == prev_sample_token:
                sample["next"] = sample_token
                break
    
    nusc_data["sample"].append(sample_data)
    
    # 快速处理传感器数据
    sensor_tokens = token_map['sensor_tokens']
    calib_sensor_tokens = token_map['calib_sensor_tokens']
    intersection_config = package_config["intersections"][intersection_key]
    
    # 相机数据处理
    for cam_info in intersection_config["cameras"]:
        if package_config["type"] == "hikvision":
            cam_folder, cam_order = cam_info
        else:
            cam_folder, _, cam_order = cam_info
        
        nusc_cam_name = CAM_ORDER_TO_NUSC.get(cam_order)
        if not nusc_cam_name or nusc_cam_name not in sensor_tokens:
            continue
        
        img_original_path = sub_packet / cam_folder / "image_dc" / f"{timestamp}.jpg"
        img_output_dir = output_root / "samples" / nusc_cam_name
        img_output_path = img_output_dir / f"{unique_key}.jpg"
        
        # 跳过检查，直接创建软链接
        img_output_dir.mkdir(parents=True, exist_ok=True)
        if not img_output_path.exists():
            try:
                os.symlink(img_original_path.resolve(), img_output_path.resolve())
            except FileExistsError:
                pass
        
        sample_data_token = id_gen.get_scoped_token("sample_data", f"{unique_key}_{nusc_cam_name}")
        nusc_data["sample_data"].append({
            "token": sample_data_token,
            "sample_token": sample_token,
            "ego_pose_token": token_map['ego_pose_token'],
            "calibrated_sensor_token": calib_sensor_tokens[nusc_cam_name],
            "filename": f"samples/{nusc_cam_name}/{unique_key}.jpg",
            "fileformat": "jpg",
            "width": 1920,
            "height": 1080,
            "timestamp": int(timestamp) if timestamp.isdigit() else 0,
            "is_key_frame": True,
            "next": "",
            "prev": ""
        })
    
    # 激光雷达数据处理
    lidar_original_npy = sub_packet / "lidar" / "pcd" / f"{timestamp}.npy"
    lidar_output_dir = output_root / "samples" / LIDAR_NUSC_NAME
    lidar_output_path = lidar_output_dir / f"{unique_key}.bin"
    
    lidar_output_dir.mkdir(parents=True, exist_ok=True)
    if not lidar_output_path.exists():
        npy_to_bin(lidar_original_npy, lidar_output_path, package_logger)
    
    lidar_sample_data_token = id_gen.get_scoped_token("sample_data", f"{unique_key}_{LIDAR_NUSC_NAME}")
    nusc_data["sample_data"].append({
        "token": lidar_sample_data_token,
        "sample_token": sample_token,
        "ego_pose_token": token_map['ego_pose_token'],
        "calibrated_sensor_token": calib_sensor_tokens[LIDAR_NUSC_NAME],
        "filename": f"samples/{LIDAR_NUSC_NAME}/{unique_key}.bin",
        "fileformat": "bin",
        "width": 0,
        "height": 0,
        "timestamp": int(timestamp) if timestamp.isdigit() else 0,
        "is_key_frame": True,
        "next": "",
        "prev": ""
    })
    
    # 处理标注数据
    label_path = sub_packet / "lidar" / "label" / f"{timestamp}.json"
    try:
        with open(label_path, 'r') as f:
            label_data = json.load(f)
        
        objects = label_data.get("objects", []) if isinstance(label_data, dict) else label_data
        
        for obj in objects:
            tyjt_cat = obj.get("type", "other")
            nusc_cat = CATEGORY_MAPPING.get(tyjt_cat, "movable_object.debris")
            
            box3d = obj.get("box3d", [0, 0, 0, 1, 1, 1])
            x, y, z, l, w, h = (box3d + [0, 0, 0, 1, 1, 1])[:6]
            
            rotation = obj.get("rotation", [0, 0, 0])
            yaw = rotation[0] if len(rotation) > 0 else 0
            quaternion = quaternion_from_euler(yaw)
            
            position_key = f"{x:.2f}_{y:.2f}_{z:.2f}_{nusc_cat}"
            instance_key = f"{unique_key}_{position_key}"
            instance_token = id_gen.get_scoped_token("instance", instance_key)
            
            category_token = id_gen.get_scoped_token("category", nusc_cat)
            
            attribute_tokens = []
            if 'vehicle' in nusc_cat:
                attribute_tokens.append(id_gen.get_scoped_token("attribute", "vehicle.moving"))
            elif 'pedestrian' in nusc_cat:
                attribute_tokens.append(id_gen.get_scoped_token("attribute", "pedestrian.moving"))
            else:
                attribute_tokens.append(id_gen.get_scoped_token("attribute", "vehicle.stopped"))
            
            instance_exists = False
            for inst in nusc_data["instance"]:
                if inst["token"] == instance_token:
                    inst["nbr_annotations"] += 1
                    inst["last_annotation_token"] = id_gen.get_scoped_token("annotation", f"{instance_token}_{timestamp}")
                    instance_exists = True
                    break
            if not instance_exists:
                first_ann_token = id_gen.get_scoped_token("annotation", f"{instance_token}_{timestamp}")
                nusc_data["instance"].append({
                    "token": instance_token,
                    "category_token": category_token,
                    "nbr_annotations": 1,
                    "first_annotation_token": first_ann_token,
                    "last_annotation_token": first_ann_token
                })
            
            ann_token = id_gen.get_scoped_token("annotation", f"{instance_token}_{timestamp}")
            nusc_data["sample_annotation"].append({
                "token": ann_token,
                "sample_token": sample_token,
                "instance_token": instance_token,
                "visibility_token": id_gen.get_scoped_token("visibility", "full"),
                "attribute_tokens": attribute_tokens,
                "translation": [float(x), float(y), float(z)],
                "size": [float(w), float(l), float(h)],
                "rotation": quaternion,
                "category_name": nusc_cat,
                "num_lidar_pts": 10,
                "num_radar_pts": 0
            })
        
    except Exception as e:
        package_logger.error(f"处理标注失败 {label_path}: {e}")
    
    return sample_token

def process_samples_from_cache(cache_samples, package_dir, output_root, package_name, 
                              package_config, global_nusc_data, global_nusc_data_lock, package_logger):
    """从缓存处理样本"""
    # 按子包分组
    samples_by_subpacket = defaultdict(list)
    for sample_key in cache_samples:
        parts = sample_key.split('|')
        if len(parts) == 4:
            pkg_name, intersection_key, subpacket_name, timestamp = parts
            if pkg_name == package_name:
                samples_by_subpacket[subpacket_name].append((timestamp, intersection_key))
    
    all_sample_tokens = []
    
    for intersection_key in package_config["intersections"]:
        # 生成基础表
        calib_data = load_calibration(package_dir, package_config, package_logger)
        if not calib_data:
            continue
        
        with global_nusc_data_lock:
            token_map = generate_basic_tables(
                global_nusc_data, package_name, calib_data, package_config, 
                intersection_key, package_logger, global_nusc_data_lock
            )
        
        # 处理这个路口的所有样本
        sample_tokens = []
        prev_token = ""
        
        # 获取这个路口的所有样本
        intersection_samples = []
        for subpacket_name, samples in samples_by_subpacket.items():
            for timestamp, sample_intersection in samples:
                if sample_intersection == intersection_key:
                    intersection_samples.append((timestamp, subpacket_name))
        
        # 按时间戳排序
        intersection_samples.sort(key=lambda x: int(x[0]) if x[0].isdigit() else 0)
        
        start_time = time.time()
        for i, (timestamp, subpacket_name) in enumerate(intersection_samples):
            sub_packet = package_dir / "datasets" / subpacket_name
            
            sample_token = process_single_sample_fast(
                global_nusc_data, sub_packet, timestamp, output_root,
                package_name, intersection_key, subpacket_name,
                prev_token, i, token_map, package_config, package_logger
            )
            
            if sample_token:
                sample_tokens.append(sample_token)
                prev_token = sample_token
            
            # 进度报告
            if i % 100 == 0 and i > 0:
                elapsed = time.time() - start_time
                speed = i / elapsed if elapsed > 0 else 0
                package_logger.info(f"进度: {i}/{len(intersection_samples)} ({i/len(intersection_samples)*100:.1f}%), "
                                  f"速度: {speed:.1f} 样本/秒")
        
        all_sample_tokens.extend(sample_tokens)
        
        # 更新场景信息
        with global_nusc_data_lock:
            for scene in global_nusc_data["scene"]:
                if scene["token"] == token_map['scene_token']:
                    scene["nbr_samples"] = len(sample_tokens)
                    if sample_tokens:
                        scene["first_sample_token"] = sample_tokens[0]
                        scene["last_sample_token"] = sample_tokens[-1]
                    break
        
        package_logger.info(f"路口 {intersection_key} 处理完成: {len(sample_tokens)} 个样本")
    
    return all_sample_tokens

# ==================== 主处理函数 ====================
def process_package_with_cache(package_dir: Path, output_root: Path, 
                             cache_samples: List[str],
                             global_nusc_data_lock, global_nusc_data):
    """使用缓存处理数据包"""
    package_name = package_dir.name
    if package_name not in DATA_PACKAGE_CONFIG:
        return
    
    thread_name = threading.current_thread().name
    package_logger = get_package_logger(package_name, log_dir)
    package_logger.info(f"=== 开始处理数据包: {package_name} (使用缓存模式) ===")
    
    start_time = time.time()
    
    try:
        package_config = DATA_PACKAGE_CONFIG[package_name]
        
        # 处理样本
        sample_tokens = process_samples_from_cache(
            cache_samples, package_dir, output_root, package_name,
            package_config, global_nusc_data, global_nusc_data_lock, package_logger
        )
        
        elapsed_time = time.time() - start_time
        package_logger.info(f"=== 数据包 {package_name} 处理完成 ===")
        package_logger.info(f"总耗时: {elapsed_time:.2f}秒，处理样本: {len(sample_tokens)}")
        
        main_logger.info(f"✅ 数据包 {package_name} 处理完成，耗时: {elapsed_time:.2f}秒")
        
    except Exception as e:
        package_logger.error(f"处理数据包 {package_name} 时出错: {e}", exc_info=True)
        main_logger.error(f"❌ 数据包 {package_name} 处理失败: {e}")

def process_package_normal(package_dir: Path, output_root: Path, 
                         global_nusc_data_lock, global_nusc_data):
    """正常模式处理数据包（不使用缓存）"""
    package_name = package_dir.name
    if package_name not in DATA_PACKAGE_CONFIG:
        return
    
    thread_name = threading.current_thread().name
    package_logger = get_package_logger(package_name, log_dir)
    package_logger.info(f"=== 开始处理数据包: {package_name} (正常模式) ===")
    
    start_time = time.time()
    
    try:
        package_config = DATA_PACKAGE_CONFIG[package_name]
        package_logger.info(f"数据包类型: {package_config['type']}")
        
        calib_data = load_calibration(package_dir, package_config, package_logger)
        
        if not calib_data:
            package_logger.error(f"数据包 {package_name} 标定加载失败，跳过")
            return
        
        for intersection_key, intersection_config in package_config["intersections"].items():
            package_logger.info(f"--- 处理路口: {intersection_key} ---")
            
            with global_nusc_data_lock:
                token_map = generate_basic_tables(
                    global_nusc_data, package_name, calib_data, package_config, 
                    intersection_key, package_logger, global_nusc_data_lock
                )
            
            # 扫描样本（原逻辑）
            datasets_dir = package_dir / "datasets"
            if not datasets_dir.exists():
                continue
            
            all_samples = []
            sub_packets = list(datasets_dir.iterdir())
            
            for sub_packet in sub_packets:
                lidar_dir = sub_packet / "lidar" / "pcd"
                if not lidar_dir.exists():
                    continue
                
                lidar_npy_files = list(lidar_dir.glob("*.npy"))
                for lidar_npy in lidar_npy_files:
                    timestamp = lidar_npy.stem
                    if not timestamp.isdigit():
                        continue
                    
                    # 检查文件
                    label_path = sub_packet / "lidar" / "label" / f"{timestamp}.json"
                    if not label_path.exists():
                        continue
                    
                    all_samples.append((lidar_npy, timestamp, sub_packet))
            
            all_samples.sort(key=lambda x: int(x[1]) if x[1].isdigit() else 0)
            
            sample_tokens = []
            prev_token = ""
            start_time = time.time()
            
            for i, (lidar_npy, timestamp, sub_packet) in enumerate(all_samples):
                sample_token = process_single_sample_fast(
                    global_nusc_data, sub_packet, timestamp, output_root,
                    package_name, intersection_key, sub_packet.name,
                    prev_token, i, token_map, package_config, package_logger
                )
                
                if sample_token:
                    sample_tokens.append(sample_token)
                    prev_token = sample_token
                
                if i % 100 == 0 and i > 0:
                    elapsed = time.time() - start_time
                    speed = i / elapsed if elapsed > 0 else 0
                    package_logger.info(f"进度: {i}/{len(all_samples)} ({i/len(all_samples)*100:.1f}%), "
                                      f"速度: {speed:.1f} 样本/秒")
            
            # 更新场景信息
            with global_nusc_data_lock:
                for scene in global_nusc_data["scene"]:
                    if scene["token"] == token_map['scene_token']:
                        scene["nbr_samples"] = len(sample_tokens)
                        if sample_tokens:
                            scene["first_sample_token"] = sample_tokens[0]
                            scene["last_sample_token"] = sample_tokens[-1]
                        break
            
            package_logger.info(f"路口 {intersection_key} 处理完成: {len(sample_tokens)} 个样本")
        
        elapsed_time = time.time() - start_time
        package_logger.info(f"=== 数据包 {package_name} 处理完成 ===")
        package_logger.info(f"总耗时: {elapsed_time:.2f}秒")
        
        main_logger.info(f"✅ 数据包 {package_name} 处理完成，耗时: {elapsed_time:.2f}秒")
        
    except Exception as e:
        package_logger.error(f"处理数据包 {package_name} 时出错: {e}", exc_info=True)
        main_logger.error(f"❌ 数据包 {package_name} 处理失败: {e}")

# ==================== 主函数 ====================
def main():
    """主函数"""
    global main_logger, log_dir
    
    # 扫描模式
    if SCAN_ONLY_MODE:
        run_scan_only_mode()
        return
    
    main_logger, log_dir = setup_logging(OUTPUT_ROOT)
    
    main_logger.info("=" * 60)
    if USE_CACHE_MODE:
        main_logger.info("🚀 Step1: TYJT→NuScenes转换工具 v9.2（缓存模式）")
        main_logger.info(f"📋 使用缓存文件: {CACHE_FILE}")
    else:
        main_logger.info("🚀 Step1: TYJT→NuScenes转换工具 v9.2（正常模式）")
    
    main_logger.info(f"📥 原数据根目录: {TYJT_ROOT}")
    main_logger.info(f"📤 输出目录: {OUTPUT_ROOT}")
    main_logger.info(f"🧵 线程数: {PROCESS_THREADS}")
    main_logger.info("=" * 60)
    
    program_start_time = time.time()
    
    # 创建输出目录
    main_logger.info("创建输出目录结构...")
    output_base = create_directory_structure(OUTPUT_ROOT, NUSC_VERSION)
    main_logger.info(f"✅ 输出目录结构创建完成")
    
    # 查找数据包
    main_logger.info("查找数据包...")
    packages = find_data_packages(TYJT_ROOT)
    if not packages:
        main_logger.error("❌ 未找到任何有效数据包，退出")
        return
    
    main_logger.info(f"✅ 找到 {len(packages)} 个有效数据包")
    
    # 初始化数据结构
    global_nusc_data = initialize_nusc_data()
    global_nusc_data_lock = threading.RLock()
    
    # 多线程处理
    main_logger.info("开始多线程处理数据包...")
    package_start_time = time.time()
    
    if USE_CACHE_MODE:
        # 缓存模式：加载缓存文件
        cache_samples = load_cache_file()
        if not cache_samples:
            main_logger.error("❌ 缓存文件为空或加载失败")
            return
        
        # 按数据包分组
        samples_by_package = defaultdict(list)
        for sample_key in cache_samples:
            parts = sample_key.split('|')
            if len(parts) == 4:
                package_name = parts[0]
                samples_by_package[package_name].append(sample_key)
        
        # 只处理有缓存样本的数据包
        packages_to_process = [p for p in packages if p.name in samples_by_package]
        main_logger.info(f"📊 缓存中包含 {len(packages_to_process)} 个数据包的样本")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=PROCESS_THREADS, 
                                                 thread_name_prefix="CacheWorker") as executor:
            futures = []
            for package in packages_to_process:
                future = executor.submit(
                    process_package_with_cache, 
                    package, 
                    output_base, 
                    samples_by_package[package.name],
                    global_nusc_data_lock,
                    global_nusc_data
                )
                futures.append(future)
            
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                    completed += 1
                    main_logger.info(f"📊 进度: {completed}/{len(packages_to_process)} 个数据包处理完成")
                except Exception as e:
                    main_logger.error(f"数据包处理异常: {e}")
                    completed += 1
    else:
        # 正常模式
        with concurrent.futures.ThreadPoolExecutor(max_workers=PROCESS_THREADS, 
                                                 thread_name_prefix="PackageWorker") as executor:
            futures = []
            for package in packages:
                future = executor.submit(
                    process_package_normal, 
                    package, 
                    output_base, 
                    global_nusc_data_lock,
                    global_nusc_data
                )
                futures.append(future)
            
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                    completed += 1
                    main_logger.info(f"📊 进度: {completed}/{len(packages)} 个数据包处理完成")
                except Exception as e:
                    main_logger.error(f"数据包处理异常: {e}")
                    completed += 1
    
    package_elapsed_time = time.time() - package_start_time
    main_logger.info(f"✅ 所有数据包处理完成，总耗时: {package_elapsed_time:.2f}秒")
    
    # Token唯一性验证
    main_logger.info("验证Token唯一性...")
    token_unique, token_stats = id_gen.validate_token_uniqueness()
    if not token_unique:
        main_logger.warning("⚠️  警告: 存在重复Token！")
    else:
        main_logger.info("✅ Token唯一性验证通过")
    
    main_logger.info(f"📊 Token统计: {token_stats}")
    
    # 保存数据
    main_logger.info("保存数据表...")
    version_dir = output_base / NUSC_VERSION
    for table_name, table_data in global_nusc_data.items():
        if table_name.startswith('_'):
            continue
        output_path = version_dir / f"{table_name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(table_data, f, indent=2, ensure_ascii=False)
        main_logger.info(f"💾 保存 {table_name}.json: {len(table_data)} 条记录")
    
    # 输出统计信息
    total_elapsed_time = time.time() - program_start_time
    main_logger.info("=" * 60)
    main_logger.info("🎉 Step1 v9.2 转换完成!")
    main_logger.info(f"⏱️  总耗时: {total_elapsed_time:.2f}秒")
    
    stats = {
        "数据包数": len(packages),
        "场景数": len(global_nusc_data['scene']),
        "样本数": len(global_nusc_data['sample']),
        "样本数据数": len(global_nusc_data['sample_data']),
        "标注数": len(global_nusc_data['sample_annotation']),
        "实例数": len(global_nusc_data['instance']),
    }
    
    for key, value in stats.items():
        main_logger.info(f"  - {key}: {value}")
    
    main_logger.info(f"📂 日志文件目录: {log_dir}")
    main_logger.info("=" * 60)
    
    # 清理缓存
    fast_checker.clear_cache()

if __name__ == "__main__":
    main()