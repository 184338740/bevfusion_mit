#!/usr/bin/env python3
"""
tyjt数据集转nuscenes格式转换工具 - v9.2.2-phase2 (紧急修复版)
版本: v9.2.2-phase2-fixed
修改日期: 2024-12-XX
紧急修复标定计算问题
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union, Any
import hashlib
import math
from collections import defaultdict
import logging
import threading
import time
import sys
from dataclasses import dataclass, asdict

# ==================== 调试配置 ====================
DEBUG_MODE = True  # 设为False以获得最佳性能
MAX_SAMPLES_PER_PACKAGE = 10      # 0表示不限制
SKIP_FILE_CREATION = False     
# BATCH_SIZE = 100

# ==================== 配置 ====================
Mode = "Local"
if Mode == "Local":
    TYJT_ROOT = "/mnt/dataset/tyjt_RawData_all/"
    OUTPUT_ROOT = "./output-1204-v9.2.2/step1/nuscenes_tyjt"
    NUSC_VERSION = "v1.0-tyjt"
    ANALYSIS_DIR = Path(OUTPUT_ROOT) / "analysis"
    DATA_PACKAGE_CONFIG = {
        # "2d3d_20250114": {
        #     "type": "hikvision",
        #     "calib_root": "calib/2d3d_20250114",
        #     "group2map_file": "group2map_calib.json",
        #     "camera2map_file": "camera2map_calib.json",
        #     "intersections": {
        #         "R9": {
        #             "group_key": "G32050700009M00",
        #             "cameras": [
        #                 ("R9_Aw_CamS", "A"),
        #                 ("R9_Bn_CamW", "B"),
        #                 ("R9_Ce_CamN", "C"),
        #                 ("R9_Ds_CamE", "D")
        #             ]
        #         }
        #     }
        # },
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
elif Mode == "A100":
    TYJT_ROOT = "/cephfsdata/users/lishan/00_Data/00_RawData"
    OUTPUT_ROOT = "./output-1204-v9.2.2/step1/nuscenes_tyjt"
    NUSC_VERSION = "v1.0-tyjt"
    ANALYSIS_DIR = Path(OUTPUT_ROOT) / "analysis"
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

CAM_ORDER_TO_NUSC = {
    "A": "CAM_FRONT",
    "B": "CAM_FRONT_RIGHT", 
    "C": "CAM_BACK",
    "D": "CAM_FRONT_LEFT"
}

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

# ==================== 修复的数学工具函数 ====================

def quaternion_from_euler(yaw: float, pitch: float = 0, roll: float = 0) -> List[float]:
    """欧拉角转四元数"""
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
    """四元数转旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ], dtype=np.float32)

def rotation_matrix_to_quaternion(matrix: np.ndarray) -> List[float]:
    """旋转矩阵转四元数"""
    m = matrix.astype(np.float32)
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


def parse_tyjt_transform_fixed(calib_params: Union[Dict, List], logger=None) -> np.ndarray:
    """修复的标定参数解析 - 根据实际数据结构"""
    try:
        # 默认返回单位矩阵
        default_result = np.eye(4, dtype=np.float32)
        
        if logger and DEBUG_MODE:
            logger.debug(f"解析标定参数，类型: {type(calib_params)}")
        
        # 1. 处理字典格式（主要格式）
        if isinstance(calib_params, Dict):
            # 1.1 相机标定格式：有translation和rotation键
            if 'translation' in calib_params and 'rotation' in calib_params:
                try:
                    translation = np.array(calib_params['translation'], dtype=np.float32)
                    rotation_quat = calib_params['rotation']
                    
                    if len(rotation_quat) == 4:
                        # 根据实际数据：rotation是[x, y, z, w]
                        x, y, z, w = rotation_quat  # x, y, z, w顺序
                        
                        if logger and DEBUG_MODE:
                            logger.debug(f"解析相机标定:")
                            logger.debug(f"  平移: {translation}")
                            logger.debug(f"  旋转: [x={x:.6f}, y={y:.6f}, z={z:.6f}, w={w:.6f}]")
                        
                        # 计算旋转矩阵（输入为[w, x, y, z]）
                        rotation_matrix = quaternion_to_rotation_matrix([w, x, y, z])
                        
                        # 构建变换矩阵
                        transform = np.eye(4, dtype=np.float32)
                        transform[:3, :3] = rotation_matrix
                        transform[:3, 3] = translation
                        
                        return transform
                    else:
                        if logger:
                            logger.warning(f"旋转四元数长度错误: {len(rotation_quat)}")
                        return default_result
                        
                except Exception as e:
                    if logger:
                        logger.error(f"解析相机标定失败: {e}")
                    return default_result
            
            # 1.2 group2map的transform格式：有tx, ty, tz, rx, ry, rz, rw键
            elif all(key in calib_params for key in ['tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'rw']):
                try:
                    translation = np.array([
                        calib_params['tx'],
                        calib_params['ty'], 
                        calib_params['tz']
                    ], dtype=np.float32)
                    
                    # 注意：rx, ry, rz, rw 就是 x, y, z, w
                    x, y, z, w = (
                        calib_params['rx'],
                        calib_params['ry'], 
                        calib_params['rz'],
                        calib_params['rw']
                    )
                    
                    if logger and DEBUG_MODE:
                        logger.debug(f"解析transform字典:")
                        logger.debug(f"  平移: {translation}")
                        logger.debug(f"  旋转: [x={x:.6f}, y={y:.6f}, z={z:.6f}, w={w:.6f}]")
                    
                    # 计算旋转矩阵（输入为[w, x, y, z]）
                    rotation_matrix = quaternion_to_rotation_matrix([w, x, y, z])
                    
                    # 构建变换矩阵
                    transform = np.eye(4, dtype=np.float32)
                    transform[:3, :3] = rotation_matrix
                    transform[:3, 3] = translation
                    
                    return transform
                    
                except Exception as e:
                    if logger:
                        logger.error(f"解析transform字典失败: {e}")
                    return default_result
            
            # 1.3 其他不支持的字典格式
            else:
                if logger:
                    logger.warning(f"不支持的字典格式，可用键: {list(calib_params.keys())}")
                return default_result
        
        # 2. 处理列表格式（可能的历史格式）
        elif isinstance(calib_params, List):
            try:
                if len(calib_params) == 7:
                    # 假设格式: [tx, ty, tz, rx, ry, rz, rw]
                    tx, ty, tz, rx, ry, rz, rw = map(float, calib_params)
                    translation = np.array([tx, ty, tz], dtype=np.float32)
                    x, y, z, w = rx, ry, rz, rw
                    
                    # 计算旋转矩阵
                    rotation_matrix = quaternion_to_rotation_matrix([w, x, y, z])
                    transform = np.eye(4, dtype=np.float32)
                    transform[:3, :3] = rotation_matrix
                    transform[:3, 3] = translation
                    
                    return transform
                
                elif len(calib_params) == 16:
                    # 可能是4x4矩阵
                    transform = np.array(calib_params, dtype=np.float32).reshape(4, 4)
                    return transform
                
                else:
                    if logger:
                        logger.warning(f"不支持的列表长度: {len(calib_params)}")
                    return default_result
                    
            except Exception as e:
                if logger:
                    logger.error(f"解析列表标定失败: {e}")
                return default_result
        
        # 3. 其他类型
        else:
            if logger:
                logger.warning(f"不支持的标定参数类型: {type(calib_params)}")
            return default_result
            
    except Exception as e:
        if logger:
            logger.error(f"parse_tyjt_transform_fixed失败: {e}")
        return np.eye(4, dtype=np.float32)


def get_sensor2ego_transform(calib_data: Dict, sensor_calib_key: str, group2map_key: str, logger=None):
    """修复的sensor→ego转换 - 根据实际标定数据结构"""
    try:
        # 1. 检查传感器标定数据是否存在
        if sensor_calib_key not in calib_data:
            if logger:
                logger.error(f"标定数据中找不到传感器标定键: {sensor_calib_key}")
                logger.info(f"可用的传感器标定键: {[k for k in calib_data.keys() if k != group2map_key]}")
            return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        
        # 2. 检查group2map标定数据是否存在
        if group2map_key not in calib_data:
            if logger:
                logger.error(f"标定数据中找不到group2map键: {group2map_key}")
                logger.info(f"可用的标定键: {list(calib_data.keys())}")
            return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        
        # 3. 获取传感器标定数据
        sensor_calib_params = calib_data[sensor_calib_key]
        if logger and DEBUG_MODE:
            logger.debug(f"传感器 {sensor_calib_key} 标定数据:")
            if isinstance(sensor_calib_params, dict):
                logger.debug(f"  类型: dict, 键: {list(sensor_calib_params.keys())}")
                if 'translation' in sensor_calib_params:
                    logger.debug(f"  平移: {sensor_calib_params['translation']}")
                if 'rotation' in sensor_calib_params:
                    logger.debug(f"  旋转: {sensor_calib_params['rotation']}")
        
        # 4. 获取group2map标定数据
        group2map_params = calib_data[group2map_key]
        if logger and DEBUG_MODE:
            logger.debug(f"group2map {group2map_key} 标定数据:")
            if isinstance(group2map_params, dict):
                logger.debug(f"  类型: dict, 键: {list(group2map_params.keys())}")
                if 'transform' in group2map_params:
                    logger.debug(f"  包含'transform'键")
        
        # 5. 处理group2map数据结构
        # 根据日志：group2map_data有 ['ref', 'device_name', 'transform'] 键
        if isinstance(group2map_params, dict) and 'transform' in group2map_params:
            # 提取transform数据
            group2map_transform = group2map_params['transform']
            if logger and DEBUG_MODE:
                logger.debug(f"从group2map中提取transform数据，类型: {type(group2map_transform)}")
                if isinstance(group2map_transform, list):
                    logger.debug(f"  transform列表长度: {len(group2map_transform)}")
            
            # 使用transform数据
            actual_group2map_params = group2map_transform
        else:
            # 如果没有transform键，直接使用原始数据
            actual_group2map_params = group2map_params
        
        # 6. 解析变换矩阵
        sensor2map = parse_tyjt_transform_fixed(sensor_calib_params, logger)
        group2map = parse_tyjt_transform_fixed(actual_group2map_params, logger)
        
        if logger and DEBUG_MODE:
            logger.debug(f"sensor2map矩阵:\n{sensor2map}")
            logger.debug(f"group2map矩阵:\n{group2map}")
        
        # 7. 计算逆矩阵 (map2group = group2map的逆)
        try:
            map2group = np.linalg.inv(group2map)
            if logger and DEBUG_MODE:
                logger.debug(f"map2group逆矩阵:\n{map2group}")
        except np.linalg.LinAlgError as e:
            if logger:
                logger.error(f"计算group2map逆矩阵失败: {e}")
                logger.error(f"group2map矩阵:\n{group2map}")
                logger.error(f"矩阵行列式: {np.linalg.det(group2map)}")
            return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        
        # 8. 计算sensor→ego变换: map2group @ sensor2map
        sensor2ego = map2group @ sensor2map
        
        # 9. 验证变换矩阵
        if not np.isfinite(sensor2ego).all():
            if logger:
                logger.error(f"无效的sensor2ego变换矩阵（包含非有限值）")
                logger.error(f"sensor2ego矩阵:\n{sensor2ego}")
            return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        
        # 10. 提取平移和旋转
        translation = sensor2ego[:3, 3].tolist()
        rotation_matrix = sensor2ego[:3, :3]
        rotation = rotation_matrix_to_quaternion(rotation_matrix)
        
        # 11. 验证是否为刚性变换
        det_rotation = np.linalg.det(rotation_matrix)
        if abs(det_rotation - 1.0) > 0.01:
            if logger:
                logger.warning(f"旋转矩阵行列式异常: {det_rotation:.6f} (应该接近1.0)")
        
        # 12. 计算相机到ego的距离
        distance = math.sqrt(sum(t*t for t in translation))
        
        if logger:
            logger.info(f"✅ 标定转换完成: {sensor_calib_key} → ego")
            logger.info(f"   平移: [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}]")
            logger.info(f"   距离: {distance:.2f}m")
            logger.info(f"   旋转四元数: [{rotation[0]:.4f}, {rotation[1]:.4f}, {rotation[2]:.4f}, {rotation[3]:.4f}]")
            
            # 检查距离是否合理
            if distance < 1.0:
                logger.warning(f"⚠️  相机距离异常近: {distance:.2f}m")
            elif distance > 100.0:
                logger.warning(f"⚠️  相机距离异常远: {distance:.2f}m")
        
        return translation, rotation
        
    except Exception as e:
        if logger:
            logger.error(f"计算sensor2ego变换时出错: {e}", exc_info=True)
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]

# ==================== Token管理器 ====================

class EnhancedIDGenerator:
    def __init__(self):
        self.uuid_map = {}
        self.used_tokens: Set[str] = set()
        self.entity_counters = defaultdict(int)
        self.lock = threading.Lock()
    
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

# ==================== 数据结构定义 ====================

@dataclass
class ValidatedSample:
    package_name: str
    intersection_key: str
    sub_packet_name: str
    timestamp: str
    files: Dict[str, Any]
    metadata: Dict[str, Any]
    unique_key: str = ""
    
    def __post_init__(self):
        if not self.unique_key:
            self.unique_key = f"{self.package_name}_{self.intersection_key}_{self.sub_packet_name}_{self.timestamp}"

@dataclass
class PackageConfig:
    name: str
    type: str
    calib_root: str
    config: Dict[str, Any]
    
    @property
    def intersections(self) -> Dict[str, Any]:
        return self.config.get("intersections", {})

# ==================== 修复的DataProcessor ====================

class DataProcessor:
    def __init__(self, output_root: str, analysis_dir: Path):
        self.output_root = Path(output_root)
        self.analysis_dir = analysis_dir
        self.id_gen = EnhancedIDGenerator()
        
        # 创建输出目录
        self.create_output_directories()
        
        # 设置日志
        self.setup_logging()
        
        # 缓存
        self.calib_cache: Dict[str, Dict] = {}
        
        # NuScenes数据结构
        self.nusc_data = self.initialize_nusc_data()
        self.nusc_data_lock = threading.Lock()
        
        # 索引
        self.sensor_index: Dict[str, str] = {}
        self.category_index: Dict[str, str] = {}
        
        self.logger.info("DataProcessor初始化完成")
    
    def setup_logging(self):
        """设置日志系统"""
        log_dir = self.output_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("processor")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            log_file = log_dir / "phase2_processing.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            console_handler = logging.StreamHandler()
            
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'  # 简化时间格式
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def create_output_directories(self):
        """创建输出目录结构"""
        base_path = self.output_root
        base_path.mkdir(parents=True, exist_ok=True)
        
        version_dir = base_path / NUSC_VERSION
        version_dir.mkdir(parents=True, exist_ok=True)
        
        for data_type in ["samples", "sweeps"]:
            root_dir = base_path / data_type
            root_dir.mkdir(parents=True, exist_ok=True)
            for nusc_cam in CAM_ORDER_TO_NUSC.values():
                (root_dir / nusc_cam).mkdir(parents=True, exist_ok=True)
            (root_dir / LIDAR_NUSC_NAME).mkdir(parents=True, exist_ok=True)
        
        maps_dir = base_path / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建空地图文件
        try:
            from PIL import Image
            empty_img = Image.new('RGB', (1, 1), color='black')
            empty_img.save(maps_dir / "empty_map.png")
        except:
            # 如果PIL不可用，创建空文件
            (maps_dir / "empty_map.png").write_bytes(b'')
    
    def initialize_nusc_data(self) -> Dict:
        return {
            "category": [], "attribute": [], "sensor": [], "calibrated_sensor": [],
            "ego_pose": [], "log": [], "scene": [], "sample": [], "sample_data": [],
            "sample_annotation": [], "instance": [], "visibility": [], "map": []
        }
    
    def load_calibration(self, package_dir: Path, package_config: PackageConfig) -> Dict:
        """加载标定数据"""
        cache_key = str(package_dir)
        if cache_key in self.calib_cache:
            return self.calib_cache[cache_key]
        
        calib_root = package_dir / package_config.calib_root
        calib_data = {}
        
        try:
            if package_config.type == "hikvision":
                group2map_file = package_config.config.get("group2map_file", "group2map_calib.json")
                camera2map_file = package_config.config.get("camera2map_file", "camera2map_calib.json")
                
                group2map_path = calib_root / group2map_file
                if group2map_path.exists():
                    with open(group2map_path, 'r') as f:
                        group_data = json.load(f)
                    calib_data.update(group_data)
                
                camera2map_path = calib_root / camera2map_file
                if camera2map_path.exists():
                    with open(camera2map_path, 'r') as f:
                        camera_data = json.load(f)
                    calib_data.update(camera_data)
            
            elif package_config.type == "all_in_one":
                sensor2map_file = package_config.config.get("sensor2map_file", "sensor2map_calib.json")
                sensor2map_path = calib_root / sensor2map_file
                if sensor2map_path.exists():
                    with open(sensor2map_path, 'r') as f:
                        sensor_data = json.load(f)
                    calib_data.update(sensor_data)
            
            self.calib_cache[cache_key] = calib_data
            self.logger.info(f"✅ 加载标定数据: {package_dir.name}, {len(calib_data)} 个键")
            
        except Exception as e:
            self.logger.error(f"❌ 加载标定数据失败: {e}")
        
        return calib_data
    
    def load_analysis_results(self) -> List[ValidatedSample]:
        """加载阶段1的分析结果"""
        valid_samples = []
        
        valid_sample_files = list(self.analysis_dir.glob("*_valid_samples.json"))
        
        if not valid_sample_files:
            self.logger.error(f"未找到有效样本文件，请先运行阶段1分析")
            return []
        
        self.logger.info(f"找到 {len(valid_sample_files)} 个有效样本文件")
        
        for file_path in valid_sample_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    samples_data = json.load(f)
                
                for sample_data in samples_data:
                    sample = ValidatedSample(**sample_data)
                    valid_samples.append(sample)
                
                self.logger.info(f"加载 {file_path.name}: {len(samples_data)} 个样本")
                
            except Exception as e:
                self.logger.error(f"加载样本文件失败 {file_path}: {e}")
        
        valid_samples.sort(key=lambda x: (
            x.package_name, 
            x.intersection_key,
            int(x.metadata.get("timestamp_int", 0))
        ))
        
        self.logger.info(f"总共加载 {len(valid_samples)} 个有效样本")
        return valid_samples
    

    def initialize_base_tables(self, package_name: str, intersection_key: str, 
                            package_config: PackageConfig, calib_data: Dict) -> Dict:
        """初始化基础数据表 - 根据实际标定数据结构修复"""
        self.logger.info(f"开始初始化路口 {intersection_key} 的基础数据表...")
        
        # 1. 详细日志标定数据
        self.logger.info(f"📊 标定数据键 ({len(calib_data)}): {list(calib_data.keys())}")
        
        token_map = {}
        intersection_config = package_config.intersections.get(intersection_key, {})
        self.logger.info(f"📊 路口配置: {intersection_config}")
        
        # 2. 检查group2map_key
        group2map_key = intersection_config.get("group_key", "")
        self.logger.info(f"🔍 group2map_key: '{group2map_key}'")
        
        if not group2map_key:
            self.logger.error(f"❌ 路口 {intersection_key} 缺少group_key配置")
            return token_map
        
        if group2map_key not in calib_data:
            self.logger.error(f"❌ group2map_key '{group2map_key}' 不在标定数据中")
            self.logger.error(f"  可用标定键: {list(calib_data.keys())}")
            return token_map
        
        # 3. 检查group2map数据结构
        group2map_data = calib_data[group2map_key]
        self.logger.info(f"📊 group2map_data类型: {type(group2map_data)}")
        
        if isinstance(group2map_data, dict):
            self.logger.info(f"📊 group2map_data键: {list(group2map_data.keys())}")
            if 'transform' in group2map_data:
                transform_data = group2map_data['transform']
                self.logger.info(f"📊 找到'transform'键，类型: {type(transform_data)}")
                if isinstance(transform_data, dict):
                    self.logger.info(f"📊 transform键: {list(transform_data.keys())}")
        else:
            self.logger.error(f"❌ group2map_data不是字典格式")
            return token_map
        
        # 4. log表
        self.logger.info("创建log表...")
        log_unique_key = f"{package_config.type}_{package_name}_{intersection_key}"
        log_token = self.id_gen.get_scoped_token("log", log_unique_key)
        token_map['log_token'] = log_token
        
        with self.nusc_data_lock:
            self.nusc_data["log"].append({
                "token": log_token,
                "logfile": f"{log_unique_key}.log",
                "vehicle": "tyjt_system",
                "date_captured": package_name.split('_')[-1] if '_' in package_name else "2024-01-01",
                "location": package_name.split('_')[-1] if package_name.endswith(('wuxi', 'weiyuan')) else "default"
            })
        
        self.logger.info(f"✅ log表创建完成，token: {log_token[:20]}...")
        
        # 5. scene表
        self.logger.info("创建scene表...")
        scene_unique_key = f"{package_name}_{intersection_key}"
        scene_token = self.id_gen.get_scoped_token("scene", scene_unique_key)
        token_map['scene_token'] = scene_token
        
        with self.nusc_data_lock:
            self.nusc_data["scene"].append({
                "token": scene_token,
                "name": scene_unique_key,
                "description": f"TYJT: {package_name} - {intersection_key}",
                "log_token": log_token,
                "nbr_samples": 0,
                "first_sample_token": "",
                "last_sample_token": ""
            })
        
        self.logger.info(f"✅ scene表创建完成，token: {scene_token[:20]}...")
        
        # 6. category表
        if not self.nusc_data["category"]:
            self.logger.info("初始化category表...")
            for tyjt_cat, nusc_cat in CATEGORY_MAPPING.items():
                category_token = self.id_gen.get_scoped_token("category", nusc_cat)
                with self.nusc_data_lock:
                    self.nusc_data["category"].append({
                        "token": category_token,
                        "name": nusc_cat,
                        "description": f"From tyjt: {tyjt_cat}"
                    })
                self.category_index[nusc_cat] = category_token
            self.logger.info(f"✅ 添加 {len(CATEGORY_MAPPING)} 个类别")
        
        # 7. attribute表
        if not self.nusc_data["attribute"]:
            self.logger.info("初始化attribute表...")
            for attr_name in ["vehicle.moving", "pedestrian.moving", "vehicle.stopped"]:
                attr_token = self.id_gen.get_scoped_token("attribute", attr_name)
                with self.nusc_data_lock:
                    self.nusc_data["attribute"].append({
                        "token": attr_token,
                        "name": attr_name,
                        "description": f"{attr_name.replace('.', ' is ')}"
                    })
            self.logger.info("✅ attribute表创建完成")
        
        # 8. visibility表
        if not self.nusc_data["visibility"]:
            vis_token = self.id_gen.get_scoped_token("visibility", "full")
            with self.nusc_data_lock:
                self.nusc_data["visibility"].append({
                    "token": vis_token,
                    "level": "full",
                    "description": "Fully visible"
                })
            self.logger.info("✅ visibility表创建完成")
        
        # 9. sensor表
        self.logger.info("创建sensor表...")
        sensor_tokens = {}
        
        # 相机传感器
        camera_count = 0
        cameras = intersection_config.get("cameras", [])
        
        self.logger.info(f"处理 {len(cameras)} 个相机...")
        
        for cam_info in cameras:
            if package_config.type == "hikvision":
                camera_folder, camera_order = cam_info
            else:
                camera_folder, calibration_key, camera_order = cam_info
            
            nusc_cam_name = CAM_ORDER_TO_NUSC.get(camera_order)
            if not nusc_cam_name:
                self.logger.warning(f"跳过无效相机顺序: {camera_order} (相机: {camera_folder})")
                continue
            
            # 检查是否已存在
            if nusc_cam_name not in self.sensor_index:
                sensor_token = self.id_gen.get_scoped_token("sensor", nusc_cam_name)
                with self.nusc_data_lock:
                    self.nusc_data["sensor"].append({
                        "token": sensor_token,
                        "channel": nusc_cam_name,
                        "modality": "camera"
                    })
                self.sensor_index[nusc_cam_name] = sensor_token
                sensor_tokens[nusc_cam_name] = sensor_token
                camera_count += 1
                self.logger.debug(f"  添加相机传感器: {nusc_cam_name}, token: {sensor_token[:20]}...")
            else:
                sensor_tokens[nusc_cam_name] = self.sensor_index[nusc_cam_name]
        
        self.logger.info(f"✅ 添加 {camera_count} 个相机传感器")
        
        # 激光雷达传感器
        if LIDAR_NUSC_NAME not in self.sensor_index:
            lidar_sensor_token = self.id_gen.get_scoped_token("sensor", LIDAR_NUSC_NAME)
            with self.nusc_data_lock:
                self.nusc_data["sensor"].append({
                    "token": lidar_sensor_token,
                    "channel": LIDAR_NUSC_NAME,
                    "modality": "lidar"
                })
            self.sensor_index[LIDAR_NUSC_NAME] = lidar_sensor_token
            sensor_tokens[LIDAR_NUSC_NAME] = lidar_sensor_token
            self.logger.debug(f"  添加LiDAR传感器: {LIDAR_NUSC_NAME}, token: {lidar_sensor_token[:20]}...")
        else:
            sensor_tokens[LIDAR_NUSC_NAME] = self.sensor_index[LIDAR_NUSC_NAME]
        
        token_map['sensor_tokens'] = sensor_tokens
        
        # 10. calibrated_sensor表 - 根据实际数据结构修复
        self.logger.info("创建calibrated_sensor表...")
        calib_sensor_tokens = {}
        calibrated_camera_count = 0
        
        for cam_info in cameras:
            if package_config.type == "hikvision":
                camera_folder, camera_order = cam_info
                sensor_calib_key = camera_folder
            else:
                camera_folder, sensor_calib_key, camera_order = cam_info
            
            nusc_cam_name = CAM_ORDER_TO_NUSC.get(camera_order)
            if not nusc_cam_name or nusc_cam_name not in sensor_tokens:
                self.logger.warning(f"跳过无效相机: {camera_folder}")
                continue
            
            # 使用修复的标定转换函数
            self.logger.info(f"🔧 计算相机 {camera_folder} 标定...")
            translation, rotation = get_sensor2ego_transform_fixed(
                calib_data, sensor_calib_key, group2map_key, self.logger
            )
            
            self.logger.debug(f"  相机 {camera_folder} → {nusc_cam_name}:")
            self.logger.debug(f"    平移: {translation}")
            self.logger.debug(f"    旋转: {rotation}")
            
            # 相机内参处理 - 根据实际数据结构修复
            cam_intrinsic = []
            
            # 检查是否有自定义内参
            if sensor_calib_key in calib_data and isinstance(calib_data[sensor_calib_key], Dict):
                cam_calib = calib_data[sensor_calib_key]
                try:
                    # 根据实际数据：相机有完整的fx, fy, cx, cy
                    if all(key in cam_calib for key in ['fx', 'fy', 'cx', 'cy']):
                        fx = float(cam_calib['fx'])
                        fy = float(cam_calib['fy'])
                        cx = float(cam_calib['cx'])
                        cy = float(cam_calib['cy'])
                        
                        cam_intrinsic = [
                            [fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]
                        ]
                        
                        if DEBUG_MODE:
                            self.logger.debug(f"  使用内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
                    else:
                        missing_keys = [k for k in ['fx', 'fy', 'cx', 'cy'] if k not in cam_calib]
                        self.logger.error(f"  缺少内参参数 {missing_keys}")
                        
                except (ValueError, TypeError, KeyError) as e:
                    self.logger.error(f"  内参解析失败: {e}")
            
            # 确保内参不为空
            if not cam_intrinsic:
                self.logger.warning(f"  内参为空，使用NuScenes时将无法投影")
            
            calib_unique_key = f"{scene_unique_key}_{nusc_cam_name}"
            calib_token = self.id_gen.get_scoped_token("calibrated_sensor", calib_unique_key)
            calib_sensor_tokens[nusc_cam_name] = calib_token
            
            with self.nusc_data_lock:
                self.nusc_data["calibrated_sensor"].append({
                    "token": calib_token,
                    "sensor_token": sensor_tokens[nusc_cam_name],
                    "translation": translation,
                    "rotation": rotation,
                    "camera_intrinsic": cam_intrinsic
                })
            
            calibrated_camera_count += 1
            distance = math.sqrt(sum(t*t for t in translation))
            self.logger.info(f"✅ 相机 {camera_folder} → {nusc_cam_name}: 距离={distance:.2f}m")
        
        self.logger.info(f"✅ 完成 {calibrated_camera_count} 个相机标定")
        
        # 激光雷达标定
        self.logger.info("创建LiDAR标定...")
        lidar_calib_unique_key = f"{scene_unique_key}_{LIDAR_NUSC_NAME}"
        lidar_calib_token = self.id_gen.get_scoped_token("calibrated_sensor", lidar_calib_unique_key)
        calib_sensor_tokens[LIDAR_NUSC_NAME] = lidar_calib_token
        
        with self.nusc_data_lock:
            self.nusc_data["calibrated_sensor"].append({
                "token": lidar_calib_token,
                "sensor_token": sensor_tokens[LIDAR_NUSC_NAME],
                "translation": [0.0, 0.0, 0.0],  # ✅ tyjt的lidar点云原点就是ego原点
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "camera_intrinsic": []
            })
        
        self.logger.info(f"✅ LiDAR标定完成")
        
        token_map['calib_sensor_tokens'] = calib_sensor_tokens
        
        # 11. ego_pose表
        self.logger.info("创建ego_pose表...")
        ego_pose_unique_key = f"{scene_unique_key}_ego"
        ego_pose_token = self.id_gen.get_scoped_token("ego_pose", ego_pose_unique_key)
        token_map['ego_pose_token'] = ego_pose_token
        
        with self.nusc_data_lock:
            self.nusc_data["ego_pose"].append({
                "token": ego_pose_token,
                "translation": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "timestamp": 0
            })
        
        self.logger.info(f"✅ ego_pose表创建完成")
        
        # 12. map表
        self.logger.info("创建map表...")
        with self.nusc_data_lock:
            if not self.nusc_data["map"]:
                map_token = self.id_gen.get_scoped_token("map", "tyjt_empty")
                self.nusc_data["map"].append({
                    "token": map_token,
                    "log_tokens": [log_token],
                    "category": "city",
                    "filename": "maps/empty_map.png",
                    "map_name": "tyjt_empty"
                })
                token_map['map_token'] = map_token
                self.logger.info(f"✅ 创建新地图表")
            else:
                map_record = self.nusc_data["map"][0]
                current_log_tokens = set(map_record.get('log_tokens', []))
                current_log_tokens.add(log_token)
                map_record['log_tokens'] = list(current_log_tokens)
                token_map['map_token'] = map_record['token']
                self.logger.info(f"✅ 更新地图表，当前log_tokens: {len(map_record['log_tokens'])}")
        
        self.logger.info(f"✅ 基础数据表生成完成")
        return token_map


    def npy_to_bin(self, npy_path: Path, bin_path: Path) -> bool:
        """npy→bin转换"""
        try:
            if not npy_path.exists():
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
            self.logger.error(f"点云转换失败 {npy_path}: {e}")
            return False
    
    def process_package_samples(self, package_samples: List[ValidatedSample], 
                              package_config: PackageConfig, package_dir: Path):
        """处理单个数据包的所有样本"""
        # 按路口分组
        samples_by_intersection = defaultdict(list)
        for sample in package_samples:
            samples_by_intersection[sample.intersection_key].append(sample)
        
        # 处理每个路口
        for intersection_key, samples in samples_by_intersection.items():
            self.logger.info(f"处理路口 {intersection_key}: {len(samples)} 个样本")
            
            # 加载标定数据
            calib_data = self.load_calibration(package_dir, package_config)
            if not calib_data:
                self.logger.error(f"路口 {intersection_key} 标定数据加载失败")
                continue
            
            # 初始化基础数据表
            with self.nusc_data_lock:
                token_map = self.initialize_base_tables(
                    package_config.name, intersection_key, package_config, calib_data
                )
            
            # 按时间戳排序
            samples.sort(key=lambda x: x.metadata.get("timestamp_int", 0))
            
            # 处理样本
            sample_tokens = []
            prev_token = ""
            
            for i, sample in enumerate(samples):
                # 创建样本记录
                sample_token = self.id_gen.get_scoped_token("sample", sample.unique_key)
                
                sample_record = {
                    "token": sample_token,
                    "timestamp": sample.metadata.get("timestamp_int", 0),
                    "scene_token": token_map['scene_token'],
                    "prev": prev_token,
                    "next": ""
                }
                
                # 更新前一个样本的next指针
                if prev_token:
                    for s in self.nusc_data["sample"]:
                        if s["token"] == prev_token:
                            s["next"] = sample_token
                            break
                
                with self.nusc_data_lock:
                    self.nusc_data["sample"].append(sample_record)
                
                # 处理传感器数据
                self.process_sample_sensors(sample, sample_token, token_map)
                
                # 处理标注数据
                self.process_sample_annotations(sample, sample_token)
                
                sample_tokens.append(sample_token)
                prev_token = sample_token
                
                # 进度显示
                if (i + 1) % 100 == 0:
                    self.logger.info(f"进度: {i+1}/{len(samples)} 个样本")
            
            # 更新scene表的样本数量
            with self.nusc_data_lock:
                for scene in self.nusc_data["scene"]:
                    if scene["token"] == token_map['scene_token']:
                        scene["nbr_samples"] = len(sample_tokens)
                        if sample_tokens:
                            scene["first_sample_token"] = sample_tokens[0]
                            scene["last_sample_token"] = sample_tokens[-1]
                        break
            
            self.logger.info(f"✅ 完成路口 {intersection_key}: {len(sample_tokens)} 个样本")
    
    def process_sample_sensors(self, sample: ValidatedSample, sample_token: str, token_map: Dict):
        """处理样本的传感器数据"""
        sensor_tokens = token_map['sensor_tokens']
        calib_sensor_tokens = token_map['calib_sensor_tokens']
        ego_pose_token = token_map['ego_pose_token']
        
        # 相机图像处理
        camera_data = sample.files.get("cameras", {})
        
        for camera_order in ["A", "B", "C", "D"]:
            if camera_order not in camera_data:
                continue
            
            camera_info = camera_data[camera_order]
            nusc_cam_name = CAM_ORDER_TO_NUSC.get(camera_order)
            if not nusc_cam_name or nusc_cam_name not in sensor_tokens:
                continue
            
            img_original_path = Path(camera_info["image_path"])
            if not img_original_path.exists():
                continue
            
            # 创建输出目录和软链接
            img_output_dir = self.output_root / "samples" / nusc_cam_name
            img_output_dir.mkdir(parents=True, exist_ok=True)
            img_output_path = img_output_dir / f"{sample.unique_key}.jpg"
            
            if not SKIP_FILE_CREATION:
                try:
                    if img_output_path.exists():
                        try:
                            img_output_path.unlink()
                        except:
                            pass
                    
                    target_path = img_original_path.resolve()
                    if target_path.exists():
                        img_output_path.symlink_to(target_path)
                except Exception as e:
                    self.logger.error(f"创建软链接失败: {e}")
            
            # 创建sample_data记录
            sample_data_token = self.id_gen.get_scoped_token("sample_data", f"{sample.unique_key}_{nusc_cam_name}")
            
            with self.nusc_data_lock:
                self.nusc_data["sample_data"].append({
                    "token": sample_data_token,
                    "sample_token": sample_token,
                    "ego_pose_token": ego_pose_token,
                    "calibrated_sensor_token": calib_sensor_tokens[nusc_cam_name],
                    "filename": f"samples/{nusc_cam_name}/{sample.unique_key}.jpg",
                    "fileformat": "jpg",
                    "width": 1920,
                    "height": 1080,
                    "timestamp": sample.metadata.get("timestamp_int", 0),
                    "is_key_frame": True,
                    "next": "",
                    "prev": ""
                })
        
        # 激光雷达处理
        lidar_npy_path = Path(sample.files.get("lidar", ""))
        if lidar_npy_path.exists() and lidar_npy_path.is_file():
            lidar_output_dir = self.output_root / "samples" / LIDAR_NUSC_NAME
            lidar_output_dir.mkdir(parents=True, exist_ok=True)
            lidar_output_path = lidar_output_dir / f"{sample.unique_key}.bin"
            
            if not SKIP_FILE_CREATION:
                if not lidar_output_path.exists():
                    if self.npy_to_bin(lidar_npy_path, lidar_output_path):
                        pass
            
            # 创建sample_data记录
            lidar_sample_data_token = self.id_gen.get_scoped_token("sample_data", f"{sample.unique_key}_{LIDAR_NUSC_NAME}")
            
            with self.nusc_data_lock:
                self.nusc_data["sample_data"].append({
                    "token": lidar_sample_data_token,
                    "sample_token": sample_token,
                    "ego_pose_token": ego_pose_token,
                    "calibrated_sensor_token": calib_sensor_tokens[LIDAR_NUSC_NAME],
                    "filename": f"samples/{LIDAR_NUSC_NAME}/{sample.unique_key}.bin",
                    "fileformat": "bin",
                    "width": 0,
                    "height": 0,
                    "timestamp": sample.metadata.get("timestamp_int", 0),
                    "is_key_frame": True,
                    "next": "",
                    "prev": ""
                })
    
    def process_sample_annotations(self, sample: ValidatedSample, sample_token: str):
        """处理样本的标注数据"""
        label_path_str = sample.files.get("label", "")
        
        if not label_path_str or not Path(label_path_str).exists():
            return
        
        label_path = Path(label_path_str)
        
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            if isinstance(label_data, dict):
                objects = label_data.get("objects", [])
            else:
                objects = label_data
            
            if not isinstance(objects, list):
                return
            
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                
                tyjt_cat = obj.get("type", "other")
                nusc_cat = CATEGORY_MAPPING.get(tyjt_cat, "movable_object.debris")
                
                box3d = obj.get("box3d", [0, 0, 0, 1, 1, 1])
                box3d_extended = box3d + [0, 0, 0, 1, 1, 1]
                x, y, z, l, w, h = box3d_extended[:6]
                
                rotation = obj.get("rotation", [0, 0, 0])
                yaw = rotation[0] if len(rotation) > 0 else 0
                quaternion = quaternion_from_euler(yaw)
                
                position_key = f"{x:.2f}_{y:.2f}_{z:.2f}_{nusc_cat}"
                instance_key = f"{sample.unique_key}_{position_key}"
                instance_token = self.id_gen.get_scoped_token("instance", instance_key)
                
                category_token = self.category_index.get(nusc_cat)
                if not category_token:
                    category_token = self.id_gen.get_scoped_token("category", nusc_cat)
                    with self.nusc_data_lock:
                        self.nusc_data["category"].append({
                            "token": category_token,
                            "name": nusc_cat,
                            "description": f"From tyjt: {tyjt_cat}"
                        })
                    self.category_index[nusc_cat] = category_token
                
                # 检查instance是否已存在
                instance_exists = False
                with self.nusc_data_lock:
                    for inst in self.nusc_data["instance"]:
                        if inst["token"] == instance_token:
                            inst["nbr_annotations"] += 1
                            inst["last_annotation_token"] = self.id_gen.get_scoped_token("annotation", f"{instance_token}_{sample.timestamp}")
                            instance_exists = True
                            break
                
                if not instance_exists:
                    first_ann_token = self.id_gen.get_scoped_token("annotation", f"{instance_token}_{sample.timestamp}")
                    with self.nusc_data_lock:
                        self.nusc_data["instance"].append({
                            "token": instance_token,
                            "category_token": category_token,
                            "nbr_annotations": 1,
                            "first_annotation_token": first_ann_token,
                            "last_annotation_token": first_ann_token
                        })
                
                # 生成annotation记录
                ann_token = self.id_gen.get_scoped_token("annotation", f"{instance_token}_{sample.timestamp}")
                
                attribute_tokens = []
                if 'vehicle' in nusc_cat:
                    attribute_tokens.append(self.id_gen.get_scoped_token("attribute", "vehicle.moving"))
                elif 'pedestrian' in nusc_cat:
                    attribute_tokens.append(self.id_gen.get_scoped_token("attribute", "pedestrian.moving"))
                else:
                    attribute_tokens.append(self.id_gen.get_scoped_token("attribute", "vehicle.stopped"))
                
                with self.nusc_data_lock:
                    self.nusc_data["sample_annotation"].append({
                        "token": ann_token,
                        "sample_token": sample_token,
                        "instance_token": instance_token,
                        "visibility_token": self.id_gen.get_scoped_token("visibility", "full"),
                        "attribute_tokens": attribute_tokens,
                        "translation": [float(x), float(y), float(z)],
                        "size": [float(w), float(l), float(h)],
                        "rotation": quaternion,
                        "category_name": nusc_cat,
                        "num_lidar_pts": 10,
                        "num_radar_pts": 0
                    })
            
        except Exception as e:
            self.logger.error(f"处理标注失败 {label_path}: {e}")
    
    def save_nusc_data(self):
        """保存NuScenes数据"""
        version_dir = self.output_root / NUSC_VERSION
        
        for table_name, table_data in self.nusc_data.items():
            if table_name.startswith('_'):
                continue
            
            output_path = version_dir / f"{table_name}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(table_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 保存 {table_name}.json: {len(table_data)} 条记录")
    
    def validate_data(self):
        """数据验证"""
        token_unique, token_stats = self.id_gen.validate_token_uniqueness()
        if not token_unique:
            self.logger.warning("⚠️  警告: 存在重复Token！")
        else:
            self.logger.info("✅ Token唯一性验证通过")
        
        self.logger.info(f"📊 Token统计: {token_stats}")
        
        stats = {
            "场景数": len(self.nusc_data['scene']),
            "样本数": len(self.nusc_data['sample']),
            "样本数据数": len(self.nusc_data['sample_data']),
            "标注数": len(self.nusc_data['sample_annotation']),
            "实例数": len(self.nusc_data['instance']),
            "类别数": len(self.nusc_data['category']),
            "传感器数": len(self.nusc_data['sensor']),
        }
        
        for key, value in stats.items():
            self.logger.info(f"  - {key}: {value}")

def main():
    """主函数"""
    processor = DataProcessor(OUTPUT_ROOT, ANALYSIS_DIR)
    
    processor.logger.info("=" * 60)
    processor.logger.info(f"🚀 TYJT数据集转换工具 v9.2.2-phase2 (修复版)")
    processor.logger.info(f"📊 调试模式: MAX_SAMPLES_PER_PACKAGE={MAX_SAMPLES_PER_PACKAGE}")
    processor.logger.info("=" * 60)
    
    # 1. 加载阶段1的分析结果
    all_samples = processor.load_analysis_results()
    
    if not all_samples:
        processor.logger.error("❌ 无有效样本，退出")
        return
    
    # 2. 应用样本限制
    if MAX_SAMPLES_PER_PACKAGE > 0:
        processor.logger.info(f"⚠️ 调试模式: 每个数据包只处理 {MAX_SAMPLES_PER_PACKAGE} 个样本")
        # 按时间戳排序并取前N个
        all_samples.sort(key=lambda x: x.metadata.get("timestamp_int", 0))
        all_samples = all_samples[:MAX_SAMPLES_PER_PACKAGE]
        processor.logger.info(f"实际处理样本数: {len(all_samples)}")
    
    # 2. 按数据包分组
    samples_by_package = defaultdict(list)
    for sample in all_samples:
        samples_by_package[sample.package_name].append(sample)
    
    processor.logger.info(f"总共 {len(samples_by_package)} 个数据包需要处理")
    
    # 3. 处理每个数据包
    processed_packages = 0
    for package_name, package_samples in samples_by_package.items():
        package_start_time = time.time()
        
        processor.logger.info(f"开始处理数据包: {package_name} ({len(package_samples)} 个样本)")
        
        # 获取数据包配置
        package_config_dict = DATA_PACKAGE_CONFIG.get(package_name)
        if not package_config_dict:
            processor.logger.warning(f"数据包 {package_name} 无配置，跳过")
            continue
        
        package_config = PackageConfig(
            name=package_name,
            type=package_config_dict["type"],
            calib_root=package_config_dict["calib_root"],
            config=package_config_dict
        )
        
        # 数据包目录
        package_dir = Path(TYJT_ROOT) / package_name
        
        # 处理数据包
        processor.process_package_samples(package_samples, package_config, package_dir)
        
        package_time = time.time() - package_start_time
        processor.logger.info(f"✅ 完成数据包 {package_name}, 耗时: {package_time:.1f}秒")
        
        processed_packages += 1
        
        # 每处理完一个数据包，保存一次中间结果
        if processed_packages % 2 == 0:
            processor.logger.info("💾 保存中间结果...")
            processor.save_nusc_data()
    
    # 4. 保存数据
    processor.logger.info("保存NuScenes数据...")
    processor.save_nusc_data()
    
    # 5. 验证数据
    processor.logger.info("验证数据...")
    processor.validate_data()
    
    # 6. 输出统计
    total_time = time.time() - program_start_time
    processor.logger.info("=" * 60)
    processor.logger.info("🎉 数据处理阶段完成!")
    processor.logger.info(f"⏱️  总耗时: {total_time:.2f}秒")
    processor.logger.info(f"📁 输出目录: {OUTPUT_ROOT}/{NUSC_VERSION}")
    processor.logger.info("=" * 60)

if __name__ == "__main__":
    main()