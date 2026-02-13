#!/usr/bin/env python3
"""
tyjt数据集转nuscenes格式转换工具 - v9.2.3-phase1 (完整数据分析阶段)
版本: v9.2.3-phase1-complete
修改日期: 2024-12-25
核心功能:
1. 数据完整性检查和标定文件完整校验
2. 时间连续性分析和场景自动划分
3. 生成包含完整标定信息的有效样本清单
4. 生成详细的无效样本报告
5. 为step1_2准备完整的组织结构
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
import time
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from datetime import datetime

# ==================== 全局配置 ====================

Mode = "Local"
if Mode == "Local":
    TYJT_ROOT = "/mnt/dataset/tyjt_RawData_all/"
    OUTPUT_ROOT = "./output-1225/step1/nuscenes_tyjt"
    NUSC_VERSION = "v1.0-tyjt"
    DATA_PACKAGE_CONFIG = {
        "2d3d_20250114": {
            "type": "hikvision",
            "calib_root": "calib/2d3d_20250114",
            "group2map_file": "group2map_calib.json",
            "camera2map_file": "camera2map_calib.json",
            "intersections": {
                "R9": {
                    "road_id": "R9",
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
                    "road_id": "R01",
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
    OUTPUT_ROOT = "./output-1225/step1/nuscenes_tyjt"
    NUSC_VERSION = "v1.0-tyjt"
    DATA_PACKAGE_CONFIG = {
        "2d3d_20250114": {
            "type": "hikvision",
            "calib_root": "calib/2d3d_20250114",
            "group2map_file": "group2map_calib.json",
            "camera2map_file": "camera2map_calib.json",
            "intersections": {
                "R9": {
                    "road_id": "R9",
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
else:
    print(f"❌【Error】: TYJT_ROOT & DATA_PACKAGE_CONFIG: None")
    sys.exit(1)

# ==================== 核心配置 ====================

# 相机映射配置
CAM_ORDER_TO_NUSC = {
    "A": "CAM_FRONT",
    "B": "CAM_FRONT_RIGHT", 
    "C": "CAM_BACK",
    "D": "CAM_FRONT_LEFT"
}

# 类别映射
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
DATASET_ROOT = str(Path(TYJT_ROOT).resolve())

# ==================== 数据结构定义 ====================

@dataclass
class CameraCalibration:
    """相机标定信息"""
    camera_order: str                    # A/B/C/D
    camera_folder: str                   # 原始相机文件夹名
    calibration_key: str                 # 标定文件中的键名
    nusc_camera_name: str               # NuScenes相机名
    
    # 原始TYJT标定信息
    raw_sensor2map: Optional[Dict] = None        # 原始sensor2map标定参数
    raw_intrinsic: Optional[Dict] = None         # 原始内参参数
    
    # 转换后的NuScenes标定信息
    nusc_translation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    nusc_rotation: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    nusc_intrinsic: List[List[float]] = field(default_factory=lambda: [
        [1680.0, 0, 960.0],
        [0, 1851.0, 540.0],
        [0, 0, 1]
    ])
    
    def __post_init__(self):
        if not self.nusc_camera_name:
            self.nusc_camera_name = CAM_ORDER_TO_NUSC.get(self.camera_order, "")

@dataclass
class RoadCalibrationSet:
    """道路标定数据集"""
    package_name: str
    package_type: str
    road_id: str
    dataset_root: str
    
    # 原始标定数据
    raw_calib_data: Dict[str, Any] = field(default_factory=dict)
    
    # 标定矩阵
    group2map_matrix: Optional[np.ndarray] = None  # 4x4 矩阵
    map2group_matrix: Optional[np.ndarray] = None  # 4x4 矩阵
    
    # 相机标定信息
    cameras: List[CameraCalibration] = field(default_factory=list)
    
    # 激光雷达标定
    lidar_calibration: Dict[str, Any] = field(default_factory=lambda: {
        "translation": [0.0, 0.0, 0.0],
        "rotation": [1.0, 0.0, 0.0, 0.0]
    })
    
    # 标定验证结果
    calibration_issues: List[str] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """检查标定数据是否有效"""
        if self.group2map_matrix is None:
            return False
        
        # 检查是否有有效的相机标定
        valid_cameras = [cam for cam in self.cameras if cam.nusc_camera_name]
        return len(valid_cameras) > 0 and len(self.calibration_issues) == 0

@dataclass
class ValidatedSample:
    """验证通过的有效样本"""
    # 基础信息
    package_name: str
    package_type: str
    road_id: str
    sub_packet_name: str
    timestamp: str
    dataset_root: str
    
    # 文件路径信息
    files: Dict[str, Any]
    
    # TYJT原始标定信息
    tyjt_calibration: Dict[str, Any] = field(default_factory=dict)
    
    # NuScenes格式标定信息
    nuscenes_calibration: Dict[str, Any] = field(default_factory=dict)
    
    # 样本元数据
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "camera_count": 0,
        "label_objects": 0,
        "timestamp_int": 0,
        "is_valid": True
    })
    
    @property
    def unique_key(self) -> str:
        """生成唯一的样本标识符"""
        return f"{self.package_name}_{self.road_id}_{self.sub_packet_name}_{self.timestamp}"
    
    @property
    def filename(self) -> str:
        """生成输出文件名"""
        return f"{self.unique_key}"

@dataclass
class InvalidSample:
    """无效样本信息"""
    package_name: str
    package_type: str
    road_id: str
    sub_packet_name: str
    timestamp: str
    
    # 详细错误信息
    error_type: str  # "missing_file", "calibration_error", "data_corrupt"
    error_details: List[str]
    missing_files: List[str]
    
    # 标定相关问题
    calibration_issues: List[str] = field(default_factory=list)
    
    # 建议解决方案
    suggestions: List[str] = field(default_factory=list)

@dataclass
class SceneInfo:
    """场景信息（在step1_1中确定）"""
    scene_token: str
    scene_name: str
    package_name: str
    road_id: str
    start_timestamp: int
    end_timestamp: int
    sample_count: int
    duration_seconds: float
    sample_tokens: List[str]  # 按时间顺序排列的样本token
    first_sample_token: str
    last_sample_token: str
    scene_index: int  # 场景序号
    
    @property
    def is_valid(self) -> bool:
        """场景有效性检查"""
        return (self.sample_count >= 10 and  # 最少10个样本（5秒）
                self.duration_seconds >= 5.0 and  # 最少5秒
                len(self.sample_tokens) == self.sample_count)

@dataclass 
class TimeAnalysisResult:
    """时间分析结果"""
    total_samples: int
    time_range: Tuple[int, int]  # (min_timestamp, max_timestamp)
    avg_time_interval: float  # 平均时间间隔（秒）
    time_interval_std: float  # 时间间隔标准差
    is_continuous: bool  # 是否时间连续
    continuity_issues: List[str]  # 连续性问题
    suggested_scenes: List[SceneInfo]  # 建议的场景划分
    recommended_sample_rate: float  # 建议的采样率（Hz）

@dataclass 
class AnalysisResult:
    """单个数据包的分析结果"""
    package_name: str
    package_type: str
    calibration_set: Optional[RoadCalibrationSet] = None
    valid_samples: List[ValidatedSample] = field(default_factory=list)
    invalid_samples: List[InvalidSample] = field(default_factory=list)
    time_analysis: Optional[TimeAnalysisResult] = None
    statistics: Dict[str, Any] = field(default_factory=dict)
    validation_report: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于JSON序列化）"""
        return {
            "package_name": self.package_name,
            "package_type": self.package_type,
            "calibration_valid": self.calibration_set.is_valid if self.calibration_set else False,
            "valid_samples_count": len(self.valid_samples),
            "invalid_samples_count": len(self.invalid_samples),
            "statistics": self.statistics,
            "validation_report": self.validation_report,
            "has_time_analysis": self.time_analysis is not None
        }

# ==================== Token生成器 ====================

class IDGenerator:
    """Token生成器"""
    
    def __init__(self):
        self.uuid_map = {}
        self.used_tokens: Set[str] = set()
        self.entity_counters = defaultdict(int)
    
    def get_scoped_token(self, entity_type: str, unique_key: str) -> str:
        """生成范围内的唯一token"""
        scope_key = f"{entity_type}::{unique_key}"
        if scope_key in self.uuid_map:
            return self.uuid_map[scope_key]
        
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
        
        return debug_uuid
    
    def validate_token_uniqueness(self) -> Tuple[bool, Dict]:
        """验证Token唯一性"""
        tokens = list(self.uuid_map.values())
        unique_tokens = set(tokens)
        stats = {
            'total_tokens': len(tokens),
            'unique_tokens': len(unique_tokens),
            'entity_counts': dict(self.entity_counters)
        }
        return len(tokens) == len(unique_tokens), stats

# 全局ID生成器
id_gen = IDGenerator()

# ==================== 数学工具函数 ====================

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

def quaternion_to_rotation_matrix(q: List[float]) -> np.ndarray:
    """四元数转旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def rotation_matrix_to_quaternion(matrix: np.ndarray) -> List[float]:
    """旋转矩阵转四元数"""
    m = np.array(matrix, dtype=np.float64)
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
    
    return [float(w), float(x), float(y), float(z)]

def parse_tyjt_transform(calib_params: Union[Dict, List]) -> np.ndarray:
    """解析TYJT标定参数为4x4变换矩阵"""
    try:
        transform = np.eye(4, dtype=np.float64)
        
        if isinstance(calib_params, Dict):
            # 格式1: 包含translation和rotation
            if 'translation' in calib_params and 'rotation' in calib_params:
                translation = np.array(calib_params['translation'], dtype=np.float64)
                rotation_quat = calib_params['rotation']
                
                if len(rotation_quat) == 4:
                    x, y, z, w = rotation_quat
                    rotation_matrix = quaternion_to_rotation_matrix([w, x, y, z])
                    transform[:3, :3] = rotation_matrix
                    transform[:3, 3] = translation
            
            # 格式2: 包含tx, ty, tz, rx, ry, rz, rw
            elif all(key in calib_params for key in ['tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'rw']):
                translation = np.array([
                    calib_params['tx'],
                    calib_params['ty'], 
                    calib_params['tz']
                ], dtype=np.float64)
                
                x, y, z, w = (
                    calib_params['rx'],
                    calib_params['ry'],
                    calib_params['rz'], 
                    calib_params['rw']
                )
                
                rotation_matrix = quaternion_to_rotation_matrix([w, x, y, z])
                transform[:3, :3] = rotation_matrix
                transform[:3, 3] = translation
        
        elif isinstance(calib_params, List):
            # 格式3: 7元素列表 [tx, ty, tz, rx, ry, rz, rw]
            if len(calib_params) == 7:
                tx, ty, tz, rx, ry, rz, rw = calib_params
                translation = np.array([tx, ty, tz], dtype=np.float64)
                
                rotation_matrix = quaternion_to_rotation_matrix([rw, rx, ry, rz])
                transform[:3, :3] = rotation_matrix
                transform[:3, 3] = translation
        
        return transform
        
    except Exception as e:
        print(f"解析标定参数失败: {e}")
        return np.eye(4, dtype=np.float64)

def compute_sensor2ego_transform(
    calib_data: Dict[str, Any],
    sensor_key: str,
    group_key: str
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """
    计算sensor到ego的变换
    """
    try:
        # 获取sensor2map变换
        if sensor_key not in calib_data:
            raise KeyError(f"标定数据中找不到传感器键: {sensor_key}")
        
        sensor_calib = calib_data[sensor_key]
        sensor2map = parse_tyjt_transform(sensor_calib)
        
        # 获取group2map变换
        if group_key not in calib_data:
            raise KeyError(f"标定数据中找不到group键: {group_key}")
        
        group_calib = calib_data[group_key]
        
        # 处理可能的嵌套格式
        if isinstance(group_calib, dict) and 'transform' in group_calib:
            group_calib = group_calib['transform']
        
        group2map = parse_tyjt_transform(group_calib)
        map2group = np.linalg.inv(group2map)
        
        # 计算sensor2ego = map2group * sensor2map
        sensor2ego = map2group @ sensor2map
        
        # 检查变换矩阵的有效性
        if not np.isfinite(sensor2ego).all():
            raise ValueError("无效的变换矩阵（包含非有限值）")
        
        # 提取平移和旋转
        translation = sensor2ego[:3, 3].tolist()
        rotation_matrix = sensor2ego[:3, :3]
        rotation = rotation_matrix_to_quaternion(rotation_matrix)
        
        # 构建详细的标定信息
        calibration_info = {
            "sensor_key": sensor_key,
            "group_key": group_key,
            "sensor2map_matrix": sensor2map.tolist(),
            "group2map_matrix": group2map.tolist(),
            "map2group_matrix": map2group.tolist(),
            "sensor2ego_matrix": sensor2ego.tolist(),
            "translation": translation,
            "rotation": rotation,
            "is_valid": True
        }
        
        return translation, rotation, calibration_info
        
    except Exception as e:
        error_msg = f"计算sensor2ego变换失败: {str(e)}"
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], {
            "sensor_key": sensor_key,
            "group_key": group_key,
            "error": error_msg,
            "is_valid": False
        }

def extract_camera_intrinsic(calib_data: Dict[str, Any], camera_key: str) -> List[List[float]]:
    """从标定数据中提取相机内参"""
    default_intrinsic = [
        [1680.0, 0, 960.0],
        [0, 1851.0, 540.0],
        [0, 0, 1]
    ]
    
    if camera_key in calib_data:
        camera_calib = calib_data[camera_key]
        
        if isinstance(camera_calib, dict):
            # 检查是否有内参参数
            if 'fx' in camera_calib and 'fy' in camera_calib:
                fx = float(camera_calib.get('fx', 1680.0))
                fy = float(camera_calib.get('fy', 1851.0))
                cx = float(camera_calib.get('cx', 960.0))
                cy = float(camera_calib.get('cy', 540.0))
                
                return [
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ]
    
    return default_intrinsic

# ==================== 时间连续性分析器 ====================

class TimeContinuityAnalyzer:
    """时间连续性分析器"""
    
    def __init__(self, expected_interval: float = 0.5):
        """
        初始化时间分析器
        expected_interval: 期望的时间间隔（秒），NuScenes默认为0.5
        """
        self.expected_interval = expected_interval
        self.interval_tolerance = 0.05  # 50ms容忍度
        self.min_scene_samples = 10     # 场景最少样本数
        self.max_scene_duration = 20.0  # 场景最大时长（秒）
        self.gap_threshold = 2.0        # 时间断层阈值（秒）
    
    def analyze_samples(self, samples: List[ValidatedSample]) -> TimeAnalysisResult:
        """分析样本的时间连续性"""
        
        if not samples:
            return TimeAnalysisResult(
                total_samples=0,
                time_range=(0, 0),
                avg_time_interval=0,
                time_interval_std=0,
                is_continuous=False,
                continuity_issues=["无样本数据"],
                suggested_scenes=[],
                recommended_sample_rate=0
            )
        
        # 按时间戳排序
        sorted_samples = sorted(samples, key=lambda x: x.metadata["timestamp_int"])
        
        # 提取时间戳
        timestamps = [s.metadata["timestamp_int"] for s in sorted_samples]
        
        # 计算时间统计
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        total_duration = (max_ts - min_ts) / 1e6  # 转换为秒
        
        # 计算时间间隔
        time_intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]) / 1e6  # 转换为秒
            time_intervals.append(interval)
        
        avg_interval = np.mean(time_intervals) if time_intervals else 0
        interval_std = np.std(time_intervals) if time_intervals else 0
        
        # 检查连续性
        is_continuous = True
        continuity_issues = []
        
        for i, interval in enumerate(time_intervals):
            expected = self.expected_interval
            if abs(interval - expected) > self.interval_tolerance:
                is_continuous = False
                issue = f"样本 {i} 到 {i+1}: 时间间隔 {interval:.3f}s, 期望 {expected:.3f}s"
                continuity_issues.append(issue)
        
        # 建议的采样率（Hz）
        if avg_interval > 0:
            recommended_rate = 1.0 / avg_interval
        else:
            recommended_rate = 2.0  # 默认2Hz（0.5秒间隔）
        
        # 自动划分场景
        suggested_scenes = self.auto_split_scenes(sorted_samples)
        
        return TimeAnalysisResult(
            total_samples=len(samples),
            time_range=(min_ts, max_ts),
            avg_time_interval=avg_interval,
            time_interval_std=interval_std,
            is_continuous=is_continuous,
            continuity_issues=continuity_issues,
            suggested_scenes=suggested_scenes,
            recommended_sample_rate=recommended_rate
        )
    
    def auto_split_scenes(self, sorted_samples: List[ValidatedSample]) -> List[SceneInfo]:
        """自动划分场景"""
        
        if not sorted_samples:
            return []
        
        scenes = []
        current_scene_samples = []
        scene_counter = 0
        
        for i, sample in enumerate(sorted_samples):
            if not current_scene_samples:
                # 开始新场景
                current_scene_samples.append(sample)
                continue
            
            # 检查是否应该开始新场景
            prev_sample = current_scene_samples[-1]
            time_diff = (sample.metadata["timestamp_int"] - 
                       prev_sample.metadata["timestamp_int"]) / 1e6
            
            # 计算当前场景时长
            current_duration = (
                current_scene_samples[-1].metadata["timestamp_int"] - 
                current_scene_samples[0].metadata["timestamp_int"]
            ) / 1e6
            
            # 判断是否开始新场景的条件：
            # 1. 时间间隔过大（大于阈值）
            # 2. 当前场景已达到最大时长
            if (time_diff > self.gap_threshold or 
                current_duration >= self.max_scene_duration):
                
                # 保存当前场景（如果样本数足够）
                if len(current_scene_samples) >= self.min_scene_samples:
                    scene_info = self.create_scene_info(
                        current_scene_samples, scene_counter
                    )
                    scenes.append(scene_info)
                    scene_counter += 1
                
                # 开始新场景
                current_scene_samples = [sample]
            else:
                current_scene_samples.append(sample)
        
        # 处理最后一个场景
        if (current_scene_samples and 
            len(current_scene_samples) >= self.min_scene_samples):
            scene_info = self.create_scene_info(
                current_scene_samples, scene_counter
            )
            scenes.append(scene_info)
        
        return scenes
    
    def create_scene_info(self, scene_samples: List[ValidatedSample], 
                         scene_index: int) -> SceneInfo:
        """创建场景信息"""
        
        if not scene_samples:
            raise ValueError("场景样本列表为空")
        
        # 获取基础信息
        first_sample = scene_samples[0]
        last_sample = scene_samples[-1]
        
        # 生成scene_token和scene_name
        scene_token = id_gen.get_scoped_token(
            "scene", 
            f"{first_sample.package_name}_{first_sample.road_id}_scene_{scene_index:03d}"
        )
        
        scene_name = f"{first_sample.package_name}_{first_sample.road_id}_scene_{scene_index:03d}"
        
        # 生成样本token（在step1_1中生成，供step1_2使用）
        sample_tokens = []
        for i, sample in enumerate(scene_samples):
            sample_token = id_gen.get_scoped_token(
                "sample",
                f"{sample.package_name}_{sample.road_id}_{sample.timestamp}"
            )
            sample_tokens.append(sample_token)
        
        # 计算时间范围和时长
        start_ts = first_sample.metadata["timestamp_int"]
        end_ts = last_sample.metadata["timestamp_int"]
        duration = (end_ts - start_ts) / 1e6
        
        return SceneInfo(
            scene_token=scene_token,
            scene_name=scene_name,
            package_name=first_sample.package_name,
            road_id=first_sample.road_id,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            sample_count=len(scene_samples),
            duration_seconds=duration,
            sample_tokens=sample_tokens,
            first_sample_token=sample_tokens[0],
            last_sample_token=sample_tokens[-1],
            scene_index=scene_index
        )

# ==================== 数据检查与分析核心逻辑 ====================

class EnhancedDataAnalyzer:
    """增强版数据检查与分析器"""
    
    def __init__(self, output_root: str):
        self.output_root = Path(output_root)
        self.analysis_dir = self.output_root / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # 时间分析器
        self.time_analyzer = TimeContinuityAnalyzer(expected_interval=0.5)
        
        # 缓存
        self.calib_cache: Dict[str, Dict] = {}
        
        # 设置日志
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志系统"""
        log_dir = self.analysis_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("enhanced_analyzer")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            # 文件处理器
            log_file = log_dir / "analysis.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def load_calibration_enhanced(self, package_path: Path, package_config: Dict) -> Dict[str, Any]:
        """增强版标定数据加载"""
        cache_key = str(package_path)
        
        if cache_key in self.calib_cache:
            return self.calib_cache[cache_key]
        
        calib_root = package_path / package_config["calib_root"]
        calib_data = {}
        
        try:
            if not calib_root.exists():
                raise FileNotFoundError(f"标定目录不存在: {calib_root}")
            
            package_type = package_config["type"]
            
            if package_type == "hikvision":
                # 加载group2map标定
                group2map_path = calib_root / package_config["group2map_file"]
                if group2map_path.exists():
                    with open(group2map_path, 'r', encoding='utf-8') as f:
                        group_data = json.load(f)
                    calib_data.update(group_data)
                    self.logger.debug(f"加载group2map: {group2map_path}")
                
                # 加载camera2map标定
                camera2map_path = calib_root / package_config["camera2map_file"]
                if camera2map_path.exists():
                    with open(camera2map_path, 'r', encoding='utf-8') as f:
                        camera_data = json.load(f)
                    calib_data.update(camera_data)
                    self.logger.debug(f"加载camera2map: {camera2map_path}")
            
            elif package_type == "all_in_one":
                # 加载sensor2map标定
                sensor2map_path = calib_root / package_config["sensor2map_file"]
                if sensor2map_path.exists():
                    with open(sensor2map_path, 'r', encoding='utf-8') as f:
                        sensor_data = json.load(f)
                    calib_data.update(sensor_data)
                    self.logger.debug(f"加载sensor2map: {sensor2map_path}")
            
            # 缓存标定数据
            if calib_data:
                self.calib_cache[cache_key] = calib_data
            
        except Exception as e:
            self.logger.error(f"加载标定数据失败 {package_path.name}: {e}")
        
        return calib_data
    
    def create_calibration_set(self, package_name: str, package_type: str, 
                              road_id: str, calib_data: Dict, 
                              road_config: Dict) -> RoadCalibrationSet:
        """创建道路标定数据集"""
        calibration_set = RoadCalibrationSet(
            package_name=package_name,
            package_type=package_type,
            road_id=road_id,
            dataset_root=DATASET_ROOT
        )
        
        # 保存原始标定数据
        calibration_set.raw_calib_data = calib_data
        
        try:
            group_key = road_config["group_key"]
            
            # 解析group2map矩阵
            if group_key in calib_data:
                group_calib = calib_data[group_key]
                if isinstance(group_calib, dict) and 'transform' in group_calib:
                    group_calib = group_calib['transform']
                
                calibration_set.group2map_matrix = parse_tyjt_transform(group_calib)
                calibration_set.map2group_matrix = np.linalg.inv(calibration_set.group2map_matrix)
            else:
                calibration_set.calibration_issues.append(f"缺失group2map标定: {group_key}")
            
            # 处理每个相机
            for cam_config in road_config["cameras"]:
                if package_type == "hikvision":
                    camera_folder, camera_order = cam_config
                    calibration_key = camera_folder
                else:
                    camera_folder, calibration_key, camera_order = cam_config
                
                nusc_camera_name = CAM_ORDER_TO_NUSC.get(camera_order, "")
                if not nusc_camera_name:
                    calibration_set.calibration_issues.append(f"无效的相机顺序: {camera_order}")
                    continue
                
                # 创建相机标定对象
                camera_calib = CameraCalibration(
                    camera_order=camera_order,
                    camera_folder=camera_folder,
                    calibration_key=calibration_key,
                    nusc_camera_name=nusc_camera_name
                )
                
                # 保存原始标定信息
                if calibration_key in calib_data:
                    camera_calib.raw_sensor2map = calib_data[calibration_key]
                
                # 提取内参
                camera_calib.nusc_intrinsic = extract_camera_intrinsic(calib_data, calibration_key)
                
                # 计算NuScenes格式的标定
                if calibration_key in calib_data and group_key in calib_data:
                    translation, rotation, calib_info = compute_sensor2ego_transform(
                        calib_data, calibration_key, group_key
                    )
                    
                    if calib_info.get("is_valid", False):
                        camera_calib.nusc_translation = translation
                        camera_calib.nusc_rotation = rotation
                    else:
                        calibration_set.calibration_issues.append(
                            f"相机 {camera_order} 标定计算失败"
                        )
                
                calibration_set.cameras.append(camera_calib)
            
            # 验证标定结果
            if not calibration_set.calibration_issues:
                self.logger.debug(f"道路 {road_id} 标定数据验证通过")
            else:
                self.logger.warning(f"道路 {road_id} 标定数据存在问题")
            
        except Exception as e:
            calibration_set.calibration_issues.append(f"创建标定集失败: {str(e)}")
            self.logger.error(f"创建道路 {road_id} 标定集失败: {e}")
        
        return calibration_set
    
    def scan_sample_files(self, sub_packet: Path, calibration_set: RoadCalibrationSet) -> Dict[str, Dict]:
        """扫描样本文件"""
        samples_info = {}
        
        # 1. 扫描lidar文件
        lidar_dir = sub_packet / "lidar" / "pcd"
        lidar_files = {}
        if lidar_dir.exists():
            for npy_file in lidar_dir.glob("*.npy"):
                if npy_file.stem.isdigit():
                    lidar_files[npy_file.stem] = str(npy_file.resolve())
        
        # 2. 扫描label文件
        label_dir = sub_packet / "lidar" / "label"
        label_files = {}
        if label_dir.exists():
            for json_file in label_dir.glob("*.json"):
                if json_file.stem.isdigit():
                    label_files[json_file.stem] = str(json_file.resolve())
        
        # 3. 扫描相机文件
        camera_files_dict = defaultdict(dict)
        for camera_calib in calibration_set.cameras:
            img_dir = sub_packet / camera_calib.camera_folder / "image_dc"
            if img_dir.exists():
                for jpg_file in img_dir.glob("*.jpg"):
                    if jpg_file.stem.isdigit():
                        camera_files_dict[camera_calib.camera_order][jpg_file.stem] = str(jpg_file.resolve())
        
        # 4. 合并所有时间戳
        all_timestamps = set(lidar_files.keys())
        all_timestamps.update(label_files.keys())
        for cam_order, img_files in camera_files_dict.items():
            all_timestamps.update(img_files.keys())
        
        # 5. 为每个时间戳构建详细文件信息
        for timestamp in sorted(all_timestamps, key=int):
            file_info = {
                "lidar": lidar_files.get(timestamp),
                "label": label_files.get(timestamp),
                "cameras": {}
            }
            
            # 收集相机文件
            for camera_calib in calibration_set.cameras:
                img_files = camera_files_dict.get(camera_calib.camera_order, {})
                img_path = img_files.get(timestamp)
                if img_path:
                    file_info["cameras"][camera_calib.camera_order] = {
                        "image_path": img_path,
                        "camera_folder": camera_calib.camera_folder,
                        "calibration_key": camera_calib.calibration_key,
                        "nusc_camera_name": camera_calib.nusc_camera_name
                    }
            
            samples_info[timestamp] = file_info
        
        return samples_info
    
    def validate_sample(self, timestamp: str, file_info: Dict, 
                       calibration_set: RoadCalibrationSet) -> Tuple[bool, List[str], List[str]]:
        """验证单个样本的完整性"""
        issues = []
        missing_files = []
        
        # 1. 检查必需文件
        if not file_info.get("lidar"):
            issues.append("缺失lidar文件")
            missing_files.append("lidar")
        
        if not file_info.get("label"):
            issues.append("缺失label文件")
            missing_files.append("label")
        
        # 2. 检查相机文件
        expected_cameras = {cam.camera_order for cam in calibration_set.cameras}
        available_cameras = set(file_info.get("cameras", {}).keys())
        missing_cameras = expected_cameras - available_cameras
        
        if missing_cameras:
            issues.append(f"缺失相机: {sorted(missing_cameras)}")
            for cam in missing_cameras:
                missing_files.append(f"camera_{cam}")
        
        # 3. 检查标定数据
        if not calibration_set.is_valid:
            issues.append("标定数据不完整")
        
        # 确定样本是否有效
        is_valid = len(issues) == 0
        
        return is_valid, issues, missing_files
    
    def analyze_road(self, package_dir: Path, package_name: str, 
                    package_config: Dict, road_id: str) -> Tuple[List[ValidatedSample], List[InvalidSample]]:
        """分析单个道路"""
        
        # 获取道路配置
        road_config = package_config["intersections"][road_id]
        
        # 加载标定数据
        calib_data = self.load_calibration_enhanced(package_dir, package_config)
        if not calib_data:
            self.logger.error(f"道路 {road_id} 标定数据加载失败")
            return [], []
        
        # 创建标定集
        calibration_set = self.create_calibration_set(
            package_name, package_config["type"], road_id, 
            calib_data, road_config
        )
        
        if not calibration_set.is_valid:
            self.logger.error(f"道路 {road_id} 标定数据验证失败")
            return [], []
        
        # 扫描子包
        datasets_dir = package_dir / "datasets"
        if not datasets_dir.exists():
            self.logger.warning(f"无datasets目录: {datasets_dir}")
            return [], []
        
        sub_packets = [d for d in datasets_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        valid_samples = []
        invalid_samples = []
        
        # 处理每个子包
        for sub_packet in sub_packets:
            sub_packet_name = sub_packet.name
            
            # 扫描样本文件
            samples_info = self.scan_sample_files(sub_packet, calibration_set)
            
            # 验证每个样本
            for timestamp, file_info in samples_info.items():
                is_valid, issues, missing_files = self.validate_sample(
                    timestamp, file_info, calibration_set
                )
                
                if is_valid:
                    # 创建有效样本
                    valid_sample = self.create_validated_sample(
                        package_name, package_config["type"], road_id,
                        sub_packet_name, timestamp, file_info, calibration_set
                    )
                    valid_samples.append(valid_sample)
                else:
                    # 创建无效样本
                    invalid_sample = InvalidSample(
                        package_name=package_name,
                        package_type=package_config["type"],
                        road_id=road_id,
                        sub_packet_name=sub_packet_name,
                        timestamp=timestamp,
                        error_type="missing_file" if missing_files else "data_error",
                        error_details=issues,
                        missing_files=missing_files,
                        calibration_issues=calibration_set.calibration_issues,
                        suggestions=self.generate_suggestions(issues, missing_files)
                    )
                    invalid_samples.append(invalid_sample)
            
            # 统计信息
            valid_count = sum(1 for ts, fi in samples_info.items() 
                            if self.validate_sample(ts, fi, calibration_set)[0])
            total_count = len(samples_info)
            
            self.logger.debug(f"子包 {sub_packet_name}: {total_count} 样本, "
                           f"有效 {valid_count}, 无效 {total_count - valid_count}")
        
        self.logger.info(f"道路 {road_id}: {len(valid_samples)} 有效样本, "
                       f"{len(invalid_samples)} 无效样本")
        
        return valid_samples, invalid_samples
    

    def create_validated_sample(self, package_name: str, package_type: str, road_id: str,
                            sub_packet_name: str, timestamp: str, file_info: Dict,
                            calibration_set: RoadCalibrationSet) -> ValidatedSample:
        """创建验证通过的有效样本"""
        
        # 1. 构建TYJT原始标定信息（包含详细的相机映射关系）
        tyjt_calibration = {
            "road_id": road_id,
            "group_key": calibration_set.raw_calib_data.get("group_key", ""),
            "cameras": {},
            "camera_mapping": {}  # 新增：明确的映射关系
        }
        
        # 存储完整的相机映射信息
        camera_mappings = {}
        for camera_calib in calibration_set.cameras:
            camera_order = camera_calib.camera_order
            nusc_camera_name = CAM_ORDER_TO_NUSC.get(camera_order, "")
            
            # 记录映射关系
            camera_mappings[camera_order] = {
                "tyjt_camera_name": camera_calib.camera_folder,
                "nuscenes_camera_name": nusc_camera_name,
                "calibration_key": camera_calib.calibration_key,
                "mapping_rule": f"A→CAM_FRONT, B→CAM_FRONT_RIGHT, C→CAM_BACK, D→CAM_FRONT_LEFT"
            }
            
            tyjt_calibration["cameras"][camera_order] = {
                "camera_folder": camera_calib.camera_folder,
                "calibration_key": camera_calib.calibration_key,
                "raw_sensor2map": camera_calib.raw_sensor2map,
                "raw_intrinsic": camera_calib.raw_intrinsic
            }
        
        tyjt_calibration["camera_mapping"] = camera_mappings
        
        # 2. 构建NuScenes格式标定信息（与TYJT标定保持一致）
        nuscenes_calibration = {
            "ego_pose": {
                "translation": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0]
            },
            "sensors": {
                "LIDAR_TOP": {
                    "translation": calibration_set.lidar_calibration["translation"],
                    "rotation": calibration_set.lidar_calibration["rotation"]
                }
            },
            "camera_mapping_explanation": "固定映射关系: A=CAM_FRONT, B=CAM_FRONT_RIGHT, C=CAM_BACK, D=CAM_FRONT_LEFT"
        }
        
        # 为每个相机添加映射说明
        for camera_calib in calibration_set.cameras:
            nusc_camera_name = CAM_ORDER_TO_NUSC.get(camera_calib.camera_order, "")
            if nusc_camera_name:
                nuscenes_calibration["sensors"][nusc_camera_name] = {
                    "translation": camera_calib.nusc_translation,
                    "rotation": camera_calib.nusc_rotation,
                    "camera_intrinsic": camera_calib.nusc_intrinsic,
                    "tyjt_camera_order": camera_calib.camera_order,
                    "tyjt_camera_name": camera_calib.camera_folder
                }
        
        # 3. 读取label文件统计对象数量
        label_objects = 0
        label_path = file_info.get("label")
        if label_path and Path(label_path).exists():
            try:
                with open(label_path, 'r') as f:
                    label_data = json.load(f)
                if isinstance(label_data, dict):
                    objects = label_data.get("objects", [])
                else:
                    objects = label_data
                label_objects = len(objects) if isinstance(objects, list) else 0
            except:
                pass
        
        # 4. 为阶段2准备额外的元数据
        nuscenes_camera_names = []
        for camera_info in file_info.get("cameras", {}).values():
            if isinstance(camera_info, dict):
                nusc_name = camera_info.get("nusc_camera_name")
                if nusc_name:
                    nuscenes_camera_names.append(nusc_name)
        
        # 5. 创建有效样本对象
        sample = ValidatedSample(
            package_name=package_name,
            package_type=package_type,
            road_id=road_id,
            sub_packet_name=sub_packet_name,
            timestamp=timestamp,
            dataset_root=DATASET_ROOT,
            files=file_info,
            tyjt_calibration=tyjt_calibration,
            nuscenes_calibration=nuscenes_calibration,
            metadata={
                "camera_count": len(file_info.get("cameras", {})),
                "label_objects": label_objects,
                "timestamp_int": int(timestamp) if timestamp.isdigit() else 0,
                "is_valid": True,
                "nuscenes_camera_names": nuscenes_camera_names,  # 明确记录NuScenes相机名
                "has_prev_next_chain": False  # 阶段1不生成链，阶段2会生成
            }
        )
        
        return sample


    def generate_suggestions(self, issues: List[str], missing_files: List[str]) -> List[str]:
        """根据问题生成建议"""
        suggestions = []
        
        if "缺失lidar文件" in issues or "lidar" in missing_files:
            suggestions.append("检查lidar/pcd目录下是否有对应的.npy文件")
        
        if "缺失label文件" in issues or "label" in missing_files:
            suggestions.append("检查lidar/label目录下是否有对应的.json文件")
        
        if any("缺失相机" in issue for issue in issues):
            suggestions.append("检查所有相机目录下的image_dc文件夹")
        
        if "标定数据不完整" in issues:
            suggestions.append("检查标定文件是否包含所有必要的相机参数")
        
        return suggestions
    
    def analyze_package(self, package_dir: Path) -> AnalysisResult:
        """分析单个数据包"""
        package_name = package_dir.name
        
        if package_name not in DATA_PACKAGE_CONFIG:
            self.logger.warning(f"数据包 {package_name} 无配置，跳过")
            return AnalysisResult(package_name=package_name, package_type="unknown")
        
        package_config = DATA_PACKAGE_CONFIG[package_name]
        package_type = package_config["type"]
        
        self.logger.info(f"开始分析数据包: {package_name} ({package_type})")
        
        result = AnalysisResult(
            package_name=package_name,
            package_type=package_type
        )
        
        start_time = time.time()
        
        # 分析每个道路
        for road_id in package_config["intersections"]:
            road_start = time.time()
            
            valid_samples, invalid_samples = self.analyze_road(
                package_dir, package_name, package_config, road_id
            )
            
            result.valid_samples.extend(valid_samples)
            result.invalid_samples.extend(invalid_samples)
            
            elapsed = time.time() - road_start
            self.logger.info(f"道路 {road_id}: {len(valid_samples)} 有效, "
                           f"{len(invalid_samples)} 无效, 耗时: {elapsed:.1f}秒")
        
        # 时间连续性分析
        if result.valid_samples:
            time_result = self.time_analyzer.analyze_samples(result.valid_samples)
            result.time_analysis = time_result
            
            self.logger.info(f"⏱️  时间分析: {len(result.valid_samples)} 个样本")
            self.logger.info(f"  平均时间间隔: {time_result.avg_time_interval:.3f}s")
            self.logger.info(f"  建议场景数: {len(time_result.suggested_scenes)}")
            
            if not time_result.is_continuous:
                self.logger.warning(f"⚠️  时间不连续，发现 {len(time_result.continuity_issues)} 个问题")
        
        # 计算统计信息
        total_time = time.time() - start_time
        result.statistics = {
            "total_valid_samples": len(result.valid_samples),
            "total_invalid_samples": len(result.invalid_samples),
            "analysis_time_seconds": total_time,
            "samples_per_second": len(result.valid_samples) / total_time if total_time > 0 else 0,
            "road_count": len(package_config["intersections"]),
            "package_type": package_type
        }
        
        if hasattr(result, 'time_analysis') and result.time_analysis:
            result.statistics.update({
                "time_interval_avg": result.time_analysis.avg_time_interval,
                "suggested_scene_count": len(result.time_analysis.suggested_scenes),
                "time_interval_std": result.time_analysis.time_interval_std,
                "is_time_continuous": result.time_analysis.is_continuous
            })
        
        # 生成验证报告
        result.validation_report = {
            "calibration_status": "PASS" if all(s.metadata["is_valid"] for s in result.valid_samples) else "FAIL",
            "file_integrity": self.check_file_integrity(result),
            "sample_consistency": self.check_sample_consistency(result),
            "recommendations": self.generate_recommendations(result)
        }
        
        # 保存分析结果
        self.save_enhanced_analysis_result(result)
        
        self.logger.info(f"✅ 完成分析数据包: {package_name}")
        self.logger.info(f"  有效样本: {len(result.valid_samples)}")
        self.logger.info(f"  无效样本: {len(result.invalid_samples)}")
        self.logger.info(f"  总耗时: {total_time:.1f}秒")
        
        return result
    
    def check_file_integrity(self, result: AnalysisResult) -> Dict[str, Any]:
        """检查文件完整性"""
        file_stats = {
            "total_files": 0,
            "missing_files": 0,
            "cameras_per_sample": []
        }
        
        for sample in result.valid_samples:
            camera_count = sample.metadata.get("camera_count", 0)
            file_stats["cameras_per_sample"].append(camera_count)
            
            files_to_check = [
                sample.files.get("lidar"),
                sample.files.get("label")
            ]
            
            for camera_info in sample.files.get("cameras", {}).values():
                if isinstance(camera_info, dict):
                    files_to_check.append(camera_info.get("image_path"))
            
            for file_path in files_to_check:
                file_stats["total_files"] += 1
                if file_path:
                    if not Path(file_path).exists():
                        file_stats["missing_files"] += 1
        
        if file_stats["cameras_per_sample"]:
            file_stats["avg_cameras_per_sample"] = sum(file_stats["cameras_per_sample"]) / len(file_stats["cameras_per_sample"])
        else:
            file_stats["avg_cameras_per_sample"] = 0
        
        return file_stats
    
    def check_sample_consistency(self, result: AnalysisResult) -> Dict[str, Any]:
        """检查样本一致性"""
        consistency = {
            "timestamp_monotonic": True,
            "camera_consistency": True,
            "time_range": {"min": None, "max": None}
        }
        
        timestamps = []
        camera_counts = set()
        
        for sample in result.valid_samples:
            ts_int = sample.metadata.get("timestamp_int", 0)
            timestamps.append(ts_int)
            
            camera_count = sample.metadata.get("camera_count", 0)
            camera_counts.add(camera_count)
        
        # 检查时间戳单调性
        if timestamps:
            sorted_timestamps = sorted(timestamps)
            consistency["timestamp_monotonic"] = timestamps == sorted_timestamps
            consistency["time_range"]["min"] = min(timestamps)
            consistency["time_range"]["max"] = max(timestamps)
        
        # 检查相机数量一致性
        consistency["camera_consistency"] = len(camera_counts) == 1
        
        return consistency
    
    def generate_recommendations(self, result: AnalysisResult) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if result.invalid_samples:
            recommendations.append(f"发现 {len(result.invalid_samples)} 个无效样本，请检查详细报告")
        
        file_stats = result.validation_report.get("file_integrity", {})
        if file_stats.get("missing_files", 0) > 0:
            recommendations.append("存在缺失文件，请检查数据完整性")
        
        if hasattr(result, 'time_analysis') and result.time_analysis:
            if not result.time_analysis.is_continuous:
                recommendations.append("时间序列不连续，可能需要重新组织样本")
        
        if not recommendations:
            recommendations.append("数据质量良好，可以开始阶段2的数据转换")
        
        return recommendations
    
    def save_enhanced_analysis_result(self, result: AnalysisResult):
        """保存增强版分析结果"""
        
        # 1. 保存完整的分析结果
        output_file = self.analysis_dir / f"{result.package_name}_analysis.json"
        
        analysis_dict = {
            "package_info": {
                "package_name": result.package_name,
                "package_type": result.package_type,
                "analysis_time": datetime.now().isoformat(),
                "camera_mapping_explanation": "A=CAM_FRONT, B=CAM_FRONT_RIGHT, C=CAM_BACK, D=CAM_FRONT_LEFT"
            },
            "statistics": result.statistics,
            "validation_report": result.validation_report,
            "sample_summary": {
                "valid_count": len(result.valid_samples),
                "invalid_count": len(result.invalid_samples),
                "first_timestamp": result.valid_samples[0].timestamp if result.valid_samples else None,
                "last_timestamp": result.valid_samples[-1].timestamp if result.valid_samples else None,
                "camera_mappings_available": True  # 明确标记映射关系已保存
            }
        }
        
        # 添加时间分析结果
        if result.time_analysis:
            time_dict = asdict(result.time_analysis)
            # 处理场景信息
            scene_dicts = []
            for scene in result.time_analysis.suggested_scenes:
                scene_dict = asdict(scene)
                # 确保scene_info有足够的元数据
                scene_dict["package_name"] = result.package_name
                scene_dict["road_id"] = result.valid_samples[0].road_id if result.valid_samples else "unknown"
                scene_dicts.append(scene_dict)
            time_dict["suggested_scenes"] = scene_dicts
            analysis_dict["time_analysis"] = time_dict
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_dict, f, indent=2, ensure_ascii=False, default=str)
        
        # 2. 保存详细的有效样本列表（为阶段2提供完整信息）
        if result.valid_samples:
            valid_samples_file = self.analysis_dir / f"{result.package_name}_valid_samples.json"
            valid_samples_data = []
            
            # 按时间戳排序
            sorted_samples = sorted(result.valid_samples, key=lambda x: x.metadata["timestamp_int"])
            
            for idx, sample in enumerate(sorted_samples):
                sample_dict = asdict(sample)
                
                # 添加处理信息（明确说明prev/next将在阶段2生成）
                sample_dict["processing_info"] = {
                    "unique_key": sample.unique_key,
                    "filename": sample.filename,
                    "nuscenes_camera_names": sample.metadata.get("nuscenes_camera_names", []),
                    "camera_mapping": {
                        "A": "CAM_FRONT",
                        "B": "CAM_FRONT_RIGHT", 
                        "C": "CAM_BACK",
                        "D": "CAM_FRONT_LEFT"
                    },
                    "prev_next_note": "prev/next链将在stage2中基于时间顺序生成",
                    "sample_index": idx,
                    "total_samples_in_scene": len(sorted_samples)
                }
                
                # 添加明确的标定映射说明
                if "tyjt_calibration" in sample_dict and "cameras" in sample_dict["tyjt_calibration"]:
                    for cam_order, cam_info in sample_dict["tyjt_calibration"]["cameras"].items():
                        if isinstance(cam_info, dict):
                            cam_info["nuscenes_mapping"] = CAM_ORDER_TO_NUSC.get(cam_order, "UNKNOWN")
                
                valid_samples_data.append(sample_dict)
            
            with open(valid_samples_file, 'w', encoding='utf-8') as f:
                json.dump(valid_samples_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"💾 保存有效样本列表: {valid_samples_file.name} "
                        f"({len(valid_samples_data)} 个样本)")
            self.logger.info(f"  📝 样本已按时间排序，prev/next链将在阶段2生成")
        
        # 3. 保存详细的无效样本报告（确保生成）
        if result.invalid_samples:
            invalid_samples_file = self.analysis_dir / f"{result.package_name}_invalid_samples.json"
            invalid_samples_data = []
            
            for idx, sample in enumerate(result.invalid_samples):
                sample_dict = asdict(sample)
                
                # 添加额外信息便于问题排查
                sample_dict["additional_info"] = {
                    "sample_index": idx,
                    "analysis_time": datetime.now().isoformat(),
                    "suggested_action": "检查缺失文件或重新采集数据"
                }
                
                invalid_samples_data.append(sample_dict)
            
            with open(invalid_samples_file, 'w', encoding='utf-8') as f:
                json.dump(invalid_samples_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"⚠️  保存无效样本报告: {invalid_samples_file.name} "
                        f"({len(invalid_samples_data)} 个问题样本)")
        else:
            # 即使没有无效样本，也创建一个空的报告文件
            invalid_samples_file = self.analysis_dir / f"{result.package_name}_invalid_samples.json"
            with open(invalid_samples_file, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)
            self.logger.info(f"✅ 无无效样本，创建空报告: {invalid_samples_file.name}")
        
        self.logger.info(f"💾 保存分析结果: {output_file.name}")

    def generate_summary_report(self, all_results: List[AnalysisResult]):
        """生成汇总报告"""
        summary = {
            "analysis_summary": {
                "total_packages": len(all_results),
                "total_valid_samples": sum(len(r.valid_samples) for r in all_results),
                "total_invalid_samples": sum(len(r.invalid_samples) for r in all_results),
                "analysis_start_time": datetime.now().isoformat(),
                "analysis_duration": time.time() - self.start_time
            },
            "package_details": [],
            "package_type_summary": defaultdict(lambda: {"count": 0, "valid_samples": 0, "invalid_samples": 0}),
            "road_summary": defaultdict(lambda: {"count": 0, "valid_samples": 0}),
            "quality_assessment": {
                "overall_quality": "GOOD",
                "issues_found": False
            }
        }
        
        total_scenes = 0
        for result in all_results:
            # 包级别统计
            package_detail = {
                "package_name": result.package_name,
                "package_type": result.package_type,
                "valid_samples": len(result.valid_samples),
                "invalid_samples": len(result.invalid_samples),
                "calibration_status": "PASS" if result.validation_report.get("calibration_status") == "PASS" else "FAIL",
                "analysis_time": result.statistics.get("analysis_time_seconds", 0)
            }
            
            # 添加时间分析信息
            if result.time_analysis:
                package_detail.update({
                    "time_interval_avg": result.time_analysis.avg_time_interval,
                    "suggested_scenes": len(result.time_analysis.suggested_scenes),
                    "is_time_continuous": result.time_analysis.is_continuous
                })
                total_scenes += len(result.time_analysis.suggested_scenes)
            
            summary["package_details"].append(package_detail)
            
            # 按类型统计
            pkg_type = result.package_type
            summary["package_type_summary"][pkg_type]["count"] += 1
            summary["package_type_summary"][pkg_type]["valid_samples"] += len(result.valid_samples)
            summary["package_type_summary"][pkg_type]["invalid_samples"] += len(result.invalid_samples)
        
        # 添加场景总数
        summary["analysis_summary"]["total_suggested_scenes"] = total_scenes
        
        # 质量评估
        total_samples = summary["analysis_summary"]["total_valid_samples"] + summary["analysis_summary"]["total_invalid_samples"]
        if total_samples > 0:
            valid_ratio = summary["analysis_summary"]["total_valid_samples"] / total_samples * 100
            if valid_ratio >= 95:
                summary["quality_assessment"]["overall_quality"] = "EXCELLENT"
            elif valid_ratio >= 85:
                summary["quality_assessment"]["overall_quality"] = "GOOD"
            elif valid_ratio >= 70:
                summary["quality_assessment"]["overall_quality"] = "FAIR"
            else:
                summary["quality_assessment"]["overall_quality"] = "POOR"
        
        if summary["analysis_summary"]["total_invalid_samples"] > 0:
            summary["quality_assessment"]["issues_found"] = True
        
        # 保存汇总报告
        summary_file = self.analysis_dir / "summary_report.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成Markdown报告
        self.generate_markdown_report(summary, all_results)
        
        return summary
    
    def generate_markdown_report(self, summary: Dict, all_results: List[AnalysisResult]):
        """生成Markdown报告"""
        report_file = self.analysis_dir / "analysis_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# TYJT数据集分析报告\n\n")
            f.write(f"**生成时间**: {summary['analysis_summary']['analysis_start_time']}\n")
            f.write(f"**分析耗时**: {summary['analysis_summary']['analysis_duration']:.2f}秒\n\n")
            
            f.write("## 📊 总体统计\n\n")
            f.write(f"- **总数据包数**: {summary['analysis_summary']['total_packages']}\n")
            f.write(f"- **总有效样本数**: {summary['analysis_summary']['total_valid_samples']}\n")
            f.write(f"- **总无效样本数**: {summary['analysis_summary']['total_invalid_samples']}\n")
            f.write(f"- **建议场景数**: {summary['analysis_summary']['total_suggested_scenes']}\n")
            
            total_samples = summary['analysis_summary']['total_valid_samples'] + summary['analysis_summary']['total_invalid_samples']
            if total_samples > 0:
                valid_ratio = summary['analysis_summary']['total_valid_samples'] / total_samples * 100
                f.write(f"- **有效样本占比**: {valid_ratio:.1f}%\n")
            
            f.write(f"- **总体质量评估**: {summary['quality_assessment']['overall_quality']}\n\n")
            
            f.write("## 📦 按数据包类型统计\n\n")
            f.write("| 类型 | 数据包数 | 有效样本数 | 无效样本数 | 有效率 |\n")
            f.write("|------|----------|------------|------------|--------|\n")
            
            for pkg_type, stats in summary["package_type_summary"].items():
                total = stats["valid_samples"] + stats["invalid_samples"]
                valid_ratio = (stats["valid_samples"] / total * 100) if total > 0 else 0
                f.write(f"| {pkg_type} | {stats['count']} | {stats['valid_samples']} | "
                       f"{stats['invalid_samples']} | {valid_ratio:.1f}% |\n")
            
            f.write("\n## 🔍 各数据包详情\n\n")
            f.write("| 数据包名称 | 类型 | 有效样本 | 无效样本 | 建议场景 | 时间连续性 | 分析耗时(秒) |\n")
            f.write("|------------|------|----------|----------|----------|------------|--------------|\n")
            
            for pkg_detail in summary["package_details"]:
                time_continuous = "✅" if pkg_detail.get("is_time_continuous", True) else "⚠️"
                f.write(f"| {pkg_detail['package_name']} | {pkg_detail['package_type']} | "
                       f"{pkg_detail['valid_samples']} | {pkg_detail['invalid_samples']} | "
                       f"{pkg_detail.get('suggested_scenes', 0)} | {time_continuous} | {pkg_detail['analysis_time']:.1f} |\n")
            
            f.write("\n## ⚠️  发现的问题\n\n")
            if summary['quality_assessment']['issues_found']:
                # 统计时间连续性问题
                time_issues = 0
                for result in all_results:
                    if result.time_analysis and not result.time_analysis.is_continuous:
                        time_issues += len(result.time_analysis.continuity_issues)
                
                if time_issues > 0:
                    f.write(f"- **时间连续性问题**: {time_issues} 处\n")
                
                if summary['analysis_summary']['total_invalid_samples'] > 0:
                    f.write(f"- **无效样本**: {summary['analysis_summary']['total_invalid_samples']} 个\n")
                
                f.write("\n### 建议解决方案\n")
                if time_issues > 0:
                    f.write("1. **时间连续性**: 检查时间戳序列，可能需要重新采样或调整场景划分\n")
                if summary['analysis_summary']['total_invalid_samples'] > 0:
                    f.write("2. **无效样本**: 检查对应数据包的问题报告文件，修复缺失文件\n")
            else:
                f.write("✅ **未发现明显问题，数据质量良好**\n")
            
            f.write("\n## 🚀 下一步建议\n\n")
            f.write("1. **数据转换**: 使用阶段2脚本处理有效样本\n")
            f.write("2. **时间对齐**: 如果存在时间连续性问题，考虑重新采样到标准频率\n")
            f.write("3. **场景优化**: 根据建议的场景划分进行数据组织\n")
            f.write("4. **问题修复**: 根据无效样本报告修复数据问题\n")
            
            f.write("\n## 📁 生成的文件\n\n")
            for result in all_results:
                f.write(f"### {result.package_name}\n")
                f.write(f"- `{result.package_name}_analysis.json` - 完整分析结果\n")
                if result.valid_samples:
                    f.write(f"- `{result.package_name}_valid_samples.json` - 有效样本清单（用于阶段2）\n")
                if result.invalid_samples:
                    f.write(f"- `{result.package_name}_invalid_samples.json` - 无效样本详细报告\n")
            
            f.write("\n### 汇总文件\n")
            f.write("- `summary_report.json` - 汇总统计信息\n")
            f.write("- `analysis_report.md` - 本报告\n")
            f.write("- `analysis.log` - 详细日志文件\n")
        
        self.logger.info(f"📋 生成分析报告: {report_file}")

# ==================== 主函数 ====================
def main():
    """主函数 - 数据检查与分析阶段"""
    analyzer = EnhancedDataAnalyzer(OUTPUT_ROOT)
    analyzer.start_time = time.time()
    
    analyzer.logger.info("=" * 60)
    analyzer.logger.info("🔍 TYJT数据集分析工具 v9.2.3-phase1")
    analyzer.logger.info(f"📁 数据集根目录: {DATASET_ROOT}")
    analyzer.logger.info(f"📊 分析输出目录: {analyzer.analysis_dir}")
    analyzer.logger.info("=" * 60)
    
    # 查找数据包
    root_path = Path(TYJT_ROOT)
    packages = []
    
    analyzer.logger.info(f"在目录 {TYJT_ROOT} 中查找数据包...")
    for item in root_path.iterdir():
        if item.is_dir() and item.name in DATA_PACKAGE_CONFIG:
            packages.append(item)
            analyzer.logger.info(f"✅ 找到数据包: {item.name}")
    
    if not packages:
        analyzer.logger.error("❌ 未找到任何有效数据包，退出")
        return
    
    analyzer.logger.info(f"📦 总共找到 {len(packages)} 个有效数据包")
    
    # 并行分析数据包
    all_results = []
    
    with ThreadPoolExecutor(max_workers=min(len(packages), 4), 
                           thread_name_prefix="EnhancedAnalyzer") as executor:
        futures = {executor.submit(analyzer.analyze_package, pkg): pkg for pkg in packages}
        
        completed = 0
        for future in as_completed(futures):
            package = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                completed += 1
                analyzer.logger.info(f"📊 进度: {completed}/{len(packages)} 个数据包分析完成")
            except Exception as e:
                analyzer.logger.error(f"❌ 数据包 {package.name} 分析失败: {e}")
                completed += 1
    
    # 生成汇总报告
    summary = analyzer.generate_summary_report(all_results)
    
    # 验证Token唯一性
    token_unique, token_stats = id_gen.validate_token_uniqueness()
    if token_unique:
        analyzer.logger.info(f"✅ Token唯一性验证通过，共生成 {token_stats['total_tokens']} 个Token")
    else:
        analyzer.logger.warning(f"⚠️  Token重复警告，唯一Token数: {token_stats['unique_tokens']}")
    
    # 输出最终统计
    total_time = time.time() - analyzer.start_time
    analyzer.logger.info("=" * 60)
    analyzer.logger.info("🎉 数据分析阶段完成!")
    analyzer.logger.info(f"⏱️  总耗时: {total_time:.2f}秒")
    analyzer.logger.info(f"📊 总有效样本数: {summary['analysis_summary']['total_valid_samples']}")
    analyzer.logger.info(f"📊 总无效样本数: {summary['analysis_summary']['total_invalid_samples']}")
    analyzer.logger.info(f"📊 建议场景数: {summary['analysis_summary']['total_suggested_scenes']}")
    analyzer.logger.info(f"📁 分析结果目录: {analyzer.analysis_dir}")
    analyzer.logger.info(f"📋 总体质量: {summary['quality_assessment']['overall_quality']}")
    
    # 输出关键文件信息
    analyzer.logger.info("\n📁 生成的关键文件:")
    for result in all_results:
        analyzer.logger.info(f"  - {result.package_name}:")
        if result.valid_samples:
            valid_file = analyzer.analysis_dir / f"{result.package_name}_valid_samples.json"
            if valid_file.exists():
                sample_data = json.loads(valid_file.read_text())
                # 检查是否包含prev/next键
                if sample_data and "prev" in sample_data[0] or "next" in sample_data[0]:
                    analyzer.logger.info(f"    ✓ valid_samples.json: {len(sample_data)} 样本，包含prev/next链")
                else:
                    analyzer.logger.info(f"    ✓ valid_samples.json: {len(sample_data)} 样本，无prev/next链")
        if result.invalid_samples:
            invalid_file = analyzer.analysis_dir / f"{result.package_name}_invalid_samples.json"
            if invalid_file.exists():
                invalid_data = json.loads(invalid_file.read_text())
                analyzer.logger.info(f"    ⚠️  invalid_samples.json: {len(invalid_data)} 无效样本")
    
    analyzer.logger.info("=" * 60)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(f"❌ 主程序运行失败: {e}")
        traceback.print_exc()
        sys.exit(1)