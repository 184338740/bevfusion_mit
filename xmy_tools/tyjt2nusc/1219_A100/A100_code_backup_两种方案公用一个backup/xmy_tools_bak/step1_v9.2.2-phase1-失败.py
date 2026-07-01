#!/usr/bin/env python3
"""
tyjt数据集转nuscenes格式转换工具 - v9.2.2-phase1 (数据检查与分析阶段)
版本: v9.2.2-phase1
修改日期: 2024-12-19
核心目标:
1. 快速扫描所有数据包，检查数据完整性
2. 生成可用样本清单和问题报告
3. 保持与v9.1完全相同的标定和映射逻辑
4. 不进行任何数据转换，只做分析和验证
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union, Any
import hashlib
import math
from collections import defaultdict
import re
import logging
import time
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from datetime import datetime  # 确保这行在文件开头
# ==================== 全局配置（与v9.1完全一致） ====================

Mode = "Local"
if Mode == "Local":
    TYJT_ROOT = "/mnt/dataset/tyjt_RawData_all/"
    OUTPUT_ROOT = "./output-1204-v9.2.2/step1/nuscenes_tyjt"
    NUSC_VERSION = "v1.0-tyjt"
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
    # 13个数据包规格配置（与v9.1完全一致）
    DATA_PACKAGE_CONFIG = {
        # 海康相机产品（9个）
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
else:
    print(f"❌【Error】: TYJT_ROOT & DATA_PACKAGE_CONFIG: None")
    sys.exit(1)

# ==================== 核心配置（与v9.1完全一致） ====================

# A/B/C/D相机→NuScenes相机固定映射（与v9.1完全一致）
CAM_ORDER_TO_NUSC = {
    "A": "CAM_FRONT",
    "B": "CAM_FRONT_RIGHT", 
    "C": "CAM_BACK",
    "D": "CAM_FRONT_LEFT"
}

# Step1类别映射 - 添加"tyjt."前缀，保持15个类别（与v9.1完全一致）
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

# ==================== 数据结构定义 ====================

@dataclass
class CameraInfo:
    """相机信息（与v9.1的映射逻辑完全一致）"""
    camera_folder: str           # 相机文件夹名，如 "R3_Aw_CamS"
    calibration_key: str         # 标定文件中的键，如 "R3_Aw_CamS" 或 "SC_2A_CamR_new"
    camera_order: str           # 相机顺序 A/B/C/D
    nusc_camera_name: str       # NuScenes相机名，如 "CAM_FRONT"
    
    def __post_init__(self):
        self.nusc_camera_name = CAM_ORDER_TO_NUSC.get(self.camera_order, "")
    
    @property
    def is_valid(self) -> bool:
        return bool(self.nusc_camera_name)

@dataclass
class SampleFileInfo:
    """样本文件信息（包含所有必要的文件路径）"""
    timestamp: str
    lidar_npy_path: Path
    label_json_path: Path
    camera_images: Dict[str, Path]  # camera_order -> 图片路径
    
    @property
    def has_lidar(self) -> bool:
        return self.lidar_npy_path.exists()
    
    @property
    def has_label(self) -> bool:
        return self.label_json_path.exists()
    
    @property
    def camera_count(self) -> int:
        return len(self.camera_images)
    
    def has_all_cameras(self, expected_cameras: List[str]) -> bool:
        """检查是否包含所有预期的相机 - 注意这不是@property！"""
        return all(cam in self.camera_images for cam in expected_cameras)

@dataclass
class ValidatedSample:
    """验证通过的样本信息（用于阶段2）"""
    package_name: str
    intersection_key: str
    sub_packet_name: str
    timestamp: str
    files: Dict[str, str]  # 文件路径映射
    metadata: Dict[str, Any]
    
    @property
    def unique_key(self) -> str:
        """生成与v9.1完全一致的unique_key"""
        return f"{self.package_name}_{self.intersection_key}_{self.sub_packet_name}_{self.timestamp}"

@dataclass
class InvalidSample:
    """无效样本信息"""
    package_name: str
    intersection_key: str
    sub_packet_name: str
    timestamp: str
    issues: List[str]
    missing_files: List[str]

@dataclass 
class AnalysisResult:
    """单个数据包的分析结果"""
    package_name: str
    valid_samples: List[ValidatedSample] = field(default_factory=list)
    invalid_samples: List[InvalidSample] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "package_name": self.package_name,
            "valid_samples": [asdict(s) for s in self.valid_samples],
            "invalid_samples": [asdict(s) for s in self.invalid_samples],
            "statistics": self.statistics
        }

# ==================== 数学工具函数（与v9.1完全一致） ====================

def quaternion_from_euler(yaw: float, pitch: float = 0, roll: float = 0) -> List[float]:
    """与v9.1完全一致的欧拉角转四元数"""
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
    """与v9.1完全一致的四元数转旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])

def rotation_matrix_to_quaternion(matrix: np.ndarray) -> List[float]:
    """与v9.1完全一致的旋转矩阵转四元数"""
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

def parse_tyjt_transform(calib_params: Union[Dict, List], logger=None) -> np.ndarray:
    """与v9.1完全一致的标定参数解析"""
    try:
        if isinstance(calib_params, Dict):
            if 'translation' in calib_params and 'rotation' in calib_params:
                translation = np.array(calib_params['translation'])
                rotation_quat = calib_params['rotation']
                
                if len(rotation_quat) == 4:
                    x, y, z, w = rotation_quat
                    w, x, y, z = w, x, y, z
                    
                    if logger:
                        logger.debug(f"海康相机格式 - 平移: {translation}")
                        logger.debug(f"海康相机格式 - 旋转四元数: [{w:.6f}, {x:.6f}, {y:.6f}, {z:.6f}]")
                else:
                    if logger:
                        logger.warning(f"不支持的旋转格式长度: {len(rotation_quat)}")
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
                
                if logger:
                    logger.debug(f"一体机格式 - 平移: {translation}")
                    logger.debug(f"一体机格式 - 旋转四元数: [{w:.6f}, {x:.6f}, {y:.6f}, {z:.6f}]")
            else:
                if logger:
                    logger.warning(f"不支持的标定格式，可用键: {list(calib_params.keys())}")
                return np.eye(4)
                
        elif isinstance(calib_params, List):
            if len(calib_params) == 7:
                tx, ty, tz, rx, ry, rz, rw = calib_params
                translation = np.array([tx, ty, tz])
                x, y, z, w = rx, ry, rz, rw
                
                if logger:
                    logger.debug(f"列表格式 - 平移: {translation}")
                    logger.debug(f"列表格式 - 旋转四元数: [{w:.6f}, {x:.6f}, {y:.6f}, {z:.6f}]")
            else:
                if logger:
                    logger.warning(f"不支持的列表格式长度: {len(calib_params)}")
                return np.eye(4)
        else:
            if logger:
                logger.warning(f"不支持的标定参数类型: {type(calib_params)}")
            return np.eye(4)
        
        rotation_matrix = quaternion_to_rotation_matrix([w, x, y, z])
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation
        
        return transform
        
    except Exception as e:
        if logger:
            logger.error(f"解析标定参数失败: {e}")
        return np.eye(4)

def get_sensor2ego_transform_fixed(calib_data: Dict, sensor_calib_key: str, group2map_key: str, logger=None) -> Tuple[List[float], List[float]]:
    """与v9.1完全一致的sensor→ego转换"""
    try:
        if sensor_calib_key not in calib_data:
            if logger:
                logger.error(f"标定数据中找不到 {sensor_calib_key}")
            return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        
        sensor_calib_params = calib_data[sensor_calib_key]
        sensor2map = parse_tyjt_transform(sensor_calib_params, logger)
        
        if group2map_key in calib_data:
            group2map_params = calib_data[group2map_key]
            
            if isinstance(group2map_params, dict) and 'transform' in group2map_params:
                group2map_params = group2map_params['transform']
            
            group2map = parse_tyjt_transform(group2map_params, logger)
            map2group = np.linalg.inv(group2map)
        else:
            if logger:
                logger.error(f"标定数据中找不到 group2map 键 {group2map_key}")
            return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        
        sensor2ego = map2group @ sensor2map
        
        if not np.isfinite(sensor2ego).all():
            if logger:
                logger.error(f"无效的变换矩阵")
            return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        
        translation = sensor2ego[:3, 3].tolist()
        rotation_matrix = sensor2ego[:3, :3]
        rotation = rotation_matrix_to_quaternion(rotation_matrix)
        
        return translation, rotation
        
    except Exception as e:
        if logger:
            logger.error(f"计算sensor2ego变换时出错: {e}")
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]

# ==================== 数据检查与分析核心逻辑 ====================

class DataAnalyzer:
    """数据检查与分析器"""
    
    def __init__(self, output_root: str):
        self.output_root = Path(output_root)
        self.analysis_dir = self.output_root / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # 缓存标定数据（避免重复加载）
        self.calib_cache: Dict[str, Dict] = {}
        
    def setup_logging(self):
        """设置日志系统"""
        log_dir = self.analysis_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("analyzer")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            log_file = log_dir / "analysis.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            console_handler = logging.StreamHandler()
            
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def load_calibration(self, package_path: Path, package_config: Dict) -> Dict:
        """加载标定数据（与v9.1完全一致）"""
        cache_key = str(package_path)
        if cache_key in self.calib_cache:
            return self.calib_cache[cache_key]
        
        calib_root = package_path / package_config["calib_root"]
        calib_data = {}
        
        try:
            if package_config["type"] == "hikvision":
                group2map_path = calib_root / package_config["group2map_file"]
                if group2map_path.exists():
                    with open(group2map_path, 'r') as f:
                        calib_data.update(json.load(f))
                
                camera2map_path = calib_root / package_config["camera2map_file"]
                if camera2map_path.exists():
                    with open(camera2map_path, 'r') as f:
                        calib_data.update(json.load(f))
            
            elif package_config["type"] == "all_in_one":
                sensor2map_path = calib_root / package_config["sensor2map_file"]
                if sensor2map_path.exists():
                    with open(sensor2map_path, 'r') as f:
                        calib_data.update(json.load(f))
            
            self.calib_cache[cache_key] = calib_data
            self.logger.info(f"✅ 加载标定数据: {package_path.name}, {len(calib_data)} 个键")
            
        except Exception as e:
            self.logger.error(f"❌ 加载标定数据失败 {package_path.name}: {e}")
        
        return calib_data
    
    def scan_sub_packet_files(self, sub_packet: Path, camera_infos: List[CameraInfo]) -> Dict[str, SampleFileInfo]:
        """扫描子包内的所有文件（高效批量扫描）"""
        samples_dict = {}
        
        # 1. 首先扫描lidar目录下的所有npy文件
        lidar_dir = sub_packet / "lidar" / "pcd"
        if lidar_dir.exists():
            lidar_files = {f.stem: f for f in lidar_dir.glob("*.npy") if f.stem.isdigit()}
        else:
            lidar_files = {}
        
        # 2. 扫描label目录下的所有json文件
        label_dir = sub_packet / "lidar" / "label"
        if label_dir.exists():
            label_files = {f.stem: f for f in label_dir.glob("*.json") if f.stem.isdigit()}
        else:
            label_files = {}
        
        # 3. 批量扫描所有相机目录
        camera_files_dict = {}
        for cam_info in camera_infos:
            img_dir = sub_packet / cam_info.camera_folder / "image_dc"
            if img_dir.exists():
                # 一次性获取所有jpg文件
                img_files = {f.stem: f for f in img_dir.glob("*.jpg") if f.stem.isdigit()}
                camera_files_dict[cam_info.camera_order] = img_files
        
        # 4. 合并所有时间戳（取交集）
        all_timestamps = set(lidar_files.keys())
        all_timestamps.update(label_files.keys())
        for cam_order, img_files in camera_files_dict.items():
            all_timestamps.update(img_files.keys())
        
        # 5. 为每个时间戳构建SampleFileInfo
        for timestamp in sorted(all_timestamps, key=int):
            # 检查是否lidar和label都存在
            lidar_path = lidar_files.get(timestamp)
            label_path = label_files.get(timestamp)
            
            # 收集可用的相机图像
            camera_images = {}
            for cam_info in camera_infos:
                cam_files = camera_files_dict.get(cam_info.camera_order, {})
                img_path = cam_files.get(timestamp)
                if img_path:
                    camera_images[cam_info.camera_order] = img_path
            
            # 创建样本信息（即使不完整也记录，用于问题分析）
            samples_dict[timestamp] = SampleFileInfo(
                timestamp=timestamp,
                lidar_npy_path=lidar_path or Path(),
                label_json_path=label_path or Path(),
                camera_images=camera_images
            )
        
        return samples_dict
    
    def validate_calibration(self, calib_data: Dict, intersection_config: Dict, 
                           camera_infos: List[CameraInfo], package_logger) -> bool:
        """验证标定数据的完整性"""
        valid = True
        group2map_key = intersection_config["group_key"]
        
        if group2map_key not in calib_data:
            package_logger.error(f"缺失group2map标定: {group2map_key}")
            valid = False
        
        for cam_info in camera_infos:
            if cam_info.calibration_key not in calib_data:
                package_logger.error(f"缺失相机标定: {cam_info.calibration_key}")
                valid = False
        
        return valid
    
    def analyze_intersection(self, package_dir: Path, package_name: str, 
                        package_config: Dict, intersection_key: str) -> Tuple[List[ValidatedSample], List[InvalidSample]]:
        """分析单个路口"""
        intersection_config = package_config["intersections"][intersection_key]
        
        # 1. 创建路口特定的logger
        intersection_logger = logging.getLogger(f"{package_name}_{intersection_key}")
        intersection_logger.setLevel(logging.INFO)
        
        # 2. 构建相机信息列表（与v9.1完全一致的映射逻辑）
        camera_infos = []
        for cam_info in intersection_config["cameras"]:
            if package_config["type"] == "hikvision":
                camera_folder, camera_order = cam_info
                calibration_key = camera_folder
            else:
                camera_folder, calibration_key, camera_order = cam_info
            
            # 显式传入nusc_camera_name
            nusc_camera_name = CAM_ORDER_TO_NUSC.get(camera_order, "")
            camera_info = CameraInfo(
                camera_folder=camera_folder,
                calibration_key=calibration_key,
                camera_order=camera_order,
                nusc_camera_name=nusc_camera_name
            )
            
            if camera_info.is_valid:
                camera_infos.append(camera_info)
        
        if not camera_infos:
            intersection_logger.warning(f"路口 {intersection_key} 无有效相机配置")
            return [], []
        
        # 3. 加载标定数据
        calib_data = self.load_calibration(package_dir, package_config)
        if not calib_data:
            return [], []
        
        # 4. 验证标定数据
        if not self.validate_calibration(calib_data, intersection_config, camera_infos, intersection_logger):
            intersection_logger.error(f"路口 {intersection_key} 标定数据不完整")
            return [], []
        
        # 5. 扫描所有子包
        datasets_dir = package_dir / "datasets"
        if not datasets_dir.exists():
            intersection_logger.warning(f"无datasets目录: {datasets_dir}")
            return [], []
        
        sub_packets = [d for d in datasets_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        intersection_logger.info(f"扫描 {len(sub_packets)} 个子包")
        
        valid_samples = []
        invalid_samples = []
        
        # 6. 处理每个子包
        for sub_packet in sub_packets:
            sub_packet_logger = logging.getLogger(f"{package_name}_{intersection_key}_{sub_packet.name}")
            
            # 批量扫描文件
            samples_dict = self.scan_sub_packet_files(sub_packet, camera_infos)
            
            # 验证每个样本
            for timestamp, sample_info in samples_dict.items():
                issues = []
                missing_files = []
                
                # 检查基本文件
                if not sample_info.has_lidar:
                    missing_files.append("lidar")
                    issues.append("缺失lidar文件")
                
                if not sample_info.has_label:
                    missing_files.append("label")
                    issues.append("缺失label文件")
                
                # 检查相机完整性
                expected_cameras = [cam.camera_order for cam in camera_infos]
                if not sample_info.has_all_cameras(expected_cameras):
                    missing_cameras = set(expected_cameras) - set(sample_info.camera_images.keys())
                    for cam in missing_cameras:
                        missing_files.append(f"camera_{cam}")
                    issues.append(f"缺失相机: {missing_cameras}")
                
                if issues:
                    # 无效样本
                    invalid_sample = InvalidSample(
                        package_name=package_name,
                        intersection_key=intersection_key,
                        sub_packet_name=sub_packet.name,
                        timestamp=timestamp,
                        issues=issues,
                        missing_files=missing_files
                    )
                    invalid_samples.append(invalid_sample)
                else:
                    # 有效样本 - 构建文件路径映射
                    files_dict = {
                        "lidar": str(sample_info.lidar_npy_path),
                        "label": str(sample_info.label_json_path),
                        "cameras": {}
                    }
                    
                    for cam_order, img_path in sample_info.camera_images.items():
                        # 查找对应的相机信息
                        for cam_info in camera_infos:
                            if cam_info.camera_order == cam_order:
                                files_dict["cameras"][cam_order] = {
                                    "image_path": str(img_path),
                                    "camera_folder": cam_info.camera_folder,
                                    "calibration_key": cam_info.calibration_key
                                }
                                break
                    
                    # 尝试读取label获取对象数量
                    label_objects = 0
                    try:
                        with open(sample_info.label_json_path, 'r') as f:
                            label_data = json.load(f)
                        objects = label_data.get("objects", []) if isinstance(label_data, dict) else label_data
                        label_objects = len(objects)
                    except:
                        pass
                    
                    # 有效样本
                    valid_sample = ValidatedSample(
                        package_name=package_name,
                        intersection_key=intersection_key,
                        sub_packet_name=sub_packet.name,
                        timestamp=timestamp,
                        files=files_dict,
                        metadata={
                            "camera_count": sample_info.camera_count,
                            "label_objects": label_objects,
                            "timestamp_int": int(timestamp) if timestamp.isdigit() else 0
                        }
                    )
                    valid_samples.append(valid_sample)
            
            # 统计有效和无效样本数量
            valid_count = len([s for s in samples_dict.values() if s.has_lidar and s.has_label])
            invalid_count = len([s for s in samples_dict.values() if not (s.has_lidar and s.has_label)])
            
            sub_packet_logger.info(f"子包 {sub_packet.name}: {len(samples_dict)} 个样本, "
                                f"有效 {valid_count}, "
                                f"无效 {invalid_count}")
        
        # 按时间戳排序
        valid_samples.sort(key=lambda x: x.metadata["timestamp_int"])
        
        intersection_logger.info(f"路口 {intersection_key}: {len(valid_samples)} 个有效样本, "
                            f"{len(invalid_samples)} 个无效样本")
        
        return valid_samples, invalid_samples

    def analyze_package(self, package_dir: Path) -> AnalysisResult:
        """分析单个数据包"""
        package_name = package_dir.name
        
        if package_name not in DATA_PACKAGE_CONFIG:
            self.logger.warning(f"数据包 {package_name} 无配置，跳过")
            return AnalysisResult(package_name=package_name)
        
        package_config = DATA_PACKAGE_CONFIG[package_name]
        self.logger.info(f"开始分析数据包: {package_name} ({package_config['type']})")
        
        result = AnalysisResult(package_name=package_name)
        start_time = time.time()
        
        # 分析每个路口
        for intersection_key in package_config["intersections"]:
            intersection_start = time.time()
            
            valid_samples, invalid_samples = self.analyze_intersection(
                package_dir, package_name, package_config, intersection_key
            )
            
            result.valid_samples.extend(valid_samples)
            result.invalid_samples.extend(invalid_samples)
            
            elapsed = time.time() - intersection_start
            self.logger.info(f"路口 {intersection_key}: {len(valid_samples)} 有效, "
                           f"{len(invalid_samples)} 无效, 耗时: {elapsed:.1f}秒")
        
        # 计算统计信息
        total_time = time.time() - start_time
        result.statistics = {
            "total_valid_samples": len(result.valid_samples),
            "total_invalid_samples": len(result.invalid_samples),
            "analysis_time_seconds": total_time,
            "samples_per_second": len(result.valid_samples) / total_time if total_time > 0 else 0,
            "intersection_count": len(package_config["intersections"]),
            "package_type": package_config["type"]
        }
        
        # 保存分析结果
        self.save_analysis_result(result)
        
        self.logger.info(f"✅ 完成分析数据包: {package_name}, "
                       f"有效样本: {len(result.valid_samples)}, "
                       f"无效样本: {len(result.invalid_samples)}, "
                       f"总耗时: {total_time:.1f}秒")
        
        return result
    
    def save_analysis_result(self, result: AnalysisResult):
        """保存分析结果到文件"""
        # 1. 保存完整分析结果
        output_file = self.analysis_dir / f"{result.package_name}_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 2. 保存简化的有效样本列表（用于阶段2）
        valid_samples_file = self.analysis_dir / f"{result.package_name}_valid_samples.json"
        valid_samples_data = []
        for sample in result.valid_samples:
            sample_dict = asdict(sample)
            # 添加unique_key用于阶段2
            sample_dict["unique_key"] = sample.unique_key
            valid_samples_data.append(sample_dict)
        
        with open(valid_samples_file, 'w', encoding='utf-8') as f:
            json.dump(valid_samples_data, f, indent=2, ensure_ascii=False)
        
        # 3. 保存问题报告
        if result.invalid_samples:
            invalid_samples_file = self.analysis_dir / f"{result.package_name}_invalid_samples.json"
            invalid_samples_data = [asdict(s) for s in result.invalid_samples]
            
            with open(invalid_samples_file, 'w', encoding='utf-8') as f:
                json.dump(invalid_samples_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 保存分析结果: {output_file.name}")
    

    def generate_summary_report(self, all_results: List[AnalysisResult]):
        """生成汇总报告"""
        summary = {
            "total_packages": len(all_results),
            "total_valid_samples": sum(len(r.valid_samples) for r in all_results),
            "total_invalid_samples": sum(len(r.invalid_samples) for r in all_results),
            "package_summary": [],
            "by_package_type": defaultdict(lambda: {"count": 0, "valid_samples": 0}),
            "analysis_start_time": datetime.now().isoformat()
        }
        
        for result in all_results:
            package_summary = {
                "package_name": result.package_name,
                "valid_samples": len(result.valid_samples),
                "invalid_samples": len(result.invalid_samples),
                "package_type": result.statistics.get("package_type", "unknown"),
                "analysis_time": result.statistics.get("analysis_time_seconds", 0)
            }
            summary["package_summary"].append(package_summary)
            
            # 按类型统计
            pkg_type = result.statistics.get("package_type", "unknown")
            summary["by_package_type"][pkg_type]["count"] += 1
            summary["by_package_type"][pkg_type]["valid_samples"] += len(result.valid_samples)
        
        # 保存汇总报告
        summary_file = self.analysis_dir / "summary_report.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 修复：只有在有结果时才生成Markdown报告
        if summary["total_packages"] > 0:
            self.generate_markdown_report(summary)
        else:
            self.logger.warning("没有分析结果，跳过Markdown报告生成")
        
        return summary


    def generate_markdown_report(self, summary: Dict):
        """生成Markdown格式的详细报告"""
        report_file = self.analysis_dir / "analysis_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# TYJT数据集分析报告\n\n")
            f.write(f"**生成时间**: {summary['analysis_start_time']}\n\n")
            
            f.write("## 总体统计\n\n")
            f.write(f"- **总数据包数**: {summary['total_packages']}\n")
            f.write(f"- **总有效样本数**: {summary['total_valid_samples']}\n")
            f.write(f"- **总无效样本数**: {summary['total_invalid_samples']}\n")
            
            # 修复：避免零除错误
            total_samples = summary['total_valid_samples'] + summary['total_invalid_samples']
            if total_samples > 0:
                valid_ratio = summary['total_valid_samples'] / total_samples * 100
                f.write(f"- **有效样本占比**: {valid_ratio:.1f}%\n\n")
            else:
                f.write(f"- **有效样本占比**: 0.0%\n\n")
            
            f.write("## 按数据包类型统计\n\n")
            f.write("| 类型 | 数据包数 | 有效样本数 | 平均每包样本数 |\n")
            f.write("|------|----------|------------|----------------|\n")
            
            for pkg_type, stats in summary["by_package_type"].items():
                avg_samples = stats["valid_samples"] / stats["count"] if stats["count"] > 0 else 0
                f.write(f"| {pkg_type} | {stats['count']} | {stats['valid_samples']} | {avg_samples:.1f} |\n")
            
            f.write("\n## 各数据包详情\n\n")
            f.write("| 数据包名称 | 类型 | 有效样本 | 无效样本 | 分析耗时(秒) |\n")
            f.write("|------------|------|----------|----------|--------------|\n")
            
            for pkg_summary in summary["package_summary"]:
                f.write(f"| {pkg_summary['package_name']} | {pkg_summary['package_type']} | "
                    f"{pkg_summary['valid_samples']} | {pkg_summary['invalid_samples']} | "
                    f"{pkg_summary['analysis_time']:.1f} |\n")
            
            f.write("\n## 建议\n\n")
            if summary["total_invalid_samples"] > 0:
                f.write("⚠️ **注意**: 发现无效样本，请检查对应数据包的问题报告文件\n")
            else:
                f.write("✅ **所有数据包均通过完整性检查**\n")
            
            f.write("\n## 文件列表\n\n")
            f.write("- `{package_name}_analysis.json` - 完整分析结果\n")
            f.write("- `{package_name}_valid_samples.json` - 有效样本清单（用于阶段2）\n")
            f.write("- `{package_name}_invalid_samples.json` - 无效样本清单（如有）\n")
            f.write("- `summary_report.json` - 汇总统计\n")
            f.write("- `analysis_report.md` - 本报告\n")
        
        self.logger.info(f"📋 生成分析报告: {report_file}")
    


# ==================== 主函数 ====================

def main():
    """主函数 - 数据检查与分析阶段"""
    analyzer = DataAnalyzer(OUTPUT_ROOT)
    
    analyzer.logger.info("=" * 60)
    analyzer.logger.info("🔍 TYJT数据集分析工具 v9.2.2-phase1")
    analyzer.logger.info(f"📥 原数据根目录: {TYJT_ROOT}")
    analyzer.logger.info(f"📊 数据包配置数量: {len(DATA_PACKAGE_CONFIG)}")
    analyzer.logger.info("=" * 60)
    
    program_start_time = time.time()
    
    # 1. 查找数据包
    root_path = Path(TYJT_ROOT)
    packages = []
    
    for item in root_path.iterdir():
        if item.is_dir() and item.name in DATA_PACKAGE_CONFIG:
            packages.append(item)
    
    if not packages:
        analyzer.logger.error("❌ 未找到任何有效数据包，退出")
        return
    
    analyzer.logger.info(f"✅ 找到 {len(packages)} 个有效数据包")
    
    # 2. 并行分析数据包
    all_results = []
    
    with ThreadPoolExecutor(max_workers=min(len(packages), 4), 
                           thread_name_prefix="Analyzer") as executor:
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
                analyzer.logger.error(f"数据包 {package.name} 分析失败: {e}")
                completed += 1
    
    # 3. 生成汇总报告
    summary = analyzer.generate_summary_report(all_results)
    
    # 4. 输出最终统计
    total_time = time.time() - program_start_time
    analyzer.logger.info("=" * 60)
    analyzer.logger.info("🎉 数据分析阶段完成!")
    analyzer.logger.info(f"⏱️  总耗时: {total_time:.2f}秒")
    analyzer.logger.info(f"📊 总有效样本数: {summary['total_valid_samples']}")
    analyzer.logger.info(f"📊 总无效样本数: {summary['total_invalid_samples']}")
    analyzer.logger.info(f"📁 分析结果目录: {analyzer.analysis_dir}")
    analyzer.logger.info("=" * 60)
    
    # 5. 生成阶段2的启动脚本
    generate_phase2_script(analyzer.analysis_dir, summary)

def generate_phase2_script(analysis_dir: Path, summary: Dict):
    """生成阶段2的启动脚本"""
    script_content = f'''#!/usr/bin/env python3
"""
TYJT数据集转换 - 阶段2：数据处理
基于阶段1的分析结果进行数据转换
有效样本数: {summary['total_valid_samples']}
生成时间: {datetime.now().isoformat()}
"""

import json
from pathlib import Path

# 配置
ANALYSIS_DIR = Path("{analysis_dir}")
OUTPUT_ROOT = Path("{OUTPUT_ROOT}")
VALID_SAMPLES_FILES = []

# 收集所有有效样本文件
for pkg_summary in {json.dumps(summary['package_summary'])}:
    package_name = pkg_summary['package_name']
    valid_file = ANALYSIS_DIR / f"{{package_name}}_valid_samples.json"
    if valid_file.exists():
        VALID_SAMPLES_FILES.append(valid_file)

print(f"找到 {{len(VALID_SAMPLES_FILES)}} 个有效样本文件")
print(f"总有效样本数: {summary['total_valid_samples']}")

# 这里可以继续实现阶段2的数据转换逻辑
# 建议使用 v9.2.2-phase2.py 进行实际转换

if __name__ == "__main__":
    print("阶段2启动脚本已生成")
    print("请运行: python v9.2.2-phase2.py")
'''

    script_file = analysis_dir / "phase2_start.py"
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 使脚本可执行
    script_file.chmod(0o755)
    
    analyzer = logging.getLogger("analyzer")
    analyzer.info(f"📜 生成阶段2启动脚本: {script_file}")

if __name__ == "__main__":
    main()