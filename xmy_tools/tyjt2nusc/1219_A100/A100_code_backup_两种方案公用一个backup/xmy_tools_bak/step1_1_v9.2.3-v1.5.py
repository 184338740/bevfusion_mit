#!/usr/bin/env python3
"""
tyjt数据集转nuscenes格式转换工具 - v9.2.3-phase1-v1.4 (信息完整精简版)
版本: v9.2.3-phase1-v1.4-fullinfo
修改日期: 2024-12-25
核心改进:
1. 统一metadata: 合并tyjt_calibration、nuscenes_calibration中的冗余信息
2. 保持三层结构: package → sub_packet → samples
3. 保留所有v1.3的信息丰富度
4. 简化内部实现，消除代码冗余
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

CAM_ORDER_TO_NUSC = {
    "A": "CAM_FRONT",
    "B": "CAM_FRONT_RIGHT", 
    "C": "CAM_BACK",
    "D": "CAM_FRONT_LEFT"
}

LIDAR_NUSC_NAME = "LIDAR_TOP"
DATASET_ROOT = str(Path(TYJT_ROOT).resolve())

# ==================== 统一数据结构定义 (v1.4核心改进) ====================

@dataclass
class CameraInfo:
    """相机信息 (V1.5增强版)"""
    order: str  # A/B/C/D
    tyjt_name: str  # 原始相机文件夹名
    calibration_key: str  # 标定文件中的键名
    nuscenes_name: str = ""  # NuScenes相机名
    
    # 文件信息
    image_path: str = ""
    
    # 标定信息 (V1.5增强)
    calibration: Dict[str, Any] = field(default_factory=lambda: {
        "tyjt": {
            "intrinsic": {},  # fx, fy, cx, cy等
            "sensor2map": {},  # sensor->map原始参数
            "sensor2map_matrix": []  # 4x4矩阵
        },
        "transform_matrices": {
            "sensor2map": [],  # 4x4矩阵
            "map2group": [],   # 4x4矩阵  
            "sensor2group": [] # 4x4矩阵
        },
        "nuscenes": {
            "translation": [],
            "rotation": [],
            "intrinsic": []
        }
    })

@dataclass
class SampleInfo:
    """样本信息 (统一metadata)"""
    # 基础标识
    unique_key: str
    package_name: str
    package_type: str
    road_id: str
    sub_packet_name: str
    timestamp: str
    
    # 文件路径
    files: Dict[str, Any] = field(default_factory=lambda: {
        "lidar": "",
        "label": "",
        "cameras": {}  # camera_order -> image_path
    })
    
    # 相机信息 (统一管理)
    cameras: Dict[str, CameraInfo] = field(default_factory=dict)  # order -> CameraInfo
    
    # 标定信息 (统一存储，避免冗余)
    calibration_summary: Dict[str, Any] = field(default_factory=lambda: {
        "group_key": "",
        "calibration_quality": "unknown",  # excellent/good/fair/poor
        "has_all_calibrations": False
    })
    
    # 元数据统计
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "camera_count": 0,
        "label_objects": 0,
        "timestamp_int": 0,
        "is_valid": True,
        "data_quality": "good"  # good/warning/error
    })
    
    # 处理状态
    processing: Dict[str, Any] = field(default_factory=lambda: {
        "phase1_analyzed": True,
        "phase1_time": "",
        "phase2_ready": False
    })
    
    def __post_init__(self):
        # 自动计算
        self.metadata["camera_count"] = len(self.cameras)
        
        # 设置分析时间
        if not self.processing["phase1_time"]:
            self.processing["phase1_time"] = datetime.now().isoformat()
        
        # 评估标定质量
        cameras_with_calib = sum(1 for cam in self.cameras.values() 
                               if cam.calibration.get("nuscenes"))
        if cameras_with_calib == 4:
            self.calibration_summary["calibration_quality"] = "excellent"
        elif cameras_with_calib >= 2:
            self.calibration_summary["calibration_quality"] = "good"
        elif cameras_with_calib >= 1:
            self.calibration_summary["calibration_quality"] = "fair"
        else:
            self.calibration_summary["calibration_quality"] = "poor"
        
        self.calibration_summary["has_all_calibrations"] = cameras_with_calib == 4
    
    @property
    def has_all_cameras(self) -> bool:
        """检查是否有所有4个相机"""
        return len(self.cameras) == 4 and all(
            cam.image_path for cam in self.cameras.values()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于JSON输出"""
        return {
            "unique_key": self.unique_key,
            "package_name": self.package_name,
            "package_type": self.package_type,
            "road_id": self.road_id,
            "sub_packet_name": self.sub_packet_name,
            "timestamp": self.timestamp,
            "files": self.files,
            "cameras": {order: {
                "order": cam.order,
                "tyjt_name": cam.tyjt_name,
                "calibration_key": cam.calibration_key,
                "nuscenes_name": cam.nuscenes_name,
                "image_path": cam.image_path,
                "calibration": cam.calibration
            } for order, cam in self.cameras.items()},
            "calibration_summary": self.calibration_summary,
            "metadata": self.metadata,
            "processing": self.processing
        }


@dataclass
class SubPacketInfo:
    """子包信息 (V1.5增强)"""
    name: str
    road_id: str
    calib_files: List[str] = field(default_factory=list)  # V1.5新增
    samples: List[SampleInfo] = field(default_factory=list)
    
    # 子包统计
    statistics: Dict[str, Any] = field(default_factory=lambda: {
        "sample_count": 0,
        "camera_count": 0,
        "time_range": {"min": None, "max": None},
        "quality_summary": {
            "excellent": 0,
            "good": 0,
            "fair": 0,
            "poor": 0
        }
    })
    
    def update_statistics(self):
        """更新统计信息"""
        if not self.samples:
            return
        
        self.statistics["sample_count"] = len(self.samples)
        
        # 相机数量
        camera_counts = [s.metadata["camera_count"] for s in self.samples]
        self.statistics["camera_count"] = max(camera_counts) if camera_counts else 0
        
        # 时间范围
        timestamps = [s.metadata["timestamp_int"] for s in self.samples if s.metadata["timestamp_int"] > 0]
        if timestamps:
            self.statistics["time_range"]["min"] = min(timestamps)
            self.statistics["time_range"]["max"] = max(timestamps)
        
        # 质量统计
        for sample in self.samples:
            quality = sample.calibration_summary.get("calibration_quality", "unknown")
            if quality in self.statistics["quality_summary"]:
                self.statistics["quality_summary"][quality] += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 (V1.5增强)"""
        self.update_statistics()
        return {
            "name": self.name,
            "road_id": self.road_id,
            "calib_files": self.calib_files,  # V1.5新增
            "samples": [s.to_dict() for s in self.samples],
            "statistics": self.statistics
        }


@dataclass
class PackageAnalysisResult:
    """数据包分析结果 (V1.5增强版)"""
    package_name: str
    package_type: str
    analysis_time: str
    dataset_root: str = DATASET_ROOT  # V1.5新增
    calib_files: Dict[str, List[str]] = field(default_factory=dict)  # V1.5新增: sub_packet -> [calib_files]
    sub_packets: Dict[str, SubPacketInfo] = field(default_factory=dict)
    
    # 总体统计
    overall_statistics: Dict[str, Any] = field(default_factory=lambda: {
        "total_samples": 0,
        "valid_samples": 0,
        "invalid_samples": 0,
        "sub_packet_count": 0,
        "quality_distribution": {
            "excellent": 0,
            "good": 0,
            "fair": 0,
            "poor": 0
        }
    })
    
    def add_sample(self, sample: SampleInfo, sub_packet_calib_files: List[str] = None):
        """添加样本到对应子包 (V1.5增强)"""
        sub_packet_name = sample.sub_packet_name
        
        if sub_packet_name not in self.sub_packets:
            self.sub_packets[sub_packet_name] = SubPacketInfo(
                name=sub_packet_name,
                road_id=sample.road_id,
                calib_files=sub_packet_calib_files if sub_packet_calib_files else []  # V1.5新增
            )
        
        self.sub_packets[sub_packet_name].samples.append(sample)
    
    def update_overall_statistics(self):
        """更新总体统计"""
        total_samples = 0
        valid_samples = 0
        
        for sub_packet in self.sub_packets.values():
            for sample in sub_packet.samples:
                total_samples += 1
                if sample.metadata["is_valid"]:
                    valid_samples += 1
                
                # 质量统计
                quality = sample.calibration_summary.get("calibration_quality", "unknown")
                if quality in self.overall_statistics["quality_distribution"]:
                    self.overall_statistics["quality_distribution"][quality] += 1
        
        self.overall_statistics.update({
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "invalid_samples": total_samples - valid_samples,
            "sub_packet_count": len(self.sub_packets)
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为三层结构的字典 (V1.5增强)"""
        self.update_overall_statistics()
        
        return {
            "package_info": {
                "name": self.package_name,
                "type": self.package_type,
                "analysis_time": self.analysis_time,
                "dataset_root": self.dataset_root,  # V1.5新增
                "data_version": "v1.5-enhanced"
            },
            "sub_packets": {name: sp.to_dict() for name, sp in self.sub_packets.items()},  # 直接调用sp.to_dict()
            "overall_statistics": self.overall_statistics,
            "camera_mapping": CAM_ORDER_TO_NUSC
        }

# ==================== 标定处理工具 ====================

class CalibrationProcessor:
    """标定数据处理"""
    
    @staticmethod
    def quaternion_to_rotation_matrix(q: List[float]) -> np.ndarray:
        """四元数转旋转矩阵"""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
    
    @staticmethod
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
    
    @staticmethod
    def parse_transform(calib_params: Union[Dict, List]) -> np.ndarray:
        """解析标定参数为4x4变换矩阵"""
        try:
            transform = np.eye(4, dtype=np.float64)
            
            if isinstance(calib_params, Dict):
                if 'translation' in calib_params and 'rotation' in calib_params:
                    translation = np.array(calib_params['translation'], dtype=np.float64)
                    rotation_quat = calib_params['rotation']
                    
                    if len(rotation_quat) == 4:
                        x, y, z, w = rotation_quat
                        rotation_matrix = CalibrationProcessor.quaternion_to_rotation_matrix([w, x, y, z])
                        transform[:3, :3] = rotation_matrix
                        transform[:3, 3] = translation
                
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
                    
                    rotation_matrix = CalibrationProcessor.quaternion_to_rotation_matrix([w, x, y, z])
                    transform[:3, :3] = rotation_matrix
                    transform[:3, 3] = translation
            
            elif isinstance(calib_params, List) and len(calib_params) == 7:
                tx, ty, tz, rx, ry, rz, rw = calib_params
                translation = np.array([tx, ty, tz], dtype=np.float64)
                
                rotation_matrix = CalibrationProcessor.quaternion_to_rotation_matrix([rw, rx, ry, rz])
                transform[:3, :3] = rotation_matrix
                transform[:3, 3] = translation
            
            return transform
            
        except Exception as e:
            print(f"解析标定参数失败: {e}")
            return np.eye(4, dtype=np.float64)
    
    @staticmethod
    def compute_sensor2ego(calib_data: Dict, sensor_key: str, group_key: str) -> Tuple[List[float], List[float], Dict]:
        """计算sensor到ego的变换"""
        try:
            # 获取sensor2map变换
            if sensor_key not in calib_data:
                raise KeyError(f"标定数据中找不到传感器键: {sensor_key}")
            
            sensor_calib = calib_data[sensor_key]
            sensor2map = CalibrationProcessor.parse_transform(sensor_calib)
            
            # 获取group2map变换
            if group_key not in calib_data:
                raise KeyError(f"标定数据中找不到group键: {group_key}")
            
            group_calib = calib_data[group_key]
            if isinstance(group_calib, dict) and 'transform' in group_calib:
                group_calib = group_calib['transform']
            
            group2map = CalibrationProcessor.parse_transform(group_calib)
            map2group = np.linalg.inv(group2map)
            
            # 计算sensor2ego = map2group * sensor2map
            sensor2ego = map2group @ sensor2map
            
            if not np.isfinite(sensor2ego).all():
                raise ValueError("无效的变换矩阵")
            
            # 提取平移和旋转
            translation = sensor2ego[:3, 3].tolist()
            rotation_matrix = sensor2ego[:3, :3]
            rotation = CalibrationProcessor.rotation_matrix_to_quaternion(rotation_matrix)
            
            # 构建详细标定信息
            calib_info = {
                "sensor_key": sensor_key,
                "group_key": group_key,
                "translation": translation,
                "rotation": rotation,
                "sensor2ego_matrix": sensor2ego.tolist(),
                "is_valid": True
            }
            
            return translation, rotation, calib_info
            
        except Exception as e:
            error_msg = f"计算sensor2ego变换失败: {str(e)}"
            return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], {
                "sensor_key": sensor_key,
                "group_key": group_key,
                "error": error_msg,
                "is_valid": False
            }

# ==================== 数据分析核心 ====================

class DataAnalyzerV1_4:
    """v1.4数据分析器"""
    
    def __init__(self, output_root: str):
        self.output_root = Path(output_root)
        self.analysis_dir = self.output_root / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        self.calib_cache = {}
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志"""
        log_dir = self.analysis_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("analyzer_v1_4")
        self.logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        self.logger.handlers = []
        
        # 文件处理器
        log_file = log_dir / "analysis_v1_4.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [v1.4] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def load_calibration(self, package_path: Path, package_config: Dict) -> Dict:
        """加载标定数据"""
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
                # 加载group2map
                group2map_path = calib_root / package_config["group2map_file"]
                if group2map_path.exists():
                    with open(group2map_path, 'r', encoding='utf-8') as f:
                        calib_data.update(json.load(f))
                
                # 加载camera2map
                camera2map_path = calib_root / package_config["camera2map_file"]
                if camera2map_path.exists():
                    with open(camera2map_path, 'r', encoding='utf-8') as f:
                        calib_data.update(json.load(f))
            
            elif package_type == "all_in_one":
                # 加载sensor2map
                sensor2map_path = calib_root / package_config["sensor2map_file"]
                if sensor2map_path.exists():
                    with open(sensor2map_path, 'r', encoding='utf-8') as f:
                        calib_data.update(json.load(f))
            
            self.calib_cache[cache_key] = calib_data
            self.logger.info(f"加载标定数据: {len(calib_data)} 个键")
            
        except Exception as e:
            self.logger.error(f"加载标定数据失败: {e}")
        
        return calib_data
    
    def analyze_package(self, package_dir: Path) -> PackageAnalysisResult:
        """分析数据包 (V1.5增强)"""
        package_name = package_dir.name
        
        if package_name not in DATA_PACKAGE_CONFIG:
            self.logger.warning(f"数据包 {package_name} 无配置，跳过")
            return None
        
        package_config = DATA_PACKAGE_CONFIG[package_name]
        self.logger.info(f"分析数据包: {package_name} ({package_config['type']})")
        
        result = PackageAnalysisResult(
            package_name=package_name,
            package_type=package_config["type"],
            analysis_time=datetime.now().isoformat(),
            dataset_root=str(package_dir.resolve())  # V1.5新增
        )
        
        start_time = time.time()
        
        # 加载标定数据
        calib_data = self.load_calibration(package_dir, package_config)
        if not calib_data:
            self.logger.error(f"数据包 {package_name} 标定数据加载失败")
            return result
        
        # V1.5: 收集标定文件路径
        calib_root = package_dir / package_config["calib_root"]
        calib_file_paths = []
        if calib_root.exists():
            # 收集所有标定JSON文件
            for json_file in calib_root.glob("*.json"):
                calib_file_paths.append(str(json_file.resolve()))
        
        # 分析每个道路
        for road_id, road_config in package_config["intersections"].items():
            self.logger.info(f"分析道路: {road_id}")
            group_key = road_config["group_key"]
            
            # 扫描数据集
            datasets_dir = package_dir / "datasets"
            if not datasets_dir.exists():
                self.logger.warning(f"无datasets目录: {datasets_dir}")
                continue
            
            # 处理每个子包
            for sub_packet in datasets_dir.iterdir():
                if not sub_packet.is_dir() or sub_packet.name.startswith('.'):
                    continue
                
                sub_packet_name = sub_packet.name
                self.logger.debug(f"处理子包: {sub_packet_name}")
                
                # 扫描lidar文件
                lidar_dir = sub_packet / "lidar" / "pcd"
                if not lidar_dir.exists():
                    continue
                
                # 处理每个npy文件
                for npy_file in lidar_dir.glob("*.npy"):
                    timestamp = npy_file.stem
                    if not timestamp.isdigit():
                        continue
                    
                    # 构建样本信息
                    sample = self.build_sample_info(
                        package_name, package_config["type"], road_id,
                        sub_packet_name, timestamp, npy_file,
                        sub_packet, road_config, calib_data, group_key
                    )
                    
                    # 添加到结果 (V1.5: 传递标定文件路径)
                    if sample:
                        result.add_sample(sample, calib_file_paths)
        
        # 更新统计
        result.update_overall_statistics()
        
        # 保存结果
        self.save_package_result(result)
        
        elapsed = time.time() - start_time
        self.logger.info(f"完成分析: {package_name}")
        self.logger.info(f"  子包数: {result.overall_statistics['sub_packet_count']}")
        self.logger.info(f"  总样本: {result.overall_statistics['total_samples']}")
        self.logger.info(f"  有效样本: {result.overall_statistics['valid_samples']}")
        self.logger.info(f"  耗时: {elapsed:.2f}秒")
        
        return result
    
    def build_sample_info(self, package_name: str, package_type: str, road_id: str,
                        sub_packet_name: str, timestamp: str, npy_file: Path,
                        sub_packet: Path, road_config: Dict, 
                        calib_data: Dict, group_key: str) -> Optional[SampleInfo]:
        """构建样本信息 (V1.5增强版)"""
        try:
            unique_key = f"{package_name}_{road_id}_{sub_packet_name}_{timestamp}"
            
            # 检查必需文件
            label_file = sub_packet / "lidar" / "label" / f"{timestamp}.json"
            if not label_file.exists():
                return None
            
            # 读取label统计
            label_objects = 0
            try:
                with open(label_file, 'r') as f:
                    label_data = json.load(f)
                if isinstance(label_data, dict):
                    objects = label_data.get("objects", [])
                else:
                    objects = label_data
                label_objects = len(objects) if isinstance(objects, list) else 0
            except:
                pass
            
            # 收集相机信息
            cameras = {}
            camera_files = {}
            
            # 计算map2group矩阵（用于所有相机）
            map2group_matrix = None
            if group_key in calib_data:
                group_calib = calib_data[group_key]
                if isinstance(group_calib, dict) and 'transform' in group_calib:
                    group_calib = group_calib['transform']
                group2map = CalibrationProcessor.parse_transform(group_calib)
                map2group_matrix = np.linalg.inv(group2map).tolist()
            
            for cam_config in road_config["cameras"]:
                if package_type == "hikvision":
                    camera_folder, camera_order = cam_config
                    calibration_key = camera_folder
                else:
                    camera_folder, calibration_key, camera_order = cam_config
                
                # 检查图像文件
                image_file = sub_packet / camera_folder / "image_dc" / f"{timestamp}.jpg"
                image_path = str(image_file.resolve()) if image_file.exists() else ""
                
                # 创建相机信息
                camera_info = CameraInfo(
                    order=camera_order,
                    tyjt_name=camera_folder,
                    calibration_key=calibration_key,
                    image_path=image_path
                )
                
                # V1.5: 增强标定信息
                if calibration_key in calib_data and group_key in calib_data:
                    # 获取原始TYJT标定参数
                    tyjt_calib = calib_data[calibration_key]
                    
                    # 解析传感器标定
                    sensor2map_matrix = None
                    sensor2map_params = {}
                    
                    if isinstance(tyjt_calib, dict):
                        # 提取内参参数
                        intrinsic_params = {}
                        if 'fx' in tyjt_calib and 'fy' in tyjt_calib:
                            intrinsic_params = {
                                "fx": tyjt_calib.get('fx', 1680.0),
                                "fy": tyjt_calib.get('fy', 1851.0),
                                "cx": tyjt_calib.get('cx', 960.0),
                                "cy": tyjt_calib.get('cy', 540.0)
                            }
                        
                        # 提取变换参数
                        if 'translation' in tyjt_calib and 'rotation' in tyjt_calib:
                            sensor2map_params = {
                                "translation": tyjt_calib['translation'],
                                "rotation": tyjt_calib['rotation']
                            }
                        elif all(k in tyjt_calib for k in ['tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'rw']):
                            sensor2map_params = {
                                "tx": tyjt_calib['tx'],
                                "ty": tyjt_calib['ty'],
                                "tz": tyjt_calib['tz'],
                                "rx": tyjt_calib['rx'],
                                "ry": tyjt_calib['ry'],
                                "rz": tyjt_calib['rz'],
                                "rw": tyjt_calib['rw']
                            }
                        
                        # 计算sensor2map矩阵
                        sensor2map_matrix = CalibrationProcessor.parse_transform(tyjt_calib).tolist()
                    
                    # 计算sensor2group矩阵
                    sensor2group_matrix = None
                    if sensor2map_matrix is not None and map2group_matrix is not None:
                        sensor2map_np = np.array(sensor2map_matrix)
                        map2group_np = np.array(map2group_matrix)
                        sensor2group_np = map2group_np @ sensor2map_np
                        sensor2group_matrix = sensor2group_np.tolist()
                        
                        # 提取NuScenes格式的translation和rotation
                        translation = sensor2group_np[:3, 3].tolist()
                        rotation = CalibrationProcessor.rotation_matrix_to_quaternion(sensor2group_np[:3, :3])
                    else:
                        translation = [0.0, 0.0, 0.0]
                        rotation = [1.0, 0.0, 0.0, 0.0]
                    
                    # 存储增强的标定信息
                    camera_info.calibration = {
                        "tyjt": {
                            "intrinsic": intrinsic_params,
                            "sensor2map": sensor2map_params,
                        },
                        "transform_matrices": {
                            "sensor2map": sensor2map_matrix,
                            "map2group": map2group_matrix,
                            "sensor2group": sensor2group_matrix
                        },
                        "nuscenes": {
                            "translation": translation,
                            "rotation": rotation,
                            "intrinsic": [
                                [intrinsic_params.get('fx', 1680.0), 0, intrinsic_params.get('cx', 960.0)],
                                [0, intrinsic_params.get('fy', 1851.0), intrinsic_params.get('cy', 540.0)],
                                [0, 0, 1]
                            ]
                        }
                    }
                
                cameras[camera_order] = camera_info
                if image_path:
                    camera_files[camera_order] = image_path
            
            # 检查样本有效性
            camera_count = len([cam for cam in cameras.values() if cam.image_path])
            is_valid = all([
                npy_file.exists(),
                label_file.exists(),
                camera_count >= 2,  # 至少2个相机
                group_key in calib_data
            ])
            
            # 创建样本信息
            sample = SampleInfo(
                unique_key=unique_key,
                package_name=package_name,
                package_type=package_type,
                road_id=road_id,
                sub_packet_name=sub_packet_name,
                timestamp=timestamp,
                files={
                    "lidar": str(npy_file.resolve()),
                    "label": str(label_file.resolve()),
                    "cameras": camera_files
                },
                cameras=cameras,
                calibration_summary={
                    "group_key": group_key,
                    "has_all_calibrations": all(cam.calibration.get("transform_matrices", {}).get("sensor2group") 
                                            for cam in cameras.values())
                },
                metadata={
                    "camera_count": camera_count,
                    "label_objects": label_objects,
                    "timestamp_int": int(timestamp) if timestamp.isdigit() else 0,
                    "is_valid": is_valid,
                    "data_quality": "good" if camera_count >= 3 and is_valid else "warning"
                }
            )
            
            return sample
            
        except Exception as e:
            self.logger.error(f"构建样本信息失败 {timestamp}: {e}")
            return None

    def save_package_result(self, result: PackageAnalysisResult):
        """保存数据包分析结果"""
        if not result:
            return
        
        # 只保存一个三层结构数据文件
        main_file = self.analysis_dir / f"{result.package_name}_structured_v1_4.json"
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"保存结构化数据: {main_file.name} ({result.overall_statistics['total_samples']} 个样本)")
    

    def generate_summary_report(self, all_results: List[PackageAnalysisResult]):
        """生成汇总报告"""
        summary = {
            "analysis_summary": {
                "total_packages": len(all_results),
                "total_samples": 0,
                "valid_samples": 0,
                "sub_packets": 0,
                "analysis_time": datetime.now().isoformat(),
                "data_version": "v1.4-unified"
            },
            "packages": []
        }
        
        for result in all_results:
            if not result:
                continue
            
            # 更新总计
            summary["analysis_summary"]["total_samples"] += result.overall_statistics["total_samples"]
            summary["analysis_summary"]["valid_samples"] += result.overall_statistics["valid_samples"]
            summary["analysis_summary"]["sub_packets"] += result.overall_statistics["sub_packet_count"]
            
            # 包级别信息
            package_info = {
                "name": result.package_name,
                "type": result.package_type,
                "sub_packets": result.overall_statistics["sub_packet_count"],
                "samples": result.overall_statistics["total_samples"],
                "valid_samples": result.overall_statistics["valid_samples"],
                "quality_distribution": result.overall_statistics["quality_distribution"]
            }
            summary["packages"].append(package_info)
        
        # 保存汇总报告
        summary_file = self.analysis_dir / "summary_report_v1_4.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 输出简要报告
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 分析汇总报告 (v1.4)")
        self.logger.info("="*60)
        self.logger.info(f"数据包数: {summary['analysis_summary']['total_packages']}")
        self.logger.info(f"总样本数: {summary['analysis_summary']['total_samples']}")
        self.logger.info(f"有效样本: {summary['analysis_summary']['valid_samples']}")
        self.logger.info(f"子包总数: {summary['analysis_summary']['sub_packets']}")
        
        for pkg in summary["packages"]:
            self.logger.info(f"\n📦 {pkg['name']} ({pkg['type']}):")
            self.logger.info(f"  子包数: {pkg['sub_packets']}")
            self.logger.info(f"  样本数: {pkg['samples']} (有效: {pkg['valid_samples']})")
            self.logger.info(f"  质量分布: {pkg['quality_distribution']}")
        
        self.logger.info("="*60)

# ==================== 主函数 ====================
def main():
    """主函数"""
    analyzer = DataAnalyzerV1_4(OUTPUT_ROOT)
    
    analyzer.logger.info("=" * 60)
    analyzer.logger.info("🔍 TYJT数据集分析工具 v9.2.3-phase1-v1.4 (信息完整版)")
    analyzer.logger.info(f"📁 数据集根目录: {TYJT_ROOT}")
    analyzer.logger.info(f"📊 输出目录: {analyzer.analysis_dir}")
    analyzer.logger.info("=" * 60)
    
    # 查找数据包
    root_path = Path(TYJT_ROOT)
    packages = []
    
    for item in root_path.iterdir():
        if item.is_dir() and item.name in DATA_PACKAGE_CONFIG:
            packages.append(item)
            analyzer.logger.info(f"✅ 找到数据包: {item.name}")
    
    if not packages:
        analyzer.logger.error("❌ 未找到任何有效数据包")
        return
    
    analyzer.logger.info(f"📦 总共找到 {len(packages)} 个有效数据包")
    
    # 分析数据包
    all_results = []
    total_start = time.time()
    
    for package in packages:
        try:
            result = analyzer.analyze_package(package)
            if result:
                all_results.append(result)
        except Exception as e:
            analyzer.logger.error(f"❌ 数据包 {package.name} 分析失败: {e}")
    
    # 生成汇总报告
    if all_results:
        analyzer.generate_summary_report(all_results)
    
    total_time = time.time() - total_start
    analyzer.logger.info(f"\n🎉 分析完成! 总耗时: {total_time:.2f}秒")
    analyzer.logger.info(f"📁 分析结果: {analyzer.analysis_dir}")
    
    # 输出关键文件信息
    analyzer.logger.info("\n📁 生成的关键文件 (v1.4):")
    for result in all_results:
        if result:
            analyzer.logger.info(f"  - {result.package_name}:")
            
            # 结构化数据文件
            structured_file = analyzer.analysis_dir / f"{result.package_name}_structured_v1_4.json"
            if structured_file.exists():
                analyzer.logger.info(f"    ✓ structured_v1_4.json: {result.overall_statistics['total_samples']} 个样本")
            
            # 汇总报告
            summary_file = analyzer.analysis_dir / "summary_report_v1_4.json"
            if summary_file.exists():
                analyzer.logger.info(f"    📊 summary_report_v1_4.json: 汇总统计")
    
    analyzer.logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 程序运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)