#!/usr/bin/env python3
"""
tyjt数据集转nuscenes格式转换工具 - v9.2.3-phase1-v1.9 (信息完整修复版)
版本: v9.2.3-phase1-v1.9-fullinfo-fixed
修改日期: 2024-12-25
核心改进:
1. 修复SampleInfo构建错误：cannot unpack non-iterable SampleInfo object
2. 添加相机径向畸变参数 (k1, k2, k3, p1, p2)
3. 添加激光雷达标定信息 (tyjt和nuscenes格式)
4. 修复子包统计为0的问题
5. 增强错误处理和日志记录
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

Mode = "A100_all"
if Mode == "Local":
    TYJT_ROOT = "/mnt/dataset/tyjt_RawData_all/"
    OUTPUT_ROOT = "./output-1226-v1.9/step1/nuscenes_tyjt"
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
elif Mode == "A100_sub":
    TYJT_ROOT = "/cephfsdata/users/lishan/00_Data/00_RawData"
    OUTPUT_ROOT = "./output-1226-v9.2.3-sub/step1/nuscenes_tyjt"
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
elif Mode == "A100_all":
    TYJT_ROOT = "/cephfsdata/users/lishan/00_Data/00_RawData"
    OUTPUT_ROOT = "./output-1226-v9.2.3-e/step1/nuscenes_tyjt"
    NUSC_VERSION = "v1.0-tyjt"
    # 13个数据包规格配置（按实际数据包名匹配，补充相机顺序标识）
    DATA_PACKAGE_CONFIG = {
        # 海康相机产品（9个）
        # "2d3d_20250114": {
        #     "type": "hikvision",
        #     "calib_root": "calib/2d3d_20250114",
        #     "group2map_file": "group2map_calib.json",
        #     "camera2map_file": "camera2map_calib.json",
        #     "intersections": {
        #         "R9": {
        #             "group_key": "G32050700009M00",
        #             "cameras": [  # (相机文件夹名, 相机顺序A/B/C/D)
        #                 ("R9_Aw_CamS", "A"),
        #                 ("R9_Bn_CamW", "B"),
        #                 ("R9_Ce_CamN", "C"),
        #                 ("R9_Ds_CamE", "D")
        #             ]
        #         }
        #     }
        # },
        # "2d3d_20250218": {
        #     "type": "hikvision",
        #     "calib_root": "calib/2d3d_20250218",
        #     "group2map_file": "group2map_calib.json",
        #     "camera2map_file": "camera2map_calib.json",
        #     "intersections": {
        #         "R03": {
        #             "group_key": "G32050700003M00",
        #             "cameras": [  # C相机缺失
        #                 ("R3_Aw_CamS", "A"),
        #                 ("R3_Bn_CamW", "B"),
        #                 ("R3_Ds_CamE", "D")
        #             ]
        #         },
        #         "R04": {
        #             "group_key": "G32050700004M00",
        #             "cameras": [  # C相机缺失
        #                 ("R4_Aw_CamS", "A"),
        #                 ("R4_Bn_CamW", "B"),
        #                 ("R4_Ds_CamE", "D")
        #             ]
        #         },
        #         "R10": {
        #             "group_key": "G32050700010M00",
        #             "cameras": [
        #                 ("R10_Aw_CamS", "A"),
        #                 ("R10_Bn_CamW", "B"),
        #                 ("R10_Ce_CamN", "C"),
        #                 ("R10_Ds_CamE", "D")
        #             ]
        #         }
        #     }
        # },
        # "2d3d_20250221": {
        #     "type": "hikvision",
        #     "calib_root": "calib/2d3d_20250221",
        #     "group2map_file": "group2map_calib.json",
        #     "camera2map_file": "camera2map_calib.json",
        #     "intersections": {
        #         "R03": {
        #             "group_key": "G32050700003M00",
        #             "cameras": [  # C相机缺失
        #                 ("R3_Aw_CamS", "A"),
        #                 ("R3_Bn_CamW", "B"),
        #                 ("R3_Ds_CamE", "D")
        #             ]
        #         },
        #         "R04": {
        #             "group_key": "G32050700004M00",
        #             "cameras": [  # C相机缺失
        #                 ("R4_Aw_CamS", "A"),
        #                 ("R4_Bn_CamW", "B"),
        #                 ("R4_Ds_CamE", "D")
        #             ]
        #         },
        #         "R10": {
        #             "group_key": "G32050700010M00",
        #             "cameras": [
        #                 ("R10_Aw_CamS", "A"),
        #                 ("R10_Bn_CamW", "B"),
        #                 ("R10_Ce_CamN", "C"),
        #                 ("R10_Ds_CamE", "D")
        #             ]
        #         }
        #     }
        # },
        # "2d3d4d_20241122_wuxi": {
        #     "type": "hikvision",
        #     "calib_root": "calib/2d3d4d_20241122",
        #     "group2map_file": "group2map_calib.json",
        #     "camera2map_file": "camera2map_calib.json",
        #     "intersections": {
        #         "R01": {"group_key": "G32020500001M00", "cameras": []}  # 无有效相机
        #     }
        # },
        # "2d3d4d_20241218": {
        #     "type": "hikvision",
        #     "calib_root": "calib/2d3d4d_20241218",
        #     "group2map_file": "group2map_calib.json",
        #     "camera2map_file": "camera2map_calib.json",
        #     "intersections": {
        #         "R02": {
        #             "group_key": "G32050700002M00",
        #             "cameras": [
        #                 ("R2_Aw_CamS", "A"),
        #                 ("R2_Bn_CamW", "B"),
        #                 ("R2_Ce_CamN", "C"),
        #                 ("R2_Ds_CamE", "D")
        #             ]
        #         }
        #     }
        # },
        # "2d3d4d_20250117": {
        #     "type": "hikvision",
        #     "calib_root": "calib/2d3d4d_20250117",
        #     "group2map_file": "group2map_calib.json",
        #     "camera2map_file": "camera2map_calib.json",
        #     "intersections": {
        #         "R26": {
        #             "group_key": "G32050700026M00",
        #             "cameras": [
        #                 ("R26_Aw_CamS", "A"),
        #                 ("R26_Bn_CamW", "B"),
        #                 ("R26_Ce_CamN", "C"),
        #                 ("R26_Ds_CamE", "D")
        #             ]
        #         }
        #     }
        # },
        # "2d3d4d_20250213": {
        #     "type": "hikvision",
        #     "calib_root": "calib/2d3d4d_20250213",
        #     "group2map_file": "group2map_calib.json",
        #     "camera2map_file": "camera2map_calib.json",
        #     "intersections": {
        #         "R26": {
        #             "group_key": "G32050700026M00",
        #             "cameras": [
        #                 ("R26_Aw_CamS", "A"),
        #                 ("R26_Bn_CamW", "B"),
        #                 ("R26_Ce_CamN", "C"),
        #                 ("R26_Ds_CamE", "D")
        #             ]
        #         }
        #     }
        # },
        # "2d3d4d_20250218": {
        #     "type": "hikvision",
        #     "calib_root": "calib/2d3d4d_20250218",
        #     "group2map_file": "group2map_calib.json",
        #     "camera2map_file": "camera2map_calib.json",
        #     "intersections": {
        #         "R26": {
        #             "group_key": "G32050700026M00",
        #             "cameras": [
        #                 ("R26_Aw_CamS", "A"),
        #                 ("R26_Bn_CamW", "B"),
        #                 ("R26_Ce_CamN", "C"),
        #                 ("R26_Ds_CamE", "D")
        #             ]
        #         }
        #     }
        # },
        # "2d3d4d_20250403": {
        #     "type": "hikvision",
        #     "calib_root": "calib/2d3d4d_20250403",
        #     "group2map_file": "group2map_calib.json",
        #     "camera2map_file": "camera2map_calib.json",
        #     "intersections": {
        #         "R26": {
        #             "group_key": "G32050700026M00",
        #             "cameras": [
        #                 ("R26_Aw_CamS", "A"),
        #                 ("R26_Bn_CamW", "B"),
        #                 ("R26_Ce_CamN", "C"),
        #                 ("R26_Ds_CamE", "D")
        #             ]
        #         }
        #     }
        # },
        # "2d3d4d_20250408": {
        #     "type": "hikvision",
        #     "calib_root": "calib/2d3d4d_20250408",
        #     "group2map_file": "group2map_calib.json",
        #     "camera2map_file": "camera2map_calib.json",
        #     "intersections": {
        #         "R26": {
        #             "group_key": "G32050700026M00",
        #             "cameras": [
        #                 ("R26_Aw_CamS", "A"),
        #                 ("R26_Bn_CamW", "B"),
        #                 ("R26_Ce_CamN", "C"),
        #                 ("R26_Ds_CamE", "D")
        #             ]
        #         }
        #     }
        # },
        # # 一体机产品（3个）
        # "2d3d4d_20250618_weiyuan": {
        #     "type": "all_in_one",
        #     "calib_root": "info/G51102400002M00",
        #     "sensor2map_file": "sensor2map_calib.json",
        #     "intersections": {
        #         "R02": {
        #             "group_key": "group2map",
        #             "cameras": [  # (相机文件夹名, 标定键, 相机顺序A/B/C/D)
        #                 ("SC_2A_CamR", "SC_2A_CamR_new", "A"),
        #                 ("SC_2B_CamR", "SC_2B_CamR_new", "B"),
        #                 ("SC_2C_CamR", "SC_2C_CamR_new", "C"),
        #                 ("SC_2D_CamR", "SC_2D_CamR_new", "D")
        #             ]
        #         }
        #     }
        # }
        # ,
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

# ==================== 核心配置 ====================

CAM_ORDER_TO_NUSC = {
    "A": "CAM_FRONT",
    "B": "CAM_FRONT_RIGHT", 
    "C": "CAM_BACK",
    "D": "CAM_FRONT_LEFT"
}

LIDAR_NUSC_NAME = "LIDAR_TOP"
DATASET_ROOT = str(Path(TYJT_ROOT).resolve())

# ==================== 统一数据结构定义 (v1.9核心改进) ====================

@dataclass
class CameraInfo:
    """相机信息 (V1.8增强版，包含径向畸变和去畸变状态)"""
    order: str  # A/B/C/D
    tyjt_name: str  # 原始相机文件夹名
    calibration_key: str  # 标定文件中的键名
    nuscenes_name: str = ""  # NuScenes相机名
    
    # 文件信息
    image_path: str = ""
    
    # 标定信息 (V1.8增强)
    calibration: Dict[str, Any] = field(default_factory=lambda: {
        "tyjt": {
            "intrinsic": {},  # fx, fy, cx, cy等
            "radial_distortion": {  # 畸变参数
                "k1": 0.0,
                "k2": 0.0,
                "k3": 0.0,
                "p1": 0.0,
                "p2": 0.0
            },
            "sensor2map": {},  # sensor->map原始参数
            "sensor2map_matrix": [],  # 4x4矩阵
            "distortion_correction_status": "unknown"  # V1.8新增：去畸变状态
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
    """样本信息 (统一metadata，包含激光雷达信息)"""
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
    
    # 激光雷达信息 (新增)
    lidar: Dict[str, Any] = field(default_factory=lambda: {
        "type": "robosense_128",
        "calibration": {
            "tyjt": {
                "sensor2map": {},  # 激光雷达到地图的变换参数
                "sensor2map_matrix": []  # 4x4矩阵
            },
            "transform_matrices": {
                "sensor2map": [],  # 4x4矩阵
                "map2group": [],   # 4x4矩阵
                "sensor2group": [] # 4x4矩阵
            },
            "nuscenes": {
                "translation": [],  # 激光雷达到ego的平移
                "rotation": []      # 激光雷达到ego的旋转
            }
        },
        "point_count": 0,
        "range_stats": {
            "max_range": 0.0,
            "min_range": 0.0,
            "avg_range": 0.0
        }
    })
    
    # 标定信息 (统一存储，避免冗余)
    calibration_summary: Dict[str, Any] = field(default_factory=lambda: {
        "group_key": "",
        "calibration_quality": "unknown",  # excellent/good/fair/poor
        "has_all_calibrations": False,
        "has_lidar_calibration": False  # 新增：激光雷达标定状态
    })
    
    # 元数据统计
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "camera_count": 0,
        "label_objects": 0,
        "timestamp_int": 0,
        "is_valid": True,
        "data_quality": "good",  # good/warning/error
        "lidar_point_count": 0,  # 新增：激光雷达点数
        "has_lidar_data": False  # 新增：是否有激光雷达数据
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
        cameras_with_calib = 0
        for cam in self.cameras.values():
            if cam.calibration.get("nuscenes", {}).get("translation") and cam.calibration.get("nuscenes", {}).get("rotation"):
                cameras_with_calib += 1
        
        if cameras_with_calib == 4:
            self.calibration_summary["calibration_quality"] = "excellent"
        elif cameras_with_calib >= 2:
            self.calibration_summary["calibration_quality"] = "good"
        elif cameras_with_calib >= 1:
            self.calibration_summary["calibration_quality"] = "fair"
        else:
            self.calibration_summary["calibration_quality"] = "poor"
        
        self.calibration_summary["has_all_calibrations"] = cameras_with_calib == 4
        
        # 激光雷达标定状态
        self.calibration_summary["has_lidar_calibration"] = bool(
            self.lidar["calibration"]["nuscenes"]["translation"] and
            self.lidar["calibration"]["nuscenes"]["rotation"]
        )
    
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
            "lidar": self.lidar,  # 新增：激光雷达信息
            "calibration_summary": self.calibration_summary,
            "metadata": self.metadata,
            "processing": self.processing
        }


@dataclass
class SubPacketInfo:
    """子包信息 (v1.9增强)"""
    name: str
    road_id: str
    calib_files: List[str] = field(default_factory=list)  # v1.9新增
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
        """转换为字典 (v1.9增强)"""
        self.update_statistics()
        return {
            "name": self.name,
            "road_id": self.road_id,
            "calib_files": self.calib_files,
            "samples": [s.to_dict() for s in self.samples],
            "statistics": self.statistics
        }


@dataclass
class PackageAnalysisResult:
    """数据包分析结果 (v1.9增强版)"""
    package_name: str
    package_type: str
    analysis_time: str
    dataset_root: str = DATASET_ROOT
    calib_files: Dict[str, List[str]] = field(default_factory=dict)  # sub_packet -> [calib_files]
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
        """转换为三层结构的字典 (v1.9增强)"""
        self.update_overall_statistics()
        
        return {
            "package_info": {
                "name": self.package_name,
                "type": self.package_type,
                "analysis_time": self.analysis_time,
                "dataset_root": self.dataset_root,
                "data_version": "v1.9-enhanced"
            },
            "sub_packets": {name: sp.to_dict() for name, sp in self.sub_packets.items()},
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
        """解析标定参数为4x4变换矩阵 - V1.9修复版"""
        try:
            transform = np.eye(4, dtype=np.float64)
            
            if isinstance(calib_params, Dict):
                # 情况1: hikvision格式 - 有嵌套的transform
                if 'transform' in calib_params:
                    # 处理: {"transform": {"tx": ..., "ty": ..., ...}}
                    actual_params = calib_params['transform']
                    return CalibrationProcessor.parse_transform(actual_params)
                
                # 情况2: all_in_one格式 - 直接的tx/ty/tz/rx/ry/rz/rw
                elif all(key in calib_params for key in ['tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'rw']):
                    translation = np.array([
                        float(calib_params['tx']),
                        float(calib_params['ty']), 
                        float(calib_params['tz'])
                    ], dtype=np.float64)
                    
                    # TYJT格式：[rx, ry, rz, rw] = [x, y, z, w]
                    rx = float(calib_params['rx'])  # x
                    ry = float(calib_params['ry'])  # y
                    rz = float(calib_params['rz'])  # z
                    rw = float(calib_params['rw'])  # w
                    
                    # 传递给函数：[rw, rx, ry, rz] = [w, x, y, z]
                    rotation_matrix = CalibrationProcessor.quaternion_to_rotation_matrix([rw, rx, ry, rz])
                    transform[:3, :3] = rotation_matrix
                    transform[:3, 3] = translation
                
                # 情况3: translation/rotation格式
                elif 'translation' in calib_params and 'rotation' in calib_params:
                    translation = np.array(calib_params['translation'], dtype=np.float64)
                    rotation_quat = calib_params['rotation']
                    
                    if len(rotation_quat) == 4:
                        # 这里需要确认格式！可能是[x, y, z, w]或[w, x, y, z]
                        # 根据TYJT惯例，假设是[x, y, z, w]
                        x, y, z, w = rotation_quat
                        # 传递给函数：[w, x, y, z]
                        rotation_matrix = CalibrationProcessor.quaternion_to_rotation_matrix([w, x, y, z])
                        transform[:3, :3] = rotation_matrix
                        transform[:3, 3] = translation
                else:
                    # 未知格式，返回单位矩阵
                    print(f"警告：未知的标定参数格式，可用键: {list(calib_params.keys())}")
                    return np.eye(4, dtype=np.float64)
            
            elif isinstance(calib_params, List):
                # 情况4: 列表格式 [tx, ty, tz, rx, ry, rz, rw]
                if len(calib_params) == 7:
                    tx, ty, tz, rx, ry, rz, rw = [float(x) for x in calib_params]
                    translation = np.array([tx, ty, tz], dtype=np.float64)
                    
                    # 传递给函数：[rw, rx, ry, rz] = [w, x, y, z]
                    rotation_matrix = CalibrationProcessor.quaternion_to_rotation_matrix([rw, rx, ry, rz])
                    transform[:3, :3] = rotation_matrix
                    transform[:3, 3] = translation
                else:
                    print(f"警告：不支持的列表格式长度: {len(calib_params)}")
                    return np.eye(4, dtype=np.float64)
            
            else:
                print(f"警告：不支持的标定参数类型: {type(calib_params)}")
                return np.eye(4, dtype=np.float64)
            
            return transform
            
        except Exception as e:
            print(f"解析标定参数失败: {e}")
            import traceback
            traceback.print_exc()
            return np.eye(4, dtype=np.float64)

# ==================== 数据分析核心 ====================

class DataAnalyzerV1_6:
    """v1.9数据分析器"""
    
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
        
        self.logger = logging.getLogger("analyzer")
        self.logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        self.logger.handlers = []
        
        # 文件处理器
        log_file = log_dir / "analysis.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [v1.9] - %(message)s',
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
                self.logger.warning(f"标定目录不存在: {calib_root}")
                return {}
            
            package_type = package_config["type"]
            
            if package_type == "hikvision":
                # 加载group2map
                group2map_path = calib_root / package_config["group2map_file"]
                if group2map_path.exists():
                    with open(group2map_path, 'r', encoding='utf-8') as f:
                        calib_data.update(json.load(f))
                    self.logger.info(f"加载group2map: {group2map_path.name}")
                
                # 加载camera2map
                camera2map_path = calib_root / package_config["camera2map_file"]
                if camera2map_path.exists():
                    with open(camera2map_path, 'r', encoding='utf-8') as f:
                        calib_data.update(json.load(f))
                    self.logger.info(f"加载camera2map: {camera2map_path.name}")
            
            elif package_type == "all_in_one":
                # 加载sensor2map
                sensor2map_path = calib_root / package_config["sensor2map_file"]
                if sensor2map_path.exists():
                    with open(sensor2map_path, 'r', encoding='utf-8') as f:
                        calib_data.update(json.load(f))
                    self.logger.info(f"加载sensor2map: {sensor2map_path.name}")
            
            self.calib_cache[cache_key] = calib_data
            self.logger.info(f"加载标定数据: {len(calib_data)} 个键")
            
        except Exception as e:
            self.logger.error(f"加载标定数据失败: {e}")
        
        return calib_data
    
    def analyze_package(self, package_dir: Path) -> PackageAnalysisResult:
        """分析数据包 (v1.9修复版)"""
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
            dataset_root=str(package_dir.resolve())
        )
        
        start_time = time.time()
        
        # 加载标定数据
        calib_data = self.load_calibration(package_dir, package_config)
        if not calib_data:
            self.logger.error(f"数据包 {package_name} 标定数据加载失败")
            return result
        
        # 收集标定文件路径
        calib_root = package_dir / package_config["calib_root"]
        calib_file_paths = []
        if calib_root.exists():
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
            sub_packet_count = 0
            valid_samples_total = 0
            for sub_packet in datasets_dir.iterdir():
                if not sub_packet.is_dir() or sub_packet.name.startswith('.'):
                    continue
                
                sub_packet_name = sub_packet.name
                sub_packet_count += 1
                
                # 创建子包信息对象
                sub_packet_info = SubPacketInfo(
                    name=sub_packet_name,
                    road_id=road_id,
                    calib_files=calib_file_paths.copy() if calib_file_paths else []
                )
                
                # 扫描lidar文件
                lidar_dir = sub_packet / "lidar" / "pcd"
                if not lidar_dir.exists():
                    self.logger.debug(f"无lidar目录: {lidar_dir}")
                    continue
                
                # 处理每个npy文件
                valid_samples_in_subpacket = 0
                npy_files = list(lidar_dir.glob("*.npy"))
                self.logger.info(f"子包 {sub_packet_name}: 找到 {len(npy_files)} 个npy文件")
                
                for npy_file in npy_files:
                    timestamp = npy_file.stem
                    if not timestamp.isdigit():
                        continue
                    
                    # 构建样本信息
                    sample = self.build_sample_info(
                        package_name, package_config["type"], road_id,
                        sub_packet_name, timestamp, npy_file,
                        sub_packet, road_config, calib_data, group_key
                    )
                    
                    # 添加到子包
                    if sample:
                        sub_packet_info.samples.append(sample)
                        valid_samples_in_subpacket += 1
                        valid_samples_total += 1
                    else:
                        self.logger.debug(f"跳过无效样本: {timestamp}")
                
                # 如果有有效样本，添加到结果
                if sub_packet_info.samples:
                    result.sub_packets[sub_packet_name] = sub_packet_info
                    self.logger.info(f"子包 {sub_packet_name}: {valid_samples_in_subpacket} 个有效样本")
                else:
                    self.logger.debug(f"子包 {sub_packet_name}: 无有效样本")
            
            self.logger.info(f"道路 {road_id}: {sub_packet_count} 个子包, {valid_samples_total} 个有效样本")
        
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
        """构建样本信息 (V1.8修复版 - 激光雷达标定=group2map)"""
        
        # 在函数开头定义所有变量
        unique_key = f"{package_name}_{road_id}_{sub_packet_name}_{timestamp}"
        label_objects = 0
        cameras = {}
        camera_files = {}
        map2group_matrix = None
        
        # 激光雷达标定 - 初始化
        lidar_calibration = {
            "tyjt": {
                "sensor2map": {},      # 这里存储group2map的原始参数
                "sensor2map_matrix": [] # group2map矩阵
            },
            "transform_matrices": {
                "sensor2map": [],      # group2map矩阵（lidar->map）
                "map2group": [],       # map->group矩阵（group2map的逆）
                "sensor2group": []     # lidar->group矩阵（单位矩阵）
            },
            "nuscenes": {
                "translation": [],     # 激光雷达坐标系=ego坐标系
                "rotation": []         # 激光雷达坐标系=ego坐标系
            }
        }
        
        lidar_point_count = 0
        lidar_range_stats = {"max_range": 0.0, "min_range": 0.0, "avg_range": 0.0}
        
        try:
            # 1. 检查必需文件
            label_file = sub_packet / "lidar" / "label" / f"{timestamp}.json"
            if not label_file.exists():
                self.logger.debug(f"标签文件不存在: {label_file}")
                return None
            
            # 2. 读取label统计
            try:
                with open(label_file, 'r') as f:
                    label_data = json.load(f)
                if isinstance(label_data, dict):
                    objects = label_data.get("objects", [])
                else:
                    objects = label_data
                label_objects = len(objects) if isinstance(objects, list) else 0
            except Exception as e:
                self.logger.warning(f"读取标签文件失败 {label_file}: {e}")
            
            # 3. 检查组标定键是否存在
            if group_key not in calib_data:
                self.logger.warning(f"标定数据中缺少group_key: {group_key}")
                return None
            
            # 4. 获取group2map参数 - 这就是激光雷达到map的变换
            group_calib = calib_data[group_key]

            # V1.9修复：统一处理两种格式
            if isinstance(group_calib, dict):
                # 检查是哪种格式
                if 'transform' in group_calib:
                    # hikvision格式：有嵌套的transform
                    group2map_params = group_calib['transform']
                    self.logger.debug(f"hikvision格式 - group2map参数: {group2map_params}")
                elif all(k in group_calib for k in ['tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'rw']):
                    # all_in_one格式：直接就是tx/ty/tz/rx/ry/rz/rw
                    group2map_params = group_calib
                    self.logger.debug(f"all_in_one格式 - group2map参数: {group2map_params}")
                else:
                    # 未知格式，尝试作为直接参数处理
                    group2map_params = group_calib
                    self.logger.warning(f"未知的group2map格式，键: {list(group_calib.keys())}")
            else:
                # 直接就是参数（列表或其他格式）
                group2map_params = group_calib
                self.logger.debug(f"直接参数格式 - group2map参数: {group2map_params}")
            # 5. 提取激光雷达标定参数（sensor2map = group2map）
            lidar_sensor2map_params = {}
            if isinstance(group2map_params, dict):
                if 'tx' in group2map_params and 'ty' in group2map_params and 'tz' in group2map_params:
                    # 提取tx, ty, tz, rx, ry, rz, rw格式
                    lidar_sensor2map_params = {
                        "tx": group2map_params['tx'],
                        "ty": group2map_params['ty'],
                        "tz": group2map_params['tz'],
                        "rx": group2map_params.get('rx', 0.0),
                        "ry": group2map_params.get('ry', 0.0),
                        "rz": group2map_params.get('rz', 0.0),
                        "rw": group2map_params.get('rw', 1.0)
                    }
                elif 'translation' in group2map_params and 'rotation' in group2map_params:
                    # 提取translation, rotation格式
                    lidar_sensor2map_params = {
                        "translation": group2map_params['translation'],
                        "rotation": group2map_params['rotation']
                    }
            
            # 6. 计算变换矩阵
            try:
                # 计算group2map矩阵（激光雷达到map）
                group2map_matrix = CalibrationProcessor.parse_transform(group2map_params).tolist()
                
                # 计算map2group矩阵（map到激光雷达）
                group2map_np = np.array(group2map_matrix)
                map2group_np = np.linalg.inv(group2map_np)
                map2group_matrix = map2group_np.tolist()
                
                # 激光雷达坐标系就是ego坐标系，所以lidar->group是单位矩阵
                lidar_sensor2group_matrix = np.eye(4).tolist()
                
                # lidar就是ego
                lidar_translation = [0.0, 0.0, 0.0]  # 单位：米
                lidar_rotation = [1.0, 0.0, 0.0, 0.0]  # 四元数 [w, x, y, z]，表示无旋转

                # 存储到激光雷达标定中
                lidar_calibration["tyjt"]["sensor2map"] = lidar_sensor2map_params
                lidar_calibration["tyjt"]["sensor2map_matrix"] = group2map_matrix
                lidar_calibration["transform_matrices"]["sensor2map"] = group2map_matrix
                lidar_calibration["transform_matrices"]["map2group"] = map2group_matrix
                lidar_calibration["transform_matrices"]["sensor2group"] = lidar_sensor2group_matrix
                lidar_calibration["nuscenes"]["translation"] = lidar_translation
                lidar_calibration["nuscenes"]["rotation"] = lidar_rotation
                
                # 激光雷达标定状态
                has_lidar_calib = True
                
                self.logger.debug(f"设置激光雷达标定: translation={lidar_translation}, rotation={lidar_rotation}")
                self.logger.debug(f"激光雷达标定处理完成: group2map = sensor2map")
                
            except Exception as e:
                self.logger.warning(f"计算激光雷达标定矩阵失败: {e}")
                map2group_matrix = np.eye(4).tolist()
                lidar_calibration["transform_matrices"]["map2group"] = map2group_matrix
                has_lidar_calib = False
            
            # 7. 激光雷达点云统计
            try:
                if npy_file.exists():
                    point_cloud = np.load(npy_file, allow_pickle=False)
                    lidar_point_count = point_cloud.shape[0]
                    
                    if point_cloud.shape[0] > 0 and point_cloud.shape[1] >= 3:
                        distances = np.linalg.norm(point_cloud[:, :3], axis=1)
                        lidar_range_stats = {
                            "max_range": float(np.max(distances)),
                            "min_range": float(np.min(distances)),
                            "avg_range": float(np.mean(distances))
                        }
            except Exception as e:
                self.logger.warning(f"读取激光雷达数据失败 {npy_file}: {e}")
            
            # 8. 处理相机信息
            camera_count = 0
            for cam_config in road_config.get("cameras", []):
                # 解析相机配置
                if package_type == "hikvision":
                    if len(cam_config) >= 2:
                        camera_folder, camera_order = cam_config[0], cam_config[1]
                        calibration_key = camera_folder
                    else:
                        continue
                else:
                    if len(cam_config) >= 3:
                        camera_folder, calibration_key, camera_order = cam_config[0], cam_config[1], cam_config[2]
                    else:
                        continue
                
                # 检查图像文件
                image_file = sub_packet / camera_folder / "image_dc" / f"{timestamp}.jpg"
                image_path = str(image_file.resolve()) if image_file.exists() else ""
                
                # 创建相机信息
                nuscenes_name = CAM_ORDER_TO_NUSC.get(camera_order, "")
                camera_info = CameraInfo(
                    order=camera_order,
                    tyjt_name=camera_folder,
                    calibration_key=calibration_key,
                    nuscenes_name=nuscenes_name,
                    image_path=image_path
                )
                
                # 处理相机标定
                if calibration_key in calib_data:
                    tyjt_calib = calib_data[calibration_key]
                    
                    # 提取标定参数
                    intrinsic_params = {}
                    radial_distortion = {"k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0}
                    sensor2map_params = {}
                    distortion_correction_status = "unknown"  # V1.8新增
                    
                    if isinstance(tyjt_calib, dict):
                        # 内参
                        if 'fx' in tyjt_calib and 'fy' in tyjt_calib:
                            intrinsic_params = {
                                "fx": tyjt_calib.get('fx', 1680.0),
                                "fy": tyjt_calib.get('fy', 1851.0),
                                "cx": tyjt_calib.get('cx', 960.0),
                                "cy": tyjt_calib.get('cy', 540.0)
                            }
                        
                        # V1.8: 处理径向畸变参数
                        if 'radial_distortion' in tyjt_calib:
                            distortion_array = tyjt_calib['radial_distortion']
                            if isinstance(distortion_array, list):
                                array_len = len(distortion_array)
                                
                                # all_in_one格式: 8个元素 [k1, k2, p1, p2, k3, k4, k5, k6]
                                if array_len == 8:
                                    radial_distortion = {
                                        "k1": float(distortion_array[0]),  # k1
                                        "k2": float(distortion_array[1]),  # k2
                                        "p1": float(distortion_array[2]),  # p1
                                        "p2": float(distortion_array[3]),  # p2
                                        "k3": float(distortion_array[4])   # k3
                                    }
                                    
                                # hikvision格式: 5个元素 [k1, k2, p1, p2, k3]
                                elif array_len == 5:
                                    radial_distortion = {
                                        "k1": float(distortion_array[0]),  # k1
                                        "k2": float(distortion_array[1]),  # k2
                                        "p1": float(distortion_array[2]),  # p1
                                        "p2": float(distortion_array[3]),  # p2
                                        "k3": float(distortion_array[4])   # k3
                                    }
                        
                        # V1.8: 判断去畸变状态
                        if 'undistorted' in tyjt_calib:
                            distortion_correction_status = "undistorted" if tyjt_calib['undistorted'] else "distorted"
                        elif 'distortion_correction' in tyjt_calib:
                            correction_status = str(tyjt_calib['distortion_correction']).lower()
                            if 'undistort' in correction_status or 'corrected' in correction_status:
                                distortion_correction_status = "undistorted"
                            else:
                                distortion_correction_status = "distorted"
                        else:
                            # 默认设为已去畸变（根据您的需求）
                            distortion_correction_status = "undistorted"
                        
                        # 变换参数
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
                    
                    # 计算相机标定矩阵
                    try:
                        sensor2map_matrix = CalibrationProcessor.parse_transform(tyjt_calib).tolist()
                        
                        # 计算sensor2group
                        translation = [0.0, 0.0, 0.0]
                        rotation = [1.0, 0.0, 0.0, 0.0]
                        sensor2group_matrix = np.eye(4).tolist()
                        
                        if sensor2map_matrix and map2group_matrix:
                            sensor2map_np = np.array(sensor2map_matrix)
                            map2group_np = np.array(map2group_matrix)
                            sensor2group_np = map2group_np @ sensor2map_np
                            sensor2group_matrix = sensor2group_np.tolist()
                            
                            translation = sensor2group_np[:3, 3].tolist()
                            rotation = CalibrationProcessor.rotation_matrix_to_quaternion(sensor2group_np[:3, :3])
                        
                        # 存储相机标定信息 - V1.8更新
                        camera_info.calibration = {
                            "tyjt": {
                                "intrinsic": intrinsic_params,
                                "radial_distortion": radial_distortion,  # V1.8: 正确的畸变参数
                                "sensor2map": sensor2map_params,
                                "sensor2map_matrix": sensor2map_matrix,
                                "distortion_correction_status": distortion_correction_status  # V1.8新增
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
                        
                    except Exception as e:
                        self.logger.warning(f"处理相机标定失败 {calibration_key}: {e}")
                
                cameras[camera_order] = camera_info
                if image_path:
                    camera_files[camera_order] = image_path
                    camera_count += 1
            
            # 9. 检查样本有效性
            is_valid = all([
                npy_file.exists(),
                label_file.exists(),
                camera_count >= 2,
                group_key in calib_data
            ])
            
            # 10. 数据质量评估
            data_quality = "good"
            if camera_count < 2:
                data_quality = "warning"
            if not is_valid:
                data_quality = "error"
            
            # 11. 创建样本信息
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
                lidar={
                    "type": "robosense_128",
                    "calibration": lidar_calibration,
                    "point_count": lidar_point_count,
                    "range_stats": lidar_range_stats
                },
                calibration_summary={
                    "group_key": group_key,
                    "has_all_calibrations": all(cam.calibration.get("transform_matrices", {}).get("sensor2group") 
                                            for cam in cameras.values()),
                    "has_lidar_calibration": has_lidar_calib
                },
                metadata={
                    "camera_count": camera_count,
                    "label_objects": label_objects,
                    "timestamp_int": int(timestamp) if timestamp.isdigit() else 0,
                    "is_valid": is_valid,
                    "data_quality": data_quality,
                    "lidar_point_count": lidar_point_count,
                    "has_lidar_data": lidar_point_count > 0
                }
            )
            
            self.logger.debug(f"成功创建样本: {unique_key}, 相机数: {camera_count}, 有效性: {is_valid}")
            return sample
            
        except Exception as e:
            self.logger.error(f"构建样本信息失败 {timestamp}: {e}", exc_info=True)
            return None



    def save_package_result(self, result: PackageAnalysisResult):
        """保存数据包分析结果"""
        if not result:
            return
        
        # 只保存一个三层结构数据文件
        main_file = self.analysis_dir / f"{result.package_name}_structured.json"
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
                "data_version": "v1.9-fixed"
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
        summary_file = self.analysis_dir / "summary_report.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 输出简要报告
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 分析汇总报告 (v1.9)")
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
    analyzer = DataAnalyzerV1_6(OUTPUT_ROOT)
    
    analyzer.logger.info("=" * 60)
    analyzer.logger.info("🔍 TYJT数据集分析工具 v9.2.3-phase1-v1.9 (信息完整修复版)")
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
    analyzer.logger.info("\n📁 生成的关键文件 (v1.9):")
    for result in all_results:
        if result:
            analyzer.logger.info(f"  - {result.package_name}:")
            
            # 结构化数据文件
            structured_file = analyzer.analysis_dir / f"{result.package_name}_structured.json"
            if structured_file.exists():
                analyzer.logger.info(f"    ✓ structured.json: {result.overall_statistics['total_samples']} 个样本")
    
    analyzer.logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 程序运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)