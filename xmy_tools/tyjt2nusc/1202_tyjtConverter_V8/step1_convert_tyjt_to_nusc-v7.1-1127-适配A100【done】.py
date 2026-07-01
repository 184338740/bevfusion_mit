#!/usr/bin/env python3
"""
tyjt数据集转nuscenes格式转换工具 - 13数据包适配+软链接优化终极版
版本: v7 (Final Fixed)
核心修正:
1. 相机映射改为：A→CAM_FRONT、B→CAM_FRONT_RIGHT、C→CAM_BACK、D→CAM_FRONT_LEFT（固定4个）
2. 移除相机朝向依赖，按A/B/C/D顺序强制映射，确保仅4个相机输出
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

# ==================== 全局配置（核心修正相机映射） ====================

Mode = "Local"
if Mode == "Local":
    TYJT_ROOT = "/mnt/dataset/tyjt_RawData_all/"
    OUTPUT_ROOT = "./output/step1/nuscenes_tyjt"
    NUSC_VERSION = "v1.0-tyjt"
    # # 指定需要处理的数据包名称列表
    # DATA_PACKAGES_TO_PROCESS = [
    #     "2d3d4d_20250728_weiyuan", 
    #     "2d3d4d_20250913_weiyuan"
    # ]
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
                    "cameras": [  # (相机文件夹名, 相机顺序A/B/C/D)
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
elif Mode == "A100":
    # TYJT_ROOT = "/cephfsdata/users/mingyuan/ws/01/00_RawData"
    TYJT_ROOT = "/cephfsdata/users/lishan/00_Data/00_RawData"
    OUTPUT_ROOT = "./output/step1-1127/nuscenes_tyjt"
    NUSC_VERSION = "v1.0-tyjt"
    # 13个数据包规格配置（按实际数据包名匹配，补充相机顺序标识）
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
                    "cameras": [  # (相机文件夹名, 相机顺序A/B/C/D)
                        ("R9_Aw_CamS", "A"),
                        ("R9_Bn_CamW", "B"),
                        ("R9_Ce_CamN", "C"),
                        ("R9_Ds_CamE", "D")
                    ]
                }
            }
        },
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
        # },
        # "2d3d4d_20250708_weiyuan": {
        #     "type": "all_in_one",
        #     "calib_root": "calib/G51102400002M00",
        #     "sensor2map_file": "sensor2map_calib.json",
        #     "intersections": {
        #         "R02": {
        #             "group_key": "group2map",
        #             "cameras": [
        #                 ("SC_2A_CamR", "SC_2A_CamR_new", "A"),
        #                 ("SC_2B_CamR", "SC_2B_CamR_new", "B"),
        #                 ("SC_2C_CamR", "SC_2C_CamR_new", "C"),
        #                 ("SC_2D_CamR", "SC_2D_CamR_new", "D")
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

    # DATA_PACKAGES_TO_PROCESS = [
    #     "2d3d_20250114",
    #     "2d3d4d_20241218",
    #     "2d3d4d_20250618_weiyuan",
    #     "2d3d4d_20250728_weiyuan",
    #     "2d3d_20250218",
    #     "2d3d4d_20250117",
    #     "2d3d4d_20250218",
    #     "2d3d4d_20250403",
    #     "2d3d4d_20250708_weiyuan",
    #     "2d3d_20250221",
    #     "2d3d4d_20250213",
    #     "2d3d4d_20241122_wuxi",
    #     "2d3d4d_20250408",
    # ]
else:
    print(f"❌【Error】: TYJT_ROOT & DATA_PACKAGES_TO_PROCESS: None")



# 核心修正：A/B/C/D相机→NuScenes相机固定映射（仅4个）
# 按相机顺序（A→B→C→D）强制映射，与朝向无关
CAM_ORDER_TO_NUSC = {
    "A": "CAM_FRONT",        # A相机→前视
    "B": "CAM_FRONT_RIGHT",  # B相机→右前视
    "C": "CAM_BACK",         # C相机→后视
    "D": "CAM_FRONT_LEFT"    # D相机→左前视
}
# 相机顺序标识（用于从相机名称中提取A/B/C/D）
CAM_ORDER_MARKERS = {"A", "B", "C", "D"}


# Step1类别映射 - 添加"tyjt."前缀，保持15个类别
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

# ==================== Token管理器（完全保留V6） ====================
class EnhancedIDGenerator:
    def __init__(self):
        self.uuid_map = {}
        self.used_tokens: Set[str] = set()
        self.entity_counters = defaultdict(int)
    
    def get_scoped_token(self, entity_type: str, unique_key: str) -> str:
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
        tokens = list(self.uuid_map.values())
        unique_tokens = set(tokens)
        stats = {
            'total_tokens': len(tokens),
            'unique_tokens': len(unique_tokens),
            'entity_counts': dict(self.entity_counters)
        }
        return len(tokens) == len(unique_tokens), stats

id_gen = EnhancedIDGenerator()

# ==================== 数学工具函数（完全保留V6标定逻辑） ====================
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


def parse_tyjt_transform(calib_params: Union[Dict, List]) -> np.ndarray:
    """解析tyjt标定参数（支持海康和一体机两种格式）为4x4变换矩阵"""
    try:
        if isinstance(calib_params, Dict):
            # 检查海康相机格式 (有 translation 和 rotation 数组)
            if 'translation' in calib_params and 'rotation' in calib_params:
                print(f"    📊 检测到海康相机格式")
                translation = np.array(calib_params['translation'])
                rotation_quat = calib_params['rotation']
                
                # 海康格式的rotation是[x, y, z, w]，需要转换为[w, x, y, z]
                if len(rotation_quat) == 4:
                    x, y, z, w = rotation_quat  # 从[x,y,z,w]提取
                    # 转换为[w,x,y,z]格式
                    w, x, y, z = w, x, y, z
                    
                    print(f"    海康格式 - 平移: {translation}")
                    print(f"    海康格式 - 旋转四元数(转换后): [{w:.6f}, {x:.6f}, {y:.6f}, {z:.6f}]")
                    
                else:
                    print(f"    ❌ 不支持的旋转格式长度: {len(rotation_quat)}")
                    return np.eye(4)
                    
            # 检查一体机格式 (有 tx,ty,tz,rx,ry,rz,rw)
            elif all(key in calib_params for key in ['tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'rw']):
                print(f"    📊 检测到一体机格式")
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
                print(f"    一体机格式 - 平移: {translation}")
                print(f"    一体机格式 - 旋转四元数: [{w:.6f}, {x:.6f}, {y:.6f}, {z:.6f}]")
                
            else:
                print(f"    ❌ 不支持的标定格式")
                print(f"    可用键: {list(calib_params.keys())}")
                return np.eye(4)
                
        elif isinstance(calib_params, List):
            # 列表格式处理
            if len(calib_params) == 7:
                tx, ty, tz, rx, ry, rz, rw = calib_params
                translation = np.array([tx, ty, tz])
                x, y, z, w = rx, ry, rz, rw
                print(f"    列表格式 - 平移: {translation}")
                print(f"    列表格式 - 旋转四元数: [{w:.6f}, {x:.6f}, {y:.6f}, {z:.6f}]")
            else:
                print(f"    ❌ 不支持的列表格式长度: {len(calib_params)}")
                return np.eye(4)
        else:
            print(f"    ❌ 不支持的标定参数类型: {type(calib_params)}")
            return np.eye(4)
        
        # 构建变换矩阵
        rotation_matrix = quaternion_to_rotation_matrix([w, x, y, z])
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation
        
        print(f"    ✅ 构建的变换矩阵:")
        for i in range(4):
            print(f"        [{transform[i,0]:.3f}, {transform[i,1]:.3f}, {transform[i,2]:.3f}, {transform[i,3]:.3f}]")
        
        return transform
        
    except Exception as e:
        print(f"    ❌ 解析标定参数失败: {e}")
        import traceback
        traceback.print_exc()
        return np.eye(4)




def get_sensor2ego_transform_fixed(calib_data: Dict, sensor_calib_key: str, group2map_key: str) -> Tuple[List[float], List[float]]:
    """修复版标定转换：sensor→map→group(ego)"""
    try:
        print(f"    🎯 开始标定转换: {sensor_calib_key}")
        
        if sensor_calib_key not in calib_data:
            print(f"    ❌ 标定数据中找不到 {sensor_calib_key}")
            return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        
        # sensor→map变换
        sensor_calib_params = calib_data[sensor_calib_key]
        sensor2map = parse_tyjt_transform(sensor_calib_params)
        
        # map→group(ego)变换（group2map的逆）
        if group2map_key in calib_data:
            group2map_params = calib_data[group2map_key]
            
            # 海康产品：group2map在transform字段中
            if isinstance(group2map_params, dict) and 'transform' in group2map_params:
                group2map_params = group2map_params['transform']
            
            group2map = parse_tyjt_transform(group2map_params)
            map2group = np.linalg.inv(group2map)
        else:
            print(f"    ❌ 标定数据中找不到 group2map 键 {group2map_key}")
            return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        
        # 核心转换：sensor2ego = map2group @ sensor2map
        sensor2ego = map2group @ sensor2map
        
        # 验证变换矩阵的有效性
        if not np.isfinite(sensor2ego).all():
            print(f"    ❌ 无效的变换矩阵")
            return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        
        translation = sensor2ego[:3, 3].tolist()
        rotation_matrix = sensor2ego[:3, :3]
        rotation = rotation_matrix_to_quaternion(rotation_matrix)
        
        distance = np.linalg.norm(translation)
        print(f"    ✅ 标定转换完成: 距离={distance:.2f}m")
        print(f"      sensor2ego平移: {translation}")
        
        return translation, rotation
        
    except Exception as e:
        print(f"    ❌ 计算sensor2ego变换时出错: {e}")
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]



# ==================== 目录结构与数据加载（适配多数据包+只读） ====================
def create_directory_structure(output_root: str, version: str):
    """创建输出目录结构（不修改原数据）"""
    base_path = Path(output_root)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # 版本目录
    version_dir = base_path / version
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # samples/sweeps目录（仅创建4个固定相机目录）
    for data_type in ["samples", "sweeps"]:
        root_dir = base_path / data_type
        root_dir.mkdir(parents=True, exist_ok=True)
        # 固定4个相机目录（与CAM_ORDER_TO_NUSC对应）
        for nusc_cam in CAM_ORDER_TO_NUSC.values():
            (root_dir / nusc_cam).mkdir(parents=True, exist_ok=True)
        # 激光雷达目录
        (root_dir / LIDAR_NUSC_NAME).mkdir(parents=True, exist_ok=True)
    
    # 空地图目录
    maps_dir = base_path / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    # 创建空地图文件
    from PIL import Image
    empty_img = Image.new('RGB', (1, 1), color='black')
    empty_img.save(maps_dir / "empty_map.png")
    
    print(f"✅ 输出目录结构创建完成: {base_path}")
    return base_path

def find_data_packages(tyjt_root: str) -> List[Path]:
    """查找13个配置内的数据包（仅读取，不修改）"""
    root_path = Path(tyjt_root)
    packages = []
    configured_package_names = set(DATA_PACKAGE_CONFIG.keys())
    
    for item in root_path.iterdir():
        if item.is_dir() and item.name in configured_package_names:
            # 验证datasets目录存在（有数据才加入）
            datasets_dir = item / "datasets"
            if datasets_dir.exists() and any(datasets_dir.iterdir()):
                packages.append(item)
                print(f"✅ 找到有效数据包: {item.name}")
            else:
                print(f"⚠️  数据包 {item.name} 无有效数据（datasets目录为空）")
    
    # 检查缺失的配置数据包
    missing = configured_package_names - {p.name for p in packages}
    if missing:
        print(f"⚠️  配置中存在但未找到的数据包: {missing}")
    
    print(f"📦 总共找到 {len(packages)} 个有效数据包")
    return packages

def load_calibration(package_path: Path, package_config: Dict) -> Dict:
    """加载标定数据（仅读取，不修改原文件）"""
    calib_root = package_path / package_config["calib_root"]
    if not calib_root.exists():
        print(f"❌ 标定目录不存在: {calib_root}")
        return {}
    
    calib_data = {}
    try:
        if package_config["type"] == "hikvision":
            # 海康相机：加载group2map和camera2map
            group2map_path = calib_root / package_config["group2map_file"]
            if group2map_path.exists():
                with open(group2map_path, 'r') as f:
                    calib_data.update(json.load(f))
            else:
                print(f"⚠️  group2map文件不存在: {group2map_path}")
            
            camera2map_path = calib_root / package_config["camera2map_file"]
            if camera2map_path.exists():
                with open(camera2map_path, 'r') as f:
                    calib_data.update(json.load(f))
            else:
                print(f"⚠️  camera2map文件不存在: {camera2map_path}")
        
        elif package_config["type"] == "all_in_one":
            # 一体机：加载sensor2map
            sensor2map_path = calib_root / package_config["sensor2map_file"]
            if sensor2map_path.exists():
                with open(sensor2map_path, 'r') as f:
                    calib_data.update(json.load(f))
            else:
                print(f"⚠️  sensor2map文件不存在: {sensor2map_path}")
        
        print(f"✅ 加载标定数据成功，共 {len(calib_data)} 个键")
    except Exception as e:
        print(f"❌ 加载标定数据失败: {e}")
    return calib_data

# ==================== 数据表初始化（完全保留V6字段格式） ====================
def initialize_nusc_data() -> Dict:
    return {
        "category": [], "attribute": [], "sensor": [], "calibrated_sensor": [],
        "ego_pose": [], "log": [], "scene": [], "sample": [], "sample_data": [],
        "sample_annotation": [], "instance": [], "visibility": [], "map": []
    }

def generate_basic_tables(nusc_data: Dict, scene_name: str, calib_data: Dict, package_config: Dict, intersection_key: str) -> Dict:
    """生成基础数据表，适配固定4个相机映射"""
    token_map = {}
    intersection_config = package_config["intersections"][intersection_key]
    
    # 1. log表（按数据包+路口区分）
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
    
    # 2. scene表（按数据包+路口区分）
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
    
    # 3. category表（去重，保留V6映射）
    existing_categories = {cat['name'] for cat in nusc_data['category']}
    for tyjt_cat, nusc_cat in CATEGORY_MAPPING.items():
        if nusc_cat not in existing_categories:
            nusc_data["category"].append({
                "token": id_gen.get_scoped_token("category", nusc_cat),
                "name": nusc_cat,
                "description": f"From tyjt: {tyjt_cat}"
            })
            existing_categories.add(nusc_cat)
    
    # 4. attribute表（保留V6）
    existing_attributes = {attr['name'] for attr in nusc_data['attribute']}
    if "vehicle.moving" not in existing_attributes:
        nusc_data["attribute"].append({
            "token": id_gen.get_scoped_token("attribute", "vehicle.moving"),
            "name": "vehicle.moving",
            "description": "Object is moving"
        })
    if "pedestrian.moving" not in existing_attributes:
        nusc_data["attribute"].append({
            "token": id_gen.get_scoped_token("attribute", "pedestrian.moving"),
            "name": "pedestrian.moving",
            "description": "Pedestrian is moving"
        })
    if "vehicle.stopped" not in existing_attributes:
        nusc_data["attribute"].append({
            "token": id_gen.get_scoped_token("attribute", "vehicle.stopped"),
            "name": "vehicle.stopped",
            "description": "Object is stopped"
        })
    
    # 5. visibility表（保留V6）
    if not nusc_data["visibility"]:
        nusc_data["visibility"].append({
            "token": id_gen.get_scoped_token("visibility", "full"),
            "level": "full",
            "description": "Fully visible"
        })
    
    # 6. sensor表（固定4个相机，按NuScenes名称去重）
    existing_sensors = {sensor['channel'] for sensor in nusc_data['sensor']}
    sensor_tokens = {}
    
    # 相机传感器（按A/B/C/D→固定NuScenes名称）
    for cam_info in intersection_config["cameras"]:
        if package_config["type"] == "hikvision":
            cam_folder, cam_order = cam_info  # 海康：(文件夹名, 顺序A/B/C/D)
            sensor_calib_key = cam_folder
        else:
            cam_folder, sensor_calib_key, cam_order = cam_info  # 一体机：(文件夹名, 标定键, 顺序A/B/C/D)
        
        # 强制按顺序映射NuScenes相机名称
        nusc_cam_name = CAM_ORDER_TO_NUSC.get(cam_order)
        if not nusc_cam_name:
            print(f"⚠️  跳过无效相机顺序: {cam_order}（{cam_folder}）")
            continue
        
        if nusc_cam_name not in existing_sensors:
            sensor_token = id_gen.get_scoped_token("sensor", nusc_cam_name)
            sensor_tokens[nusc_cam_name] = sensor_token
            nusc_data["sensor"].append({
                "token": sensor_token,
                "channel": nusc_cam_name,
                "modality": "camera"
            })
            existing_sensors.add(nusc_cam_name)
        else:
            # 复用已存在的sensor token
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
    token_map['group2map_key'] = intersection_config["group_key"]
    
    # 7. calibrated_sensor表（按场景+传感器区分）
    calib_sensor_tokens = {}
    group2map_key = intersection_config["group_key"]
    
    print(f"🔧 生成 {intersection_key} 路口标定数据（固定4相机映射）:")
    # 相机标定
    for cam_info in intersection_config["cameras"]:
        if package_config["type"] == "hikvision":
            cam_folder, cam_order = cam_info
            sensor_calib_key = cam_folder
        else:
            cam_folder, sensor_calib_key, cam_order = cam_info
        
        nusc_cam_name = CAM_ORDER_TO_NUSC.get(cam_order)
        if not nusc_cam_name or nusc_cam_name not in sensor_tokens:
            print(f"⚠️  跳过无传感器token的相机: {cam_folder}（{cam_order}）")
            continue
        
        sensor_token = sensor_tokens[nusc_cam_name]
        
        # 计算sensor→ego变换（保留V6逻辑）
        translation, rotation = get_sensor2ego_transform_fixed(
            calib_data, sensor_calib_key, group2map_key
        )
        
        # 生成calibrated_sensor token
        calib_unique_key = f"{scene_unique_key}_{nusc_cam_name}"
        calib_token = id_gen.get_scoped_token("calibrated_sensor", calib_unique_key)
        calib_sensor_tokens[nusc_cam_name] = calib_token
        
        # 相机内参（默认值，可根据实际标定调整）
        cam_intrinsic = [
            [1680.0, 0, 960.0],
            [0, 1851.0, 540.0],
            [0, 0, 1]
        ]
        # 尝试从标定数据中读取内参（如果存在）
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
        print(f"   ✅ {nusc_cam_name} (原文件夹: {cam_folder}, 顺序: {cam_order}): 标定完成")
    
    # 激光雷达标定（保留V6：ego坐标系原点）
    lidar_calib_unique_key = f"{scene_unique_key}_{LIDAR_NUSC_NAME}"
    lidar_calib_token = id_gen.get_scoped_token("calibrated_sensor", lidar_calib_unique_key)
    calib_sensor_tokens[LIDAR_NUSC_NAME] = lidar_calib_token
    nusc_data["calibrated_sensor"].append({
        "token": lidar_calib_token,
        "sensor_token": sensor_tokens[LIDAR_NUSC_NAME],
        "translation": [0.0, 0.0, 0.0],
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "camera_intrinsic": []
    })
    
    token_map['calib_sensor_tokens'] = calib_sensor_tokens
    
    # 8. ego_pose表（按场景+路口区分）
    ego_pose_unique_key = f"{scene_unique_key}_ego"
    ego_pose_token = id_gen.get_scoped_token("ego_pose", ego_pose_unique_key)
    token_map['ego_pose_token'] = ego_pose_token
    nusc_data["ego_pose"].append({
        "token": ego_pose_token,
        "translation": [0.0, 0.0, 0.0],
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "timestamp": 0
    })
    
    # 9. map表（全局唯一）
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
    
    return token_map

# ==================== 样本处理（软链接+只读原数据） ====================
def npy_to_bin(npy_path: Path, bin_path: Path) -> bool:
    """npy→bin转换（原npy只读，不修改）"""
    try:
        if not npy_path.exists():
            print(f"❌ npy文件不存在: {npy_path}")
            return False
        
        # 只读加载npy
        point_cloud = np.load(npy_path, allow_pickle=False)
        
        # 格式处理（保留V6逻辑）
        if point_cloud.shape[1] >= 5:
            points_5d = point_cloud[:, :5].astype(np.float32)
        elif point_cloud.shape[1] >= 3:
            xyz = point_cloud[:, :3].astype(np.float32)
            n_points = xyz.shape[0]
            points_5d = np.zeros((n_points, 5), dtype=np.float32)
            points_5d[:, :3] = xyz
            points_5d[:, 3] = 1.0  # intensity
            points_5d[:, 4] = 0.0  # ring
        
        # 保存bin文件
        points_5d.tofile(bin_path)
        return True
    except Exception as e:
        print(f"❌ 点云转换失败 {npy_path}: {e}")
        return False

def process_single_sample(
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
    package_config: Dict
) -> str:
    """处理单个样本：图像软链接，LIDAR bin生成（固定4相机）"""
    # 唯一标识（避免跨数据包文件冲突）
    unique_key = f"{package_name}_{intersection_key}_{sub_packet_name}_{timestamp}"
    sample_token = id_gen.get_scoped_token("sample", unique_key)
    
    # 1. sample表
    sample_data = {
        "token": sample_token,
        "timestamp": int(timestamp) if timestamp.isdigit() else sample_index * 1000000,
        "scene_token": token_map['scene_token'],
        "prev": prev_sample_token,
        "next": ""
    }
    # 更新前一个样本的next字段
    if prev_sample_token:
        for sample in nusc_data["sample"]:
            if sample["token"] == prev_sample_token:
                sample["next"] = sample_token
                break
    nusc_data["sample"].append(sample_data)
    
    # 2. 传感器数据处理
    process_sensor_data(
        nusc_data, sub_packet, timestamp, output_root, sample_token,
        unique_key, token_map, package_config, intersection_key
    )
    
    # 3. 标注数据处理
    process_annotation_data(
        nusc_data, sub_packet, timestamp, sample_token, unique_key
    )
    
    return sample_token

def process_sensor_data(
    nusc_data: Dict,
    sub_packet: Path,
    timestamp: str,
    output_root: Path,
    sample_token: str,
    unique_key: str,
    token_map: Dict,
    package_config: Dict,
    intersection_key: str
):
    """处理传感器数据：固定4相机软链接，LIDAR bin生成"""
    sensor_tokens = token_map['sensor_tokens']
    calib_sensor_tokens = token_map['calib_sensor_tokens']
    intersection_config = package_config["intersections"][intersection_key]
    
    # 相机数据：创建软链接（不复制，固定4相机）
    print(f"    📷 处理相机数据（软链接，固定4相机）: {timestamp}")
    for cam_info in intersection_config["cameras"]:
        if package_config["type"] == "hikvision":
            cam_folder, cam_order = cam_info
        else:
            cam_folder, _, cam_order = cam_info
        
        nusc_cam_name = CAM_ORDER_TO_NUSC.get(cam_order)
        if not nusc_cam_name or nusc_cam_name not in sensor_tokens:
            print(f"    ⚠️  跳过无效相机: {cam_folder}（{cam_order}）")
            continue
        
        # 原图像路径
        img_original_path = sub_packet / cam_folder / "image_dc" / f"{timestamp}.jpg"
        if not img_original_path.exists():
            print(f"    ⚠️  图像文件不存在: {img_original_path}")
            continue
        
        # 输出软链接路径
        img_output_dir = output_root / "samples" / nusc_cam_name
        img_output_path = img_output_dir / f"{unique_key}.jpg"
        
        # 创建软链接（避免重复）
        if not img_output_path.exists():
            os.symlink(img_original_path.resolve(), img_output_path.resolve())
            print(f"    ✅ 创建软链接: {nusc_cam_name}/{img_output_path.name} -> {img_original_path.name}")
        else:
            print(f"    ⚠️  软链接已存在: {nusc_cam_name}/{img_output_path.name}")
        
        # 生成sample_data表条目
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
    
    # 激光雷达数据：npy→bin（原npy只读）
    print(f"    📡 处理激光雷达数据: {timestamp}")
    lidar_original_npy = sub_packet / "lidar" / "pcd" / f"{timestamp}.npy"
    if not lidar_original_npy.exists():
        print(f"    ⚠️  lidar npy文件不存在: {lidar_original_npy}")
        return
    
    # 输出bin路径
    lidar_output_dir = output_root / "samples" / LIDAR_NUSC_NAME
    lidar_output_path = lidar_output_dir / f"{unique_key}.bin"
    
    if lidar_output_path.exists():
        print(f"    ⚠️  lidar bin文件已存在: {lidar_output_path.name}")
    else:
        if npy_to_bin(lidar_original_npy, lidar_output_path):
            print(f"    ✅ 生成bin文件: {lidar_output_path.name}")
        else:
            print(f"    ❌ bin文件生成失败: {lidar_original_npy.name}")
            return
    
    # 生成sample_data表条目
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

def process_annotation_data(
    nusc_data: Dict,
    sub_packet: Path,
    timestamp: str,
    sample_token: str,
    unique_key: str
):
    """处理标注数据（完全保留V6逻辑）"""
    label_path = sub_packet / "lidar" / "label" / f"{timestamp}.json"
    if not label_path.exists():
        print(f"    ⚠️  标注文件不存在: {label_path}")
        return
    
    try:
        # 只读加载标注文件
        with open(label_path, 'r') as f:
            label_data = json.load(f)
        
        objects = label_data.get("objects", []) if isinstance(label_data, dict) else label_data
        for obj_idx, obj in enumerate(objects):
            tyjt_cat = obj.get("type", "other")
            nusc_cat = CATEGORY_MAPPING.get(tyjt_cat, "movable_object.debris")
            
            # 3D框处理
            box3d = obj.get("box3d", [0, 0, 0, 1, 1, 1])
            x, y, z, l, w, h = (box3d + [0, 0, 0, 1, 1, 1])[:6]  # 补全默认值
            
            # 旋转处理
            rotation = obj.get("rotation", [0, 0, 0])
            yaw = rotation[0] if len(rotation) > 0 else 0
            quaternion = quaternion_from_euler(yaw)
            
            # Instance token生成（保留V6逻辑）
            position_key = f"{x:.2f}_{y:.2f}_{z:.2f}_{nusc_cat}"
            instance_key = f"{unique_key}_{position_key}"
            instance_token = id_gen.get_scoped_token("instance", instance_key)
            
            # Category token（保留V6）
            category_token = id_gen.get_scoped_token("category", nusc_cat)
            
            # Attribute tokens（保留V6）
            attribute_tokens = []
            if 'vehicle' in nusc_cat:
                attribute_tokens.append(id_gen.get_scoped_token("attribute", "vehicle.moving"))
            elif 'pedestrian' in nusc_cat:
                attribute_tokens.append(id_gen.get_scoped_token("attribute", "pedestrian.moving"))
            else:
                attribute_tokens.append(id_gen.get_scoped_token("attribute", "vehicle.stopped"))
            
            # 更新Instance表（保留V6）
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
            
            # 生成Sample Annotation（保留V6字段）
            ann_token = id_gen.get_scoped_token("annotation", f"{instance_token}_{timestamp}")
            nusc_data["sample_annotation"].append({
                "token": ann_token,
                "sample_token": sample_token,
                "instance_token": instance_token,
                "visibility_token": id_gen.get_scoped_token("visibility", "full"),
                "attribute_tokens": attribute_tokens,
                "translation": [float(x), float(y), float(z)],
                "size": [float(w), float(l), float(h)],  # NuScenes格式：[w, l, h]
                "rotation": quaternion,
                "category_name": nusc_cat,
                "num_lidar_pts": 10,
                "num_radar_pts": 0
            })
        
        print(f"    ✅ 处理标注: {len(objects)} 个目标")
    except Exception as e:
        print(f"    ❌ 处理标注失败 {label_path}: {e}")

# ==================== 场景处理（适配多数据包+多路口） ====================
def process_intersection_samples(
    nusc_data: Dict,
    package_dir: Path,
    output_root: Path,
    package_name: str,
    intersection_key: str,
    package_config: Dict,
    token_map: Dict
) -> List[str]:
    """处理单个路口的所有样本（固定4相机）"""
    datasets_dir = package_dir / "datasets"
    if not datasets_dir.exists():
        print(f"⚠️  无datasets目录: {datasets_dir}")
        return []
    
    # 收集所有有效样本（按时间戳去重）
    all_samples = []
    processed_timestamps = set()
    sub_packets = [d for d in datasets_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    for sub_packet in sub_packets:
        print(f"  📁 处理子包: {sub_packet.name}")
        lidar_dir = sub_packet / "lidar" / "pcd"
        if not lidar_dir.exists():
            print(f"  ⚠️  无子包lidar目录: {lidar_dir}")
            continue
        
        # 收集npy文件（按时间戳排序）
        lidar_npy_files = sorted(lidar_dir.glob("*.npy"), key=lambda x: x.stem)
        for lidar_npy in lidar_npy_files:
            timestamp = lidar_npy.stem
            if not timestamp.isdigit():
                continue
            
            # 去重：同一路口+时间戳只处理一次
            unique_ts_key = f"{intersection_key}_{timestamp}"
            if unique_ts_key in processed_timestamps:
                print(f"  ⚠️  跳过重复时间戳: {timestamp}")
                continue
            
            # 检查标注文件是否存在
            label_path = sub_packet / "lidar" / "label" / f"{timestamp}.json"
            if not label_path.exists():
                print(f"  ⚠️  标注文件缺失，跳过: {timestamp}")
                continue
            
            all_samples.append((lidar_npy, timestamp, sub_packet))
            processed_timestamps.add(unique_ts_key)
    
    # 按时间戳排序
    all_samples.sort(key=lambda x: int(x[1]))
    print(f"  📊 去重后有效样本数: {len(all_samples)}")
    
    # 处理所有样本
    sample_tokens = []
    prev_token = ""
    for i, (lidar_npy, timestamp, sub_packet) in enumerate(all_samples):
        sample_token = process_single_sample(
            nusc_data, sub_packet, timestamp, output_root,
            package_name, intersection_key, sub_packet.name,
            prev_token, i, token_map, package_config
        )
        if sample_token:
            sample_tokens.append(sample_token)
            prev_token = sample_token
    
    return sample_tokens

def process_package(nusc_data: Dict, package_dir: Path, output_root: Path):
    """处理单个数据包（含多个路口，固定4相机）"""
    package_name = package_dir.name
    if package_name not in DATA_PACKAGE_CONFIG:
        print(f"❌ 数据包 {package_name} 无配置，跳过")
        return
    
    package_config = DATA_PACKAGE_CONFIG[package_name]
    print(f"\n🎬 开始处理数据包: {package_name} (类型: {package_config['type']})")
    
    # 加载标定数据（只读）
    calib_data = load_calibration(package_dir, package_config)
    if not calib_data:
        print(f"❌ 数据包 {package_name} 标定加载失败，跳过")
        return
    
    # 处理每个路口
    for intersection_key, intersection_config in package_config["intersections"].items():
        print(f"\n🔹 处理路口: {intersection_key}")
        
        # 生成基础数据表
        token_map = generate_basic_tables(
            nusc_data, package_name, calib_data, package_config, intersection_key
        )
        
        # 处理路口样本
        sample_tokens = process_intersection_samples(
            nusc_data, package_dir, output_root, package_name,
            intersection_key, package_config, token_map
        )
        
        # 更新scene表信息
        for scene in nusc_data["scene"]:
            if scene["token"] == token_map['scene_token']:
                scene["nbr_samples"] = len(sample_tokens)
                if sample_tokens:
                    scene["first_sample_token"] = sample_tokens[0]
                    scene["last_sample_token"] = sample_tokens[-1]
                print(f"  ✅ 路口 {intersection_key} 处理完成: {len(sample_tokens)} 个样本")
                break

# ==================== 数据验证（保留V6逻辑+固定4相机校验） ====================
def validate_data_consistency(nusc_data: Dict, output_root: Path) -> Tuple[bool, List[str]]:
    """验证数据一致性（软链接有效性+固定4相机校验）"""
    print("\n🔍 开始数据一致性验证...")
    errors = []
    
    # 构建索引
    indices = {}
    for table_name, records in nusc_data.items():
        if table_name.startswith('_'):
            continue
        indices[table_name] = {r['token']: r for r in records}
    
    # 1. 外键关系验证（保留V6）
    tables_to_check = {
        'sample_data': ['sample_token', 'ego_pose_token', 'calibrated_sensor_token'],
        'sample_annotation': ['sample_token', 'instance_token', 'visibility_token'],
        'instance': ['category_token'],
        'scene': ['log_token'],
        'sample': ['scene_token']
    }
    for table, foreign_keys in tables_to_check.items():
        for record in nusc_data.get(table, []):
            for fk in foreign_keys:
                if fk in record and record[fk]:
                    ref_table = fk.replace('_token', '')
                    if ref_table in indices and record[fk] not in indices[ref_table]:
                        errors.append(f"{table} {record['token'][:8]} 引用了不存在的 {fk}")
    
    # 2. 样本链验证（保留V6）
    sample_tokens = {s['token']: s for s in nusc_data.get('sample', [])}
    for sample in nusc_data.get('sample', []):
        if sample['prev'] and sample['prev'] not in sample_tokens:
            errors.append(f"sample {sample['token'][:8]} 的prev指向不存在的sample")
        if sample['next'] and sample['next'] not in sample_tokens:
            errors.append(f"sample {sample['token'][:8]} 的next指向不存在的sample")
    
    # 3. 文件有效性验证（适配软链接）
    file_paths = set()
    for sample_data in nusc_data.get('sample_data', []):
        file_path = output_root / sample_data['filename']
        file_key = sample_data['filename']
        
        # 检查文件是否存在（软链接是否有效）
        if not file_path.exists():
            errors.append(f"sample_data {sample_data['token'][:8]} 引用的文件不存在/软链接失效: {file_path}")
        
        # 检查文件路径唯一性
        if file_key in file_paths:
            errors.append(f"文件路径重复: {file_key}")
        else:
            file_paths.add(file_key)
    
    # 4. 固定4相机校验
    camera_sensors = [s for s in nusc_data.get('sensor', []) if s['modality'] == 'camera']
    camera_channels = {s['channel'] for s in camera_sensors}
    required_cameras = set(CAM_ORDER_TO_NUSC.values())
    missing_cameras = required_cameras - camera_channels
    extra_cameras = camera_channels - required_cameras
    if missing_cameras:
        errors.append(f"缺失必要相机: {missing_cameras}（需固定4个：{required_cameras}）")
    if extra_cameras:
        errors.append(f"存在多余相机: {extra_cameras}（仅允许固定4个）")
    
    # 5. Token唯一性验证
    all_tokens = []
    for table_name, records in nusc_data.items():
        if table_name.startswith('_'):
            continue
        for record in records:
            if 'token' in record:
                all_tokens.append(record['token'])
    duplicate_tokens = [t for t in all_tokens if all_tokens.count(t) > 1]
    if duplicate_tokens:
        errors.append(f"存在重复token: {list(set(duplicate_tokens))[:5]}...")
    
    is_valid = len(errors) == 0
    if is_valid:
        print("✅ 数据一致性验证通过")
        print(f"📊 统计: 唯一文件数 {len(file_paths)}, 总token数 {len(all_tokens)}, 相机数 {len(camera_sensors)}（均为固定4个）")
    else:
        print(f"❌ 发现 {len(errors)} 个数据一致性问题")
        for error in errors[:10]:
            print(f"  {error}")
    
    return is_valid, errors

# ==================== 主函数 ====================
def main():
    print("🚀 Step1: TYJT→NuScenes转换工具 v7（13数据包+固定4相机+软链接）")
    print(f"📥 原数据根目录（只读）: {TYJT_ROOT}")
    print(f"📤 输出目录: {OUTPUT_ROOT}")
    print(f"📷 相机映射规则（固定4个）: A→CAM_FRONT, B→CAM_FRONT_RIGHT, C→CAM_BACK, D→CAM_FRONT_LEFT")
    
    # 1. 创建输出目录结构
    output_base = create_directory_structure(OUTPUT_ROOT, NUSC_VERSION)
    
    # 2. 查找13个数据包
    packages = find_data_packages(TYJT_ROOT)
    if not packages:
        print("❌ 未找到任何有效数据包，退出")
        return
    
    # 3. 初始化NuScenes数据结构
    global_nusc_data = initialize_nusc_data()
    
    # 4. 处理所有数据包
    for package in packages:
        process_package(global_nusc_data, package, output_base)
    
    # 5. 验证Token唯一性（保留V6）
    token_unique, token_stats = id_gen.validate_token_uniqueness()
    if not token_unique:
        print("⚠️  警告: 存在重复Token！")
    print(f"\n📊 Token统计: {token_stats}")
    
    # 6. 数据一致性验证（含固定4相机校验）
    data_valid, errors = validate_data_consistency(global_nusc_data, output_base)
    
    # 7. 保存数据（保留V6字段格式）
    version_dir = output_base / NUSC_VERSION
    for table_name, table_data in global_nusc_data.items():
        if table_name.startswith('_'):
            continue
        output_path = version_dir / f"{table_name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(table_data, f, indent=2, ensure_ascii=False)
        print(f"💾 保存 {table_name}.json: {len(table_data)} 条记录")
    
    # 8. 输出统计信息
    print(f"\n🎉 Step1 v7 转换完成!")
    print(f"📁 输出目录: {version_dir}")
    print(f"📊 最终统计:")
    print(f"  - 数据包数: {len(packages)}")
    print(f"  - 场景数（数据包+路口）: {len(global_nusc_data['scene'])}")
    print(f"  - 样本数: {len(global_nusc_data['sample'])}")
    print(f"  - 标注数: {len(global_nusc_data['sample_annotation'])}")
    print(f"  - 传感器数: 相机{len([s for s in global_nusc_data['sensor'] if s['modality'] == 'camera'])}个（固定4个） + 激光雷达1个")
    
    if not data_valid:
        print(f"\n⚠️  注意: 数据存在 {len(errors)} 个一致性问题，请查看日志")

if __name__ == "__main__":
    main()