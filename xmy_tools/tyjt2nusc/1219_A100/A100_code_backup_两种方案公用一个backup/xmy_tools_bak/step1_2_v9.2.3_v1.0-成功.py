#!/usr/bin/env python3
"""
tyjt数据集转nuscenes格式转换工具 - v9.2.3-phase2-v1.9 (数据处理版)
版本: v9.2.3-phase2-v1.9-data-processing
修改日期: 2024-12-26
核心功能:
1. 基于step1_1的分析结果进行数据转换
2. 创建NuScenes格式目录结构
3. 处理图像数据（软链接）
4. 处理激光雷达数据（npy→bin转换）
5. 生成NuScenes格式的JSON数据表
6. 处理标定数据和地图数据
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
from dataclasses import dataclass, field
import shutil
from datetime import datetime
import sys
from tqdm import tqdm
import concurrent.futures

# ==================== 全局配置 ====================

Mode = "Local"
if Mode == "Local":
    TYJT_ROOT = "/mnt/dataset/tyjt_RawData_all/"
    OUTPUT_ROOT = "./output-1226-v1.9/step1/nuscenes_tyjt"
    ANALYSIS_ROOT = "./output-1226-v1.9/step1/nuscenes_tyjt/analysis"
    NUSC_VERSION = "v1.0-tyjt"
else:
    print(f"❌【Error】: 配置模式错误")
    sys.exit(1)

# ==================== 核心配置 ====================

CAM_ORDER_TO_NUSC = {
    "A": "CAM_FRONT",
    "B": "CAM_FRONT_RIGHT", 
    "C": "CAM_BACK",
    "D": "CAM_FRONT_LEFT"
}

LIDAR_NUSC_NAME = "LIDAR_TOP"

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

# ==================== 数据结构 ====================

class EnhancedIDGenerator:
    """Token生成器"""
    def __init__(self):
        self.uuid_map = {}
        self.used_tokens: Set[str] = set()
        self.entity_counters = defaultdict(int)
    
    def get_scoped_token(self, entity_type: str, unique_key: str) -> str:
        """确保相同的unique_key生成相同的token"""
        scope_key = f"{entity_type}::{unique_key}"
        if scope_key not in self.uuid_map:
            # 使用确定性的哈希
            hash_input = f"{entity_type}_{unique_key}"
            hash_obj = hashlib.sha256(hash_input.encode())
            hash_hex = hash_obj.hexdigest()[:32]
            debug_uuid = f"{hash_hex[:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:32]}"
            
            # 确保唯一性
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

# ==================== 数据处理核心 ====================

class DataProcessor:
    """v1.9数据处理器"""
    
    def __init__(self, output_root: str, analysis_root: str):
        self.output_root = Path(output_root)
        self.analysis_root = Path(analysis_root)
        self.nusc_version = NUSC_VERSION
        
        # 初始化ID生成器
        self.id_gen = EnhancedIDGenerator()
        
        # 初始化NuScenes数据结构
        self.nusc_data = self.initialize_nusc_data()

        self.log_registry = {}
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志"""
        log_dir = self.output_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("processor_v1_9")
        self.logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        self.logger.handlers = []
        
        # 文件处理器
        log_file = log_dir / "processing_v1_9.log"
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
    
    def initialize_nusc_data(self) -> Dict:
        """初始化NuScenes数据结构"""
        return {
            "category": [],
            "attribute": [],
            "sensor": [],
            "calibrated_sensor": [],
            "ego_pose": [],
            "log": [],
            "scene": [],
            "sample": [],
            "sample_data": [],
            "sample_annotation": [],
            "instance": [],
            "visibility": [],
            "map": []
        }
    
    def create_directory_structure(self):
        """创建NuScenes格式目录结构"""
        self.logger.info("创建目录结构...")
        
        # 基础目录
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 版本目录
        version_dir = self.output_root / self.nusc_version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # samples目录（关键帧）
        samples_dir = self.output_root / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # sweeps目录（非关键帧）
        sweeps_dir = self.output_root / "sweeps"
        sweeps_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建相机目录
        for nusc_cam in CAM_ORDER_TO_NUSC.values():
            (samples_dir / nusc_cam).mkdir(parents=True, exist_ok=True)
            (sweeps_dir / nusc_cam).mkdir(parents=True, exist_ok=True)
        
        # 创建激光雷达目录
        (samples_dir / LIDAR_NUSC_NAME).mkdir(parents=True, exist_ok=True)
        (sweeps_dir / LIDAR_NUSC_NAME).mkdir(parents=True, exist_ok=True)
        
        # maps目录
        maps_dir = self.output_root / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建空地图文件
        from PIL import Image
        empty_img = Image.new('RGB', (100, 100), color='black')
        empty_img.save(maps_dir / "empty_map.png")
        
        self.logger.info(f"✅ 目录结构创建完成: {self.output_root}")
        
        return version_dir
    
    def load_analysis_results(self) -> List[Dict]:
        """加载step1_1的分析结果"""
        analysis_files = list(self.analysis_root.glob("*_structured.json"))
        
        if not analysis_files:
            self.logger.error(f"❌ 未找到分析结果文件: {self.analysis_root}")
            return []
        
        all_results = []
        for analysis_file in analysis_files:
            try:
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    all_results.append(result)
                self.logger.info(f"✅ 加载分析结果: {analysis_file.name}")
            except Exception as e:
                self.logger.error(f"❌ 加载分析结果失败 {analysis_file}: {e}")
        
        return all_results
    
    def generate_basic_tables(self, analysis_result: Dict):
        """生成基础数据表"""
        package_info = analysis_result["package_info"]
        package_name = package_info["name"]
        
        self.logger.info(f"为数据包 {package_name} 生成基础数据表...")
        
        # 1. category表
        self.generate_categories()
        
        # 2. attribute表
        self.generate_attributes()
        
        # 3. visibility表
        self.generate_visibility()
        
        # 4. sensor表
        self.generate_sensors()
        
        # 5. map表
        self.generate_map(package_info)
        
        self.logger.info(f"✅ 基础数据表生成完成")
    
    def generate_categories(self):
        """生成category表"""
        if self.nusc_data["category"]:
            return
        
        for tyjt_cat, nusc_cat in CATEGORY_MAPPING.items():
            self.nusc_data["category"].append({
                "token": self.id_gen.get_scoped_token("category", nusc_cat),
                "name": nusc_cat,
                "description": f"From tyjt: {tyjt_cat}"
            })
        
        self.logger.info(f"添加 {len(CATEGORY_MAPPING)} 个类别")
    
    def generate_attributes(self):
        """生成attribute表"""
        if self.nusc_data["attribute"]:
            return
        
        attributes = [
            "vehicle.moving",
            "vehicle.stopped",
            "pedestrian.moving",
            "pedestrian.sitting_lying_down",
            "pedestrian.standing",
            "cycle.with_rider",
            "cycle.without_rider"
        ]
        
        for attr_name in attributes:
            self.nusc_data["attribute"].append({
                "token": self.id_gen.get_scoped_token("attribute", attr_name),
                "name": attr_name,
                "description": f"{attr_name.replace('.', ' is ')}"
            })
        
        self.logger.info(f"添加 {len(attributes)} 个属性")
    
    def generate_visibility(self):
        """生成visibility表"""
        if self.nusc_data["visibility"]:
            return
        
        visibility_levels = [
            ("v0-40", "visibility 0-40%"),
            ("v40-60", "visibility 40-60%"),
            ("v60-80", "visibility 60-80%"),
            ("v80-100", "visibility 80-100%")
        ]
        
        for level, desc in visibility_levels:
            self.nusc_data["visibility"].append({
                "token": self.id_gen.get_scoped_token("visibility", level),
                "level": level,
                "description": desc
            })
        
        self.logger.info(f"添加 {len(visibility_levels)} 个可见性级别")
    
    def generate_sensors(self):
        """生成sensor表"""
        if self.nusc_data["sensor"]:
            return
        
        # 相机传感器
        for nusc_cam in CAM_ORDER_TO_NUSC.values():
            self.nusc_data["sensor"].append({
                "token": self.id_gen.get_scoped_token("sensor", nusc_cam),
                "channel": nusc_cam,
                "modality": "camera"
            })
        
        # 激光雷达传感器
        self.nusc_data["sensor"].append({
            "token": self.id_gen.get_scoped_token("sensor", LIDAR_NUSC_NAME),
            "channel": LIDAR_NUSC_NAME,
            "modality": "lidar"
        })
        
        self.logger.info(f"添加 {len(CAM_ORDER_TO_NUSC) + 1} 个传感器")
    

    def generate_map(self, package_info: Dict):
        """确保map表正确关联所有log"""
        if self.nusc_data["map"]:
            map_record = self.nusc_data["map"][0]
            # 获取当前数据包的log_token
            current_log_token = self.log_registry.get(package_info['name'])
            if current_log_token:
                # 确保不重复添加
                current_log_tokens = map_record.get('log_tokens', [])
                if current_log_token not in current_log_tokens:
                    map_record['log_tokens'] = current_log_tokens + [current_log_token]
                    self.logger.debug(f"更新map.log_tokens，新增log: {current_log_token[:16]}...")
            return
        
        # 创建新map记录（理论上只运行一次）
        map_token = self.id_gen.get_scoped_token("map", "tyjt_empty")
        # 初始化时，log_tokens应为空，后续由各个数据包添加
        self.nusc_data["map"].append({
            "token": map_token,
            "log_tokens": [],  # 初始为空
            "category": "city",
            "filename": "maps/empty_map.png",
            "map_name": "tyjt_empty"
        })
        self.logger.info("创建新map记录")


    def process_package(self, analysis_result: Dict):
        """处理单个数据包"""
        package_info = analysis_result["package_info"]
        package_name = package_info["name"]
        
        self.logger.info(f"开始处理数据包: {package_name}")
        
        # 生成log表
        log_token = self.generate_log(package_info)
        
        # 处理每个子包
        total_samples = 0
        for sub_packet_name, sub_packet_info in analysis_result["sub_packets"].items():
            samples_processed = self.process_sub_packet(
                sub_packet_name, sub_packet_info, package_info, log_token
            )
            total_samples += samples_processed
        
        self.logger.info(f"✅ 数据包 {package_name} 处理完成: {total_samples} 个样本")
        
        return total_samples
    
    def generate_log(self, package_info: Dict) -> str:
        """生成或获取log表记录，确保每个数据包全局唯一，并更新map表"""
        package_name = package_info['name']
        log_name = f"{package_name}_{package_info['type']}"
        
        # 关键修复：检查是否已为该包创建过log
        if package_name in self.log_registry:
            self.logger.debug(f"数据包 {package_name} 的log已存在，直接返回token")
            return self.log_registry[package_name]
        
        # 创建新的log记录
        log_token = self.id_gen.get_scoped_token("log", log_name)
        
        self.nusc_data["log"].append({
            "token": log_token,
            "logfile": f"{log_name}.log",
            "vehicle": "tyjt_system",
            "date_captured": package_info.get("analysis_time", "2025-12-26").split('T')[0],
            "location": package_info.get("dataset_root", "").split('/')[-1] if '/' in package_info.get("dataset_root", "") else "default"
        })
        
        # 注册这个log
        self.log_registry[package_name] = log_token
        
        # !!! 关键修复：将这个log_token添加到map表的log_tokens数组中 !!!
        if self.nusc_data["map"]:
            map_record = self.nusc_data["map"][0]
            current_log_tokens = map_record.get('log_tokens', [])
            # 确保不重复添加
            if log_token not in current_log_tokens:
                map_record['log_tokens'] = current_log_tokens + [log_token]
                self.logger.debug(f"将log_token {log_token[:16]}... 添加到 map.log_tokens")
        else:
            # 理论上这不会发生，因为generate_map会先被调用
            self.logger.warning("map表不存在，无法添加log_token")
        
        self.logger.info(f"为数据包 {package_name} 创建新log: {log_token[:16]}...")
        
        return log_token
    
    def process_sub_packet(self, sub_packet_name: str, sub_packet_info: Dict, 
                          package_info: Dict, log_token: str) -> int:
        """处理单个子包"""
        road_id = sub_packet_info["road_id"]
        scene_name = f"{package_info['name']}_{road_id}_{sub_packet_name}"
        
        self.logger.info(f"处理子包: {sub_packet_name} (道路: {road_id})")
        
        # 生成scene表
        scene_token = self.id_gen.get_scoped_token("scene", scene_name)
        
        self.nusc_data["scene"].append({
            "token": scene_token,
            "name": scene_name,
            "description": f"TYJT: {package_info['name']} - {road_id} - {sub_packet_name}",
            "log_token": log_token,
            "nbr_samples": 0,
            "first_sample_token": "",
            "last_sample_token": ""
        })
        
        # 处理每个样本
        samples_processed = 0
        sample_tokens = []
        
        for sample_info in tqdm(sub_packet_info["samples"], desc=f"处理样本", leave=False):
            if not sample_info["metadata"]["is_valid"]:
                continue
            
            sample_token = self.process_sample(
                sample_info, package_info, scene_token, sub_packet_name
            )
            
            if sample_token:
                sample_tokens.append(sample_token)
                samples_processed += 1
        
        # 更新scene的样本信息
        for scene in self.nusc_data["scene"]:
            if scene["token"] == scene_token:
                scene["nbr_samples"] = len(sample_tokens)
                if sample_tokens:
                    scene["first_sample_token"] = sample_tokens[0]
                    scene["last_sample_token"] = sample_tokens[-1]
                break
        
        self.logger.info(f"子包 {sub_packet_name}: 处理 {samples_processed} 个有效样本")
        
        return samples_processed
    
    def process_sample(self, sample_info: Dict, package_info: Dict, 
                      scene_token: str, sub_packet_name: str) -> Optional[str]:
        """处理单个样本"""
        try:
            # 生成sample表
            sample_token = self.id_gen.get_scoped_token("sample", sample_info["unique_key"])
            
            self.nusc_data["sample"].append({
                "token": sample_token,
                "timestamp": sample_info["metadata"]["timestamp_int"],
                "scene_token": scene_token,
                "prev": "",
                "next": ""
            })
            
            # 生成ego_pose
            ego_pose_token = self.generate_ego_pose(sample_info["unique_key"])
            
            # 处理传感器数据
            self.process_sensor_data(sample_info, package_info, sample_token, ego_pose_token)
            
            # 处理标注数据
            self.process_annotation_data(sample_info, sample_token)
            
            return sample_token
            
        except Exception as e:
            self.logger.error(f"处理样本失败 {sample_info.get('unique_key', 'unknown')}: {e}")
            return None
    
    def generate_ego_pose(self, unique_key: str) -> str:
        """生成ego_pose表记录"""
        ego_pose_token = self.id_gen.get_scoped_token("ego_pose", f"{unique_key}_ego")
        
        self.nusc_data["ego_pose"].append({
            "token": ego_pose_token,
            "translation": [0.0, 0.0, 0.0],  # ego坐标系原点
            "rotation": [1.0, 0.0, 0.0, 0.0],  # 无旋转
            "timestamp": 0
        })
        
        return ego_pose_token
    
    def process_sensor_data(self, sample_info: Dict, package_info: Dict, 
                           sample_token: str, ego_pose_token: str):
        """处理传感器数据"""
        unique_key = sample_info["unique_key"]
        
        # 处理相机数据
        for camera_order, camera_info in sample_info["cameras"].items():
            if not camera_info["image_path"]:
                continue
            
            nusc_cam_name = CAM_ORDER_TO_NUSC.get(camera_order)
            if not nusc_cam_name:
                continue
            
            # 生成calibrated_sensor
            calib_token = self.generate_camera_calibration(
                camera_info, nusc_cam_name, unique_key
            )
            
            # 创建图像软链接
            image_success = self.create_image_symlink(
                camera_info["image_path"], unique_key, nusc_cam_name
            )
            
            if image_success:
                # 生成sample_data
                sample_data_token = self.id_gen.get_scoped_token(
                    "sample_data", f"{unique_key}_{nusc_cam_name}"
                )
                
                self.nusc_data["sample_data"].append({
                    "token": sample_data_token,
                    "sample_token": sample_token,
                    "ego_pose_token": ego_pose_token,
                    "calibrated_sensor_token": calib_token,
                    "filename": f"samples/{nusc_cam_name}/{unique_key}.jpg",
                    "fileformat": "jpg",
                    "width": 1920,
                    "height": 1080,
                    "timestamp": sample_info["metadata"]["timestamp_int"],
                    "is_key_frame": True,
                    "next": "",
                    "prev": ""
                })
        
        # 处理激光雷达数据
        lidar_success = self.process_lidar_data(
            sample_info, sample_token, ego_pose_token, unique_key
        )
    
    def generate_camera_calibration(self, camera_info: Dict, 
                                   nusc_cam_name: str, unique_key: str) -> str:
        """生成相机标定数据"""
        calib_token = self.id_gen.get_scoped_token(
            "calibrated_sensor", f"{unique_key}_{nusc_cam_name}"
        )
        
        # 获取传感器token
        sensor_token = None
        for sensor in self.nusc_data["sensor"]:
            if sensor["channel"] == nusc_cam_name:
                sensor_token = sensor["token"]
                break
        
        if not sensor_token:
            self.logger.warning(f"找不到传感器token: {nusc_cam_name}")
            sensor_token = self.id_gen.get_scoped_token("sensor", nusc_cam_name)
        
        # 从分析结果中获取标定参数
        calib_data = camera_info.get("calibration", {})
        nuscenes_calib = calib_data.get("nuscenes", {})
        
        translation = nuscenes_calib.get("translation", [0.0, 0.0, 0.0])
        rotation = nuscenes_calib.get("rotation", [1.0, 0.0, 0.0, 0.0])
        intrinsic = nuscenes_calib.get("intrinsic", [[1680.0, 0, 960.0], [0, 1851.0, 540.0], [0, 0, 1]])
        
        self.nusc_data["calibrated_sensor"].append({
            "token": calib_token,
            "sensor_token": sensor_token,
            "translation": translation,
            "rotation": rotation,
            "camera_intrinsic": intrinsic
        })
        
        return calib_token
    
    def create_image_symlink(self, image_path: str, unique_key: str, 
                            nusc_cam_name: str) -> bool:
        """创建图像软链接"""
        try:
            source_path = Path(image_path)
            if not source_path.exists():
                self.logger.warning(f"源图像不存在: {image_path}")
                return False
            
            target_dir = self.output_root / "samples" / nusc_cam_name
            target_path = target_dir / f"{unique_key}.jpg"
            
            if not target_path.exists():
                os.symlink(source_path.resolve(), target_path)
                return True
            else:
                # 文件已存在
                return True
                
        except Exception as e:
            self.logger.error(f"创建图像软链接失败: {e}")
            return False
    
    def process_lidar_data(self, sample_info: Dict, sample_token: str,
                          ego_pose_token: str, unique_key: str) -> bool:
        """处理激光雷达数据"""
        try:
            lidar_path = sample_info["files"]["lidar"]
            if not lidar_path or not Path(lidar_path).exists():
                return False
            
            # 生成calibrated_sensor
            lidar_calib_token = self.generate_lidar_calibration(sample_info, unique_key)
            
            # 转换npy到bin
            bin_success = self.convert_npy_to_bin(lidar_path, unique_key)
            
            if bin_success:
                # 生成sample_data
                sample_data_token = self.id_gen.get_scoped_token(
                    "sample_data", f"{unique_key}_{LIDAR_NUSC_NAME}"
                )
                
                self.nusc_data["sample_data"].append({
                    "token": sample_data_token,
                    "sample_token": sample_token,
                    "ego_pose_token": ego_pose_token,
                    "calibrated_sensor_token": lidar_calib_token,
                    "filename": f"samples/{LIDAR_NUSC_NAME}/{unique_key}.bin",
                    "fileformat": "bin",
                    "width": 0,
                    "height": 0,
                    "timestamp": sample_info["metadata"]["timestamp_int"],
                    "is_key_frame": True,
                    "next": "",
                    "prev": ""
                })
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"处理激光雷达数据失败: {e}")
            return False
    
    def generate_lidar_calibration(self, sample_info: Dict, unique_key: str) -> str:
        """生成激光雷达标定数据"""
        calib_token = self.id_gen.get_scoped_token(
            "calibrated_sensor", f"{unique_key}_{LIDAR_NUSC_NAME}"
        )
        
        # 获取传感器token
        sensor_token = None
        for sensor in self.nusc_data["sensor"]:
            if sensor["channel"] == LIDAR_NUSC_NAME:
                sensor_token = sensor["token"]
                break
        
        if not sensor_token:
            sensor_token = self.id_gen.get_scoped_token("sensor", LIDAR_NUSC_NAME)
        
        # 从分析结果中获取标定参数
        lidar_calib = sample_info.get("lidar", {}).get("calibration", {})
        nuscenes_calib = lidar_calib.get("nuscenes", {})
        
        translation = nuscenes_calib.get("translation", [0.0, 0.0, 0.0])
        rotation = nuscenes_calib.get("rotation", [1.0, 0.0, 0.0, 0.0])
        
        self.nusc_data["calibrated_sensor"].append({
            "token": calib_token,
            "sensor_token": sensor_token,
            "translation": translation,
            "rotation": rotation,
            "camera_intrinsic": []  # 激光雷达没有内参
        })
        
        return calib_token
    
    def convert_npy_to_bin(self, npy_path: str, unique_key: str) -> bool:
        """转换npy到bin格式"""
        try:
            source_path = Path(npy_path)
            if not source_path.exists():
                return False
            
            target_dir = self.output_root / "samples" / LIDAR_NUSC_NAME
            target_path = target_dir / f"{unique_key}.bin"
            
            if target_path.exists():
                return True
            
            # 加载npy数据
            point_cloud = np.load(source_path, allow_pickle=False)
            
            # 确保是5维数据 (x, y, z, intensity, ring)
            if point_cloud.shape[1] >= 5:
                points_5d = point_cloud[:, :5].astype(np.float32)
            elif point_cloud.shape[1] >= 3:
                xyz = point_cloud[:, :3].astype(np.float32)
                n_points = xyz.shape[0]
                points_5d = np.zeros((n_points, 5), dtype=np.float32)
                points_5d[:, :3] = xyz
                points_5d[:, 3] = 1.0  # 默认强度
                points_5d[:, 4] = 0.0  # 默认ring
            else:
                self.logger.warning(f"点云维度不足: {point_cloud.shape}")
                return False
            
            # 保存为bin文件
            points_5d.tofile(target_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"转换npy到bin失败 {npy_path}: {e}")
            return False
    
    def process_annotation_data(self, sample_info: Dict, sample_token: str):
        """处理标注数据"""
        try:
            # 这里需要加载实际的标注文件
            label_path = Path(sample_info["files"]["label"])
            if not label_path.exists():
                return
            
            with open(label_path, 'r') as f:
                label_data = json.load(f)
            
            objects = []
            if isinstance(label_data, dict):
                objects = label_data.get("objects", [])
            elif isinstance(label_data, list):
                objects = label_data
            
            for obj_idx, obj in enumerate(objects):
                self.process_single_annotation(obj, sample_token, obj_idx)
                
        except Exception as e:
            self.logger.error(f"处理标注数据失败: {e}")
    
    def process_single_annotation(self, obj: Dict, sample_token: str, obj_idx: int):
        """处理单个标注"""
        try:
            # 获取类别
            tyjt_cat = obj.get("type", "other")
            nusc_cat = CATEGORY_MAPPING.get(tyjt_cat, "tyjt.other")
            
            # 获取3D框
            box3d = obj.get("box3d", [0, 0, 0, 1, 1, 1])
            x, y, z, l, w, h = (box3d + [0, 0, 0, 1, 1, 1])[:6]
            
            # 获取旋转
            rotation = obj.get("rotation", [0, 0, 0])
            yaw = rotation[0] if len(rotation) > 0 else 0
            
            # 计算四元数
            cy = math.cos(yaw * 0.5)
            sy = math.sin(yaw * 0.5)
            quaternion = [cy, 0, 0, sy]  # [w, x, y, z] 格式
            
            # 生成instance token
            position_key = f"{x:.2f}_{y:.2f}_{z:.2f}_{nusc_cat}"
            instance_key = f"{sample_token}_{position_key}_{obj_idx}"
            instance_token = self.id_gen.get_scoped_token("instance", instance_key)
            
            # 检查instance是否已存在
            instance_exists = False
            for inst in self.nusc_data["instance"]:
                if inst["token"] == instance_token:
                    inst["nbr_annotations"] += 1
                    inst["last_annotation_token"] = self.id_gen.get_scoped_token(
                        "annotation", f"{instance_token}_{sample_token}"
                    )
                    instance_exists = True
                    break
            
            if not instance_exists:
                first_ann_token = self.id_gen.get_scoped_token(
                    "annotation", f"{instance_token}_{sample_token}"
                )
                
                category_token = self.id_gen.get_scoped_token("category", nusc_cat)
                
                self.nusc_data["instance"].append({
                    "token": instance_token,
                    "category_token": category_token,
                    "nbr_annotations": 1,
                    "first_annotation_token": first_ann_token,
                    "last_annotation_token": first_ann_token
                })
            
            # 生成annotation
            ann_token = self.id_gen.get_scoped_token(
                "annotation", f"{instance_token}_{sample_token}"
            )
            
            # 属性
            attribute_tokens = []
            if 'vehicle' in nusc_cat:
                attribute_tokens.append(self.id_gen.get_scoped_token("attribute", "vehicle.moving"))
            elif 'pedestrian' in nusc_cat:
                attribute_tokens.append(self.id_gen.get_scoped_token("attribute", "pedestrian.moving"))
            else:
                attribute_tokens.append(self.id_gen.get_scoped_token("attribute", "vehicle.stopped"))
            
            self.nusc_data["sample_annotation"].append({
                "token": ann_token,
                "sample_token": sample_token,
                "instance_token": instance_token,
                "visibility_token": self.id_gen.get_scoped_token("visibility", "v80-100"),
                "attribute_tokens": attribute_tokens,
                "translation": [float(x), float(y), float(z)],
                "size": [float(w), float(l), float(h)],  # NuScenes格式: width, length, height
                "rotation": quaternion,
                "category_name": nusc_cat,
                "num_lidar_pts": obj.get("num_points", 10),
                "num_radar_pts": 0
            })
            
        except Exception as e:
            self.logger.error(f"处理单个标注失败: {e}")
    
    def save_nuscenes_data(self, version_dir: Path):
        """保存NuScenes数据表"""
        self.logger.info("保存NuScenes数据表...")
        
        for table_name, table_data in self.nusc_data.items():
            if table_name.startswith('_'):
                continue
            
            output_path = version_dir / f"{table_name}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(table_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 保存 {table_name}.json: {len(table_data)} 条记录")
    
    def validate_data(self):
        """验证数据一致性"""
        self.logger.info("验证数据一致性...")
        
        # 验证Token唯一性
        token_valid, token_stats = self.id_gen.validate_token_uniqueness()
        if not token_valid:
            self.logger.warning("⚠️  Token唯一性验证失败")
        else:
            self.logger.info("✅ Token唯一性验证通过")
        
        self.logger.info(f"📊 Token统计: {token_stats}")
        
        # 基本统计
        total_samples = len(self.nusc_data["sample"])
        total_annotations = len(self.nusc_data["sample_annotation"])
        total_scenes = len(self.nusc_data["scene"])
        
        self.logger.info(f"📊 数据统计:")
        self.logger.info(f"  场景数: {total_scenes}")
        self.logger.info(f"  样本数: {total_samples}")
        self.logger.info(f"  标注数: {total_annotations}")
        self.logger.info(f"  实例数: {len(self.nusc_data['instance'])}")
        self.logger.info(f"  传感器数: {len(self.nusc_data['sensor'])}")
        
        return token_valid
    
    def run(self):
        """主运行函数"""
        self.logger.info("=" * 60)
        self.logger.info("🚀 TYJT数据集转换工具 v9.2.3-phase2-v1.9 (数据处理版)")
        self.logger.info(f"📁 输出目录: {self.output_root}")
        self.logger.info(f"📊 分析结果: {self.analysis_root}")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # 1. 创建目录结构
            version_dir = self.create_directory_structure()
            
            # 2. 加载分析结果
            analysis_results = self.load_analysis_results()
            if not analysis_results:
                self.logger.error("❌ 无分析结果可用，退出")
                return
            
            # 3. 生成基础数据表
            self.generate_basic_tables(analysis_results[0])
            
            # 4. 处理每个数据包
            total_samples_all = 0
            for result in analysis_results:
                samples_processed = self.process_package(result)
                total_samples_all += samples_processed
            
            # 5. 保存数据
            self.save_nuscenes_data(version_dir)
            
            # 6. 数据验证
            self.validate_data()
            
            # 7. 输出统计信息
            elapsed_time = time.time() - start_time
            self.logger.info("=" * 60)
            self.logger.info("🎉 数据处理完成!")
            self.logger.info(f"⏱️  总耗时: {elapsed_time:.2f}秒")
            self.logger.info(f"📁 输出目录: {version_dir}")
            self.logger.info(f"📦 处理数据包数: {len(analysis_results)}")
            self.logger.info(f"📊 总样本数: {total_samples_all}")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"❌ 数据处理失败: {e}", exc_info=True)
            raise

# ==================== 主函数 ====================
def main():
    """主函数"""
    try:
        processor = DataProcessor(OUTPUT_ROOT, ANALYSIS_ROOT)
        processor.run()
    except Exception as e:
        print(f"❌ 程序运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()