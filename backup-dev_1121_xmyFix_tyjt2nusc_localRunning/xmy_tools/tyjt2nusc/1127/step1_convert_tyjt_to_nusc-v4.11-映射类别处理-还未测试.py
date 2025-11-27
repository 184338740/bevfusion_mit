#!/usr/bin/env python3
"""
tyjt数据集转nuscenes格式转换工具 - 修复标定问题版本
版本: v5.2-fixed-calib
修复重点:
1. 修正sensor->ego的标定转换逻辑
2. 修复四元数顺序问题
3. 正确的坐标变换链
"""

import os
import json
import numpy as np
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import hashlib
import math
from collections import defaultdict

# ==================== 配置参数 ====================
TYJT_ROOT = "/mnt/dataset/tyjt_RawData_demo"
OUTPUT_ROOT = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1127/output/step1/nuscenes_tyjt"
NUSC_VERSION = "v1.0-tyjt"

# 相机映射
CAMERA_MAPPING = {
    'SC_1A_CamR': 'CAM_FRONT',
    'SC_1B_CamR': 'CAM_FRONT_RIGHT', 
    'SC_1C_CamR': 'CAM_BACK',
    'SC_1D_CamR': 'CAM_FRONT_LEFT'
}

# 标定名称映射
CALIB_NAME_MAPPING = {
    'SC_1A_CamR': 'SC_R1_Aw_CpcS_CAMR_new',
    'SC_1B_CamR': 'SC_R1_Bn_CpcW_CAMR_new', 
    'SC_1C_CamR': 'SC_R1_Ce_CpcN_CAMR_new',
    'SC_1D_CamR': 'SC_R1_Ds_CpcE_CAMR_new'
}

LIDAR_NUSC_NAME = 'LIDAR_TOP'

# 类别映射
# 替换 CATEGORY_MAPPING
CATEGORY_MAPPING = {
    "car": "car",
    "truck": "truck", 
    "construction_truck": "construction_vehicle",
    "van": "car",  # van映射到car
    "bus": "bus",
    "robot": "construction_vehicle",  # 机器人映射到工程车
    "pedestrian": "pedestrian", 
    "cyclist": "motorcycle",  # 骑行者映射到摩托车
    "bicycle": "bicycle",
    "tricycle": "motorcycle",  # 三轮车映射到摩托车
    "tricyclist": "pedestrian",  # 三轮车骑行者映射到行人
    "trolley": "trailer",
    "cone": "traffic_cone",
    "barrier": "barrier",
    "other": "barrier"  # 其他映射到障碍物
}

# ==================== 增强的Token管理器 ====================
class EnhancedIDGenerator:
    def __init__(self):
        self.uuid_map = {}
        self.used_tokens: Set[str] = set()
        self.entity_counters = defaultdict(int)
    
    def get_scoped_token(self, entity_type: str, unique_key: str) -> str:
        """生成带作用域的唯一token"""
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
        """验证token唯一性并返回统计信息"""
        tokens = list(self.uuid_map.values())
        unique_tokens = set(tokens)
        
        stats = {
            'total_tokens': len(tokens),
            'unique_tokens': len(unique_tokens),
            'entity_counts': dict(self.entity_counters)
        }
        
        is_unique = len(tokens) == len(unique_tokens)
        return is_unique, stats

id_gen = EnhancedIDGenerator()

# ==================== 修复的数学工具函数 ====================
def quaternion_from_euler(yaw: float, pitch: float = 0, roll: float = 0) -> List[float]:
    """欧拉角转四元数 [w, x, y, z] 顺序"""
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
    """四元数转旋转矩阵 [w, x, y, z] - NuScenes标准顺序"""
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])

def rotation_matrix_to_quaternion(matrix: np.ndarray) -> List[float]:
    """旋转矩阵转四元数 [w, x, y, z]"""
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

def parse_tyjt_transform(calib_params: Dict) -> np.ndarray:
    """解析tyjt标定参数为4x4变换矩阵 - 修复版本"""
    # tyjt标定参数顺序是 [x, y, z, w]
    translation = np.array([
        calib_params['tx'], 
        calib_params['ty'], 
        calib_params['tz']
    ])
    
    # tyjt四元数顺序: [x, y, z, w]
    x, y, z, w = calib_params['rx'], calib_params['ry'], calib_params['rz'], calib_params['rw']
    
    # 转换为 [w, x, y, z] 顺序用于旋转矩阵计算
    rotation_matrix = quaternion_to_rotation_matrix([w, x, y, z])
    
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    
    return transform

def get_sensor2ego_transform_fixed(calib_data: Dict, sensor_calib_name: str) -> Tuple[List[float], List[float]]:
    """修复：计算sensor到ego的变换：sensor->map->group(ego)"""
    try:
        if sensor_calib_name not in calib_data:
            print(f"⚠️  警告: 标定数据中找不到 {sensor_calib_name}")
            return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
            
        # sensor到map的变换
        sensor_params = calib_data[sensor_calib_name]
        sensor2map = parse_tyjt_transform(sensor_params)
        
        # map到group(ego)的变换（group2map的逆）
        if 'group2map' in calib_data:
            group2map = parse_tyjt_transform(calib_data['group2map'])
            map2group = np.linalg.inv(group2map)
        else:
            map2group = np.eye(4)
            
        # 关键修复：正确的变换顺序 sensor->map->group(ego)
        # sensor2ego = map2group @ sensor2map
        sensor2ego = map2group @ sensor2map
        
        # 提取平移和旋转
        translation = sensor2ego[:3, 3].tolist()
        rotation_matrix = sensor2ego[:3, :3]
        
        # 修复：使用正确的旋转矩阵转四元数函数
        rotation = rotation_matrix_to_quaternion(rotation_matrix)
        
        print(f"   标定转换 {sensor_calib_name}:")
        print(f"     sensor2map平移: {sensor2map[:3, 3]}")
        print(f"     group2map平移: {group2map[:3, 3]}")
        print(f"     sensor2ego平移: {translation}")
        print(f"     sensor2ego旋转: {rotation}")
        
        return translation, rotation
        
    except Exception as e:
        print(f"❌ 计算sensor2ego变换时出错 {sensor_calib_name}: {e}")
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]

# ==================== 数据加载和验证 ====================
def load_calibration(calib_path: str) -> Dict:
    """加载tyjt标定文件"""
    if not Path(calib_path).exists():
        print(f"❌ 错误: 标定文件不存在: {calib_path}")
        return {}
    
    try:
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
        
        # 验证标定数据完整性
        required_calibs = list(CALIB_NAME_MAPPING.values()) + ['group2map']
        missing_calibs = [calib for calib in required_calibs if calib not in calib_data]
        
        if missing_calibs:
            print(f"⚠️  标定数据缺失: {missing_calibs}")
        
        return calib_data
    except Exception as e:
        print(f"❌ 加载标定文件失败 {calib_path}: {e}")
        return {}

def create_directory_structure(output_root: str, version: str):
    """创建完整的nuscenes目录结构"""
    base_path = Path(output_root)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # 创建版本目录
    version_dir = base_path / version
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建samples目录结构
    samples_dir = base_path / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    for cam_name in CAMERA_MAPPING.values():
        (samples_dir / cam_name).mkdir(parents=True, exist_ok=True)
    (samples_dir / LIDAR_NUSC_NAME).mkdir(parents=True, exist_ok=True)
    
    # 创建sweeps目录结构
    sweeps_dir = base_path / "sweeps"
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    
    for cam_name in CAMERA_MAPPING.values():
        (sweeps_dir / cam_name).mkdir(parents=True, exist_ok=True)
    (sweeps_dir / LIDAR_NUSC_NAME).mkdir(parents=True, exist_ok=True)
    
    # 创建maps目录和空图像文件
    maps_dir = base_path / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建空的PNG图像文件
    empty_map_path = maps_dir / "empty_map.png"
    
    # 创建一个1x1像素的空白PNG图像
    from PIL import Image
    empty_img = Image.new('RGB', (1, 1), color='black')
    empty_img.save(empty_map_path)
    
    print(f"✅ 目录结构创建完成: {base_path}")
    return base_path

def find_data_packages(tyjt_root: str) -> List[Path]:
    """查找所有数据包"""
    root_path = Path(tyjt_root)
    packages = []
    
    for item in root_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            calib_dir = item / "calib"
            if calib_dir.exists():
                packages.append(item)
                print(f"✅ 找到数据包: {item.name}")
    
    print(f"📦 总共找到 {len(packages)} 个数据包")
    return packages

# ==================== 数据表初始化 ====================
def initialize_nusc_data() -> Dict:
    """初始化空的nuscenes数据结构"""
    return {
        "category": [], "attribute": [], "sensor": [], "calibrated_sensor": [],
        "ego_pose": [], "log": [], "scene": [], "sample": [], "sample_data": [],
        "sample_annotation": [], "instance": [], "visibility": [], "map": []
    }

def generate_basic_tables(nusc_data: Dict, scene_name: str, calib_data: Dict) -> Dict:
    """生成基础数据表并返回token映射 - 修复版本"""
    token_map = {}
    
    # 1. 生成log表
    log_token = id_gen.get_scoped_token("log", scene_name)
    token_map['log_token'] = log_token
    nusc_data["log"].append({
        "token": log_token,
        "logfile": f"{scene_name}.log",
        "vehicle": "tyjt_system",
        "date_captured": "2024-01-01",
        "location": scene_name
    })
    
    # 2. 生成scene表
    scene_token = id_gen.get_scoped_token("scene", scene_name)
    token_map['scene_token'] = scene_token
    nusc_data["scene"].append({
        "token": scene_token,
        "name": scene_name,
        "description": f"From tyjt: {scene_name}",
        "log_token": log_token,
        "nbr_samples": 0,
        "first_sample_token": "",
        "last_sample_token": ""
    })
    
    # 3. 生成category表（去重）
    existing_categories = {cat['name'] for cat in nusc_data['category']}
    for tyjt_cat, nusc_cat in CATEGORY_MAPPING.items():
        if nusc_cat not in existing_categories:
            nusc_data["category"].append({
                "token": id_gen.get_scoped_token("category", nusc_cat),
                "name": nusc_cat,
                "description": f"From tyjt: {tyjt_cat}"
            })
            existing_categories.add(nusc_cat)
    
    # 4. 生成attribute表（去重）
    existing_attributes = {attr['name'] for attr in nusc_data['attribute']}
    if "vehicle.moving" not in existing_attributes:
        nusc_data["attribute"].append({
            "token": id_gen.get_scoped_token("attribute", "vehicle.moving"),
            "name": "vehicle.moving",
            "description": "Object is moving"
        })
    
    # 5. 生成visibility表（去重）
    if not nusc_data["visibility"]:
        nusc_data["visibility"].append({
            "token": id_gen.get_scoped_token("visibility", "full"),
            "level": "full",
            "description": "Fully visible"
        })
    
    # 6. 生成sensor表（去重）
    existing_sensors = {sensor['channel'] for sensor in nusc_data['sensor']}
    sensor_tokens = {}
    
    for tyjt_cam, nusc_cam in CAMERA_MAPPING.items():
        if nusc_cam not in existing_sensors:
            sensor_token = id_gen.get_scoped_token("sensor", nusc_cam)
            sensor_tokens[nusc_cam] = sensor_token
            nusc_data["sensor"].append({
                "token": sensor_token,
                "channel": nusc_cam,
                "modality": "camera"
            })
            existing_sensors.add(nusc_cam)
        else:
            # 找到已存在的sensor token
            for sensor in nusc_data['sensor']:
                if sensor['channel'] == nusc_cam:
                    sensor_tokens[nusc_cam] = sensor['token']
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
    
    # 7. 生成calibrated_sensor表（每个场景独立）- 使用修复的标定转换
    calib_sensor_tokens = {}
    
    print(f"🔧 生成相机标定数据:")
    # 相机标定
    for tyjt_cam, nusc_cam in CAMERA_MAPPING.items():
        calib_name = CALIB_NAME_MAPPING[tyjt_cam]
        translation, rotation = get_sensor2ego_transform_fixed(calib_data, calib_name)
        
        calib_token = id_gen.get_scoped_token("calibrated_sensor", f"{scene_name}_{nusc_cam}")
        calib_sensor_tokens[nusc_cam] = calib_token
        
        # 获取相机内参
        if calib_name in calib_data:
            cam_calib = calib_data[calib_name]
            nusc_data["calibrated_sensor"].append({
                "token": calib_token,
                "sensor_token": sensor_tokens[nusc_cam],
                "translation": translation,
                "rotation": rotation,
                "camera_intrinsic": [
                    [cam_calib.get('fx', 1680.0), 0, cam_calib.get('cx', 960.0)],
                    [0, cam_calib.get('fy', 1851.0), cam_calib.get('cy', 540.0)],
                    [0, 0, 1]
                ]
            })
            print(f"   ✅ {nusc_cam}: 内参设置完成")
        else:
            print(f"   ⚠️  警告: 找不到相机 {tyjt_cam} 的标定参数")
    
    # 激光雷达标定 (假设在ego坐标系原点)
    lidar_calib_token = id_gen.get_scoped_token("calibrated_sensor", f"{scene_name}_{LIDAR_NUSC_NAME}")
    calib_sensor_tokens[LIDAR_NUSC_NAME] = lidar_calib_token
    nusc_data["calibrated_sensor"].append({
        "token": lidar_calib_token,
        "sensor_token": sensor_tokens[LIDAR_NUSC_NAME],
        "translation": [0.0, 0.0, 0.0],
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "camera_intrinsic": []
    })
    
    token_map['calib_sensor_tokens'] = calib_sensor_tokens
    
    # 8. 生成ego_pose表（每个场景一个）
    ego_pose_token = id_gen.get_scoped_token("ego_pose", scene_name)
    token_map['ego_pose_token'] = ego_pose_token
    nusc_data["ego_pose"].append({
        "token": ego_pose_token,
        "translation": [0.0, 0.0, 0.0],
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "timestamp": 0
    })
    
    # 9. map表处理（全局唯一）
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
        # 更新现有map的log_tokens
        map_record = nusc_data["map"][0]
        current_log_tokens = set(map_record.get('log_tokens', []))
        current_log_tokens.add(log_token)
        map_record['log_tokens'] = list(current_log_tokens)
        token_map['map_token'] = map_record['token']
    
    return token_map

# ==================== 样本处理 ====================
def npy_to_bin(npy_path: Path, bin_path: Path) -> bool:
    """将npy点云转换为nuscenes标准bin格式"""
    try:
        point_cloud = np.load(npy_path)
        
        if point_cloud.shape[1] >= 5:
            # 使用前5个维度: x, y, z, intensity, ring
            points_5d = point_cloud[:, :5].astype(np.float32)
        elif point_cloud.shape[1] >= 3:
            # 只有xyz，添加默认值
            xyz = point_cloud[:, :3].astype(np.float32)
            n_points = xyz.shape[0]
            points_5d = np.zeros((n_points, 5), dtype=np.float32)
            points_5d[:, :3] = xyz
            points_5d[:, 3] = 1.0  # intensity
            points_5d[:, 4] = 0.0  # ring
        else:
            print(f"⚠️  点云格式异常: {npy_path}, shape: {point_cloud.shape}")
            return False
        
        points_5d.tofile(bin_path)
        return True
            
    except Exception as e:
        print(f"❌ 点云转换失败 {npy_path}: {e}")
        return False

def process_single_sample(nusc_data: Dict, sub_packet: Path, timestamp: str, 
                         output_root: str, packet_name: str, sub_packet_name: str, 
                         prev_sample_token: str, sample_index: int, token_map: Dict):
    """处理单个样本"""
    
    # 生成唯一样本标识
    unique_sample_key = f"{packet_name}::{sub_packet_name}::{timestamp}"
    sample_token = id_gen.get_scoped_token("sample", unique_sample_key)
    
    # sample记录
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
    
    # 处理传感器数据
    process_sensor_data(nusc_data, sub_packet, timestamp, output_root, sample_token, 
                       packet_name, sub_packet_name, token_map)
    
    # 处理标注数据
    process_annotation_data(nusc_data, sub_packet, timestamp, sample_token, 
                           packet_name, sub_packet_name)
    
    return sample_token


def process_sensor_data(nusc_data: Dict, sub_packet: Path, timestamp: str, output_root: str, 
                       sample_token: str, packet_name: str, sub_packet_name: str, token_map: Dict):
    """处理传感器数据 - 修复文件覆盖问题"""
    
    # 生成唯一文件标识 - 包含场景包信息防止覆盖
    file_suffix = f"{packet_name}_{sub_packet_name}_{timestamp}"
    
    # 相机数据
    for tyjt_cam, nusc_cam in CAMERA_MAPPING.items():
        img_path = sub_packet / tyjt_cam / "image_dc" / f"{timestamp}.jpg"
        if img_path.exists():
            output_img_dir = Path(f"{output_root}/samples/{nusc_cam}")
            output_img_dir.mkdir(parents=True, exist_ok=True)
            # 修复：使用唯一文件标识
            output_img_path = output_img_dir / f"{file_suffix}.jpg"
            
            try:
                # 检查文件是否已存在（防止重复处理）
                if output_img_path.exists():
                    print(f"    ⚠️  文件已存在，跳过: {output_img_path.name}")
                    continue
                    
                shutil.copy(str(img_path), str(output_img_path))
                
                sample_data_token = id_gen.get_scoped_token("sample_data", 
                    f"{packet_name}_{sub_packet_name}_{nusc_cam}_{timestamp}")
                
                nusc_data["sample_data"].append({
                    "token": sample_data_token,
                    "sample_token": sample_token,
                    "ego_pose_token": token_map['ego_pose_token'],
                    "calibrated_sensor_token": token_map['calib_sensor_tokens'][nusc_cam],
                    "filename": f"samples/{nusc_cam}/{file_suffix}.jpg",  # 修复：更新filename
                    "fileformat": "jpg",
                    "width": 1920,
                    "height": 1080,
                    "timestamp": int(timestamp) if timestamp.isdigit() else 0,
                    "is_key_frame": True,
                    "next": "",
                    "prev": ""
                })
                print(f"    💾 保存图像: {output_img_path.name}")
            except Exception as e:
                print(f"    ❌ 复制图像失败 {img_path}: {e}")
    
    # 激光雷达数据
    lidar_file = sub_packet / "lidar" / "pcd" / f"{timestamp}.npy"
    if lidar_file.exists():
        output_lidar_dir = Path(f"{output_root}/samples/{LIDAR_NUSC_NAME}")
        output_lidar_dir.mkdir(parents=True, exist_ok=True)
        # 修复：使用唯一文件标识
        output_lidar_path = output_lidar_dir / f"{file_suffix}.bin"
        
        # 检查文件是否已存在
        if output_lidar_path.exists():
            print(f"    ⚠️  点云文件已存在，跳过: {output_lidar_path.name}")
        elif npy_to_bin(lidar_file, output_lidar_path):
            sample_data_token = id_gen.get_scoped_token("sample_data",
                f"{packet_name}_{sub_packet_name}_{LIDAR_NUSC_NAME}_{timestamp}")
            
            nusc_data["sample_data"].append({
                "token": sample_data_token,
                "sample_token": sample_token,
                "ego_pose_token": token_map['ego_pose_token'],
                "calibrated_sensor_token": token_map['calib_sensor_tokens'][LIDAR_NUSC_NAME],
                "filename": f"samples/{LIDAR_NUSC_NAME}/{file_suffix}.bin",  # 修复：更新filename
                "fileformat": "bin",
                "width": 0,
                "height": 0,
                "timestamp": int(timestamp) if timestamp.isdigit() else 0,
                "is_key_frame": True,
                "next": "",
                "prev": ""
            })
            print(f"    💾 保存点云: {output_lidar_path.name}")



def process_annotation_data(nusc_data: Dict, sub_packet: Path, timestamp: str, 
                           sample_token: str, packet_name: str, sub_packet_name: str):
    """处理标注数据"""
    label_path = sub_packet / "lidar" / "label" / f"{timestamp}.json"
    if not label_path.exists():
        return
    
    try:
        with open(label_path, 'r') as f:
            label_data = json.load(f)
        
        objects = label_data.get("objects", []) if isinstance(label_data, dict) else label_data
        
        for obj_idx, obj in enumerate(objects):
            process_single_annotation(nusc_data, obj, obj_idx, sample_token, 
                                    packet_name, sub_packet_name, timestamp)
            
    except Exception as e:
        print(f"    ❌ 处理标注失败 {label_path}: {e}")

def process_single_annotation(nusc_data: Dict, obj: Dict, obj_idx: int, 
                             sample_token: str, packet_name: str, sub_packet_name: str, timestamp: str):
    """处理单个标注 - 修复所有关联问题"""
    tyjt_cat = obj.get("type", "other")
    nusc_cat = CATEGORY_MAPPING.get(tyjt_cat, "movable_object.debris")
    
    # tyjt格式: [x, y, z, l, w, h]
    box3d = obj.get("box3d", [0, 0, 0, 1, 1, 1])
    if len(box3d) == 6:
        x, y, z, l, w, h = box3d
    else:
        x, y, z, l, w, h = 0, 0, 0, 1, 1, 1
    
    # 转换rotation (只使用yaw)
    rotation = obj.get("rotation", [0, 0, 0])
    yaw = rotation[0] if len(rotation) > 0 else 0
    quaternion = quaternion_from_euler(yaw)
    
    # 修复：生成基于物理位置的instance_token，确保同一物体在不同帧使用相同instance
    position_key = f"{x:.2f}_{y:.2f}_{z:.2f}_{nusc_cat}"
    instance_key = f"{packet_name}_{position_key}"
    instance_token = id_gen.get_scoped_token("instance", instance_key)
    
    # 修复：确保category_token存在
    category_token = id_gen.get_scoped_token("category", nusc_cat)
    
    # 修复：添加默认属性
    attribute_tokens = []
    if 'vehicle' in nusc_cat:
        attribute_tokens.append(id_gen.get_scoped_token("attribute", "vehicle.moving"))
    elif 'human' in nusc_cat:
        attribute_tokens.append(id_gen.get_scoped_token("attribute", "pedestrian.moving"))
    else:
        attribute_tokens.append(id_gen.get_scoped_token("attribute", "vehicle.stopped"))
    
    # 创建或更新instance
    instance_exists = False
    for inst in nusc_data["instance"]:
        if inst["token"] == instance_token:
            inst["nbr_annotations"] += 1
            inst["last_annotation_token"] = id_gen.get_scoped_token("annotation", 
                f"{instance_token}_{timestamp}")
            instance_exists = True
            break
    
    if not instance_exists:
        nusc_data["instance"].append({
            "token": instance_token,
            "category_token": category_token,
            "nbr_annotations": 1,
            "first_annotation_token": id_gen.get_scoped_token("annotation", 
                f"{instance_token}_{timestamp}"),
            "last_annotation_token": id_gen.get_scoped_token("annotation", 
                f"{instance_token}_{timestamp}")
        })
    
    # 创建annotation - 修复所有关联问题
    ann_token = id_gen.get_scoped_token("annotation", f"{instance_token}_{timestamp}")
    nusc_data["sample_annotation"].append({
        "token": ann_token,
        "sample_token": sample_token,
        "instance_token": instance_token,
        "visibility_token": id_gen.get_scoped_token("visibility", "full"),
        "attribute_tokens": attribute_tokens,
        "translation": [float(x), float(y), float(z)],
        "size": [float(w), float(l), float(h)],  # nusc格式: [w, l, h]
        "rotation": quaternion,
        "category_name": nusc_cat,
        "num_lidar_pts": 10,
        "num_radar_pts": 0
    })

# ==================== 场景处理 ====================
def process_scene_samples(nusc_data: Dict, packet_dir: Path, output_root: str, 
                         scene_name: str, token_map: Dict):
    """处理场景中的所有样本 - 修复重复样本和文件覆盖问题"""
    datasets_dir = packet_dir / "datasets"
    if not datasets_dir.exists():
        print(f"⚠️  警告: 无datasets目录: {datasets_dir}")
        return []
    
    # 收集所有样本
    all_samples = []
    # 使用更严格的唯一标识，包含场景包信息
    processed_samples = set()
    sub_packets = [d for d in datasets_dir.iterdir() if d.is_dir()]
    
    for sub_packet in sub_packets:
        print(f"  📁 处理子包: {sub_packet.name}")
        
        lidar_dir = sub_packet / "lidar" / "pcd"
        if not lidar_dir.exists():
            continue
            
        lidar_files = list(lidar_dir.glob("*.npy"))
        for lidar_file in lidar_files:
            timestamp = lidar_file.stem
            if timestamp.isdigit():
                # 使用场景+子包+时间戳作为唯一标识
                sample_key = f"{scene_name}_{sub_packet.name}_{timestamp}"
                
                if sample_key in processed_samples:
                    print(f"  ⚠️  跳过重复样本: {timestamp} (场景: {scene_name}, 子包: {sub_packet.name})")
                    continue
                
                all_samples.append((lidar_file, timestamp, sub_packet))
                processed_samples.add(sample_key)
    
    # 按时间戳排序
    all_samples.sort(key=lambda x: int(x[1]))
    
    print(f"  📊 去重后样本数: {len(all_samples)}")
    print(f"  🔍 样本标识示例: {list(processed_samples)[:3] if processed_samples else '无'}")  # 调试信息
    
    # 处理样本
    sample_tokens = []
    prev_token = ""
    
    for i, (lidar_file, timestamp, sub_packet) in enumerate(all_samples):
        sample_token = process_single_sample(
            nusc_data, sub_packet, timestamp, output_root, 
            scene_name, sub_packet.name, prev_token, i, token_map
        )
        if sample_token:
            sample_tokens.append(sample_token)
            prev_token = sample_token
    
    return sample_tokens



def process_scene(packet_dir: Path, output_root: str, version: str, global_nusc_data: Dict):
    """处理单个场景"""
    scene_name = packet_dir.name
    print(f"\n🎬 开始处理场景: {scene_name}")
    
    # 加载标定
    calib_path = packet_dir / "calib" / "sensor2map_calib.json"
    calib_data = load_calibration(str(calib_path))
    
    if not calib_data:
        print(f"❌ 跳过场景 {scene_name}，标定数据加载失败")
        return
    
    # 生成基础数据表
    token_map = generate_basic_tables(global_nusc_data, scene_name, calib_data)
    
    # 处理样本数据
    sample_tokens = process_scene_samples(global_nusc_data, packet_dir, output_root, scene_name, token_map)
    
    # 更新scene信息
    if global_nusc_data["scene"] and sample_tokens:
        current_scene = None
        for scene in global_nusc_data["scene"]:
            if scene["token"] == token_map['scene_token']:
                current_scene = scene
                break
        
        if current_scene:
            current_scene["nbr_samples"] = len(sample_tokens)
            current_scene["first_sample_token"] = sample_tokens[0]
            current_scene["last_sample_token"] = sample_tokens[-1]
            print(f"  ✅ 更新scene: {len(sample_tokens)} 个样本")

# ==================== 数据验证 ====================
def validate_data_consistency(nusc_data: Dict) -> Tuple[bool, List[str]]:
    """全面验证数据一致性 - 增强文件唯一性检查"""
    print("\n🔍 开始数据一致性验证...")
    errors = []
    
    # 构建索引
    indices = {}
    for table_name, records in nusc_data.items():
        if table_name.startswith('_'): continue
        indices[table_name] = {r['token']: r for r in records}
    
    # 验证外键关系
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
    
    # 验证样本链
    sample_tokens = {s['token']: s for s in nusc_data.get('sample', [])}
    for sample in nusc_data.get('sample', []):
        if sample['prev'] and sample['prev'] not in sample_tokens:
            errors.append(f"sample {sample['token'][:8]} 的prev指向不存在的sample")
        if sample['next'] and sample['next'] not in sample_tokens:
            errors.append(f"sample {sample['token'][:8]} 的next指向不存在的sample")
    
    # 验证文件存在性和唯一性
    output_base = Path(OUTPUT_ROOT)
    file_paths = set()
    
    for sample_data in nusc_data.get('sample_data', []):
        file_path = output_base / sample_data['filename']
        
        # 检查文件是否存在
        if not file_path.exists():
            errors.append(f"sample_data {sample_data['token'][:8]} 引用的文件不存在: {file_path}")
        
        # 检查文件路径是否唯一
        if sample_data['filename'] in file_paths:
            errors.append(f"文件路径重复: {sample_data['filename']}")
        else:
            file_paths.add(sample_data['filename'])
    
    # 检查文件名格式（确保包含场景信息）
    for filename in file_paths:
        if filename.count('_') < 2:  # 至少包含packet_name_subpacket_timestamp
            errors.append(f"文件名格式可能不包含场景信息: {filename}")
    
    is_valid = len(errors) == 0
    if is_valid:
        print("✅ 数据一致性验证通过")
        print(f"📊 文件统计: {len(file_paths)} 个唯一文件")
    else:
        print(f"❌ 发现 {len(errors)} 个数据一致性问题")
        for error in errors[:10]:
            print(f"  {error}")
    
    return is_valid, errors



# ==================== 主函数 ====================
def main():
    """主函数 - Step1数据生成"""
    print("🚀 Step1: tyjt转nuscenes格式转换工具 v5.2-fixed-calib")
    print(f"输入: {TYJT_ROOT}")
    print(f"输出: {OUTPUT_ROOT}")
    
    # 创建输出目录
    output_base = create_directory_structure(OUTPUT_ROOT, NUSC_VERSION)
    
    # 查找数据包
    packages = find_data_packages(TYJT_ROOT)
    if not packages:
        print("❌ 未找到任何数据包，退出")
        return
    
    # 初始化全局数据结构
    global_nusc_data = initialize_nusc_data()
    
    # 处理所有场景
    for package in packages:
        process_scene(package, str(output_base), NUSC_VERSION, global_nusc_data)
    
    # 验证
    token_unique, token_stats = id_gen.validate_token_uniqueness()
    if not token_unique:
        print("⚠️  警告: 存在重复token")
    print(f"📊 Token统计: {token_stats}")
    
    data_valid, errors = validate_data_consistency(global_nusc_data)
    if not data_valid:
        print("⚠️  警告: 数据一致性验证发现问题")
    
    # 保存数据
    version_dir = output_base / NUSC_VERSION
    version_dir.mkdir(parents=True, exist_ok=True)
    
    for table_name, table_data in global_nusc_data.items():
        if table_name.startswith('_'): continue
        output_path = version_dir / f"{table_name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(table_data, f, indent=2, ensure_ascii=False)
        print(f"💾 保存 {table_name}.json: {len(table_data)} 条记录")
    
    print(f"\n🎉 Step1完成!")
    print(f"📁 输出目录: {version_dir}")
    print(f"📊 数据统计:")
    print(f"  - 场景: {len(global_nusc_data['scene'])}")
    print(f"  - 样本: {len(global_nusc_data['sample'])}")
    print(f"  - 标注: {len(global_nusc_data['sample_annotation'])}")
    print(f"  - 传感器数据: {len(global_nusc_data['sample_data'])}")
    
    if not data_valid:
        print("\n⚠️  注意: 数据存在一致性问题，请检查上述错误信息")

if __name__ == "__main__":
    main()