# step1_analyze_bevfusion_pkl_format.py
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

def analyze_bevfusion_pkl_format():
    """分析BEVFusion PKL格式并生成文档"""
    
    # 加载官方PKL文件分析结构
    nuscenes_pkl_path = "/mnt/bevfusion_mit_xmy/data/nuscenes/nuscenes_infos_train.pkl"
    
    with open(nuscenes_pkl_path, 'rb') as f:
        nuscenes_data = pickle.load(f)
    
    print("BEVFusion PKL格式分析报告")
    print("=" * 50)
    
    # 分析顶层结构
    print(f"顶层键: {nuscenes_data.keys()}")
    
    if 'infos' in nuscenes_data:
        sample = nuscenes_data['infos'][0]
        print(f"\n样本键: {sample.keys()}")
        
        # 详细分析每个字段
        analyze_sample_structure(sample)
        
        # 生成格式规范文档
        generate_format_spec(sample, nuscenes_data.get('metadata', {}))
    
    return nuscenes_data

def analyze_sample_structure(sample: Dict):
    """分析样本结构"""
    print("\n详细字段分析:")
    print("-" * 30)
    
    # 基础信息字段
    base_fields = ['token', 'timestamp', 'scene_token', 'lidar_path', 'sweeps']
    for field in base_fields:
        if field in sample:
            print(f"{field}: {type(sample[field])} - {sample[field]}")
    
    # 相机数据
    if 'cams' in sample:
        print(f"\n相机数据 (共{len(sample['cams'])}个):")
        for cam_name, cam_data in list(sample['cams'].items())[:2]:  # 只显示前2个
            print(f"  {cam_name}: {cam_data.keys()}")
            
            # 相机字段详情
            for key, value in list(cam_data.items())[:3]:  # 只显示前3个字段
                if hasattr(value, 'shape'):
                    print(f"    {key}: {type(value)} - 形状: {value.shape} - dtype: {value.dtype}")
                elif isinstance(value, list):
                    print(f"    {key}: {type(value)} - 长度: {len(value)}")
                else:
                    print(f"    {key}: {type(value)}")
    
    # 标注信息 - 关键分析
    if 'ann_info' in sample:
        print(f"\n标注信息 (ann_info):")
        ann_info = sample['ann_info']
        for key, value in ann_info.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {type(value)} - 形状: {value.shape} - dtype: {value.dtype}")
            elif isinstance(value, list):
                print(f"  {key}: {type(value)} - 长度: {len(value)}")
            else:
                print(f"  {key}: {type(value)}")
    
    # 其他重要字段
    other_fields = ['lidar2ego_rotation', 'lidar2ego_translation', 
                   'ego2global_rotation', 'ego2global_translation']
    for field in other_fields:
        if field in sample:
            value = sample[field]
            if isinstance(value, list):
                print(f"{field}: list - 长度: {len(value)}")
            else:
                print(f"{field}: {type(value)}")

def generate_format_spec(sample: Dict, metadata: Dict):
    """生成格式规范文档"""
    
    spec = {
        "format_version": "1.0",
        "description": "BEVFusion PKL格式规范 - 基于NuScenes数据集",
        "top_level_structure": {
            "infos": "List[Dict] - 样本列表",
            "metadata": "Dict - 元数据信息"
        },
        "sample_structure": {
            "required_fields": {
                "token": "str - 样本唯一标识",
                "timestamp": "int - 时间戳",
                "scene_token": "str - 场景标识", 
                "lidar_path": "str - LiDAR数据路径",
                "sweeps": "List - 历史帧信息",
                "cams": "Dict - 相机数据",
                "ann_info": "Dict - 标注信息"
            },
            "camera_structure": generate_camera_spec(sample.get('cams', {})),
            "annotation_structure": generate_annotation_spec(sample.get('ann_info', {})),
            "coordinate_fields": generate_coordinate_spec(sample)
        },
        "data_types": generate_data_types_spec(sample),
        "tyjt_adaptation_notes": generate_adaptation_notes()
    }
    
    # 保存规范文档
    with open("bevfusion_pkl_format_spec.json", "w") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 格式规范已保存: bevfusion_pkl_format_spec.json")

def generate_camera_spec(cams: Dict) -> Dict:
    """生成相机数据规范"""
    if not cams:
        return {}
    
    first_cam = list(cams.values())[0]
    spec = {
        "required_fields": {
            "data_path": "str - 图像文件路径",
            "cam_intrinsic": "np.ndarray (3,3) float32 - 相机内参",
            "sensor2lidar_rotation": "np.ndarray (3,3) float32 - 传感器到LiDAR旋转",
            "sensor2lidar_translation": "np.ndarray (3,) float32 - 传感器到LiDAR平移"
        },
        "optional_fields": {
            "sensor2ego_rotation": "list (3x3) - 传感器到ego旋转",
            "sensor2ego_translation": "list (3,) - 传感器到ego平移", 
            "ego2global_rotation": "list (3x3) - ego到全局旋转",
            "ego2global_translation": "list (3,) - ego到全局平移"
        }
    }
    return spec

def generate_annotation_spec(ann_info: Dict) -> Dict:
    """生成标注信息规范"""
    spec = {
        "required_fields": {
            "gt_bboxes_3d": "np.ndarray (N,7) float32 - 3D边界框 [x,y,z,l,w,h,yaw]",
            "gt_labels_3d": "np.ndarray (N,) int64 - 类别标签索引",
            "valid_flag": "np.ndarray (N,) bool - 有效标志"  # 关键字段！
        },
        "optional_fields": {
            "gt_velocity": "np.ndarray (N,2) float32 - 速度信息",
            "num_lidar_pts": "np.ndarray (N,) int64 - LiDAR点数",
            "num_radar_pts": "np.ndarray (N,) int64 - 雷达点数"
        }
    }
    return spec

def generate_coordinate_spec(sample: Dict) -> Dict:
    """生成坐标系字段规范"""
    spec = {}
    coord_fields = [
        'lidar2ego_rotation', 'lidar2ego_translation',
        'ego2global_rotation', 'ego2global_translation'
    ]
    
    for field in coord_fields:
        if field in sample:
            value = sample[field]
            if isinstance(value, list):
                spec[field] = f"list - 长度: {len(value)}"
            else:
                spec[field] = f"{type(value).__name__}"
    
    return spec

def generate_data_types_spec(sample: Dict) -> Dict:
    """生成数据类型规范"""
    spec = {
        "numpy_arrays": [
            "cam_intrinsic: (3,3) float32",
            "sensor2lidar_rotation: (3,3) float32", 
            "sensor2lidar_translation: (3,) float32",
            "gt_bboxes_3d: (N,7) float32",
            "gt_labels_3d: (N,) int64",
            "valid_flag: (N,) bool",  # 关键！
            "gt_velocity: (N,2) float32",
            "num_lidar_pts: (N,) int64",
            "num_radar_pts: (N,) int64"
        ],
        "lists": [
            "sensor2ego_rotation: 3x3",
            "sensor2ego_translation: 3 elements", 
            "ego2global_rotation: 3x3",
            "ego2global_translation: 3 elements",
            "lidar2ego_rotation: 3x3", 
            "lidar2ego_translation: 3 elements",
            "ego2global_rotation: 3x3",
            "ego2global_translation: 3 elements"
        ],
        "strings": ["token", "scene_token", "lidar_path", "data_path"],
        "integers": ["timestamp"]
    }
    return spec

def generate_adaptation_notes() -> Dict:
    """生成TYJT适配说明"""
    return {
        "critical_notes": [
            "必须包含 valid_flag 字段在 ann_info 中",
            "所有numpy数组必须使用正确的dtype",
            "相机数量为4个，需要正确映射",
            "标注框格式为 [x,y,z,l,w,h,yaw]",
            "类别索引从0开始，必须连续"
        ],
        "camera_mapping": {
            "SC_1A_CamR": "CAM_FRONT_RIGHT",
            "SC_1B_CamR": "CAM_FRONT",
            "SC_1C_CamR": "CAM_BACK", 
            "SC_1D_CamR": "CAM_FRONT_LEFT"
        },
        "class_names": [
            "car", "truck", "bus", "pedestrian", 
            "bicycle", "motorcycle", "traffic_cone"
        ]
    }

if __name__ == "__main__":
    analyze_bevfusion_pkl_format()