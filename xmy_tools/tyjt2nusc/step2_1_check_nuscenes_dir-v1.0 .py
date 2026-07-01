import os
import json
from typing import List, Dict, Any

# ==================== 配置参数 ====================
NUSC_ROOT = "./output/nuscenes_tyjt"  # 数据集根目录
NUSC_VERSION = "v1.0-tyjt"           # 数据集版本
# JSON文件读取路径（与转换脚本保存路径匹配）
JSON_DIR = os.path.join(NUSC_ROOT, NUSC_VERSION, NUSC_VERSION)

# ==================== 标准定义 ====================
# 必需的目录结构（相对根目录）
REQUIRED_DIRS = [
    f"{NUSC_VERSION}/{NUSC_VERSION}",
    f"{NUSC_VERSION}/samples/LIDAR_TOP",
    f"{NUSC_VERSION}/samples/CAM_FRONT",
    f"{NUSC_VERSION}/samples/CAM_FRONT_LEFT",
    f"{NUSC_VERSION}/samples/CAM_FRONT_RIGHT",
    f"{NUSC_VERSION}/samples/CAM_BACK",
    f"{NUSC_VERSION}/sweeps/LIDAR_TOP",
    f"{NUSC_VERSION}/sweeps/CAM_FRONT",
    f"{NUSC_VERSION}/sweeps/CAM_FRONT_LEFT",
    f"{NUSC_VERSION}/sweeps/CAM_FRONT_RIGHT",
    f"{NUSC_VERSION}/sweeps/CAM_BACK",
    f"{NUSC_VERSION}/maps"
]

# 各JSON文件的校验规则
JSON_SCHEMAS = {
    "attribute.json": {
        "type": list,
        "item_type": dict,
        "required_keys": ["token", "name", "description"],
        "value_types": {
            "token": str,
            "name": str,
            "description": str
        }
    },
    "calibrated_sensor.json": {
        "type": list,
        "item_type": dict,
        "required_keys": ["token", "sensor_token", "translation", "rotation", "camera_intrinsic"],
        "value_types": {
            "token": str,
            "sensor_token": str,
            "translation": (list, 3, float),  # (类型, 长度, 元素类型)
            "rotation": (list, 4, float),
            "camera_intrinsic": (list, 3, list)  # 3x3矩阵（内层list元素为float）
        }
    },
    "category.json": {
        "type": list,
        "item_type": dict,
        "required_keys": ["token", "name", "description"],
        "value_types": {
            "token": str,
            "name": str,
            "description": str
        }
    },
    "ego_pose.json": {
        "type": list,
        "item_type": dict,
        "required_keys": ["token", "translation", "rotation", "timestamp"],
        "value_types": {
            "token": str,
            "translation": (list, 3, float),
            "rotation": (list, 4, float),
            "timestamp": int
        }
    },
    "instance.json": {
        "type": list,
        "item_type": dict,
        "required_keys": ["token", "category_token", "nbr_annotations", "first_annotation_token", "last_annotation_token"],
        "value_types": {
            "token": str,
            "category_token": str,
            "nbr_annotations": int,
            "first_annotation_token": str,
            "last_annotation_token": str
        }
    },
    "log.json": {
        "type": list,
        "item_type": dict,
        "required_keys": ["token", "logfile", "vehicle", "date_captured", "location", "map_token"],
        "value_types": {
            "token": str,
            "logfile": str,
            "vehicle": str,
            "date_captured": str,
            "location": str,
            "map_token": str
        }
    },
    "map.json": {
        "type": list,
        "item_type": dict,
        "required_keys": ["token", "name", "description", "filename"],
        "value_types": {
            "token": str,
            "name": str,
            "description": str,
            "filename": str
        }
    },
    "sample.json": {
        "type": list,
        "item_type": dict,
        "required_keys": ["token", "scene_token", "timestamp", "prev", "next"],
        "value_types": {
            "token": str,
            "scene_token": str,
            "timestamp": int,
            "prev": str,
            "next": str
        }
    },
    "sample_annotation.json": {
        "type": list,
        "item_type": dict,
        "required_keys": ["token", "sample_token", "instance_token", "translation", "size", "rotation",
                          "num_lidar_pts", "num_radar_pts", "attribute_tokens", "visibility_token", "prev", "next"],
        "value_types": {
            "token": str,
            "sample_token": str,
            "instance_token": str,
            "translation": (list, 3, float),
            "size": (list, 3, float),
            "rotation": (list, 4, float),
            "num_lidar_pts": int,
            "num_radar_pts": int,
            "attribute_tokens": list,
            "visibility_token": str,
            "prev": str,
            "next": str
        }
    },
    "sample_data.json": {
        "type": list,
        "item_type": dict,
        "required_keys": ["token", "sample_token", "ego_pose_token", "calibrated_sensor_token",
                          "filename", "timestamp", "is_key_frame", "sensor_token", "channel", "prev", "next"],
        "value_types": {
            "token": str,
            "sample_token": str,
            "ego_pose_token": str,
            "calibrated_sensor_token": str,
            "filename": str,
            "timestamp": int,
            "is_key_frame": bool,
            "sensor_token": str,
            "channel": str,
            "prev": str,
            "next": str
        }
    },
    "scene.json": {
        "type": list,
        "item_type": dict,
        "required_keys": ["token", "name", "description", "log_token", "nbr_samples", "first_sample_token", "last_sample_token"],
        "value_types": {
            "token": str,
            "name": str,
            "description": str,
            "log_token": str,
            "nbr_samples": int,
            "first_sample_token": str,
            "last_sample_token": str
        }
    },
    "sensor.json": {
        "type": list,
        "item_type": dict,
        "required_keys": ["token", "channel", "modality", "description"],
        "value_types": {
            "token": str,
            "channel": str,
            "modality": str,
            "description": str
        }
    },
    "visibility.json": {
        "type": list,
        "item_type": dict,
        "required_keys": ["token", "level", "description"],
        "value_types": {
            "token": str,
            "level": int,
            "description": str
        }
    }
}

# ==================== 校验函数 ====================
def check_directory_structure() -> bool:
    """验证目录结构是否完整"""
    print("=== 验证目录结构 ===")
    all_ok = True
    for dir_rel in REQUIRED_DIRS:
        dir_path = os.path.join(NUSC_ROOT, dir_rel)
        if not os.path.exists(dir_path):
            print(f"❌ 缺失目录: {dir_path}")
            all_ok = False
        else:
            print(f"✅ 存在目录: {dir_path}")
    return all_ok

def check_json_file(json_file: str) -> bool:
    """验证单个JSON文件的结构和内容合法性"""
    print(f"\n=== 验证JSON文件: {json_file} ===")
    file_path = os.path.join(JSON_DIR, json_file)
    
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    # 2. 加载并解析JSON
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON格式错误: {str(e)}")
        return False
    
    # 3. 验证顶层数据类型
    schema = JSON_SCHEMAS[json_file]
    if not isinstance(data, schema["type"]):
        print(f"❌ 顶层类型错误: 预期{schema['type'].__name__}，实际{type(data).__name__}")
        return False
    
    # 4. 空列表警告（格式合法但无数据）
    if len(data) == 0:
        print(f"⚠️ 警告: 文件内容为空列表")
        return True
    
    # 5. 验证列表项类型
    item_type = schema["item_type"]
    for i, item in enumerate(data[:5]):  # 抽样检查前5项
        if not isinstance(item, item_type):
            print(f"❌ 第{i}项类型错误: 预期{item_type.__name__}，实际{type(item).__name__}")
            return False
    
    # 6. 验证必填字段
    required_keys = schema["required_keys"]
    for i, item in enumerate(data[:5]):
        missing_keys = [k for k in required_keys if k not in item]
        if missing_keys:
            print(f"❌ 第{i}项缺失字段: {missing_keys}")
            return False
    
    # 7. 验证字段值类型
    value_types = schema["value_types"]
    for i, item in enumerate(data[:5]):
        for key, vtype in value_types.items():
            val = item[key]
            # 处理复杂类型（如：3个float元素的列表）
            if isinstance(vtype, tuple) and vtype[0] == list:
                # 验证是否为列表
                if not isinstance(val, list):
                    print(f"❌ 第{i}项'{key}'类型错误: 预期list，实际{type(val).__name__}")
                    return False
                # 验证列表长度
                if len(val) != vtype[1]:
                    print(f"❌ 第{i}项'{key}'长度错误: 预期{vtype[1]}，实际{len(val)}")
                    return False
                # 验证列表元素类型
                elem_type = vtype[2]
                for elem in val:
                    if not isinstance(elem, elem_type):
                        print(f"❌ 第{i}项'{key}'元素类型错误: 预期{elem_type.__name__}，实际{type(elem).__name__}")
                        return False
            # 处理普通类型
            else:
                if not isinstance(val, vtype):
                    print(f"❌ 第{i}项'{key}'类型错误: 预期{vtype.__name__}，实际{type(val).__name__}")
                    return False
    
    print(f"✅ JSON文件验证通过")
    return True

def check_token_references() -> bool:
    """验证关键Token引用关系的有效性"""
    print("\n=== 验证Token引用关系 ===")
    try:
        # 加载需要验证的核心数据
        def load_json(file_name):
            path = os.path.join(JSON_DIR, file_name)
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        samples = load_json("sample.json")
        scenes = load_json("scene.json")
        sample_datas = load_json("sample_data.json")
        
        # 构建Token集合（用于快速查询）
        scene_tokens = {s["token"] for s in scenes}
        sample_tokens = {s["token"] for s in samples}
        
        # 验证Sample引用的SceneToken
        for sample in samples[:50]:  # 抽样检查前50条
            if sample["scene_token"] not in scene_tokens:
                print(f"❌ Sample[{sample['token']}]引用无效SceneToken: {sample['scene_token']}")
                return False
        
        # 验证SampleData引用的SampleToken
        for sd in sample_datas[:50]:  # 抽样检查前50条
            if sd["sample_token"] not in sample_tokens:
                print(f"❌ SampleData[{sd['token']}]引用无效SampleToken: {sd['sample_token']}")
                return False
        
        print("✅ Token引用关系验证通过")
        return True
    except Exception as e:
        print(f"❌ Token验证失败: {str(e)}")
        return False

# ==================== 主函数 ====================
def main():
    print("===== NuScenes数据集完整性验证工具 =====")
    
    # 步骤1: 验证目录结构
    dir_ok = check_directory_structure()
    
    # 步骤2: 验证所有JSON文件
    json_ok = True
    for json_file in JSON_SCHEMAS.keys():
        if not check_json_file(json_file):
            json_ok = False
    
    # 步骤3: 验证Token引用关系
    token_ok = check_token_references() if (dir_ok and json_ok) else False
    
    # 输出最终结果
    print("\n===== 验证总结 =====")
    if dir_ok and json_ok and token_ok:
        print("🎉 数据集格式验证通过！")
    else:
        print("❌ 数据集存在问题，请根据上述提示修复")

if __name__ == "__main__":
    main()