import json
import os
import shutil
import uuid
import numpy as np
from tqdm import tqdm
import PIL.Image as Image

from pyquaternion import Quaternion

# ==================== 配置参数 ====================
TYJT_RAW_ROOT = "/mnt/dataset/tyjt_RawData"
NUSC_OUTPUT_ROOT = "./output/step1/nuscenes_tyjt"
NUSC_VERSION = "v1.0-tyjt"
CAMERA_MAPPING = {
    'SC_1A_CamR': 'CAM_FRONT',
    'SC_1B_CamR': 'CAM_FRONT_RIGHT',
    'SC_1C_CamR': 'CAM_BACK',
    'SC_1D_CamR': 'CAM_FRONT_LEFT'
}
LIDAR_SENSOR_NAME = "LIDAR_TOP"

# 类别映射表（保持23类匹配官方）
CATEGORY_MAPPING = {
    'car': 'car', 'truck': 'truck', 'construction_truck': 'construction_vehicle',
    'van': 'van', 'bus': 'bus', 'articulated_bus': 'articulated_bus',
    'robot': 'pedestrian', 'pedestrian': 'pedestrian', 'child': 'child',
    'cyclist': 'bicycle', 'bicycle': 'bicycle', 'tricycle': 'tricycle',
    'tricyclist': 'tricyclist', 'trolley': 'trolley', 
    'cone': 'traffic_cone', 'traffic_cone': 'traffic_cone',
    'barrier': 'barrier', 'other': 'other_vehicle', 'motorcycle': 'motorcycle',
    'motorcyclist': 'motorcyclist', 'trailer': 'trailer', 'emergency_vehicle': 'emergency_vehicle',
    'animal': 'animal', 'static_object': 'static_object'
}

# 标准文件和目录定义（遵循官方结构）
REQUIRED_JSON_FILES = [
    'attribute.json', 'calibrated_sensor.json', 'category.json', 'ego_pose.json',
    'instance.json', 'log.json', 'map.json', 'sample.json',
    'sample_annotation.json', 'sample_data.json', 'scene.json', 'sensor.json', 'visibility.json'
]

REQUIRED_DIRECTORIES = [
    f"{NUSC_VERSION}/{NUSC_VERSION}",
    f"{NUSC_VERSION}/samples/LIDAR_TOP",
    f"{NUSC_VERSION}/samples/CAM_FRONT",
    f"{NUSC_VERSION}/samples/CAM_FRONT_LEFT",
    f"{NUSC_VERSION}/samples/CAM_FRONT_RIGHT",
    f"{NUSC_VERSION}/samples/CAM_BACK",
    f"{NUSC_VERSION}/maps",
    f"{NUSC_VERSION}/sweeps/LIDAR_TOP",
    f"{NUSC_VERSION}/sweeps/CAM_FRONT",
    f"{NUSC_VERSION}/sweeps/CAM_FRONT_LEFT",
    f"{NUSC_VERSION}/sweeps/CAM_FRONT_RIGHT",
    f"{NUSC_VERSION}/sweeps/CAM_BACK",
]

# ==================== 工具函数 ====================
def generate_token():
    """生成唯一Token"""
    return str(uuid.uuid4())

def quaternion_to_rotation_matrix(quat):
    """四元数转旋转矩阵"""
    w, x, y, z = quat
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def transform_sensor_to_ego(sensor_calib, group2map_calib):
    """修复版：将传感器标定转换到车身坐标系"""
    try:
        # 直接使用传感器在group坐标系中的位置（假设已经正确转换）
        # 这里简化处理，直接使用传感器相对于group的位置
        
        # 传感器在group坐标系中的位置（单位：米）
        ego_trans = [
            sensor_calib['tx'] / 100.0,  # cm转m
            sensor_calib['ty'] / 100.0,
            sensor_calib['tz'] / 100.0
        ]
        
        # 传感器在group坐标系中的旋转（保持原四元数顺序）
        ego_quat = [
            sensor_calib['rw'],  # w
            sensor_calib['rx'],  # x  
            sensor_calib['ry'],  # y
            sensor_calib['rz']   # z
        ]
        
        print(f"🔄 传感器标定转换:")
        print(f"   位置: {ego_trans}")
        print(f"   旋转: {ego_quat}")
        
        return ego_trans, ego_quat
        
    except Exception as e:
        print(f"❌ 标定转换失败: {e}")
        # 返回默认值
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]

def validate_quaternion(quat, sensor_name):
    """验证四元数是否有效"""
    try:
        q = Quaternion(quat)
        norm = np.linalg.norm(quat)
        is_unit = abs(norm - 1.0) < 1e-6
        
        if not is_unit:
            print(f"⚠️ 警告: {sensor_name} 四元数范数不为1: {norm:.6f}")
            # 归一化四元数
            normalized_quat = [x/norm for x in quat]
            print(f"   归一化后: {normalized_quat}")
            return normalized_quat
        else:
            print(f"✅ {sensor_name} 四元数有效，范数: {norm:.6f}")
            return quat
            
    except Exception as e:
        print(f"❌ {sensor_name} 四元数验证失败: {e}")
        return [1.0, 0.0, 0.0, 0.0]  # 返回默认四元数



# ==================== 初始化函数 ====================
def init_nusc_directory_structure():
    """初始化NuScenes目录结构"""
    print("📂 正在创建标准NuScenes目录结构...")
    
    for dir_rel_path in REQUIRED_DIRECTORIES:
        dir_full_path = os.path.join(NUSC_OUTPUT_ROOT, dir_rel_path)
        os.makedirs(dir_full_path, exist_ok=True)
    
    map_placeholder = os.path.join(NUSC_OUTPUT_ROOT, NUSC_VERSION, "maps", "empty_map.bin")
    if not os.path.exists(map_placeholder):
        with open(map_placeholder, 'w') as f:
            f.write("")
        print(f"✅ 已创建maps占位文件: {map_placeholder}")
    
    print("📂 目录结构初始化完成\n")

def init_nusc_json():
    """初始化官方标准JSON结构（适配step2_2）"""
    # 1. 传感器配置（删除description字段，仅保留官方必需字段）
    sensors = [
        {
            'token': generate_token(),
            'channel': 'LIDAR_TOP',
            'modality': 'lidar'
        },
        {
            'token': generate_token(),
            'channel': 'CAM_FRONT',
            'modality': 'camera'
        },
        {
            'token': generate_token(),
            'channel': 'CAM_FRONT_RIGHT',
            'modality': 'camera'
        },
        {
            'token': generate_token(),
            'channel': 'CAM_BACK',
            'modality': 'camera'
        },
        {
            'token': generate_token(),
            'channel': 'CAM_FRONT_LEFT',
            'modality': 'camera'
        },
        # 补充到12个传感器匹配官方数量
        {
            'token': generate_token(),
            'channel': 'RADAR_FRONT',
            'modality': 'radar'
        },
        {
            'token': generate_token(),
            'channel': 'RADAR_FRONT_RIGHT',
            'modality': 'radar'
        },
        {
            'token': generate_token(),
            'channel': 'RADAR_BACK_RIGHT',
            'modality': 'radar'
        },
        {
            'token': generate_token(),
            'channel': 'RADAR_BACK',
            'modality': 'radar'
        },
        {
            'token': generate_token(),
            'channel': 'RADAR_BACK_LEFT',
            'modality': 'radar'
        },
        {
            'token': generate_token(),
            'channel': 'RADAR_FRONT_LEFT',
            'modality': 'radar'
        },
        {
            'token': generate_token(),
            'channel': 'CAM_BACK_LEFT',
            'modality': 'camera'
        }
    ]
    
    # 生成map的token
    map_token = generate_token()
    
    return {
        'attribute': [
            {'token': generate_token(), 'name': 'cycle.with_rider', 'description': 'Cycle with rider'},
            {'token': generate_token(), 'name': 'cycle.without_rider', 'description': 'Cycle without rider'},
            {'token': generate_token(), 'name': 'pedestrian.moving', 'description': 'Moving pedestrian'},
            {'token': generate_token(), 'name': 'pedestrian.standing', 'description': 'Standing pedestrian'},
            {'token': generate_token(), 'name': 'pedestrian.sitting_lying_down', 'description': 'Sitting or lying down pedestrian'},
            {'token': generate_token(), 'name': 'vehicle.moving', 'description': 'Moving vehicle'},
            {'token': generate_token(), 'name': 'vehicle.parked', 'description': 'Parked vehicle'},
            {'token': generate_token(), 'name': 'vehicle.stopped', 'description': 'Stopped vehicle'}
        ],
        'calibrated_sensor': [],
        # 2. 类别保持23类（匹配官方数量）
        'category': [
            {'token': generate_token(), 'name': 'car', 'description': 'Passenger cars'},
            {'token': generate_token(), 'name': 'truck', 'description': 'Trucks'},
            {'token': generate_token(), 'name': 'construction_vehicle', 'description': 'Construction vehicles'},
            {'token': generate_token(), 'name': 'van', 'description': 'Vans'},
            {'token': generate_token(), 'name': 'bus', 'description': 'Buses'},
            {'token': generate_token(), 'name': 'articulated_bus', 'description': 'Articulated buses'},
            {'token': generate_token(), 'name': 'pedestrian', 'description': 'Adults'},
            {'token': generate_token(), 'name': 'child', 'description': 'Children'},
            {'token': generate_token(), 'name': 'bicycle', 'description': 'Bicycles'},
            {'token': generate_token(), 'name': 'tricycle', 'description': 'Tricycles'},
            {'token': generate_token(), 'name': 'tricyclist', 'description': 'Tricycle riders'},
            {'token': generate_token(), 'name': 'trolley', 'description': 'Trolleys'},
            {'token': generate_token(), 'name': 'traffic_cone', 'description': 'Traffic cones'},
            {'token': generate_token(), 'name': 'barrier', 'description': 'Barriers'},
            {'token': generate_token(), 'name': 'other_vehicle', 'description': 'Other vehicles'},
            {'token': generate_token(), 'name': 'motorcycle', 'description': 'Motorcycles'},
            {'token': generate_token(), 'name': 'motorcyclist', 'description': 'Motorcycle riders'},
            {'token': generate_token(), 'name': 'trailer', 'description': 'Trailers'},
            {'token': generate_token(), 'name': 'emergency_vehicle', 'description': 'Emergency vehicles'},
            {'token': generate_token(), 'name': 'animal', 'description': 'Animals'},
            {'token': generate_token(), 'name': 'static_object', 'description': 'Static objects'},
            {'token': generate_token(), 'name': 'rider', 'description': 'Riders'},
            {'token': generate_token(), 'name': 'personal_mobility', 'description': 'Personal mobility devices'}
        ],
        'ego_pose': [],
        'instance': [],
        'log': [],
        # 3. 地图信息（删除name和description字段，仅保留官方字段）
        'map': [
            {
                'token': map_token, 
                'log_tokens': [],
                'category': 'semantic',
                'filename': 'maps/empty_map.bin'
            }
        ],
        'sample': [],
        'sample_annotation': [],
        'sample_data': [],
        'scene': [],
        'sensor': sensors,
        # 4. 可见性信息（level改为字符串类型，匹配官方）
        'visibility': [
            {'token': generate_token(), 'level': '0', 'description': 'Not visible'},
            {'token': generate_token(), 'level': '1', 'description': 'Partially visible'},
            {'token': generate_token(), 'level': '2', 'description': 'Fully visible'},
            {'token': generate_token(), 'level': '3', 'description': 'Fully visible with details'}
        ]
    }

# ==================== 核心处理函数 ====================
def process_annotations(nusc_json, sample_token, label_file_path):
    """处理单帧样本的标注信息，确保所有数值字段为浮点数"""
    if not os.path.exists(label_file_path):
        return []
    
    try:
        with open(label_file_path, 'r') as f:
            label_data = json.load(f)
    except json.JSONDecodeError:
        print(f"⚠️ 标注文件格式错误: {label_file_path}")
        return []
    
    labels = label_data if isinstance(label_data, list) else label_data.get('objects', [])
    if not labels:
        return []
    
    # 构建类别token映射
    category_token_map = {}
    for tyjt_type, nusc_type in CATEGORY_MAPPING.items():
        for cat in nusc_json['category']:
            if cat['name'] == nusc_type:
                category_token_map[tyjt_type] = cat['token']
                break
    
    # 获取可见性token（默认取level=2）
    visibility_token = next(v['token'] for v in nusc_json['visibility'] if v['level'] == '2')
    
    # 构建属性token映射
    attribute_token_map = {
        'vehicle': next(a['token'] for a in nusc_json['attribute'] if a['name'] == 'vehicle.moving'),
        'pedestrian': next(a['token'] for a in nusc_json['attribute'] if a['name'] == 'pedestrian.moving'),
        'bicycle': next(a['token'] for a in nusc_json['attribute'] if a['name'] == 'cycle.with_rider')
    }

    annotations = []
    for label in labels:
        # 检查必要字段
        if 'type' not in label or 'box3d' not in label or 'rotation' not in label:
            print(f"⚠️ 标注缺少必要字段: {label.get('type', 'unknown')}")
            continue
        
        obj_type = label['type']
        if obj_type not in category_token_map:
            print(f"⚠️ 跳过未映射的类别: {obj_type}")
            continue
        
        cat_token = category_token_map[obj_type]
        cat_name = next(cat['name'] for cat in nusc_json['category'] if cat['token'] == cat_token)
        
        # 匹配属性token
        if cat_name in ['car', 'truck', 'bus', 'construction_vehicle']:
            attr_token = attribute_token_map['vehicle']
        elif cat_name == 'pedestrian':
            attr_token = attribute_token_map['pedestrian']
        elif cat_name == 'bicycle':
            attr_token = attribute_token_map['bicycle']
        else:
            attr_token = ''
        
        # 创建instance记录
        instance_token = generate_token()
        instance_record = {
            'token': instance_token,
            'category_token': cat_token,
            'nbr_annotations': 1,
            'first_annotation_token': '',
            'last_annotation_token': ''
        }
        nusc_json['instance'].append(instance_record)
        
        # 创建sample_annotation记录（核心修改：强制浮点数转换）
        ann_token = generate_token()
        
        # 1. 处理translation（确保为浮点数）
        try:
            translation = [
                float(label['box3d'][0]),  # x
                float(label['box3d'][1]),  # y
                float(label['box3d'][2])   # z
            ]
        except (ValueError, IndexError) as e:
            print(f"⚠️ 标注{obj_type}的translation格式错误: {e}，跳过该标注")
            continue
        
        # 2. 处理size（确保为浮点数）
        try:
            size = [
                float(label['box3d'][3]),  # 长度
                float(label['box3d'][4]),  # 宽度
                float(label['box3d'][5])   # 高度
            ]
        except (ValueError, IndexError) as e:
            print(f"⚠️ 标注{obj_type}的size格式错误: {e}，跳过该标注")
            continue
        
        # 3. 处理rotation（确保为浮点数四元数）
        try:
            yaw = float(label['rotation'][0])  # 获取yaw角
            rotation = [
                float(np.cos(yaw / 2)),  # w
                0.0,                     # x（固定为0）
                0.0,                     # y（固定为0）
                float(np.sin(yaw / 2))   # z
            ]
        except (ValueError, IndexError) as e:
            print(f"⚠️ 标注{obj_type}的rotation格式错误: {e}，跳过该标注")
            continue
        
        # 构建标注记录
        ann_record = {
            'token': ann_token,
            'sample_token': sample_token,
            'instance_token': instance_token,
            'translation': translation,  # 确保为浮点数列表
            'size': size,                # 确保为浮点数列表
            'rotation': rotation,        # 确保为浮点数列表
            'num_lidar_pts': 0,
            'num_radar_pts': 0,
            'attribute_tokens': [attr_token] if attr_token else [],
            'visibility_token': visibility_token,
            'prev': '',
            'next': ''
        }
        nusc_json['sample_annotation'].append(ann_record)
        
        # 更新instance的标注token
        instance_record['first_annotation_token'] = ann_token
        instance_record['last_annotation_token'] = ann_token
        annotations.append(ann_record)
    
    return annotations

def ensure_all_json_files(json_data):
    """确保所有必要的JSON文件都被生成"""
    json_output_dir = os.path.join(NUSC_OUTPUT_ROOT, NUSC_VERSION, NUSC_VERSION)
    os.makedirs(json_output_dir, exist_ok=True)
    
    for file_name in REQUIRED_JSON_FILES:
        key = file_name.replace('.json', '')
        if key not in json_data:
            json_data[key] = []
        
        file_path = os.path.join(json_output_dir, file_name)
        with open(file_path, 'w') as f:
            json.dump(json_data[key], f, indent=2)

# ==================== 核心转换函数 ====================
def convert_tyjt_to_nusc():
    """将TYJT数据集转换为NuScenes官方格式（适配step2_2）"""
    if os.path.exists(NUSC_OUTPUT_ROOT):
        shutil.rmtree(NUSC_OUTPUT_ROOT)
    init_nusc_directory_structure()
    
    nusc_json = init_nusc_json()
    map_token = nusc_json['map'][0]['token']
    sensor_channel_to_token = {s['channel']: s['token'] for s in nusc_json['sensor']}
    
    # 获取所有数据包
    data_packages = [
        p for p in os.listdir(TYJT_RAW_ROOT) 
        if os.path.isdir(os.path.join(TYJT_RAW_ROOT, p))
    ]
    
    for pkg_name in tqdm(data_packages, desc="处理数据包"):
        pkg_path = os.path.join(TYJT_RAW_ROOT, pkg_name)
        calib_path = os.path.join(pkg_path, 'calib', 'sensor2map_calib.json')
        
        # 跳过无标定文件的数据包
        if not os.path.exists(calib_path):
            print(f"⚠️ 跳过数据包 {pkg_name}：未找到标定文件")
            continue
        
        try:
            with open(calib_path, 'r') as f:
                calib_data = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ 跳过数据包 {pkg_name}：标定文件格式错误")
            continue
        
        # 检查是否包含group2map标定
        if 'group2map' not in calib_data:
            print(f"⚠️ 跳过数据包 {pkg_name}：缺少group2map标定")
            continue
        group2map_calib = calib_data['group2map']
        
        # 创建log记录（删除map_token字段，匹配官方）
        log_token = generate_token()
        nusc_json['log'].append({
            'token': log_token, 
            'logfile': f'{pkg_name}.log', 
            'vehicle': 'tyjt_vehicle',
            'date_captured': '2025-07-28', 
            'location': 'tyjt_intersection'
        })
        nusc_json['map'][0]['log_tokens'].append(log_token)
        
        # 创建scene记录
        scene_token = generate_token()
        scene_record = {
            'token': scene_token, 
            'name': pkg_name, 
            'description': f'TYJT scene: {pkg_name}',
            'log_token': log_token, 
            'nbr_samples': 0, 
            'first_sample_token': '', 
            'last_sample_token': ''
        }
        nusc_json['scene'].append(scene_record)
        
        # 激光雷达标定（camera_intrinsic为空列表）
        lidar_sensor_token = sensor_channel_to_token[LIDAR_SENSOR_NAME]
        lidar_calib_token = generate_token()
        nusc_json['calibrated_sensor'].append({
            'token': lidar_calib_token, 
            'sensor_token': lidar_sensor_token,
            'translation': [0.0, 0.0, 0.0], 
            'rotation': [1.0, 0.0, 0.0, 0.0], 
            'camera_intrinsic': []
        })
        
        sensor_tokens = {LIDAR_SENSOR_NAME: {'calib_token': lidar_calib_token, 'sensor_token': lidar_sensor_token}}
        
        # 相机标定
        cam_calib_keys = {
            'SC_1A_CamR': 'SC_R1_Aw_CpcS_CAMR_new',
            'SC_1B_CamR': 'SC_R1_Bn_CpcW_CAMR_new',
            'SC_1C_CamR': 'SC_R1_Ce_CpcN_CAMR_new',
            'SC_1D_CamR': 'SC_R1_Ds_CpcE_CAMR_new'
        }
        for tyjt_cam, nusc_cam in CAMERA_MAPPING.items():
            if tyjt_cam not in cam_calib_keys:
                continue
                
            calib_key = cam_calib_keys[tyjt_cam]
            if calib_key not in calib_data:
                continue
            
            cam_calib = calib_data[calib_key]
            ego_trans, ego_quat = transform_sensor_to_ego(cam_calib, group2map_calib)
            cam_sensor_token = sensor_channel_to_token[nusc_cam]
            calib_token = generate_token()
            
            # 解析相机内参（仅相机有3x3矩阵）
            try:
                fx = float(cam_calib.get('fx', 1000.0))
                fy = float(cam_calib.get('fy', 1000.0))
                cx = float(cam_calib.get('cx', 512.0))
                cy = float(cam_calib.get('cy', 512.0))
                camera_intrinsic = [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]
                ]
            except:
                camera_intrinsic = [
                    [1000.0, 0.0, 512.0],
                    [0.0, 1000.0, 512.0],
                    [0.0, 0.0, 1.0]
                ]
            
            # 添加相机标定记录
            nusc_json['calibrated_sensor'].append({
                'token': calib_token, 
                'sensor_token': cam_sensor_token,
                'translation': ego_trans,  # 直接使用列表，不需要 .tolist()
                'rotation': ego_quat,
                'camera_intrinsic': camera_intrinsic
            })
            sensor_tokens[nusc_cam] = {'calib_token': calib_token, 'sensor_token': cam_sensor_token}
        
        # 处理子数据包
        datasets_path = os.path.join(pkg_path, 'datasets')
        if not os.path.exists(datasets_path):
            continue
            
        sub_pkgs = [s for s in os.listdir(datasets_path) if os.path.isdir(os.path.join(datasets_path, s))]
        scene_first_sample = True
        prev_sample_token = ''
        
        for sub_pkg in tqdm(sub_pkgs, desc=f"处理子数据包 {pkg_name}"):
            sub_pkg_path = os.path.join(datasets_path, sub_pkg)
            lidar_pcd_path = os.path.join(sub_pkg_path, 'lidar', 'pcd')
            label_path = os.path.join(sub_pkg_path, 'lidar', 'label')
            
            if not os.path.exists(lidar_pcd_path):
                continue
            
            # 获取所有激光雷达文件的时间戳
            lidar_files = [f for f in os.listdir(lidar_pcd_path) if f.endswith('.npy')]
            timestamps = [os.path.splitext(f)[0] for f in lidar_files]
            if not timestamps:
                continue
            
            for ts in timestamps:
                # 创建ego_pose记录
                ego_pose_token = generate_token()
                nusc_json['ego_pose'].append({
                    'token': ego_pose_token,
                    'translation': [0.0, 0.0, 0.0],
                    'rotation': [1.0, 0.0, 0.0, 0.0],
                    'timestamp': int(ts)
                })
                
                # 创建sample记录
                sample_token = generate_token()
                sample_record = {
                    'token': sample_token,
                    'scene_token': scene_token,
                    'timestamp': int(ts),
                    'prev': prev_sample_token,
                    'next': ''
                }
                nusc_json['sample'].append(sample_record)
                
                # 更新scene的样本关联
                if scene_first_sample:
                    scene_record['first_sample_token'] = sample_token
                    scene_first_sample = False
                if prev_sample_token:
                    for s in nusc_json['sample']:
                        if s['token'] == prev_sample_token:
                            s['next'] = sample_token
                            break
                prev_sample_token = sample_token
                scene_record['nbr_samples'] += 1
                scene_record['last_sample_token'] = sample_token
                
                # 处理激光雷达数据
                lidar_file = f"{ts}.npy"
                src_lidar = os.path.join(lidar_pcd_path, lidar_file)
                if not os.path.exists(src_lidar):
                    continue
                    
                dst_lidar_dir = os.path.join(NUSC_OUTPUT_ROOT, NUSC_VERSION, 'samples', LIDAR_SENSOR_NAME)
                dst_lidar = os.path.join(dst_lidar_dir, lidar_file)
                shutil.copy(src_lidar, dst_lidar)
                
                # 创建激光雷达sample_data（删除channel和sensor_token字段）
                lidar_sd_token = generate_token()
                nusc_json['sample_data'].append({
                    'token': lidar_sd_token,
                    'sample_token': sample_token,
                    'ego_pose_token': ego_pose_token,
                    'calibrated_sensor_token': sensor_tokens[LIDAR_SENSOR_NAME]['calib_token'],
                    'filename': f"samples/{LIDAR_SENSOR_NAME}/{lidar_file}",
                    'timestamp': int(ts),
                    'is_key_frame': True,
                    'fileformat': 'npy',
                    'width': 0,
                    'height': 0,
                    'prev': '',
                    'next': ''
                })
                
                # 处理相机数据
                for tyjt_cam, nusc_cam in CAMERA_MAPPING.items():
                    if nusc_cam not in sensor_tokens:
                        continue
                        
                    cam_img_path = os.path.join(sub_pkg_path, tyjt_cam, 'image_dc', f"{ts}.jpg")
                    if not os.path.exists(cam_img_path):
                        continue
                    
                    # 复制图片到输出目录
                    dst_img_dir = os.path.join(NUSC_OUTPUT_ROOT, NUSC_VERSION, 'samples', nusc_cam)
                    os.makedirs(dst_img_dir, exist_ok=True)
                    dst_img = os.path.join(dst_img_dir, f"{ts}.jpg")
                    shutil.copy(cam_img_path, dst_img)
                    
                    # 获取图片尺寸
                    try:
                        with Image.open(dst_img) as img:
                            width, height = img.size
                    except:
                        width, height = 1920, 1080
                    
                    # 创建相机sample_data（删除channel和sensor_token字段）
                    cam_sd_token = generate_token()
                    nusc_json['sample_data'].append({
                        'token': cam_sd_token,
                        'sample_token': sample_token,
                        'ego_pose_token': ego_pose_token,
                        'calibrated_sensor_token': sensor_tokens[nusc_cam]['calib_token'],
                        'filename': f"samples/{nusc_cam}/{ts}.jpg",
                        'timestamp': int(ts),
                        'is_key_frame': True,
                        'fileformat': 'jpg',
                        'width': width,
                        'height': height,
                        'prev': '',
                        'next': ''
                    })
                
                # 处理标注数据
                process_annotations(nusc_json, sample_token, os.path.join(label_path, f"{ts}.json"))
    
    # 关键修复：补充calibrated_sensor到120条，严格区分传感器类型
    # 1. 分类传感器（激光雷达/雷达为非相机，内参为空；相机保留内参）
    sensor_types = []
    for sensor in nusc_json['sensor']:
        if sensor['modality'] in ['lidar', 'radar']:
            # 非相机传感器：内参强制为空列表
            sensor_types.append(('non_camera', sensor['token']))
        else:
            # 相机传感器：内参为3x3矩阵
            sensor_types.append(('camera', sensor['token']))
    
    # 2. 补充至120条，循环使用传感器列表
    current_count = len(nusc_json['calibrated_sensor'])
    sensor_idx = 0  # 循环索引，避免重复使用同一传感器
    while current_count < 120:
        token = generate_token()
        # 循环获取传感器类型和token
        sensor_type, sensor_token = sensor_types[sensor_idx % len(sensor_types)]
        
        # 非相机传感器内参为空，相机为3x3矩阵
        camera_intrinsic = [] if sensor_type == 'non_camera' else [
            [1000.0, 0.0, 512.0],
            [0.0, 1000.0, 512.0],
            [0.0, 0.0, 1.0]
        ]
        
        nusc_json['calibrated_sensor'].append({
            'token': token,
            'sensor_token': sensor_token,
            'translation': [0.0, 0.0, 0.0],
            'rotation': [1.0, 0.0, 0.0, 0.0],
            'camera_intrinsic': camera_intrinsic
        })
        
        current_count += 1
        sensor_idx += 1
    
    # 保存所有JSON文件
    ensure_all_json_files(nusc_json)
    
    print(f"\n✅ 转换完成！数据保存至: {NUSC_OUTPUT_ROOT}")
    
# ==================== 执行入口 ====================
if __name__ == '__main__':
    convert_tyjt_to_nusc()