import json
import os
import shutil
import uuid
import numpy as np
from tqdm import tqdm

# 配置参数
TYJT_RAW_ROOT = "/mnt/dataset/tyjt_RawData"
NUSC_OUTPUT_ROOT = "./output/nuscenes_tyjt"
NUSC_VERSION = "v1.0-tyjt"
CAMERA_MAPPING = {
    'SC_1A_CamR': 'CAM_FRONT',
    'SC_1B_CamR': 'CAM_FRONT_RIGHT',
    'SC_1C_CamR': 'CAM_BACK',
    'SC_1D_CamR': 'CAM_FRONT_LEFT'
}
LIDAR_SENSOR_NAME = "LIDAR_TOP"

# 完整类别映射表
CATEGORY_MAPPING = {
    'car': 'car', 'truck': 'truck', 'construction_truck': 'construction_vehicle',
    'van': 'car', 'bus': 'bus', 'robot': 'pedestrian', 'pedestrian': 'pedestrian',
    'cyclist': 'bicycle', 'bicycle': 'bicycle', 'tricycle': 'bicycle', 'tricyclist': 'bicycle',
    'trolley': 'bicycle', 'cone': 'barrier', 'barrier': 'barrier', 'other': 'barrier',
    'motorcycle': 'bicycle', 'traffic_cone': 'barrier'
}

# 标准文件和目录定义
REQUIRED_JSON_FILES = [
    'attribute.json', 'calibrated_sensor.json', 'category.json', 'ego_pose.json',
    'instance.json', 'log.json', 'map.json', 'sample.json',
    'sample_annotation.json', 'sample_data.json', 'scene.json', 'sensor.json', 'visibility.json'
]

# 必须创建的目录结构
REQUIRED_DIRECTORIES = [
    # JSON文件目录
    f"{NUSC_VERSION}/{NUSC_VERSION}",
    # 关键帧传感器数据目录
    f"{NUSC_VERSION}/samples/LIDAR_TOP",
    f"{NUSC_VERSION}/samples/CAM_FRONT",
    f"{NUSC_VERSION}/samples/CAM_FRONT_LEFT",
    f"{NUSC_VERSION}/samples/CAM_FRONT_RIGHT",
    f"{NUSC_VERSION}/samples/CAM_BACK",
    # 新增maps目录
    f"{NUSC_VERSION}/maps",
    # 非关键帧传感器数据目录（可选但建议创建）
    f"{NUSC_VERSION}/sweeps/LIDAR_TOP",
    f"{NUSC_VERSION}/sweeps/CAM_FRONT",
    f"{NUSC_VERSION}/sweeps/CAM_FRONT_LEFT",
    f"{NUSC_VERSION}/sweeps/CAM_FRONT_RIGHT",
    f"{NUSC_VERSION}/sweeps/CAM_BACK",
]

def generate_token():
    return str(uuid.uuid4())

def quaternion_to_rotation_matrix(quat):
    w, x, y, z = quat
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def transform_sensor_to_ego(sensor_calib, group2map_calib):
    """计算传感器到 ego 坐标系的变换矩阵（保持原逻辑）"""
    # 传感器到地图的变换
    sensor_quat = [sensor_calib['rw'], sensor_calib['rx'], sensor_calib['ry'], sensor_calib['rz']]
    sensor_trans = np.array([sensor_calib['tx'], sensor_calib['ty'], sensor_calib['tz']]) / 100.0  # cm -> m
    
    # 地图到group（ego）的变换
    group_quat = [group2map_calib['rw'], group2map_calib['rx'], group2map_calib['ry'], group2map_calib['rz']]
    group_trans = np.array([group2map_calib['tx'], group2map_calib['ty'], group2map_calib['tz']]) / 100.0
    group_rot_mat = quaternion_to_rotation_matrix(group_quat)
    map2group_rot = group_rot_mat.T  # 逆旋转
    map2group_trans = -map2group_rot @ group_trans
    
    # 传感器到ego的变换 = 传感器到地图 × 地图到ego
    sensor_rot_mat = quaternion_to_rotation_matrix(sensor_quat)
    ego_rot = sensor_rot_mat @ map2group_rot
    ego_trans = sensor_rot_mat @ map2group_trans + sensor_trans
    
    # 旋转矩阵转四元数（使用更精确的转换）
    # 修复：确保四元数为float类型
    tr = ego_rot[0,0] + ego_rot[1,1] + ego_rot[2,2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (ego_rot[2,1] - ego_rot[1,2]) / S
        qy = (ego_rot[0,2] - ego_rot[2,0]) / S
        qz = (ego_rot[1,0] - ego_rot[0,1]) / S
    elif (ego_rot[0,0] > ego_rot[1,1]) and (ego_rot[0,0] > ego_rot[2,2]):
        S = np.sqrt(1.0 + ego_rot[0,0] - ego_rot[1,1] - ego_rot[2,2]) * 2
        qw = (ego_rot[2,1] - ego_rot[1,2]) / S
        qx = 0.25 * S
        qy = (ego_rot[0,1] + ego_rot[1,0]) / S
        qz = (ego_rot[0,2] + ego_rot[2,0]) / S
    elif ego_rot[1,1] > ego_rot[2,2]:
        S = np.sqrt(1.0 + ego_rot[1,1] - ego_rot[0,0] - ego_rot[2,2]) * 2
        qw = (ego_rot[0,2] - ego_rot[2,0]) / S
        qx = (ego_rot[0,1] + ego_rot[1,0]) / S
        qy = 0.25 * S
        qz = (ego_rot[1,2] + ego_rot[2,1]) / S
    else:
        S = np.sqrt(1.0 + ego_rot[2,2] - ego_rot[0,0] - ego_rot[1,1]) * 2
        qw = (ego_rot[1,0] - ego_rot[0,1]) / S
        qx = (ego_rot[0,2] + ego_rot[2,0]) / S
        qy = (ego_rot[1,2] + ego_rot[2,1]) / S
        qz = 0.25 * S
    
    return ego_trans, [qw, qx, qy, qz]  # 确保返回float类型四元数

# ---------------------------
# 新增：目录结构初始化函数
# ---------------------------
def init_nusc_directory_structure():
    """初始化完整的NuScenes目录结构，包含maps文件夹及占位文件"""
    print("📂 正在创建标准NuScenes目录结构...")
    
    # 必须创建的目录结构（包含maps）
    REQUIRED_DIRECTORIES = [
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
        f"{NUSC_VERSION}/maps"  # 新增maps目录
    ]
    
    for dir_rel_path in REQUIRED_DIRECTORIES:
        dir_full_path = os.path.join(NUSC_OUTPUT_ROOT, dir_rel_path)
        os.makedirs(dir_full_path, exist_ok=True)
        
        if os.path.exists(dir_full_path):
            print(f"✅ 已创建目录: {dir_full_path}")
        else:
            print(f"❌ 无法创建目录: {dir_full_path}")
    
    # 创建maps目录下的占位文件
    map_placeholder = os.path.join(NUSC_OUTPUT_ROOT, NUSC_VERSION, "maps", "empty_map.bin")
    if not os.path.exists(map_placeholder):
        with open(map_placeholder, 'w') as f:
            f.write("")  # 空文件作为占位符
        print(f"✅ 已在maps目录创建占位文件: {map_placeholder}")
    else:
        print(f"ℹ️ maps目录占位文件已存在: {map_placeholder}")
    
    print("📂 目录结构初始化完成\n")


def init_nusc_json():
    """初始化包含所有必要字段的NuScenes JSON结构，确保包含sensor字段"""
    # 传感器定义（sensor.json内容）
    sensors = [
        {
            'token': generate_token(),
            'channel': 'LIDAR_TOP',
            'modality': 'lidar',
            'description': 'Top-mounted lidar sensor'
        },
        {
            'token': generate_token(),
            'channel': 'CAM_FRONT',
            'modality': 'camera',
            'description': 'Front-facing camera'
        },
        {
            'token': generate_token(),
            'channel': 'CAM_FRONT_RIGHT',
            'modality': 'camera',
            'description': 'Front-right facing camera'
        },
        {
            'token': generate_token(),
            'channel': 'CAM_BACK',
            'modality': 'camera',
            'description': 'Rear-facing camera'
        },
        {
            'token': generate_token(),
            'channel': 'CAM_FRONT_LEFT',
            'modality': 'camera',
            'description': 'Front-left facing camera'
        }
    ]
    
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
        'category': [
            {'token': generate_token(), 'name': 'car', 'description': 'Passenger cars and vans'},
            {'token': generate_token(), 'name': 'truck', 'description': 'Trucks and heavy vehicles'},
            {'token': generate_token(), 'name': 'bus', 'description': 'Buses and large passenger vehicles'},
            {'token': generate_token(), 'name': 'pedestrian', 'description': 'Pedestrians and human-like robots'},
            {'token': generate_token(), 'name': 'bicycle', 'description': 'Bicycles, tricycles, motorcycles and their riders'},
            {'token': generate_token(), 'name': 'construction_vehicle', 'description': 'Construction and work vehicles'},
            {'token': generate_token(), 'name': 'barrier', 'description': 'Traffic barriers, cones and obstacles'},
            {'token': generate_token(), 'name': 'trailer', 'description': 'Trailers attached to vehicles'}
        ],
        'ego_pose': [],
        'instance': [],
        'log': [],
        'map': [
            {
                'token': generate_token(), 
                'name': 'empty_map', 
                'description': 'TYJT dataset map',
                'filename': 'maps/empty_map.bin'
            }
        ],
        'sample': [],
        'sample_annotation': [],
        'sample_data': [],
        'scene': [],
        'sensor': sensors,  # 明确包含sensor字段
        'visibility': [
            {'token': generate_token(), 'level': 0, 'description': 'Not visible'},
            {'token': generate_token(), 'level': 1, 'description': 'Partially visible'},
            {'token': generate_token(), 'level': 2, 'description': 'Fully visible'}
        ]
    }


def process_annotations(nusc_json, sample_token, label_file_path):
    """处理标注文件并转换为NuScenes格式的sample_annotation"""
    if not os.path.exists(label_file_path):
        return []
    
    try:
        with open(label_file_path, 'r') as f:
            label_data = json.load(f)
    except json.JSONDecodeError:
        print(f"⚠️ 标注文件格式错误: {label_file_path}")
        return []
    
    # 兼容两种标注格式：直接列表或包含在objects字段中
    if isinstance(label_data, list):
        labels = label_data
    else:
        labels = label_data.get('objects', [])
    
    if not labels:
        return []
    
    # 1. 预构建必要的token映射表
    # 类别映射（TYJT类型 -> NuScenes类别token）
    category_token_map = {}
    for tyjt_type, nusc_type in CATEGORY_MAPPING.items():
        for cat in nusc_json['category']:
            if cat['name'] == nusc_type:
                category_token_map[tyjt_type] = cat['token']
                break
    
    # 可见性token映射（默认使用完全可见）
    visibility_token = next(v['token'] for v in nusc_json['visibility'] if v['level'] == 2)
    
    # 属性token映射表（按类别预设属性）
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
        # 处理未映射的类别
        if obj_type not in category_token_map:
            print(f"⚠️ 跳过未映射的类别: {obj_type}")
            continue
        
        # 获取类别token
        cat_token = category_token_map[obj_type]
        
        # 2. 确定目标属性（根据类别设置默认属性）
        # 查找类别名称（用于确定属性）
        cat_name = next(cat['name'] for cat in nusc_json['category'] if cat['token'] == cat_token)
        
        # 根据类别分配默认属性token
        if cat_name in ['car', 'truck', 'bus', 'construction_vehicle']:
            attr_token = attribute_token_map['vehicle']
        elif cat_name == 'pedestrian':
            attr_token = attribute_token_map['pedestrian']
        elif cat_name == 'bicycle':
            attr_token = attribute_token_map['bicycle']
        else:
            attr_token = ''  # 其他类别无属性
        
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
        
        # 创建sample_annotation记录（确保包含visibility_token和attribute_tokens）
        ann_token = generate_token()
        # 转换3D框格式（修复：确保所有数值为float类型）
        translation = [float(x) for x in label['box3d'][:3]]  # x, y, z
        size = [
            float(label['box3d'][3]),  # 长度
            float(label['box3d'][4]),  # 宽度
            float(label['box3d'][5])   # 高度
        ]
        yaw = float(label['rotation'][0])  # 偏航角（修复：转为float）
        # 转换为四元数（修复：确保为float类型）
        rotation = [
            float(np.cos(yaw/2)), 
            0.0, 
            0.0, 
            float(np.sin(yaw/2))
        ]
        
        ann_record = {
            'token': ann_token,
            'sample_token': sample_token,
            'instance_token': instance_token,
            'translation': translation,
            'size': size,
            'rotation': rotation,
            'num_lidar_pts': 0,  # 可根据实际点云计数填充
            'num_radar_pts': 0,
            'attribute_tokens': [attr_token] if attr_token else [],  # 属性token列表
            'visibility_token': visibility_token,  # 可见性token
            'prev': '',
            'next': ''
        }
        nusc_json['sample_annotation'].append(ann_record)
        
        # 更新instance的标注关联
        instance_record['first_annotation_token'] = ann_token
        instance_record['last_annotation_token'] = ann_token
        annotations.append(ann_record)
    
    return annotations


def ensure_all_json_files(json_data):
    """确保所有必要的JSON文件都被生成"""
    json_output_dir = os.path.join(NUSC_OUTPUT_ROOT, NUSC_VERSION, NUSC_VERSION)
    
    for file_name in REQUIRED_JSON_FILES:
        key = file_name.replace('.json', '')
        if key not in json_data:
            json_data[key] = []
        
        file_path = os.path.join(json_output_dir, file_name)
        with open(file_path, 'w') as f:
            json.dump(json_data[key], f, indent=2)
    
    generated_files = [f for f in os.listdir(json_output_dir) if f.endswith('.json')]
    missing = set(REQUIRED_JSON_FILES) - set(generated_files)
    if missing:
        print(f"⚠️ 警告: 仍缺失JSON文件 {missing}")
    else:
        print(f"✅ 已生成所有{len(REQUIRED_JSON_FILES)}个标准JSON文件")

# ---------------------------
# 修改：核心转换函数（添加目录初始化）
# ---------------------------
def convert_tyjt_to_nusc():
    """核心转换函数：将TYJT数据集转换为标准NuScenes格式"""
    # 初始化目录结构
    print("📂 初始化NuScenes目录结构...")
    init_nusc_directory_structure()
    
    # 初始化NuScenes JSON结构
    nusc_json = init_nusc_json()
    map_token = nusc_json['map'][0]['token']  # 获取地图token
    sensor_channel_to_token = {s['channel']: s['token'] for s in nusc_json['sensor']}
    
    # 遍历数据包
    data_packages = [
        p for p in os.listdir(TYJT_RAW_ROOT) 
        if os.path.isdir(os.path.join(TYJT_RAW_ROOT, p))
    ]
    
    for pkg_name in tqdm(data_packages, desc="处理数据包"):
        pkg_path = os.path.join(TYJT_RAW_ROOT, pkg_name)
        calib_path = os.path.join(pkg_path, 'calib', 'sensor2map_calib.json')
        
        if not os.path.exists(calib_path):
            print(f"⚠️ 跳过数据包 {pkg_name}：未找到标定文件")
            continue
        
        try:
            with open(calib_path, 'r') as f:
                calib_data = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ 跳过数据包 {pkg_name}：标定文件格式错误")
            continue
        
        if 'group2map' not in calib_data:
            print(f"⚠️ 跳过数据包 {pkg_name}：缺少group2map标定")
            continue
        group2map_calib = calib_data['group2map']
        
        # 创建log和scene
        log_token = generate_token()
        nusc_json['log'].append({
            'token': log_token, 'logfile': f'{pkg_name}.log', 'vehicle': 'tyjt_vehicle',
            'date_captured': '2025-07-28', 'location': 'tyjt_intersection', 'map_token': map_token
        })
        
        scene_token = generate_token()
        scene_record = {
            'token': scene_token, 'name': pkg_name, 'description': f'TYJT scene: {pkg_name}',
            'log_token': log_token, 'nbr_samples': 0, 'first_sample_token': '', 'last_sample_token': ''
        }
        nusc_json['scene'].append(scene_record)
        
        # 激光雷达标定（修复：设置camera_intrinsic为空列表）
        lidar_sensor_token = sensor_channel_to_token[LIDAR_SENSOR_NAME]
        lidar_calib_token = generate_token()
        nusc_json['calibrated_sensor'].append({
            'token': lidar_calib_token, 'sensor_token': lidar_sensor_token,
            'translation': [0.0, 0.0, 0.0], 'rotation': [1.0, 0.0, 0.0, 0.0], 
            'camera_intrinsic': []  # 符合官方规范：激光雷达为空列表
        })
        
        sensor_tokens = {LIDAR_SENSOR_NAME: {'calib_token': lidar_calib_token, 'sensor_token': lidar_sensor_token}}
        
        # 相机标定（修复：确保camera_intrinsic为3x3矩阵）
        cam_calib_keys = {
            'SC_1A_CamR': 'SC_R1_Aw_CpcS_CAMR_new',
            'SC_1B_CamR': 'SC_R1_Bn_CpcW_CAMR_new',
            'SC_1C_CamR': 'SC_R1_Ce_CpcN_CAMR_new',
            'SC_1D_CamR': 'SC_R1_Ds_CpcE_CAMR_new'
        }
        for tyjt_cam, nusc_cam in CAMERA_MAPPING.items():
            if tyjt_cam not in cam_calib_keys:
                print(f"⚠️ 跳过相机 {tyjt_cam}：无对应标定键")
                continue
                
            calib_key = cam_calib_keys[tyjt_cam]
            if calib_key not in calib_data:
                print(f"⚠️ 跳过相机 {tyjt_cam}：标定文件中缺少 {calib_key}")
                continue
            
            cam_calib = calib_data[calib_key]
            ego_trans, ego_quat = transform_sensor_to_ego(cam_calib, group2map_calib)
            cam_sensor_token = sensor_channel_to_token[nusc_cam]
            calib_token = generate_token()
            
            # 修复：确保camera_intrinsic是3x3矩阵
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
            except (KeyError, TypeError, ValueError):
                print(f"⚠️ 相机 {tyjt_cam} 内参数据无效，使用默认值")
                camera_intrinsic = [
                    [1000.0, 0.0, 512.0],
                    [0.0, 1000.0, 512.0],
                    [0.0, 0.0, 1.0]
                ]
            
            nusc_json['calibrated_sensor'].append({
                'token': calib_token, 'sensor_token': cam_sensor_token,
                'translation': ego_trans.tolist(), 'rotation': ego_quat,
                'camera_intrinsic': camera_intrinsic  # 确保3x3矩阵
            })
            sensor_tokens[nusc_cam] = {'calib_token': calib_token, 'sensor_token': cam_sensor_token}
        
        # 处理子数据包
        datasets_path = os.path.join(pkg_path, 'datasets')
        if not os.path.exists(datasets_path):
            print(f"⚠️ 数据包 {pkg_name} 中未找到datasets目录")
            continue
            
        sub_pkgs = [s for s in os.listdir(datasets_path) if os.path.isdir(os.path.join(datasets_path, s))]
        scene_first_sample = True
        prev_sample_token = ''
        
        for sub_pkg in tqdm(sub_pkgs, desc=f"处理子数据包 {pkg_name}"):
            sub_pkg_path = os.path.join(datasets_path, sub_pkg)
            lidar_pcd_path = os.path.join(sub_pkg_path, 'lidar', 'pcd')
            label_path = os.path.join(sub_pkg_path, 'lidar', 'label')
            
            if not os.path.exists(lidar_pcd_path):
                print(f"⚠️ 跳过子数据包 {sub_pkg}：未找到激光雷达目录")
                continue
            
            lidar_files = [f for f in os.listdir(lidar_pcd_path) if f.endswith('.npy')]
            timestamps = [os.path.splitext(f)[0] for f in lidar_files]
            if not timestamps:
                print(f"⚠️ 子数据包 {sub_pkg} 中未找到激光雷达文件")
                continue
            
            for ts in timestamps:
                # 创建ego_pose
                ego_pose_token = generate_token()
                nusc_json['ego_pose'].append({
                    'token': ego_pose_token,
                    'translation': [0.0, 0.0, 0.0],
                    'rotation': [1.0, 0.0, 0.0, 0.0],
                    'timestamp': int(ts)
                })
                
                # 创建sample
                sample_token = generate_token()
                sample_record = {
                    'token': sample_token,
                    'scene_token': scene_token,
                    'timestamp': int(ts),
                    'prev': prev_sample_token,
                    'next': ''
                }
                nusc_json['sample'].append(sample_record)
                
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
                    print(f"⚠️ 跳过激光雷达文件 {src_lidar}")
                    continue
                    
                dst_lidar_dir = os.path.join(NUSC_OUTPUT_ROOT, NUSC_VERSION, 'samples', LIDAR_SENSOR_NAME)
                dst_lidar = os.path.join(dst_lidar_dir, lidar_file)
                shutil.copy(src_lidar, dst_lidar)
                
                # 创建激光雷达sample_data
                lidar_sd_token = generate_token()
                nusc_json['sample_data'].append({
                    'token': lidar_sd_token,
                    'sample_token': sample_token,
                    'ego_pose_token': ego_pose_token,
                    'calibrated_sensor_token': sensor_tokens[LIDAR_SENSOR_NAME]['calib_token'],
                    'filename': f"samples/{LIDAR_SENSOR_NAME}/{lidar_file}",
                    'timestamp': int(ts),
                    'is_key_frame': True,
                    'sensor_token': sensor_tokens[LIDAR_SENSOR_NAME]['sensor_token'],
                    'channel': LIDAR_SENSOR_NAME,
                    'prev': '',
                    'next': ''
                })
                
                # 处理相机数据（JPG格式）
                for tyjt_cam, nusc_cam in CAMERA_MAPPING.items():
                    if nusc_cam not in sensor_tokens:
                        continue
                        
                    cam_img_path = os.path.join(sub_pkg_path, tyjt_cam, 'image_dc', f"{ts}.jpg")
                    if not os.path.exists(cam_img_path):
                        print(f"⚠️ 跳过相机图像 {cam_img_path}")
                        continue
                    
                    dst_img_dir = os.path.join(NUSC_OUTPUT_ROOT, NUSC_VERSION, 'samples', nusc_cam)
                    dst_img = os.path.join(dst_img_dir, f"{ts}.jpg")
                    shutil.copy(cam_img_path, dst_img)
                    
                    # 创建相机sample_data
                    cam_sd_token = generate_token()
                    nusc_json['sample_data'].append({
                        'token': cam_sd_token,
                        'sample_token': sample_token,
                        'ego_pose_token': ego_pose_token,
                        'calibrated_sensor_token': sensor_tokens[nusc_cam]['calib_token'],
                        'filename': f"samples/{nusc_cam}/{ts}.jpg",
                        'timestamp': int(ts),
                        'is_key_frame': True,
                        'sensor_token': sensor_tokens[nusc_cam]['sensor_token'],
                        'channel': nusc_cam,
                        'prev': '',
                        'next': ''
                    })
                
                # 处理标注数据
                process_annotations(nusc_json, sample_token, os.path.join(label_path, f"{ts}.json"))
    
    # 保存所有JSON文件
    ensure_all_json_files(nusc_json)
    
    print(f"\n✅ 转换完成！数据保存至: {NUSC_OUTPUT_ROOT}")


def generate_calibrated_sensor(sensor_tokens, sensor_modalities):
    """生成calibrated_sensor数据，按传感器类型设置camera_intrinsic"""
    calibrated_sensors = []
    for idx, (sensor_token, modality) in enumerate(zip(sensor_tokens, sensor_modalities)):
        calib_token = str(uuid.uuid4())
        # 基础校准参数（实际项目中需替换为真实校准数据）
        translation = [0.0, 0.0, 0.0]
        rotation = [1.0, 0.0, 0.0, 0.0]
        
        # 按传感器类型设置camera_intrinsic
        if modality == "camera":
            # 相机：3x3内参矩阵（示例值，需替换为真实内参）
            camera_intrinsic = [
                [1200.0, 0.0, 960.0],
                [0.0, 1200.0, 540.0],
                [0.0, 0.0, 1.0]
            ]
        else:
            # 激光雷达/雷达：空列表
            camera_intrinsic = []
        
        calibrated_sensor = {
            "token": calib_token,
            "sensor_token": sensor_token,
            "translation": translation,
            "rotation": rotation,
            "camera_intrinsic": camera_intrinsic
        }
        calibrated_sensors.append(calibrated_sensor)
    return calibrated_sensors

if __name__ == '__main__':
    convert_tyjt_to_nusc()
    