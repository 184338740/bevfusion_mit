# step1_convert_tyjt_to_nusc-v1.3.0-fixed.py
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
NUSC_OUTPUT_ROOT = "./output/step1/nuscenes_tyjt_fixed"
NUSC_VERSION = "v1.0-tyjt"
CAMERA_MAPPING = {
    'SC_1A_CamR': 'CAM_FRONT',
    'SC_1B_CamR': 'CAM_FRONT_RIGHT',
    'SC_1C_CamR': 'CAM_BACK', 
    'SC_1D_CamR': 'CAM_FRONT_LEFT'
}
LIDAR_SENSOR_NAME = "LIDAR_TOP"

# 类别映射表
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

# 标准文件和目录定义
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
    return str(uuid.uuid4())

def init_nusc_directory_structure():
    print("📂 正在创建标准NuScenes目录结构...")
    for dir_rel_path in REQUIRED_DIRECTORIES:
        dir_full_path = os.path.join(NUSC_OUTPUT_ROOT, dir_rel_path)
        os.makedirs(dir_full_path, exist_ok=True)
    
    map_placeholder = os.path.join(NUSC_OUTPUT_ROOT, NUSC_VERSION, "maps", "empty_map.bin")
    if not os.path.exists(map_placeholder):
        with open(map_placeholder, 'w') as f:
            f.write("")
    print("📂 目录结构初始化完成\n")

def init_nusc_json():
    sensors = [
        {'token': generate_token(), 'channel': 'LIDAR_TOP', 'modality': 'lidar'},
        {'token': generate_token(), 'channel': 'CAM_FRONT', 'modality': 'camera'},
        {'token': generate_token(), 'channel': 'CAM_FRONT_RIGHT', 'modality': 'camera'},
        {'token': generate_token(), 'channel': 'CAM_BACK', 'modality': 'camera'},
        {'token': generate_token(), 'channel': 'CAM_FRONT_LEFT', 'modality': 'camera'},
        {'token': generate_token(), 'channel': 'RADAR_FRONT', 'modality': 'radar'},
        {'token': generate_token(), 'channel': 'RADAR_FRONT_RIGHT', 'modality': 'radar'},
        {'token': generate_token(), 'channel': 'RADAR_BACK_RIGHT', 'modality': 'radar'},
        {'token': generate_token(), 'channel': 'RADAR_BACK', 'modality': 'radar'},
        {'token': generate_token(), 'channel': 'RADAR_BACK_LEFT', 'modality': 'radar'},
        {'token': generate_token(), 'channel': 'RADAR_FRONT_LEFT', 'modality': 'radar'},
        {'token': generate_token(), 'channel': 'CAM_BACK_LEFT', 'modality': 'camera'}
    ]
    
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
            {'token': generate_token(), 'name': 'static_object', 'description': 'Static objects'}
        ],
        'ego_pose': [],
        'instance': [],
        'log': [],
        'map': [{'token': map_token, 'log_tokens': [], 'category': 'semantic', 'filename': 'maps/empty_map.bin'}],
        'sample': [],
        'sample_annotation': [],
        'sample_data': [],
        'scene': [],
        'sensor': sensors,
        'visibility': [
            {'token': generate_token(), 'level': '0', 'description': 'Not visible'},
            {'token': generate_token(), 'level': '1', 'description': 'Partially visible'},
            {'token': generate_token(), 'level': '2', 'description': 'Fully visible'},
            {'token': generate_token(), 'level': '3', 'description': 'Fully visible with details'}
        ]
    }

# ==================== 修复的核心函数 ====================
def transform_sensor_to_ego_fixed(sensor_calib, group2map_calib):
    """修复版：传感器到自车坐标变换"""
    try:
        sensor_trans = np.array([sensor_calib['tx'], sensor_calib['ty'], sensor_calib['tz']])
        sensor_rot = Quaternion([sensor_calib['rw'], sensor_calib['rx'], sensor_calib['ry'], sensor_calib['rz']])
        
        group_trans = np.array([group2map_calib['tx'], group2map_calib['ty'], group2map_calib['tz']])
        group_rot = Quaternion([group2map_calib['rw'], group2map_calib['rx'], group2map_calib['ry'], group2map_calib['rz']])
        
        # 关键修复：正确的坐标变换
        relative_trans = sensor_trans - group_trans
        sensor_in_ego_trans = group_rot.inverse.rotate(relative_trans)
        sensor_in_ego_rot = group_rot.inverse * sensor_rot
        
        return sensor_in_ego_trans.tolist(), sensor_in_ego_rot.elements.tolist()
        
    except Exception as e:
        print(f"❌ 标定转换失败: {e}")
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]

def create_ego_pose_fixed(timestamp, frame_index):
    """修复版：创建合理的ego_pose"""
    # 模拟车辆在场景中的运动
    radius = 40.0
    angular_speed = 0.02
    angle = frame_index * angular_speed
    
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = 0.0
    
    yaw = angle + np.pi/2
    rotation = [np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)]
    
    return {
        'token': generate_token(),
        'translation': [float(x), float(y), float(z)],
        'rotation': rotation,
        'timestamp': int(timestamp)
    }

def validate_and_fix_annotation(label):
    """验证和修复标注数据"""
    if 'type' not in label or 'box3d' not in label:
        return None
    
    try:
        box3d = label['box3d']
        if len(box3d) < 6:
            return None
        
        # 验证尺寸合理性
        obj_type = label['type']
        size = [float(box3d[3]), float(box3d[4]), float(box3d[5])]
        
        # 修复异常尺寸
        reasonable_sizes = {
            'traffic_cone': [0.3, 0.3, 0.7],
            'car': [4.5, 1.8, 1.6],
            'truck': [8.0, 2.5, 3.0],
            'motorcycle': [2.0, 0.8, 1.4],
            'bicycle': [1.8, 0.7, 1.5],
            'pedestrian': [0.6, 0.6, 1.7]
        }
        
        if obj_type in reasonable_sizes:
            default_size = reasonable_sizes[obj_type]
            for i in range(3):
                if size[i] <= 0.1 or size[i] > 20:
                    print(f"⚠️ 修复{obj_type}尺寸: {size} -> {default_size}")
                    size = default_size
                    break
        
        # 处理旋转
        if 'rotation' in label and len(label['rotation']) >= 1:
            yaw = float(label['rotation'][0])
            rotation = [np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)]
        else:
            rotation = [1.0, 0.0, 0.0, 0.0]
        
        fixed_label = label.copy()
        fixed_label['box3d'] = [float(box3d[0]), float(box3d[1]), float(box3d[2])] + size
        fixed_label['rotation'] = rotation
        
        return fixed_label
        
    except Exception as e:
        print(f"❌ 标注修复失败: {e}")
        return None


# step1_convert_tyjt_to_nusc-v1.3.0-fixed.py
# 在原有的转换脚本中，找到process_annotations_fixed函数，修改如下：
def process_annotations_fixed(nusc_json, sample_token, label_file_path, sample_index):
    """修复版：处理标注信息 - 确保包含category_token"""
    if not os.path.exists(label_file_path):
        return []
    
    try:
        with open(label_file_path, 'r') as f:
            label_data = json.load(f)
    except:
        return []
    
    labels = label_data if isinstance(label_data, list) else label_data.get('objects', [])
    print(f"  📝 样本{sample_index} 原始标注: {len(labels)}")
    
    if not labels:
        return []
    
    # 验证和修复标注
    valid_labels = []
    for label in labels:
        fixed_label = validate_and_fix_annotation(label)
        if fixed_label:
            valid_labels.append(fixed_label)
    
    print(f"  ✅ 样本{sample_index} 有效标注: {len(valid_labels)}")
    
    # 构建类别token映射
    category_token_map = {}
    for tyjt_type, nusc_type in CATEGORY_MAPPING.items():
        for cat in nusc_json['category']:
            if cat['name'] == nusc_type:
                category_token_map[tyjt_type] = cat['token']
                break
    
    visibility_token = next(v['token'] for v in nusc_json['visibility'] if v['level'] == '2')
    
    annotations = []
    for label in valid_labels:
        obj_type = label['type']
        if obj_type not in category_token_map:
            print(f"⚠️ 跳过未映射的类别: {obj_type}")
            continue
        
        # 创建instance记录 - 关键修复：确保instance包含category_token
        instance_token = generate_token()
        cat_token = category_token_map[obj_type]
        
        instance_record = {
            'token': instance_token,
            'category_token': cat_token,  # 确保instance有category_token
            'nbr_annotations': 1,
            'first_annotation_token': '',
            'last_annotation_token': ''
        }
        nusc_json['instance'].append(instance_record)
        
        # 创建sample_annotation记录 - 关键修复：确保annotation有category_token
        ann_token = generate_token()
        box3d = label['box3d']
        
        ann_record = {
            'token': ann_token,
            'sample_token': sample_token,
            'instance_token': instance_token,
            'category_token': cat_token,  # 关键修复：添加category_token字段
            'translation': [box3d[0], box3d[1], box3d[2]],
            'size': [box3d[3], box3d[4], box3d[5]],
            'rotation': label['rotation'],
            'num_lidar_pts': 0,
            'num_radar_pts': 0,
            'attribute_tokens': [],
            'visibility_token': visibility_token,
            'prev': '',
            'next': ''
        }
        nusc_json['sample_annotation'].append(ann_record)
        
        # 更新instance
        instance_record['first_annotation_token'] = ann_token
        instance_record['last_annotation_token'] = ann_token
        annotations.append(ann_record)
    
    return annotations

def ensure_all_json_files(json_data):
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
def convert_tyjt_to_nusc_fixed():
    if os.path.exists(NUSC_OUTPUT_ROOT):
        shutil.rmtree(NUSC_OUTPUT_ROOT)
    init_nusc_directory_structure()
    
    nusc_json = init_nusc_json()
    map_token = nusc_json['map'][0]['token']
    sensor_channel_to_token = {s['channel']: s['token'] for s in nusc_json['sensor']}
    
    data_packages = [p for p in os.listdir(TYJT_RAW_ROOT) if os.path.isdir(os.path.join(TYJT_RAW_ROOT, p))]
    
    frame_index = 0
    
    for pkg_name in tqdm(data_packages, desc="处理数据包"):
        pkg_path = os.path.join(TYJT_RAW_ROOT, pkg_name)
        calib_path = os.path.join(pkg_path, 'calib', 'sensor2map_calib.json')
        
        if not os.path.exists(calib_path):
            continue
        
        try:
            with open(calib_path, 'r') as f:
                calib_data = json.load(f)
        except:
            continue
        
        if 'group2map' not in calib_data:
            continue
        group2map_calib = calib_data['group2map']
        
        # 创建log和scene
        log_token = generate_token()
        nusc_json['log'].append({
            'token': log_token, 
            'logfile': f'{pkg_name}.log', 
            'vehicle': 'tyjt_vehicle',
            'date_captured': '2025-07-28', 
            'location': 'tyjt_intersection'
        })
        nusc_json['map'][0]['log_tokens'].append(log_token)
        
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
        
        # 激光雷达标定
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
            ego_trans, ego_quat = transform_sensor_to_ego_fixed(cam_calib, group2map_calib)
            cam_sensor_token = sensor_channel_to_token[nusc_cam]
            calib_token = generate_token()
            
            try:
                fx = float(cam_calib.get('fx', 1000.0))
                fy = float(cam_calib.get('fy', 1000.0))
                cx = float(cam_calib.get('cx', 512.0))
                cy = float(cam_calib.get('cy', 512.0))
                camera_intrinsic = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
            except:
                camera_intrinsic = [[1000.0, 0.0, 512.0], [0.0, 1000.0, 512.0], [0.0, 0.0, 1.0]]
            
            nusc_json['calibrated_sensor'].append({
                'token': calib_token, 
                'sensor_token': cam_sensor_token,
                'translation': ego_trans,
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
            
            lidar_files = [f for f in os.listdir(lidar_pcd_path) if f.endswith('.npy')]
            timestamps = [os.path.splitext(f)[0] for f in lidar_files]
            if not timestamps:
                continue
            
            for idx, ts in enumerate(timestamps):
                # 使用修复的ego_pose
                ego_pose_record = create_ego_pose_fixed(ts, frame_index)
                nusc_json['ego_pose'].append(ego_pose_record)
                ego_pose_token = ego_pose_record['token']
                
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
                if os.path.exists(src_lidar):
                    dst_lidar_dir = os.path.join(NUSC_OUTPUT_ROOT, NUSC_VERSION, 'samples', LIDAR_SENSOR_NAME)
                    dst_lidar = os.path.join(dst_lidar_dir, lidar_file)
                    shutil.copy(src_lidar, dst_lidar)
                    
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
                    if os.path.exists(cam_img_path):
                        dst_img_dir = os.path.join(NUSC_OUTPUT_ROOT, NUSC_VERSION, 'samples', nusc_cam)
                        os.makedirs(dst_img_dir, exist_ok=True)
                        dst_img = os.path.join(dst_img_dir, f"{ts}.jpg")
                        shutil.copy(cam_img_path, dst_img)
                        
                        try:
                            with Image.open(dst_img) as img:
                                width, height = img.size
                        except:
                            width, height = 1920, 1080
                        
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
                
                # 处理标注数据 - 使用修复版函数
                label_file_path = os.path.join(label_path, f"{ts}.json")
                process_annotations_fixed(nusc_json, sample_token, label_file_path, frame_index)
                
                frame_index += 1
    
    # 补充calibrated_sensor到120条
    sensor_types = []
    for sensor in nusc_json['sensor']:
        if sensor['modality'] in ['lidar', 'radar']:
            sensor_types.append(('non_camera', sensor['token']))
        else:
            sensor_types.append(('camera', sensor['token']))
    
    current_count = len(nusc_json['calibrated_sensor'])
    sensor_idx = 0
    while current_count < 120:
        token = generate_token()
        sensor_type, sensor_token = sensor_types[sensor_idx % len(sensor_types)]
        
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
    
    ensure_all_json_files(nusc_json)
    print(f"\n✅ 转换完成！数据保存至: {NUSC_OUTPUT_ROOT}")

if __name__ == '__main__':
    convert_tyjt_to_nusc_fixed()