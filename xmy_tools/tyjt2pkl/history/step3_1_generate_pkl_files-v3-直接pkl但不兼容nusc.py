# step3_1_generate_pkl_files_simple.py
import pickle
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def create_pkl_format_guide(output_dir):
    """创建PKL格式说明文档"""
    guide_content = {
        "PKL文件格式说明": {
            "文件用途": "TYJT数据集转换为MIT-BEVFusion兼容格式的PKL文件",
            "目标框架": "MIT-BEVFusion"
        },
        "BEVFusion必需字段": {
            "cams": "相机数据字典，包含所有相机的图像路径和标定参数",
            "lidar_path": "激光雷达点云文件路径", 
            "lidar2ego_rotation/translation": "激光雷达到自车的变换",
            "ego2global_rotation/translation": "自车到全局坐标的变换",
            "timestamp": "时间戳用于数据关联"
        },
        "标注规范": {
            "type": "类别名称，范围: ['car', 'truck', 'construction_truck', 'van', 'bus', 'robot', 'pedestrian', 'cyclist', 'bicycle', 'tricycle', 'tricyclist', 'trolley', 'cone', 'barrier', 'other']",
            "id": "物体唯一标识符，字符串类型",
            "box3d": "3D边界框，格式: [cx, cy, cz, dx, dy, dz]",
            "rotation": "旋转角度，格式: [yaw, roll, pitch]，yaw∈[-π,π)，roll/pitch∈[-0.1,0.1]",
            "pointNum": "点云点数，允许为0（数据补全情况）"
        },
        "BEVFusion边界框格式": {
            "说明": "使用7参数边界框格式，与BEVFusion保持一致",
            "格式": "[x, y, z, dx, dy, dz, yaw]",
            "参数说明": [
                "x: 中心点x坐标",
                "y: 中心点y坐标", 
                "z: 中心点z坐标",
                "dx: 长度(length)",
                "dy: 宽度(width)",
                "dz: 高度(height)",
                "yaw: 偏航角(绕z轴旋转)"
            ],
            "注意": "只使用rotation中的yaw角，忽略roll和pitch"
        }
    }
    
    guide_file = output_dir / "pkl_format_guide.json"
    with open(guide_file, 'w', encoding='utf-8') as f:
        json.dump(guide_content, f, indent=2, ensure_ascii=False)
    
    print(f"PKL格式说明文档已保存: {guide_file}")
    return guide_content

def convert_numpy_to_serializable(obj):
    """将numpy数组和数据类型转换为JSON可序列化的格式"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_serializable(item) for item in obj]
    else:
        return obj

def save_json_with_numpy_conversion(data, file_path):
    """保存JSON文件，自动转换numpy数据类型"""
    serializable_data = convert_numpy_to_serializable(data)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)

def validate_annotation(obj):
    """验证标注对象的属性是否符合规范"""
    warnings = []  # 收集警告信息
    try:
        # 检查必需字段
        required_fields = ['type', 'box3d']
        for field in required_fields:
            if field not in obj:
                return False, f"缺少必需字段: {field}", warnings
        
        # 检查type字段
        valid_types = [
            "car", "truck", "construction_truck", "van", "bus", "robot",
            "pedestrian", "cyclist", "bicycle", "tricycle", "tricyclist", 
            "trolley", "cone", "barrier", "other"
        ]
        if obj['type'] not in valid_types:
            return False, f"无效的type: {obj['type']}", warnings
        
        # 检查box3d字段
        if not isinstance(obj['box3d'], list) or len(obj['box3d']) != 6:
            return False, f"box3d格式错误: {obj['box3d']}", warnings
        
        # 检查box3d尺寸是否大于0
        dx, dy, dz = obj['box3d'][3], obj['box3d'][4], obj['box3d'][5]
        if dx <= 0 or dy <= 0 or dz <= 0:
            return False, f"box3d尺寸无效: [{dx}, {dy}, {dz}]", warnings
        
        # 检查rotation字段（如果存在）
        if 'rotation' in obj:
            if not isinstance(obj['rotation'], list) or len(obj['rotation']) != 3:
                return False, f"rotation格式错误: {obj['rotation']}", warnings
            
            yaw, roll, pitch = obj['rotation']
            # 检查yaw范围 [-π, π)
            if not (-np.pi <= yaw < np.pi):
                return False, f"yaw超出范围: {yaw}", warnings
            
            # 检查roll/pitch范围 [-0.1, 0.1]
            zeta = 0.1
            if not (-zeta <= roll <= zeta) or not (-zeta <= pitch <= zeta):
                return False, f"roll/pitch超出范围: roll={roll}, pitch={pitch}", warnings
        
        # 检查pointNum字段（如果存在）- 改为warning而不是error
        if 'pointNum' in obj:
            if not isinstance(obj['pointNum'], int):
                warnings.append(f"pointNum类型无效: {type(obj['pointNum'])}")
            elif obj['pointNum'] < 0:
                warnings.append(f"pointNum为负数: {obj['pointNum']}")
            elif obj['pointNum'] == 0:
                warnings.append(f"pointNum为0（可能是数据补全）: {obj['pointNum']}")
        
        return True, "验证通过", warnings
        
    except Exception as e:
        return False, f"验证过程中出错: {e}", warnings

def main():
    # 配置路径
    dataset_root = Path("/mnt/dataset/tyjt_RawData")
    calib_dir = Path("./step2_output")
    output_dir = Path("./step3_1_output")
    output_dir.mkdir(exist_ok=True)
    
    print("=== 开始生成PKL文件 ===")
    
    # 定义场景名称
    scene_name1 = "2d3d4d_20250728_weiyuan"
    scene_name2 = "G51102400001M00_20250728191726_2.0HZ"
    dataset_path = dataset_root / scene_name1 / "datasets" / scene_name2
    
    # 首先创建格式说明文档
    guide_content = create_pkl_format_guide(output_dir)
    
    # 相机映射 - 按照ABCD相机顺时针顺序
    camera_mapping = {
        'SC_1A_CamR': 'CAM_FRONT',        # A相机 - 前向
        'SC_1B_CamR': 'CAM_FRONT_LEFT',   # B相机 - 左前（顺时针下一个）
        'SC_1C_CamR': 'CAM_BACK',         # C相机 - 后向  
        'SC_1D_CamR': 'CAM_FRONT_RIGHT'   # D相机 - 右前
    }
    
    calib_key_mapping = {
        'SC_1A_CamR': 'SC_R1_Aw_CpcS_CAMR_new',
        'SC_1B_CamR': 'SC_R1_Bn_CpcW_CAMR_new',
        'SC_1C_CamR': 'SC_R1_Ce_CpcN_CAMR_new',
        'SC_1D_CamR': 'SC_R1_Ds_CpcE_CAMR_new'
    }
    
    # 根据TYJT标注规范定义类别列表
    class_names = [
        "car", "truck", "construction_truck", "van", "bus", "robot",
        "pedestrian", "cyclist", "bicycle", "tricycle", "tricyclist", 
        "trolley", "cone", "barrier", "other"
    ]
    
    # === 步骤1: 加载标定数据 ===
    print("\n=== 步骤1: 加载标定数据 ===")
    try:
        with open(calib_dir / "sensor2group_calib.json", 'r') as f:
            calib_data = json.load(f)
        print("  标定数据加载成功")
    except Exception as e:
        print(f"  标定数据加载失败: {e}")
        return
    
    # === 步骤2: 扫描数据集, 获取list ===
    print("\n=== 步骤2: 扫描数据集, 获取list ===")
    frames = []
    
    if not dataset_path.exists():
        print(f"  数据集路径不存在: {dataset_path}")
        return
    
    # 使用第一个相机的时间戳作为基准
    first_cam = 'SC_1A_CamR'
    image_dir = dataset_path / first_cam / "image_dc"
    
    if not image_dir.exists():
        print(f"  图像目录不存在: {image_dir}")
        return
    
    # --- 步骤2.1: 获取全部图像文件list ==> image_files  --- 
    print("\n  --- 步骤2.1: 获取全部图像文件list ==> image_files ---")
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    print(f"  找到 {len(image_files)} 个图像文件")

    
    # --- 步骤2.2: 获取图像、Lidar、Label路径，筛选有效帧 ==> frames ---
    print("\n  --- 步骤2.2: 获取图像文件并筛选有效帧 ==> frames ---")
    for i, image_file in enumerate(image_files):
        if i % 100 == 0:
            print(f"    扫描进度: {i}/{len(image_files)}")
        
        try:
            timestamp = int(image_file.stem)
        except ValueError:
            print(f"    跳过无效文件名: {image_file.name}")
            continue
        
        frame_data = {
            'timestamp': timestamp,
            'cameras': {},
            'lidar_path': None,
            'annotation_path': None
        }
        
        # 收集所有相机图像
        for tyjt_cam in camera_mapping.keys():
            cam_image_path = dataset_path / tyjt_cam / "image_dc" / image_file.name
            if cam_image_path.exists():
                frame_data['cameras'][tyjt_cam] = cam_image_path
        
        # 激光雷达数据
        lidar_path = dataset_path / "lidar" / "pcd" / f"{timestamp}.pcd"
        if lidar_path.exists():
            frame_data['lidar_path'] = lidar_path
        else:
            continue
        
        # 标注数据
        annotation_path = dataset_path / "lidar" / "label" / f"{timestamp}.json"
        if annotation_path.exists():
            frame_data['annotation_path'] = annotation_path
        
        if frame_data['cameras'] and frame_data['lidar_path']:
            frames.append(frame_data)
    
    print(f"  有效帧数量: {len(frames)}")
    
    # === 步骤3: 转换数据格式 ===
    print("\n=== 步骤3: 转换数据格式 ===")
    bevfusion_infos = []
    
    # 统计信息
    validation_stats = {
        'total_objects': 0,
        'valid_objects': 0,
        'invalid_objects': 0,
        'validation_errors': [],
        'validation_warnings': []
    }
    
    # --- 步骤3.1: 遍历frames，构建BEVFusion格式数据 ---
    print("\n  --- 步骤3.1: 遍历frames构建BEVFusion格式数据 ---")
    for i, frame_data in enumerate(frames):
        if i % 10 == 0:
            print(f"\n    [Running] 处理进度: {i}/{len(frames)}")

        # 3.1.1 初始化bevfusion_info结构 & 填充基础信息
        if i == 0:
            print("""
    --- 步骤3.1.1: 初始化 metadata::基础信息 ---
    bevfusion_info/
    ├── token ✓
    ├── timestamp ✓  
    ├── scene_token ✓
    ├── lidar_path ✓
    ├── sweeps ✓
    ├── cams [待填充]
    ├── lidar2ego_rotation ✓
    ├── lidar2ego_translation ✓
    ├── ego2global_rotation ✓
    └── ego2global_translation ✓
            """)
        
        bevfusion_info = {
            'token': f"tyjt_{frame_data['timestamp']}",
            'timestamp': frame_data['timestamp'],
            'scene_token': f'tyjt_{scene_name1}_{scene_name2}',
            'lidar_path': str(frame_data['lidar_path']),
            'sweeps': [],
            'cams': {},
            'lidar2ego_rotation': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            'lidar2ego_translation': [0.0, 0.0, 0.0],
            'ego2global_rotation': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            'ego2global_translation': [0.0, 0.0, 0.0],
        }
        
        if i == 0:
            print("    ✅ 基础信息初始化完成")

        # 3.1.2 处理相机数据 metadata::相机信息
        if i == 0:
            print("""
    --- 步骤3.1.2: 填充 metadata::相机信息 ---
    bevfusion_info/
    ├── cams/
    │   ├── CAM_FRONT/
    │   │   ├── data_path ✓
    │   │   ├── cam_intrinsic ✓
    │   │   ├── sensor2lidar_rotation ✓
    │   │   ├── sensor2lidar_translation ✓
    │   │   ├── sensor2ego_rotation ✓
    │   │   ├── sensor2ego_translation ✓
    │   │   ├── ego2global_rotation ✓
    │   │   ├── ego2global_translation ✓
    │   │   ├── sample_data_token ✓
    │   │   └── timestamp ✓
    │   ├── CAM_FRONT_LEFT/
    │   │   ├── data_path ✓
    │   │   ├── cam_intrinsic ✓
    │   │   ├── sensor2lidar_rotation ✓
    │   │   ├── sensor2lidar_translation ✓
    │   │   ├── sensor2ego_rotation ✓
    │   │   ├── sensor2ego_translation ✓
    │   │   ├── ego2global_rotation ✓
    │   │   ├── ego2global_translation ✓
    │   │   ├── sample_data_token ✓
    │   │   └── timestamp ✓
    │   ├── CAM_BACK/
    │   │   ├── data_path ✓
    │   │   ├── cam_intrinsic ✓
    │   │   ├── sensor2lidar_rotation ✓
    │   │   ├── sensor2lidar_translation ✓
    │   │   ├── sensor2ego_rotation ✓
    │   │   ├── sensor2ego_translation ✓
    │   │   ├── ego2global_rotation ✓
    │   │   ├── ego2global_translation ✓
    │   │   ├── sample_data_token ✓
    │   │   └── timestamp ✓
    │   └── CAM_FRONT_RIGHT/
    │       ├── data_path ✓
    │       ├── cam_intrinsic ✓
    │       ├── sensor2lidar_rotation ✓
    │       ├── sensor2lidar_translation ✓
    │       ├── sensor2ego_rotation ✓
    │       ├── sensor2ego_translation ✓
    │       ├── ego2global_rotation ✓
    │       ├── ego2global_translation ✓
    │       ├── sample_data_token ✓
    │       └── timestamp ✓
            """)
        
        for tyjt_cam, image_path in frame_data['cameras'].items():
            nusc_cam = camera_mapping[tyjt_cam]
            calib_key = calib_key_mapping[tyjt_cam]
            
            if calib_key not in calib_data['cameras']:
                print(f"    标定数据中找不到相机: {calib_key}")
                continue
                
            cam_calib = calib_data['cameras'][calib_key]
            transform = cam_calib['sensor2group_transform']
        
            # 直接从标定文件获取rotation_matrix
            rotation_matrix = np.array(transform['rotation_matrix'], dtype=np.float64)
            
            # 构建相机数据
            camera_data = {
                'data_path': str(image_path),
                'cam_intrinsic': np.array(cam_calib['intrinsics']['matrix'], dtype=np.float64),
                'sensor2lidar_rotation': rotation_matrix.astype(np.float64),
                'sensor2lidar_translation': np.array(transform['translation'], dtype=np.float64),
                'sensor2ego_rotation': rotation_matrix.tolist(),
                'sensor2ego_translation': transform['translation'],
                'ego2global_rotation': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                'ego2global_translation': [0.0, 0.0, 0.0],
                'sample_data_token': f"tyjt_{tyjt_cam}_{frame_data['timestamp']}",
                'timestamp': frame_data['timestamp'],
            }
            
            bevfusion_info['cams'][nusc_cam] = camera_data
        
        if i == 0:
            print(f"    ✅ 相机信息填充完成 - 共 {len(bevfusion_info['cams'])} 个相机")

        # 3.1.3 处理标注Label数据 metadata::标注信息
        if i == 0:
            print("""
    --- 步骤3.1.3: 构建 metadata::标注信息 ---
    bevfusion_info/
    ├── gt_boxes [待填充]
    └── gt_names [待填充]
            """)
        
        if frame_data['annotation_path'] and frame_data['annotation_path'].exists():
            try:
                with open(frame_data['annotation_path'], 'r') as f:
                    annotation_data = json.load(f)
                
                gt_bboxes = []
                gt_names = []
                
                if isinstance(annotation_data, list):
                    for obj in annotation_data:
                        validation_stats['total_objects'] += 1
                        
                        # 验证标注对象
                        is_valid, error_msg, warnings = validate_annotation(obj)
                        
                        # 记录警告信息
                        for warning in warnings:
                            validation_stats['validation_warnings'].append({
                                'file': str(frame_data['annotation_path']),
                                'object': obj.get('id', 'unknown'),
                                'warning': warning
                            })
                        
                        if is_valid:
                            box3d = obj['box3d']
                            rotation = obj.get('rotation', [0, 0, 0])
                            
                            bbox = [
                                float(box3d[0]), float(box3d[1]), float(box3d[2]),
                                float(box3d[3]), float(box3d[4]), float(box3d[5]),
                                float(rotation[0])
                            ]
                            gt_bboxes.append(bbox)
                            gt_names.append(obj['type'])
                            validation_stats['valid_objects'] += 1
                        else:
                            validation_stats['invalid_objects'] += 1
                            validation_stats['validation_errors'].append({
                                'file': str(frame_data['annotation_path']),
                                'object': obj.get('id', 'unknown'),
                                'error': error_msg
                            })
                
                if gt_bboxes:
                    bevfusion_info['gt_boxes'] = np.array(gt_bboxes, dtype=np.float32)
                    bevfusion_info['gt_names'] = np.array(gt_names, dtype='<U32')
                    if i == 0:
                        print(f"    ✅ 标注信息填充完成 - {len(gt_bboxes)} 个有效标注")
                else:
                    bevfusion_info['gt_boxes'] = np.zeros((0, 7), dtype=np.float32)
                    bevfusion_info['gt_names'] = np.array([], dtype='<U32')
                    if i == 0:
                        print("    ✅ 标注信息填充完成 - 无标注数据")
                    
            except Exception as e:
                print(f"    加载标注失败: {e}")
                bevfusion_info['gt_boxes'] = np.zeros((0, 7), dtype=np.float32)
                bevfusion_info['gt_names'] = np.array([], dtype='<U32')
                if i == 0:
                    print("    ⚠️ 标注信息填充完成 - 加载失败，使用空标注")
        else:
            bevfusion_info['gt_boxes'] = np.zeros((0, 7), dtype=np.float32)
            bevfusion_info['gt_names'] = np.array([], dtype='<U32')
            if i == 0:
                print("    ✅ 标注信息填充完成 - 无标注文件")

        # 完成当前帧处理
        if i == 0:
            print("""
    --- 当前帧处理完成 ---
    bevfusion_info/
    ├── token ✓
    ├── timestamp ✓
    ├── scene_token ✓
    ├── lidar_path ✓
    ├── sweeps ✓
    ├── cams ✓ ({count}个相机)
    ├── lidar2ego_rotation ✓
    ├── lidar2ego_translation ✓
    ├── ego2global_rotation ✓
    ├── ego2global_translation ✓
    ├── gt_boxes ✓ ({bbox_count}个标注)
    └── gt_names ✓ ({name_count}个类别)
            """.format(
                count=len(bevfusion_info['cams']),
                bbox_count=len(bevfusion_info['gt_boxes']),
                name_count=len(bevfusion_info['gt_names'])
            ))

        bevfusion_infos.append(bevfusion_info)
    
    # === 步骤4: 分割数据集 ===
    print("\n=== 步骤4: 分割数据集 ===")
    split_idx = int(len(bevfusion_infos) * 0.8)
    train_infos = bevfusion_infos[:split_idx]
    val_infos = bevfusion_infos[split_idx:]
    print(f"  训练集: {len(train_infos)} 帧")
    print(f"  验证集: {len(val_infos)} 帧")
    
    # === 步骤5: 生成JSON文件 ===
    print("\n=== 步骤5: 生成JSON分析文件 ===")
    
    train_json_path = output_dir / "tyjt_infos_train.json"
    save_json_with_numpy_conversion(train_infos, train_json_path)
    print(f"  训练集JSON: {train_json_path}")
    
    val_json_path = output_dir / "tyjt_infos_val.json"
    save_json_with_numpy_conversion(val_infos, val_json_path)
    print(f"  验证集JSON: {val_json_path}")
    
    # === 步骤6: 生成PKL文件 ===
    print("\n=== 步骤6: 生成PKL文件 ===")
    
    train_pkl_path = output_dir / "tyjt_infos_train.pkl"
    with open(train_pkl_path, 'wb') as f:
        pickle.dump(train_infos, f)
    print(f"  训练集PKL: {train_pkl_path}")
    
    val_pkl_path = output_dir / "tyjt_infos_val.pkl"
    with open(val_pkl_path, 'wb') as f:
        pickle.dump(val_infos, f)
    print(f"  验证集PKL: {val_pkl_path}")
    
    # === 步骤7: 保存统计信息 ===
    print("\n=== 步骤7: 保存统计信息 ===")
    
    stats = {
        'total_frames': len(frames),
        'train_frames': len(train_infos),
        'val_frames': len(val_infos),
        'cameras_used': list(camera_mapping.values()),
        'class_names': class_names,
        'class_count': len(class_names),
        'validation_stats': validation_stats,
    }
    
    with open(output_dir / "conversion_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  统计信息: {output_dir / 'conversion_stats.json'}")
    
    print("\n=== 所有文件生成完成 ===")
    print(f"输出目录: {output_dir}")
    
    # 最终统计信息
    print("\n=== 最终统计信息 ===")
    print(f"总帧数: {len(bevfusion_infos)}")
    print(f"训练集: {len(train_infos)} 帧")
    print(f"验证集: {len(val_infos)} 帧")
    print(f"总标注物体: {validation_stats['total_objects']}")
    print(f"有效标注物体: {validation_stats['valid_objects']}")
    print(f"无效标注物体: {validation_stats['invalid_objects']}")
    print(f"警告数量: {len(validation_stats['validation_warnings'])}")
    print(f"类别数量: {len(class_names)}")

if __name__ == "__main__":
    main()