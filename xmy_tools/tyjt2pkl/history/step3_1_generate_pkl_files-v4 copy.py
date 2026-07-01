# step3_1_generate_pkl_files-v4.py
import pickle
import json
import numpy as np
from pathlib import Path
import mmcv
from os import path as osp
from mmdet3d.datasets import Custom3DDataset
from mmdet3d.datasets.builder import DATASETS

# 注册TYJT数据集
@DATASETS.register_module()
class TYJTDataset(Custom3DDataset):
    """TYJT数据集类"""
    
    CLASSES = (
        "car", "truck", "construction_truck", "van", "bus", "robot",
        "pedestrian", "cyclist", "bicycle", "tricycle", "tricyclist", 
        "trolley", "cone", "barrier", "other"
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_data_info(self, index):
        """获取数据信息"""
        info = self.data_infos[index]
        
        # 构建输入字典
        input_dict = dict(
            sample_idx=info['token'],
            timestamp=info['timestamp'],
            lidar_path=info['lidar_path'],
            sweeps=info['sweeps'],
            cams=dict(),
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
        )
        
        # 处理相机数据
        for cam_name, cam_info in info['cams'].items():
            input_dict['cams'][cam_name] = dict(
                data_path=cam_info['data_path'],
                cam_intrinsic=cam_info['cam_intrinsic'],
                sensor2lidar_translation=cam_info['sensor2lidar_translation'],
                sensor2lidar_rotation=cam_info['sensor2lidar_rotation'],
                sensor2ego_translation=cam_info['sensor2ego_translation'],
                sensor2ego_rotation=cam_info['sensor2ego_rotation'],
                ego2global_translation=cam_info['ego2global_translation'],
                ego2global_rotation=cam_info['ego2global_rotation'],
                timestamp=cam_info['timestamp'],
            )
        
        # 如果有标注，添加标注信息
        if 'gt_boxes' in info and 'gt_names' in info:
            input_dict['ann_info'] = dict(
                gt_bboxes_3d=info['gt_boxes'],
                gt_labels_3d=info['gt_names'],
            )

        return input_dict


def create_pkl_format_guide(output_dir):
    """创建PKL格式说明文档"""
    guide_content = {
        "PKL文件格式说明": {
            "文件用途": "TYJT数据集转换为MIT-BEVFusion兼容格式的PKL文件",
            "目标框架": "MIT-BEVFusion",
            "注意": "此版本移除了BEVFusion不需要的mask相关处理"
        },
        "BEVFusion必需字段": {
            "cams": "相机数据字典，包含所有相机的图像路径和标定参数",
            "lidar_path": "激光雷达点云文件路径", 
            "lidar2ego_rotation/translation": "激光雷达到自车的变换",
            "ego2global_rotation/translation": "自车到全局坐标的变换",
            "timestamp": "时间戳用于数据关联"
        },
        "GT数据库用途": {
            "目的": "为ObjectSample数据增强提供物体实例",
            "内容": "每个物体的点云文件(.bin)和元信息",
            "使用方式": "训练时从数据库采样物体插入当前场景"
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
    warnings = []
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
        
        return True, "验证通过", warnings
        
    except Exception as e:
        return False, f"验证过程中出错: {e}", warnings


def create_tyjt_groundtruth_database(
    data_path,
    info_prefix,
    info_path=None,
    database_save_path=None,
    db_info_save_path=None,
    used_classes=None
):
    """为TYJT数据集创建GT数据库（移除了BEVFusion不需要的mask处理）"""
    print("🔵[xmy]>>> 创建TYJT GT数据库（BEVFusion兼容版）")
    
    from mmdet3d.core.bbox import box_np_ops
    from mmdet3d.datasets import build_dataset
    
    # 构建TYJT数据集配置 - 关键修改：使用正确的标注字段名
    dataset_cfg = dict(
        type='TYJTDataset',
        dataset_root=data_path,
        ann_file=info_path,
        test_mode=False,
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=6,  # TYJT点云维度: x, y, z, intensity, ring, time
                use_dim=[0, 1, 2, 3],  # 只使用前4维: x, y, z, intensity
            ),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,   # 加载3D边界框
                with_label_3d=True,  # 加载3D标签
            ),
        ]
    )
    
    # 构建数据集
    dataset = build_dataset(dataset_cfg)
    
    # 添加调试：检查第一个样本的标注信息
    print("🔵[xmy]>>> 检查数据集标注信息...")
    if len(dataset) > 0:
        test_input = dataset.get_data_info(0)
        dataset.pre_pipeline(test_input)
        test_example = dataset.pipeline(test_input)
        
        print(f"示例数据包含的键: {test_example.keys()}")
        if 'ann_info' in test_example:
            print(f"ann_info包含的键: {test_example['ann_info'].keys()}")
        else:
            print("没有找到 ann_info")
    
    # 设置保存路径
    if database_save_path is None:
        database_save_path = osp.join(data_path, f"{info_prefix}_gt_database")
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path, f"{info_prefix}_dbinfos_train.pkl")
    
    mmcv.mkdir_or_exist(database_save_path)
    all_db_infos = dict()
    group_counter = 0
    
    print("🔵[xmy]>>> 开始处理样本生成GT数据库...")
    
    for j in mmcv.track_iter_progress(range(len(dataset))):
        # 获取处理后的数据
        input_dict = dataset.get_data_info(j)
        dataset.pre_pipeline(input_dict)
        example = dataset.pipeline(input_dict)
        
        # 调试：检查example的结构
        if j == 0:  # 只检查第一个样本
            print(f"第一个样本的键: {example.keys()}")
            if 'ann_info' in example:
                print(f"第一个样本ann_info的键: {example['ann_info'].keys()}")
        
        # 提取标注信息
        if 'ann_info' not in example:
            print(f"跳过样本 {example.get('sample_idx', 'unknown')}: 没有ann_info")
            continue
            
        annos = example["ann_info"]
        image_idx = example["sample_idx"]

        # 安全地提取points
        if hasattr(example["points"], 'tensor'):
            points = example["points"].tensor.numpy()
        else:
            points = example["points"]

        print(f"\n=== 调试: 处理样本 {image_idx} ===")
        print(f"点云形状: {points.shape}")
        print(f"ann_info包含: {annos.keys()}")
        
        # 检查标注是否存在 - 使用正确的字段名
        if "gt_bboxes_3d" not in annos or "gt_labels_3d" not in annos:
            print(f"跳过样本 {image_idx}: 没有找到3D标注信息")
            print(f"可用的标注键: {list(annos.keys())}")
            continue
            
        gt_boxes_3d = annos["gt_bboxes_3d"]
        names = annos["gt_labels_3d"]

        # 确保都是numpy数组
        if hasattr(gt_boxes_3d, 'numpy'):
            gt_boxes_3d = gt_boxes_3d.numpy()
        if hasattr(names, 'numpy'):
            names = names.numpy()

        print(f"3D边界框数量: {len(gt_boxes_3d)}")
        print(f"3D边界框形状: {gt_boxes_3d.shape}")
        
        # 检查标注是否为空
        if len(gt_boxes_3d) == 0:
            print(f"跳过样本 {image_idx}: gt_boxes_3d 为空")
            continue

        # 计算每个点属于哪些3D框
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)
        num_obj = gt_boxes_3d.shape[0]
        
        print(f"point_indices形状: {point_indices.shape}")
        print(f"每个物体的点数: {point_indices.sum(axis=0)}")
        
        # 如果所有物体都没有点，跳过这个样本
        if point_indices.sum() == 0:
            print(f"警告: 样本 {image_idx} 中所有物体都没有点云数据")
            continue

        # 设置默认难度（TYJT可能没有难度标注）
        difficulty = np.zeros(num_obj, dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]
        
        # 设置默认group_ids
        group_ids = np.arange(num_obj, dtype=np.int64)
        group_dict = {}
        
        # 处理每个物体
        obj_count = 0
        for i in range(num_obj):
            # 生成文件名
            filename = f"{image_idx}_{names[i]}_{i}.bin"
            abs_filepath = osp.join(database_save_path, filename)
            rel_filepath = osp.join(f"{info_prefix}_gt_database", filename)
            
            # 提取并中心化点云
            gt_points = points[point_indices[:, i]]
            
            print(f"  物体 {i}({names[i]}): 提取到 {len(gt_points)} 个点")
            
            # 如果点云为空，跳过
            if len(gt_points) == 0:
                print(f"    跳过: 没有点云数据")
                continue
            
            gt_points[:, :3] -= gt_boxes_3d[i, :3]  # 中心化到物体坐标系
            
            # 保存点云文件（BEVFusion只需要这个）
            with open(abs_filepath, "wb") as f:
                gt_points.tofile(f)
            
            print(f"    保存: {filename}")
            obj_count += 1
            
            # 记录数据库信息（过滤指定类别）
            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    "name": names[i],
                    "path": rel_filepath,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes_3d[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                }
                
                # 处理分组信息
                local_group_id = group_ids[i]
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                
                # 按类别组织
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]
        
        print(f"样本 {image_idx} 成功保存 {obj_count} 个物体")
    
    # 保存数据库信息
    with open(db_info_save_path, "wb") as f:
        pickle.dump(all_db_infos, f)
    
    # 打印统计信息
    print("🔵[xmy]>>> GT数据库生成完成")
    for k, v in all_db_infos.items():
        print(f"  {k}: {len(v)} 个实例")
    
    return all_db_infos

def main():
    # 配置路径
    dataset_root = Path("/mnt/dataset/tyjt_RawData")
    calib_dir = Path("./step2_output")
    output_dir = Path("./output/step3_1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔵[xmy]>>> === 开始生成BEVFusion兼容的PKL文件 ===")
    
    # 定义场景名称
    scene_name1 = "2d3d4d_20250728_weiyuan"
    scene_name2 = "G51102400001M00_20250728191726_2.0HZ"
    dataset_path = dataset_root / scene_name1 / "datasets" / scene_name2
    
    # 首先创建格式说明文档
    guide_content = create_pkl_format_guide(output_dir)
    
    # 相机映射
    camera_mapping = {
        'SC_1A_CamR': 'CAM_FRONT',
        'SC_1B_CamR': 'CAM_FRONT_LEFT', 
        'SC_1C_CamR': 'CAM_BACK',
        'SC_1D_CamR': 'CAM_FRONT_RIGHT'
    }
    
    calib_key_mapping = {
        'SC_1A_CamR': 'SC_R1_Aw_CpcS_CAMR_new',
        'SC_1B_CamR': 'SC_R1_Bn_CpcW_CAMR_new',
        'SC_1C_CamR': 'SC_R1_Ce_CpcN_CAMR_new',
        'SC_1D_CamR': 'SC_R1_Ds_CpcE_CAMR_new'
    }
    
    # TYJT类别列表
    class_names = [
        "car", "truck", "construction_truck", "van", "bus", "robot",
        "pedestrian", "cyclist", "bicycle", "tricycle", "tricyclist", 
        "trolley", "cone", "barrier", "other"
    ]
    
    # === 步骤1: 加载标定数据 ===
    print("🔵[xmy]>>> === 步骤1: 加载标定数据 ===")
    try:
        with open(calib_dir / "sensor2group_calib.json", 'r') as f:
            calib_data = json.load(f)
        print("  ✅ 标定数据加载成功")
    except Exception as e:
        print(f"  ❌ 标定数据加载失败: {e}")
        return
    
    # === 步骤2: 扫描数据集, 获取帧列表 ===
    print("🔵[xmy]>>> === 步骤2: 扫描数据集, 获取帧列表 ===")
    frames = []
    
    if not dataset_path.exists():
        print(f"  ❌ 数据集路径不存在: {dataset_path}")
        return
    
    # 使用第一个相机的时间戳作为基准
    first_cam = 'SC_1A_CamR'
    image_dir = dataset_path / first_cam / "image_dc"
    
    if not image_dir.exists():
        print(f"  ❌ 图像目录不存在: {image_dir}")
        return
    
    # 获取图像文件列表
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    print(f"  📁 找到 {len(image_files)} 个图像文件")
    
    # 筛选有效帧
    print("🔵[xmy]>>> 筛选有效帧...")
    for i, image_file in enumerate(image_files):
        if i % 100 == 0:
            print(f"    扫描进度: {i}/{len(image_files)}")
        
        try:
            timestamp = int(image_file.stem)
        except ValueError:
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
        
        # 激光雷达数据 - 优先使用.npy文件
        lidar_npy_path = dataset_path / "lidar" / "pcd" / f"{timestamp}.npy"
        lidar_pcd_path = dataset_path / "lidar" / "pcd" / f"{timestamp}.pcd"

        if lidar_npy_path.exists():
            frame_data['lidar_path'] = lidar_npy_path  # 使用.npy文件
        elif lidar_pcd_path.exists():
            frame_data['lidar_path'] = lidar_pcd_path  # 备用.pcd文件
        else:
            continue
                
        # 标注数据
        annotation_path = dataset_path / "lidar" / "label" / f"{timestamp}.json"
        if annotation_path.exists():
            frame_data['annotation_path'] = annotation_path
        
        if frame_data['cameras'] and frame_data['lidar_path']:
            frames.append(frame_data)
    
    print(f"  ✅ 有效帧数量: {len(frames)}")
    
    # === 步骤3: 转换数据格式 ===
    print("🔵[xmy]>>> === 步骤3: 转换数据格式 ===")
    bevfusion_infos = []
    
    # 统计信息
    validation_stats = {
        'total_objects': 0,
        'valid_objects': 0,
        'invalid_objects': 0,
        'validation_errors': [],
        'validation_warnings': []
    }
    
    # 遍历frames构建BEVFusion格式数据
    print("🔵[xmy]>>> 遍历frames构建BEVFusion格式数据...")
    for i, frame_data in enumerate(frames):
        if i % 10 == 0:
            print(f"   处理进度: {i}/{len(frames)}")
        
        # 3.1 初始化bevfusion_info结构
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
        
        # 3.2 处理相机数据
        for tyjt_cam, image_path in frame_data['cameras'].items():
            nusc_cam = camera_mapping[tyjt_cam]
            calib_key = calib_key_mapping[tyjt_cam]
            
            if calib_key not in calib_data['cameras']:
                continue
                
            cam_calib = calib_data['cameras'][calib_key]
            transform = cam_calib['sensor2group_transform']
            rotation_matrix = np.array(transform['rotation_matrix'], dtype=np.float64)
            
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
        
        # 3.3 处理标注数据
        if frame_data['annotation_path'] and frame_data['annotation_path'].exists():
            try:
                with open(frame_data['annotation_path'], 'r') as f:
                    annotation_data = json.load(f)
                
                gt_bboxes = []
                gt_names = []
                
                if isinstance(annotation_data, list):
                    for obj in annotation_data:
                        validation_stats['total_objects'] += 1
                        
                        is_valid, error_msg, warnings = validate_annotation(obj)
                        
                        for warning in warnings:
                            validation_stats['validation_warnings'].append({
                                'file': str(frame_data['annotation_path']),
                                'object': obj.get('id', 'unknown'),
                                'warning': warning
                            })
                        
                        if is_valid:
                            box3d = obj['box3d']
                            rotation = obj.get('rotation', [0, 0, 0])
                            
                            # 转换为BEVFusion格式: [x, y, z, dx, dy, dz, yaw]
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
                else:
                    bevfusion_info['gt_boxes'] = np.zeros((0, 7), dtype=np.float32)
                    bevfusion_info['gt_names'] = np.array([], dtype='<U32')
                    
            except Exception as e:
                print(f"    加载标注失败: {e}")
                bevfusion_info['gt_boxes'] = np.zeros((0, 7), dtype=np.float32)
                bevfusion_info['gt_names'] = np.array([], dtype='<U32')
        else:
            bevfusion_info['gt_boxes'] = np.zeros((0, 7), dtype=np.float32)
            bevfusion_info['gt_names'] = np.array([], dtype='<U32')
        
        bevfusion_infos.append(bevfusion_info)
    
    # === 步骤4: 分割数据集 ===
    print("🔵[xmy]>>> === 步骤4: 分割数据集 ===")
    split_idx = int(len(bevfusion_infos) * 0.8)
    train_infos = bevfusion_infos[:split_idx]
    val_infos = bevfusion_infos[split_idx:]
    print(f"  训练集: {len(train_infos)} 帧")
    print(f"  验证集: {len(val_infos)} 帧")
    
    # === 步骤5: 生成JSON文件 ===
    print("🔵[xmy]>>> === 步骤5: 生成JSON分析文件 ===")
    
    train_json_path = output_dir / "tyjt_infos_train.json"
    save_json_with_numpy_conversion(train_infos, train_json_path)
    print(f"  ✅ 训练集JSON: {train_json_path}")
    
    val_json_path = output_dir / "tyjt_infos_val.json"
    save_json_with_numpy_conversion(val_infos, val_json_path)
    print(f"  ✅ 验证集JSON: {val_json_path}")
    
    # === 步骤6: 生成PKL文件 ===
    print("🔵[xmy]>>> === 步骤6: 生成PKL文件 ===")
    
    train_pkl_path = output_dir / "tyjt_infos_train.pkl"
    with open(train_pkl_path, 'wb') as f:
        pickle.dump(train_infos, f)
    print(f"  ✅ 训练集PKL: {train_pkl_path}")
    
    val_pkl_path = output_dir / "tyjt_infos_val.pkl"
    with open(val_pkl_path, 'wb') as f:
        pickle.dump(val_infos, f)
    print(f"  ✅ 验证集PKL: {val_pkl_path}")
    
    # === 步骤7: 生成GT数据库 ===
    print("🔵[xmy]>>> === 步骤7: 生成GT数据库 ===")
    
    # 创建GT数据库（移除了mask处理）
    db_infos = create_tyjt_groundtruth_database(
        data_path=str(output_dir),
        info_prefix='tyjt',
        info_path=str(train_pkl_path),
        used_classes=['car', 'truck', 'pedestrian', 'cyclist']  # 常用类别
    )
    
    # === 步骤8: 保存统计信息 ===
    print("🔵[xmy]>>> === 步骤8: 保存统计信息 ===")
    
    stats = {
        'total_frames': len(frames),
        'train_frames': len(train_infos),
        'val_frames': len(val_infos),
        'cameras_used': list(camera_mapping.values()),
        'class_names': class_names,
        'class_count': len(class_names),
        'validation_stats': validation_stats,
        'gt_database_stats': {k: len(v) for k, v in db_infos.items()}
    }
    
    stats_path = output_dir / "conversion_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✅ 统计信息: {stats_path}")
    
    print("🔵[xmy]>>> === 所有文件生成完成 ===")
    print(f"输出目录: {output_dir}")
    
    # 最终统计信息
    print("\n🔵[xmy]>>> === 最终统计信息 ===")
    print(f"总帧数: {len(bevfusion_infos)}")
    print(f"训练集: {len(train_infos)} 帧")
    print(f"验证集: {len(val_infos)} 帧")
    print(f"总标注物体: {validation_stats['total_objects']}")
    print(f"有效标注物体: {validation_stats['valid_objects']}")
    print(f"无效标注物体: {validation_stats['invalid_objects']}")
    print(f"GT数据库实例: {sum(len(v) for v in db_infos.values())}")

if __name__ == "__main__":
    main()