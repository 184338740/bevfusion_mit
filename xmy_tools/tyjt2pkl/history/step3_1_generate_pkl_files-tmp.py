# step3_1_generate_pkl_files.py
import json
import pickle
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

class TyjtToPklConverter:
    def __init__(self, dataset_root: str, calib_dir: str = "./step2_output", output_dir: str = "./step3_1_output"):
        self.dataset_root = Path(dataset_root)
        self.calib_dir = Path(calib_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 相机名称映射
        self.camera_mapping = {
            'SC_1A_CamR': 'CAM_FRONT_RIGHT',
            'SC_1B_CamR': 'CAM_FRONT', 
            'SC_1C_CamR': 'CAM_BACK',
            'SC_1D_CamR': 'CAM_FRONT_LEFT'
        }
        
        # 标定键映射
        self.calib_key_mapping = {
            'SC_1A_CamR': 'SC_R1_Aw_CpcS_CAMR_new',
            'SC_1B_CamR': 'SC_R1_Bn_CpcW_CAMR_new',
            'SC_1C_CamR': 'SC_R1_Ce_CpcN_CAMR_new',
            'SC_1D_CamR': 'SC_R1_Ds_CpcE_CAMR_new'
        }
        
        # 类别名称
        self.class_names = ['car', 'truck', 'bus', 'pedestrian', 'bicycle', 'motorcycle', 'traffic_cone']
        
        # 加载标定数据
        self.calib_data = self.load_calibration_data()
    
    def load_calibration_data(self):
        """加载标定数据"""
        calib_file = self.calib_dir / "sensor2group_calib.json"
        with open(calib_file, 'r') as f:
            return json.load(f)
    
    def scan_tyjt_dataset(self):
        """扫描TYJT数据集"""
        print("扫描TYJT数据集...")
        
        frames = []
        subscene = "2d3d4d_20250728_weiyuan"
        subdataset = "G51102400001M00_20250728191726_2.0HZ"
        dataset_path = self.dataset_root / subscene / "datasets" / subdataset
        
        # 使用第一个相机的时间戳作为基准
        first_cam = 'SC_1A_CamR'
        image_dir = dataset_path / first_cam / "image_dc"
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        for image_file in image_files:
            timestamp = int(Path(image_file).stem)
            
            frame_data = {
                'timestamp': timestamp,
                'cameras': {},
                'lidar_path': dataset_path / "lidar" / "pcd" / f"{timestamp}.pcd",
                'annotation_path': dataset_path / "lidar" / "label" / f"{timestamp}.json"
            }
            
            # 收集所有相机
            for tyjt_cam in self.camera_mapping.keys():
                cam_path = dataset_path / tyjt_cam / "image_dc" / image_file.name
                if cam_path.exists():
                    frame_data['cameras'][tyjt_cam] = cam_path
            
            if frame_data['cameras'] and frame_data['lidar_path'].exists():
                frames.append(frame_data)
        
        print(f"有效帧数量: {len(frames)}")
        return frames
    
    def build_camera_data(self, tyjt_cam, image_path, timestamp):
        """构建相机数据 - 匹配NuScenes格式"""
        nusc_cam = self.camera_mapping[tyjt_cam]
        calib_key = self.calib_key_mapping[tyjt_cam]
        
        cam_calib = self.calib_data['cameras'][calib_key]
        transform = cam_calib['sensor2group_transform']
        
        # 计算旋转矩阵
        rotation_quaternion = transform['rotation_quaternion']
        rotation_matrix = R.from_quat(rotation_quaternion).as_matrix()
        
        # 关键：使用NuScenes原始格式
        return nusc_cam, {
            'data_path': str(image_path),
            'type': 'camera',
            'sample_data_token': f"tyjt_{tyjt_cam}_{timestamp}",
            'sensor2ego_translation': transform['translation'],
            'sensor2ego_rotation': rotation_quaternion,  # 四元数格式
            'ego2global_translation': [0.0, 0.0, 0.0],
            'ego2global_rotation': [1.0, 0.0, 0.0, 0.0],  # 四元数格式
            'timestamp': timestamp,
            'sensor2lidar_rotation': np.array(rotation_matrix, dtype=np.float32),
            'sensor2lidar_translation': np.array(transform['translation'], dtype=np.float32),
            'cam_intrinsic': np.array(cam_calib['intrinsics']['matrix'], dtype=np.float32),
        }
    
    def load_annotation(self, annotation_path):
        """加载标注数据 - 使用NuScenes原始格式"""
        if not annotation_path.exists():
            return self.create_empty_annotation()
            
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查objects字段是否存在
            if 'objects' not in data:
                return self.create_empty_annotation()
            
            gt_boxes = []
            gt_names = []
            
            for obj in data['objects']:
                if 'box3d' in obj and 'type' in obj:
                    box3d = obj['box3d']
                    rotation = obj.get('rotation', [0, 0, 0])
                    
                    # 构建7元素边界框 [x, y, z, l, w, h, yaw]
                    bbox = [
                        float(box3d[0]), float(box3d[1]), float(box3d[2]),
                        float(box3d[3]), float(box3d[4]), float(box3d[5]),
                        float(rotation[0]) if len(rotation) > 0 else 0.0
                    ]
                    gt_boxes.append(bbox)
                    gt_names.append(obj['type'])
            
            if gt_boxes:
                return {
                    'gt_boxes': np.array(gt_boxes, dtype=np.float32),
                    'gt_names': np.array(gt_names, dtype='<U32'),
                    'gt_velocity': np.zeros((len(gt_boxes), 2), dtype=np.float32),
                    'num_lidar_pts': np.full(len(gt_boxes), 10, dtype=np.int64),
                    'num_radar_pts': np.zeros(len(gt_boxes), dtype=np.int64),
                    'valid_flag': np.ones(len(gt_boxes), dtype=bool)
                }
            else:
                return self.create_empty_annotation()
                
        except Exception as e:
            print(f"加载标注失败 {annotation_path}: {e}")
            return self.create_empty_annotation()
    
    def create_empty_annotation(self):
        """创建空的标注数据 - NuScenes格式"""
        return {
            'gt_boxes': np.zeros((0, 7), dtype=np.float32),
            'gt_names': np.array([], dtype='<U32'),
            'gt_velocity': np.zeros((0, 2), dtype=np.float32),
            'num_lidar_pts': np.array([], dtype=np.int64),
            'num_radar_pts': np.array([], dtype=np.int64),
            'valid_flag': np.array([], dtype=bool)
        }
    
    def convert_frame(self, frame_data):
        """转换单帧数据 - 使用NuScenes原始格式"""
        info = {
            'token': f"tyjt_{frame_data['timestamp']}",
            'timestamp': frame_data['timestamp'],
            'scene_token': 'tyjt_scene',
            'lidar_path': str(frame_data['lidar_path']),
            'sweeps': [],
            'lidar2ego_translation': [0.0, 0.0, 0.0],
            'lidar2ego_rotation': [1.0, 0.0, 0.0, 0.0],  # 四元数格式
            'ego2global_translation': [0.0, 0.0, 0.0],
            'ego2global_rotation': [1.0, 0.0, 0.0, 0.0],  # 四元数格式
            'prev': '',
            'prev_token': '',
            'cams': {},
            'radars': {},  # 空雷达数据
        }
        
        # 添加相机数据
        for tyjt_cam, image_path in frame_data['cameras'].items():
            try:
                nusc_cam, camera_data = self.build_camera_data(tyjt_cam, image_path, frame_data['timestamp'])
                info['cams'][nusc_cam] = camera_data
            except Exception as e:
                print(f"转换相机 {tyjt_cam} 失败: {e}")
        
        # 添加标注数据 - 直接添加到info中（NuScenes格式）
        annotation_data = self.load_annotation(frame_data['annotation_path'])
        info.update(annotation_data)  # 关键：直接合并到info中
        
        return info
    
    def generate_pkl_files(self, frames, split_ratio=0.8):
        """生成PKL文件"""
        print("开始转换数据格式...")
        
        bevfusion_infos = []
        for i, frame_data in enumerate(frames):
            if i % 50 == 0:
                print(f"  转换进度: {i}/{len(frames)}")
            
            try:
                converted_frame = self.convert_frame(frame_data)
                bevfusion_infos.append(converted_frame)
            except Exception as e:
                print(f"转换帧 {frame_data['timestamp']} 失败: {e}")
        
        # 分割数据集
        split_idx = int(len(bevfusion_infos) * split_ratio)
        train_infos = bevfusion_infos[:split_idx]
        val_infos = bevfusion_infos[split_idx:]
        
        # 保存训练集
        train_data = {
            'infos': train_infos,
            'metadata': {
                'version': 'tyjt_1.0',
                'class_names': self.class_names
            }
        }
        
        # 保存验证集
        val_data = {
            'infos': val_infos,
            'metadata': train_data['metadata']
        }
        
        train_pkl_path = self.output_dir / "tyjt_infos_train.pkl"
        val_pkl_path = self.output_dir / "tyjt_infos_val.pkl"
        
        with open(train_pkl_path, 'wb') as f:
            pickle.dump(train_data, f)
        
        with open(val_pkl_path, 'wb') as f:
            pickle.dump(val_data, f)
        
        print(f"✅ PKL文件生成完成！")
        print(f"训练集: {len(train_infos)} 样本")
        print(f"验证集: {len(val_infos)} 样本")
        
        # 验证关键字段
        self.validate_key_fields(train_infos, "训练集")
        
        return train_pkl_path, val_pkl_path
    
    def validate_key_fields(self, infos, dataset_name):
        """验证关键字段"""
        valid_flag_count = 0
        gt_boxes_count = 0
        camera_count = 0
        
        for info in infos[:5]:  # 检查前5个样本
            if 'valid_flag' in info:
                valid_flag_count += 1
            if 'gt_boxes' in info:
                gt_boxes_count += 1
            if 'cams' in info:
                camera_count += len(info['cams'])
        
        print(f"✅ {dataset_name}验证:")
        print(f"  - {valid_flag_count}/5 样本有valid_flag")
        print(f"  - {gt_boxes_count}/5 样本有gt_boxes") 
        print(f"  - 平均相机数: {camera_count/5:.1f}")

def main():
    dataset_root = "/mnt/dataset/tyjt_RawData"
    calib_dir = "./step2_output"
    output_dir = "./step3_1_output"
    
    converter = TyjtToPklConverter(dataset_root, calib_dir, output_dir)
    frames = converter.scan_tyjt_dataset()
    
    if frames:
        converter.generate_pkl_files(frames)
        print("🎉 Step3 完成！使用NuScenes原始格式")
    else:
        print("❌ 未找到有效数据")

if __name__ == "__main__":
    main()