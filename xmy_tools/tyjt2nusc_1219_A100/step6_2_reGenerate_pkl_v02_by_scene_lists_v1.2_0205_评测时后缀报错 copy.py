#!/usr/bin/env python3
"""
step6_2_reGenerate_pkl_v02_by_scene_lists_v1.3_final.py
完整修复版：包含版本兼容性和dbinfos结构修复
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import argparse
import logging
import sys
from datetime import datetime
import shutil

# ==================== 配置日志 ====================
def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("pkl_regenerator_final")
    logger.setLevel(logging.INFO)
    
    logger.handlers.clear()
    
    log_file = log_dir / f"regenerate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

# ==================== 场景列表读取器 ====================
def read_scene_list(file_path: Path) -> Set[str]:
    scenes = set()
    
    if not file_path.exists():
        raise FileNotFoundError(f"场景列表文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            scenes.add(line)
    
    return scenes

# ==================== 场景映射建立器 ====================
class SceneMapper:
    def __init__(self, nuscenes_dir: Path, pkl_dir: Path, logger: logging.Logger):
        self.nuscenes_dir = nuscenes_dir
        self.pkl_dir = pkl_dir
        self.logger = logger
        
        self.sample_to_scene = {}
        self.scene_to_samples = defaultdict(list)
        
        self.version_dir = self._detect_version_dir()
        self._load_mappings()
    
    def _read_version_from_pkl(self) -> str:
        """从原始pkl读取版本"""
        train_pkl = self.pkl_dir / "tyjt_infos_train.pkl"
        
        try:
            with open(train_pkl, 'rb') as f:
                data = pickle.load(f)
            
            if 'metadata' in data and 'version' in data['metadata']:
                version = data['metadata']['version']
                self.logger.info(f"从pkl读取版本: {version}")
                return version
        except Exception as e:
            self.logger.warning(f"从pkl读取版本失败: {e}")
        
        return "v1.0-tyjt-trainval"
    
    def _detect_version_dir(self) -> Path:
        """检测版本目录"""
        # 从pkl读取版本
        pkl_version = self._read_version_from_pkl()
        
        # 提取基础目录名
        if pkl_version.endswith('-trainval'):
            base_name = pkl_version[:-9]
        elif pkl_version.endswith('-test'):
            base_name = pkl_version[:-5]
        else:
            base_name = pkl_version
        
        # 检查目录是否存在
        version_dir = self.nuscenes_dir / base_name
        if version_dir.exists():
            self.logger.info(f"使用版本目录: {version_dir.name}")
            return version_dir
        
        # 如果不存在，查找v开头的目录
        for item in self.nuscenes_dir.iterdir():
            if item.is_dir() and item.name.startswith("v"):
                if (item / "scene.json").exists() and (item / "sample.json").exists():
                    self.logger.info(f"使用备选目录: {item.name}")
                    return item
        
        raise FileNotFoundError(f"找不到版本目录: {self.nuscenes_dir}")
    
    def _load_mappings(self):
        """加载并建立映射关系"""
        try:
            sample_file = self.version_dir / "sample.json"
            scene_file = self.version_dir / "scene.json"
            
            with open(sample_file, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            with open(scene_file, 'r', encoding='utf-8') as f:
                scenes = json.load(f)
            
            scene_token_to_name = {}
            for scene in scenes:
                scene_token_to_name[scene['token']] = scene['name']
            
            for sample in samples:
                sample_token = sample['token']
                scene_token = sample['scene_token']
                
                if scene_token in scene_token_to_name:
                    scene_name = scene_token_to_name[scene_token]
                    self.sample_to_scene[sample_token] = scene_name
                    self.scene_to_samples[scene_name].append(sample_token)
            
            self.logger.info(f"映射完成: {len(self.sample_to_scene)}样本, {len(self.scene_to_samples)}场景")
            
        except Exception as e:
            self.logger.error(f"建立映射失败: {e}")
            raise
    
    def get_scene_for_sample(self, sample_token: str) -> str:
        return self.sample_to_scene.get(sample_token, "unknown")

# ==================== PKL重新生成器 ====================
class PklRegenerator:
    def __init__(self, args, logger: logging.Logger):
        self.args = args
        self.logger = logger
        
        # 源数据路径
        self.src_nuscenes_dir = Path(args.src_nuscenes_dir)
        self.src_pkl_dir = Path(args.src_pkl_dir)
        self.src_train_scenes = Path(args.src_train_scenes)
        self.src_val_scenes = Path(args.src_val_scenes)
        
        # ========== 简化版本处理 ==========
        # 用户输入的版本（如 "v02-tyjt" 或 "v02-tyjt-trainval"）
        user_version = args.dst-version
        
        # 检查用户输入的版本后缀
        if user_version.endswith('-trainval'):
            # 已经是训练/验证集版本
            final_version = user_version
            version_type = "trainval"
            
        elif user_version.endswith('-test'):
            # 已经是测试集版本
            final_version = user_version
            version_type = "test"
            
        else:
            # 没有后缀，默认添加 -trainval（训练/验证集）
            final_version = f"{user_version}-trainval"
            version_type = "trainval"
            self.logger.info(f"🔧 自动添加-trainval后缀: {user_version} -> {final_version}")

        self.version = final_version  # 唯一版本
        self.logger.info(f"📋 版本类型: {version_type}")
        
        self.version = final_version  # 唯一版本，如 "v02-tyjt-trainval"
        
        # 提取基础目录名（移除后缀）
        if self.version.endswith('-trainval'):
            base_dir_name = self.version[:-9]  # "v02-tyjt"
        else:
            base_dir_name = self.version[:-5]  # "v02-tyjt"（如果是-test）
        
        self.logger.info(f"📋 版本配置:")
        self.logger.info(f"  最终版本: {self.version}")
        self.logger.info(f"  对应目录: {base_dir_name}")
        
        # 检查对应JSON目录是否存在
        json_dir = self.src_nuscenes_dir / base_dir_name
        if not json_dir.exists():
            self.logger.error(f"❌ 找不到JSON目录: {json_dir}")
            self.logger.error(f"   请确保目录 {base_dir_name} 存在于 {self.src_nuscenes_dir}")
            raise FileNotFoundError(f"JSON目录不存在: {base_dir_name}")
        
        self.logger.info(f"✅ JSON目录存在: {json_dir}")
        # ========== 简化结束 ==========
        
        # 输出目录
        self.dst_root = Path(args.dst_root)
        self.dst_version_dir = self.dst_root / self.version
        
        # 创建输出目录
        self.dst_version_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取场景列表
        self.logger.info("读取场景列表...")
        self.train_scenes = read_scene_list(self.src_train_scenes)
        self.val_scenes = read_scene_list(self.src_val_scenes)
        
        self.logger.info(f"  训练集场景: {len(self.train_scenes)}")
        self.logger.info(f"  验证集场景: {len(self.val_scenes)}")
        
        # 建立映射（使用base_dir_name对应的目录）
        self.logger.info("\n建立样本-场景映射...")
        self.scene_mapper = SceneMapper(self.src_nuscenes_dir, base_dir_name, logger)
    
    def _read_version_from_source(self) -> str:
        """从原始pkl文件读取版本"""
        src_train_pkl = self.src_pkl_dir / "tyjt_infos_train.pkl"
        
        try:
            with open(src_train_pkl, 'rb') as f:
                data = pickle.load(f)
            
            if 'metadata' in data and 'version' in data['metadata']:
                version = data['metadata']['version']
                self.logger.info(f"读取源版本: {version}")
                return version
        except Exception as e:
            self.logger.warning(f"读取源版本失败: {e}")
        
        return "v1.0-tyjt-trainval"
    
    def _extract_sample_token(self, info: Dict) -> str:
        """从info中提取sample_token"""
        for field in ['token', 'sample_token', 'lidar_token']:
            if field in info and isinstance(info[field], str):
                return info[field]
        
        if 'lidar_path' in info:
            path = info['lidar_path']
            if 'samples/LIDAR_TOP/' in path:
                return Path(path).stem
        
        if 'cams' in info:
            for cam_info in info['cams'].values():
                if 'data_path' in cam_info:
                    path = cam_info['data_path']
                    if 'samples/CAM_' in path:
                        return Path(path).stem
        
        import hashlib
        info_str = str(info).encode('utf-8')
        return f"hash_{hashlib.md5(info_str).hexdigest()[:16]}"
    
    def run(self):
        """主运行函数"""
        self.logger.info("\n开始重新生成pkl文件...")
        
        # 1. 加载原始pkl
        src_train_pkl = self.src_pkl_dir / "tyjt_infos_train.pkl"
        src_val_pkl = self.src_pkl_dir / "tyjt_infos_val.pkl"
        
        if not src_train_pkl.exists():
            raise FileNotFoundError(f"找不到训练集pkl: {src_train_pkl}")
        
        with open(src_train_pkl, 'rb') as f:
            original_train = pickle.load(f)
        
        # 加载验证集
        all_infos = original_train['infos']
        if src_val_pkl.exists():
            with open(src_val_pkl, 'rb') as f:
                original_val = pickle.load(f)
            if 'infos' in original_val:
                all_infos.extend(original_val['infos'])
        
        self.logger.info(f"加载 {len(all_infos)} 个样本")
        
        # 2. 按场景重新划分
        train_infos = []
        val_infos = []
        
        for info in all_infos:
            sample_token = self._extract_sample_token(info)
            scene_name = self.scene_mapper.get_scene_for_sample(sample_token)
            
            if scene_name == "unknown":
                train_infos.append(info)
                continue
            
            if scene_name in self.train_scenes:
                train_infos.append(info)
            elif scene_name in self.val_scenes:
                val_infos.append(info)
            else:
                train_infos.append(info)
        
        # 3. 统计
        self.logger.info(f"划分结果:")
        self.logger.info(f"  训练集样本: {len(train_infos)}")
        self.logger.info(f"  验证集样本: {len(val_infos)}")
        
        # 4. 创建新的pkl
        new_train_pkl = {
            'infos': train_infos,
            'metadata': {
                'version': self.output_version,
                'dataset': 'tyjt',
                'original_version': self.base_version,
                'custom_tag': self.custom_tag,
                'scene_split': {
                    'train_scenes': len(self.train_scenes),
                    'val_scenes': len(self.val_scenes),
                    'time': datetime.now().isoformat()
                }
            }
        }
        
        new_val_pkl = {
            'infos': val_infos,
            'metadata': {
                'version': self.output_version,
                'dataset': 'tyjt',
                'original_version': self.base_version,
                'custom_tag': self.custom_tag
            }
        }
        
        # 5. 保存pkl文件
        train_output = self.dst_version_dir / "tyjt_infos_train.pkl"
        val_output = self.dst_version_dir / "tyjt_infos_val.pkl"
        
        with open(train_output, 'wb') as f:
            pickle.dump(new_train_pkl, f)
        with open(val_output, 'wb') as f:
            pickle.dump(new_val_pkl, f)
        
        self.logger.info(f"保存pkl文件:")
        self.logger.info(f"  训练集: {train_output}")
        self.logger.info(f"  验证集: {val_output}")
        self.logger.info(f"  版本: {self.output_version}")
        
        # 6. 处理dbinfos
        self.process_dbinfos()
        
        # 7. 复制JSON文件
        self.copy_json_files()
        
        # 8. 保存场景列表
        self.save_scene_lists()
        
        return train_output, val_output
    
    def process_dbinfos(self):
        """处理dbinfos文件"""
        src_dbinfos = self.src_pkl_dir / "tyjt_dbinfos_train.pkl"
        dst_dbinfos = self.dst_version_dir / "tyjt_dbinfos_train.pkl"
        
        if not src_dbinfos.exists():
            self.logger.warning(f"找不到原始dbinfos: {src_dbinfos}")
            return
        
        try:
            # 加载并修复dbinfos结构
            with open(src_dbinfos, 'rb') as f:
                original = pickle.load(f)
            
            # 确保正确的结构
            if isinstance(original, dict) and 'dbinfos' in original:
                fixed = original
                self.logger.info("dbinfos格式正确")
            elif isinstance(original, dict):
                # 添加外层dbinfos键
                fixed = {'dbinfos': original}
                self.logger.info("修复dbinfos结构")
            else:
                fixed = {'dbinfos': {}}
                self.logger.warning("创建空dbinfos")
            
            # 添加版本信息
            fixed['metadata'] = {
                'version': self.output_version,
                'source': 'step6_2_regenerated'
            }
            
            # 保存修复后的文件
            with open(dst_dbinfos, 'wb') as f:
                pickle.dump(fixed, f)
            
            self.logger.info(f"保存dbinfos: {dst_dbinfos}")
            
        except Exception as e:
            self.logger.error(f"处理dbinfos失败: {e}")
            # 直接复制
            shutil.copy2(src_dbinfos, dst_dbinfos)
            self.logger.info(f"直接复制dbinfos: {dst_dbinfos}")
    
    def copy_json_files(self):
        """复制JSON文件"""
        version_dir = self.scene_mapper.version_dir
        
        json_count = 0
        for json_file in version_dir.glob("*.json"):
            dst_file = self.dst_version_dir / json_file.name
            shutil.copy2(json_file, dst_file)
            json_count += 1
        
        self.logger.info(f"复制 {json_count} 个JSON文件")
    
    def save_scene_lists(self):
        """保存场景列表"""
        # 训练集
        with open(self.dst_version_dir / "train_scenes.txt", 'w', encoding='utf-8') as f:
            f.write(f"# 训练集场景列表\n")
            f.write(f"# 版本: {self.output_version}\n")
            f.write(f"# 场景数: {len(self.train_scenes)}\n")
            f.write("#\n")
            for scene in sorted(self.train_scenes):
                f.write(f"{scene}\n")
        
        # 验证集
        with open(self.dst_version_dir / "val_scenes.txt", 'w', encoding='utf-8') as f:
            f.write(f"# 验证集场景列表\n")
            f.write(f"# 版本: {self.output_version}\n")
            f.write(f"# 场景数: {len(self.val_scenes)}\n")
            f.write("#\n")
            for scene in sorted(self.val_scenes):
                f.write(f"{scene}\n")
        
        self.logger.info("保存场景列表")

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(
        description='根据场景列表重新生成pkl文件 - 完整修复版'
    )
    
    # 源数据
    parser.add_argument('--src_nuscenesdir', type=str, required=True,
                       help='nuscenes_tyjt数据集目录')
    parser.add_argument('--src-pkl-dir', type=str, required=True,
                       help='原始pkl文件目录')
    parser.add_argument('--src-train-scenes', type=str, required=True,
                       help='训练集场景列表文件')
    parser.add_argument('--src-val-scenes', type=str, required=True,
                       help='验证集场景列表文件')
    
    # 输出
    parser.add_argument('--dst-root', type=str, default='./output',
                       help='输出根目录')
    parser.add_argument('--dst-version', type=str, default='V02-TYJT',
                       help='自定义版本标签')
    
    args = parser.parse_args()
    
    # 创建日志目录
    log_dir = Path(args.dst_root) / "logs"
    logger, log_file = setup_logging(log_dir)
    
    logger.info("=" * 60)
    logger.info("重新生成pkl文件 - 完整修复版")
    logger.info("=" * 60)
    
    try:
        # 运行主逻辑
        regenerator = PklRegenerator(args, logger)
        train_path, val_path = regenerator.run()
        
        # 输出结果
        logger.info("\n" + "=" * 60)
        logger.info("✅ 处理完成!")
        logger.info("=" * 60)
        logger.info(f"输出目录: {regenerator.dst_version_dir}")
        logger.info(f"版本: {regenerator.output_version}")
        logger.info(f"日志: {log_file}")
        
    except Exception as e:
        logger.error(f"❌ 处理失败: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()