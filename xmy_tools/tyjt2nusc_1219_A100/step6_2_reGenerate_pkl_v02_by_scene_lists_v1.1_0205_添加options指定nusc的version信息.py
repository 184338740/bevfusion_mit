#!/usr/bin/env python3
"""
根据场景列表重新生成pkl文件 - V02版本
输入：train_scenes.txt, val_scenes.txt
输出：场景完全隔离的V02版本pkl文件
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import argparse
import logging
import sys
from datetime import datetime


# ==================== 配置日志 ====================
def setup_logging(output_dir: Path):
    """配置日志"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger("regenerate_pkl")
    logger.setLevel(logging.INFO)
    
    # 清除现有处理器
    logger.handlers.clear()
    
    # 文件处理器
    log_file = log_dir / f"regenerate_v02_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 格式
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ==================== 场景列表读取器 ====================
def read_scene_list(file_path: Path) -> Set[str]:
    """读取场景列表文件"""
    scenes = set()
    
    if not file_path.exists():
        raise FileNotFoundError(f"场景列表文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue
            scenes.add(line)
    
    return scenes

# ==================== 场景映射建立器 ====================
class SceneMapper:
    """建立sample_token到scene_name的映射"""
    
    def __init__(self, nuscenes_root: Path, logger: logging.Logger):
        self.nuscenes_root = nuscenes_root
        self.logger = logger
        
        # 映射表
        self.sample_to_scene = {}      # sample_token -> scene_name
        self.scene_to_samples = defaultdict(list)  # scene_name -> [sample_tokens]
        
        # 加载数据
        self._load_mappings()
    
    def _load_mappings(self):
        """加载并建立映射关系"""
        try:
            # 加载sample.json和scene.json
            version_dir = self.nuscenes_root / "v1.0-tyjt"
            
            # 1. 加载sample.json
            sample_file = version_dir / "sample.json"
            if not sample_file.exists():
                raise FileNotFoundError(f"找不到sample.json: {sample_file}")
            
            with open(sample_file, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            
            # 2. 加载scene.json
            scene_file = version_dir / "scene.json"
            if not scene_file.exists():
                raise FileNotFoundError(f"找不到scene.json: {scene_file}")
            
            with open(scene_file, 'r', encoding='utf-8') as f:
                scenes = json.load(f)
            
            # 3. 建立scene_token到scene_name的映射
            scene_token_to_name = {}
            for scene in scenes:
                scene_token_to_name[scene['token']] = scene['name']
            
            # 4. 建立sample_token到scene_name的映射
            scene_sample_count = defaultdict(int)
            
            for sample in samples:
                sample_token = sample['token']
                scene_token = sample['scene_token']
                
                if scene_token in scene_token_to_name:
                    scene_name = scene_token_to_name[scene_token]
                    self.sample_to_scene[sample_token] = scene_name
                    self.scene_to_samples[scene_name].append(sample_token)
                    scene_sample_count[scene_name] += 1
            
            # 统计信息
            total_samples = len(self.sample_to_scene)
            total_scenes = len(self.scene_to_samples)
            
            self.logger.info(f"✅ 映射建立完成:")
            self.logger.info(f"  总样本数: {total_samples}")
            self.logger.info(f"  总场景数: {total_scenes}")
            
            # 输出前10个场景的样本数
            sorted_scenes = sorted(self.scene_to_samples.items(), 
                                  key=lambda x: len(x[1]), reverse=True)[:10]
            self.logger.info("  样本最多的前10个场景:")
            for scene_name, samples_list in sorted_scenes:
                self.logger.info(f"    {scene_name}: {len(samples_list)} 个样本")
            
        except Exception as e:
            self.logger.error(f"❌ 建立映射失败: {e}")
            raise
    
    def get_scene_for_sample(self, sample_token: str) -> str:
        """获取样本所属的场景"""
        return self.sample_to_scene.get(sample_token, "unknown")
    
    def get_samples_for_scene(self, scene_name: str) -> List[str]:
        """获取场景的所有样本"""
        return self.scene_to_samples.get(scene_name, [])

# ==================== PKL处理器 ====================
class PklRegenerator:
    """PKL重新生成器"""
    def __init__(self, input_dir: Path, scene_mapper: SceneMapper,  # 改为input_dir
                 train_scenes: Set[str], val_scenes: Set[str], 
                 logger: logging.Logger, version: str, 
                 output_dir: Path):
        self.input_dir = input_dir      # 原始pkl目录
        self.scene_mapper = scene_mapper
        self.train_scenes = train_scenes
        self.val_scenes = val_scenes
        self.logger = logger
        self.version = version
        self.output_dir = output_dir    # 新pkl输出目录

        self.logger.info(f"🔖 使用版本: {self.version}")
        self.logger.info(f"📥 输入目录: {self.input_dir}")
        self.logger.info(f"📤 输出目录: {self.output_dir}")

        # 检查场景列表
        self._validate_scene_lists()
    
    def _validate_scene_lists(self):
        """验证场景列表"""
        # 检查交集
        intersection = self.train_scenes & self.val_scenes
        if intersection:
            self.logger.warning(f"⚠️  训练集和验证集有 {len(intersection)} 个重叠场景:")
            for scene in list(intersection)[:5]:  # 只显示前5个
                self.logger.warning(f"    {scene}")
        
        # 检查场景是否存在
        all_known_scenes = set(self.scene_mapper.scene_to_samples.keys())
        
        train_unknown = self.train_scenes - all_known_scenes
        val_unknown = self.val_scenes - all_known_scenes
        
        if train_unknown:
            self.logger.warning(f"⚠️  训练集有 {len(train_unknown)} 个未知场景")
        if val_unknown:
            self.logger.warning(f"⚠️  验证集有 {len(val_unknown)} 个未知场景")
        
        # 统计
        train_sample_count = sum(len(self.scene_mapper.scene_to_samples.get(s, [])) 
                               for s in self.train_scenes)
        val_sample_count = sum(len(self.scene_mapper.scene_to_samples.get(s, [])) 
                             for s in self.val_scenes)
        
        self.logger.info(f"📊 场景划分统计:")
        self.logger.info(f"  训练集场景数: {len(self.train_scenes)}")
        self.logger.info(f"  验证集场景数: {len(self.val_scenes)}")
        self.logger.info(f"  训练集预估样本数: {train_sample_count}")
        self.logger.info(f"  验证集预估样本数: {val_sample_count}")
    
    def _extract_sample_token(self, info: Dict) -> str:
        """从info中提取sample_token"""
        # 方法1: 直接检查token字段
        for field in ['token', 'sample_token', 'lidar_token']:
            if field in info and isinstance(info[field], str):
                return info[field]
        
        # 方法2: 从lidar_path提取
        if 'lidar_path' in info:
            path = info['lidar_path']
            # 格式: samples/LIDAR_TOP/{token}.bin
            if 'samples/LIDAR_TOP/' in path:
                return Path(path).stem
        
        # 方法3: 从cams信息提取
        if 'cams' in info:
            for cam_info in info['cams'].values():
                if 'data_path' in cam_info:
                    path = cam_info['data_path']
                    if 'samples/CAM_' in path:
                        return Path(path).stem
        
        # 如果都失败，返回一个基于内容的哈希
        import hashlib
        info_str = str(info).encode('utf-8')
        return f"hash_{hashlib.md5(info_str).hexdigest()[:16]}"
    
    def regenerate_pkl(self):
        """重新生成pkl文件"""
        self.logger.info("开始重新生成pkl文件...")
        
        # 1. 从self.input_dir加载原始pkl
        original_train_path = self.input_dir / "tyjt_infos_train.pkl"
        original_val_path = self.input_dir / "tyjt_infos_val.pkl"
        
        if not original_train_path.exists():
            raise FileNotFoundError(f"找不到原始训练集pkl: {original_train_path}")
        
        with open(original_train_path, 'rb') as f:
            original_train = pickle.load(f)
        
        # 加载验证集（如果存在）
        original_val = None
        if original_val_path.exists():
            with open(original_val_path, 'rb') as f:
                original_val = pickle.load(f)
        
        # 2. 合并所有infos
        all_infos = original_train['infos']
        if original_val and 'infos' in original_val:
            all_infos.extend(original_val['infos'])
        
        self.logger.info(f"合并原始infos: {len(all_infos)} 个样本")
        
        # 3. 按场景重新划分
        train_infos = []
        val_infos = []
        unknown_scene_infos = []
        unknown_token_infos = []
        
        for info in all_infos:
            # 提取sample_token
            sample_token = self._extract_sample_token(info)
            
            # 查找场景
            scene_name = self.scene_mapper.get_scene_for_sample(sample_token)
            
            if scene_name == "unknown":
                unknown_token_infos.append((sample_token, info))
                continue
            
            # 分配到训练集或验证集
            if scene_name in self.train_scenes:
                train_infos.append(info)
            elif scene_name in self.val_scenes:
                val_infos.append(info)
            else:
                unknown_scene_infos.append((scene_name, info))
        
        # 4. 统计信息
        self.logger.info("📊 重新划分结果:")
        self.logger.info(f"  训练集样本: {len(train_infos)}")
        self.logger.info(f"  验证集样本: {len(val_infos)}")
        self.logger.info(f"  未知场景样本: {len(unknown_scene_infos)}")
        self.logger.info(f"  未知token样本: {len(unknown_token_infos)}")
        
        # 输出未知场景（用于调试）
        if unknown_scene_infos:
            unknown_scenes = set(scene for scene, _ in unknown_scene_infos)
            self.logger.warning(f"⚠️  发现 {len(unknown_scenes)} 个不在train/val列表中的场景:")
            for scene in sorted(list(unknown_scenes))[:10]:  # 只显示前10个
                count = sum(1 for s, _ in unknown_scene_infos if s == scene)
                self.logger.warning(f"    {scene}: {count} 个样本")
        
        # 5. 创建新的pkl结构
        # 训练集pkl
        new_train_pkl = {
            'infos': train_infos,
            'metadata': {
                'version': self.version, #'v2.0_scene_isolated',
                'dataset': 'tyjt',
                'scene_split': {
                    'train_scenes': sorted(list(self.train_scenes)),
                    'val_scenes': sorted(list(self.val_scenes)),
                    'generation_time': datetime.now().isoformat()
                }
            }
        }
        
        # 验证集pkl
        new_val_pkl = {
            'infos': val_infos,
            'metadata': {
                'version': 'v2.0_scene_isolated',
                'dataset': 'tyjt'
            }
        }
        
        # 6. 保存新的pkl文件
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        train_output_path = self.output_dir / "tyjt_infos_train.pkl"
        val_output_path = self.output_dir / "tyjt_infos_val.pkl"
        
        with open(train_output_path, 'wb') as f:
            pickle.dump(new_train_pkl, f)
        
        with open(val_output_path, 'wb') as f:
            pickle.dump(new_val_pkl, f)
        
        self.logger.info(f"✅ 保存新的pkl文件:")
        self.logger.info(f"  训练集: {train_output_path}")
        self.logger.info(f"  验证集: {val_output_path}")
        
        # 7. 生成dbinfos（可选）
        self._generate_dbinfos(train_infos, self.output_dir)
        
        return train_output_path, val_output_path
    
    def _generate_dbinfos(self, train_infos: List[Dict], output_dir: Path):
        """生成新的dbinfos文件"""
        try:
            # 从原始dbinfos复制或创建新的
            original_dbinfos_path = self.input_dir / "tyjt_dbinfos_train.pkl"
            new_dbinfos_path = output_dir / "tyjt_dbinfos_train.pkl"
            
            if original_dbinfos_path.exists():
                with open(original_dbinfos_path, 'rb') as f:
                    original_dbinfos = pickle.load(f)
                
                # 简单复制（如果结构相同）
                with open(new_dbinfos_path, 'wb') as f:
                    pickle.dump(original_dbinfos, f)
                
                self.logger.info(f"✅ 复制dbinfos: {new_dbinfos_path}")
            else:
                self.logger.warning("⚠️  找不到原始dbinfos文件")
                
        except Exception as e:
            self.logger.warning(f"生成dbinfos失败: {e}")

# ==================== 主函数 ====================
def main():
    # input_dir = args.pkl_dir          # 输入：原始pkl所在目录
    # output_dir = args.output_dir      # 输出：顶级输出目录
    # version_output_dir = output_dir / args.version / pkl # 版本子目录
    parser = argparse.ArgumentParser(description='根据场景列表重新生成pkl文件 - V02等指定版本')
    parser.add_argument('--nuscenes-dir', type=str, required=True,
                       help='nuscenes_tyjt数据集目录')
    parser.add_argument('--pkl_dir', type=str, required=True,
                       help='V01版本pkl文件目录')
    parser.add_argument('--train-scenes', type=str, required=True,
                       help='训练集场景列表文件 (每行一个场景名)')
    parser.add_argument('--val-scenes', type=str, required=True,
                       help='验证集场景列表文件 (每行一个场景名)')
    parser.add_argument('--output-dir', type=str, default='./v02_output',
                       help='输出目录')
    # 新增version参数
    parser.add_argument('--version', type=str, required=True,
                       help='--version：pkl中metadata的version字段，对应数据目录名(如：V02-TYJT)； 若未指定，则代码内会自动设置为创建日期')
    
    args = parser.parse_args()
    # 如果没有指定version，生成一个带日期的
    if args.version is None:
        from datetime import datetime
        args.version = f"V02-TYJT-{datetime.now().strftime('%Y%m%d')}"
        logger.info(f"未指定version，使用默认(日期): {args.version}")
    
    # 创建输出目录
    version_output_dir = Path(args.output_dir) / args.version / 'pkl'
    version_output_dir.mkdir(parents=True, exist_ok=True)
    
    # output_dir = Path(args.output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(version_output_dir)
    
    logger.info("=" * 60)
    logger.info("🔄 开始重新生成V02版本pkl文件")
    logger.info("=" * 60)
    
    try:
        # 1. 读取场景列表
        logger.info("📋 读取场景列表...")
        train_scenes = read_scene_list(Path(args.train_scenes))
        val_scenes = read_scene_list(Path(args.val_scenes))
        
        logger.info(f"  训练集场景数: {len(train_scenes)}")
        logger.info(f"  验证集场景数: {len(val_scenes)}")
        
        # 2. 建立映射关系
        logger.info("\n🗺️  建立样本-场景映射...")
        scene_mapper = SceneMapper(Path(args.nuscenes_dir), logger)
        
        # 3. 重新生成pkl
        logger.info("\n🔄 重新生成pkl文件...")
        # 3.1 初始化
        regenerator = PklRegenerator(
            input_dir=Path(args.pkl_dir),       # 改为更清晰的input_dir
            scene_mapper=scene_mapper,
            train_scenes=train_scenes,
            val_scenes=val_scenes,
            logger=logger,
            version=args.version,
            output_dir=version_output_dir       # 新pkl的输出目录
        )
        # 构建
        train_path, val_path = regenerator.regenerate_pkl()
        
        # 4. 输出完成信息
        logger.info("\n" + "=" * 60)
        logger.info("✅ V02版本pkl文件生成完成!")
        logger.info("=" * 60)
        logger.info(f"📁 输出目录: {version_output_dir}")
        logger.info(f"📄 训练集: {train_path}")
        logger.info(f"📄 验证集: {val_path}")
        logger.info(f"📄 dbinfos: {version_output_dir}/tyjt_dbinfos_train_v02.pkl")
        
        # 5. 创建场景列表备份
        with open(version_output_dir / "train_scenes_backup.txt", 'w') as f:
            for scene in sorted(train_scenes):
                f.write(f"{scene}\n")
        
        with open(version_output_dir / "val_scenes_backup.txt", 'w') as f:
            for scene in sorted(val_scenes):
                f.write(f"{scene}\n")
        
        logger.info("💾 场景列表已备份到输出目录")
        
    except Exception as e:
        logger.error(f"❌ 处理失败: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()