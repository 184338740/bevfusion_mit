#!/usr/bin/env python3
"""
step6_2_reGenerate_pkl_v02_by_scene_lists_final_simple.py
简化版：基于确认的PKL格式
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
import shutil
# ==================== BEVFusion 仓库路径 ====================
BEVFUSION_PATH = Path("/data2/xmy/01_project/bevfusion_mit_xmy_1205")  # 请确认路径正确


# ==================== 配置日志 ====================
def setup_logging(log_dir: Path) -> Tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("pkl_regenerator_simple")
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
        pkl_version = self._read_version_from_pkl()
        
        if pkl_version.endswith('-trainval'):
            base_name = pkl_version[:-9]
        elif pkl_version.endswith('-test'):
            base_name = pkl_version[:-5]
        else:
            base_name = pkl_version
        
        version_dir = self.nuscenes_dir / base_name
        if version_dir.exists():
            self.logger.info(f"使用版本目录: {version_dir.name}")
            return version_dir
        
        for item in self.nuscenes_dir.iterdir():
            if item.is_dir():
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
        return self.sample_to_scene.get(sample_token, "")

# ==================== PKL重新生成器 ====================
class PklRegenerator:
    def __init__(self, args, logger: logging.Logger):
        self.args = args
        self.logger = logger
        
        self.src_nuscenes_dir = Path(args.src_nuscenes_dir)
        self.src_pkl_dir = Path(args.src_pkl_dir)
        self.src_train_scenes = Path(args.src_train_scenes)
        self.src_val_scenes = Path(args.src_val_scenes)
        
        self.dst_root = Path(args.dst_root)
        self.dst_version = args.dst_version
        
        self.dst_dir = self.dst_root / self.dst_version
        self.dst_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("读取场景列表...")
        self.train_scenes = read_scene_list(self.src_train_scenes)
        self.val_scenes = read_scene_list(self.src_val_scenes)
        
        self.logger.info(f"训练集场景: {len(self.train_scenes)}")
        self.logger.info(f"验证集场景: {len(self.val_scenes)}")
        
        self.logger.info("\n建立样本-场景映射...")
        self.scene_mapper = SceneMapper(
            nuscenes_dir=self.src_nuscenes_dir,
            pkl_dir=self.src_pkl_dir,
            logger=logger
        )

    def run(self):
        """主运行函数 - 按场景列表重新划分并生成完整 V02 数据集"""
        self.logger.info("\n" + "="*60)
        self.logger.info("PKL重新生成")
        self.logger.info("="*60)

        try:
            # ---------- 1. 加载原始 PKL ----------
            src_train_pkl = self.src_pkl_dir / "tyjt_infos_train.pkl"
            with open(src_train_pkl, 'rb') as f:
                original_train = pickle.load(f)

            src_val_pkl = self.src_pkl_dir / "tyjt_infos_val.pkl"
            original_val = None
            if src_val_pkl.exists():
                with open(src_val_pkl, 'rb') as f:
                    original_val = pickle.load(f)

            # 校验 PKL 格式
            if not isinstance(original_train, dict) or 'infos' not in original_train:
                raise ValueError(f"PKL格式错误，期望dict with 'infos', 实际: {type(original_train)}")

            # 打印原始结构（调试用）
            self.logger.info(f"原始train.pkl类型: {type(original_train)}")
            self.logger.info(f"原始train.pkl键: {list(original_train.keys())}")
            if 'metadata' in original_train:
                self.logger.info(f"原始train.pkl metadata: {original_train['metadata']}")

            # ---------- 2. 合并所有样本（train + val）----------
            all_infos = original_train['infos'].copy()
            if original_val and 'infos' in original_val:
                all_infos.extend(original_val['infos'])

            self.logger.info(f"总样本数: {len(all_infos)}")

            # ---------- 3. 按场景列表重新划分 ----------
            train_infos = []
            val_infos = []

            for info in all_infos:
                sample_token = info['token']
                scene_name = self.scene_mapper.get_scene_for_sample(sample_token)

                if not scene_name:
                    self.logger.warning(f"样本 {sample_token[:16]}... 没有场景映射，默认放入训练集")
                    train_infos.append(info)
                    continue

                if scene_name in self.train_scenes:
                    train_infos.append(info)
                elif scene_name in self.val_scenes:
                    val_infos.append(info)
                else:
                    self.logger.warning(f"场景 {scene_name} 不在任何列表中，默认放入训练集")
                    train_infos.append(info)

            self.logger.info(f"划分结果: 训练集 {len(train_infos)} 样本, 验证集 {len(val_infos)} 样本")

            # ---------- 4. 创建新的 PKL 数据字典 ----------
            # 提取原始 metadata 基础信息
            if 'metadata' in original_train and isinstance(original_train['metadata'], dict):
                dataset_type = original_train['metadata'].get('dataset', 'tyjt')
            else:
                dataset_type = 'tyjt'

            # 构建新 metadata（使用命令行传入的 dst_version）
            new_metadata = {
                'version': self.dst_version,
                'dataset': dataset_type,
                'regenerated': datetime.now().isoformat(),
                'source_version': original_train.get('metadata', {}).get('version', 'unknown')
            }

            new_train_data = {'infos': train_infos, 'metadata': new_metadata.copy()}
            new_val_data   = {'infos': val_infos,   'metadata': new_metadata.copy()}

            # ---------- 5. 保存新的 PKL 文件 ----------
            train_output = self.dst_dir / "tyjt_infos_train.pkl"
            val_output   = self.dst_dir / "tyjt_infos_val.pkl"

            with open(train_output, 'wb') as f:
                pickle.dump(new_train_data, f)
            with open(val_output, 'wb') as f:
                pickle.dump(new_val_data, f)

            # 同时保存 JSON 格式方便查看
            with open(train_output.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(new_train_data, f, indent=2, ensure_ascii=False, default=str)
            with open(val_output.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(new_val_data, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"\n✅ PKL文件保存:")
            self.logger.info(f"  训练集: {train_output}")
            self.logger.info(f"  验证集: {val_output}")

            # ---------- 6. 创建数据软链接（必须）----------
            self._create_data_symlinks()

            # ---------- 7. 重新生成 dbinfos 和 gt_database ----------
            self._regenerate_dbinfos(train_output)

            # ---------- 8. 复制元数据 JSON 文件 ----------
            self._copy_json_files()

            # ---------- 9. 保存场景列表 ----------
            self._save_scene_lists()

            return train_output, val_output

        except Exception as e:
            self.logger.error(f"处理失败: {e}", exc_info=True)
            raise
    
    def _copy_dbinfos(self):
        """复制dbinfos文件"""
        src_dbinfos = self.src_pkl_dir / "tyjt_dbinfos_train.pkl"
        dst_dbinfos = self.dst_dir / "tyjt_dbinfos_train.pkl"
        
        if src_dbinfos.exists():
            shutil.copy2(src_dbinfos, dst_dbinfos)
            self.logger.info(f"复制dbinfos: {dst_dbinfos}")
        else:
            self.logger.warning(f"找不到原始dbinfos: {src_dbinfos}")
    
    def _copy_json_files(self):
        """复制JSON文件"""
        version_dir = self.scene_mapper.version_dir
        
        json_count = 0
        for json_file in version_dir.glob("*.json"):
            dst_file = self.dst_dir / json_file.name
            shutil.copy2(json_file, dst_file)
            json_count += 1
        
        self.logger.info(f"复制 {json_count} 个JSON文件")
    
    def _save_scene_lists(self):
        """保存场景列表"""
        train_list_file = self.dst_dir / "train_scenes.txt"
        with open(train_list_file, 'w', encoding='utf-8') as f:
            f.write(f"# 训练集场景列表\n# 版本: {self.dst_version}\n#\n")
            for scene in sorted(self.train_scenes):
                f.write(f"{scene}\n")
        
        val_list_file = self.dst_dir / "val_scenes.txt"
        with open(val_list_file, 'w', encoding='utf-8') as f:
            f.write(f"# 验证集场景列表\n# 版本: {self.dst_version}\n#\n")
            for scene in sorted(self.val_scenes):
                f.write(f"{scene}\n")
        
        self.logger.info("保存场景列表")

    def _create_data_symlinks(self):
        """创建指向原始 nuscenes_tyjt 数据目录的软链接"""
        src_root = Path(self.src_nuscenes_dir)
        dst_root = self.dst_dir

        for subdir in ['samples', 'sweeps', 'maps']:
            src = src_root / subdir
            dst = dst_root / subdir
            if src.exists() and not dst.exists():
                dst.symlink_to(src, target_is_directory=True)
                self.logger.info(f"创建软链接: {dst} -> {src}")
            elif src.exists() and dst.exists():
                self.logger.info(f"软链接已存在: {dst}")
            else:
                self.logger.warning(f"源目录不存在: {src}，跳过软链接创建")

    def _regenerate_dbinfos(self, train_infos_path: Path):
        """仅使用新划分的训练集重新生成 dbinfos 和 gt_database"""
        import sys

        # 确保 BEVFusion 根目录在 sys.path 中
        if str(BEVFUSION_PATH) not in sys.path:
            sys.path.insert(0, str(BEVFUSION_PATH))

        try:
            from tools.data_converter.create_gt_database import create_groundtruth_database
        except ImportError as e:
            self.logger.error(f"导入 create_groundtruth_database 失败: {e}")
            self.logger.error("请检查 BEVFUSION_PATH 是否正确，或手动添加仓库路径")
            raise

        self.logger.info("开始重新生成 tyjt_dbinfos_train.pkl ...")

        if not train_infos_path.exists():
            raise FileNotFoundError(f"训练集 pkl 未找到: {train_infos_path}")

        # 调用官方函数
        create_groundtruth_database(
            dataset_class_name='TYJTDataset',      # 您的数据集类名
            data_path=str(self.src_nuscenes_dir),  # 原始数据根目录（必须包含 samples/ 等）
            info_prefix='tyjt',                   # 与训练配置一致
            info_path=str(train_infos_path),      # 新生成的训练集 pkl
            database_save_path=str(self.dst_dir / "tyjt_gt_database"),
            db_info_save_path=str(self.dst_dir / "tyjt_dbinfos_train.pkl"),
            with_mask=False,
            load_augmented=None,
            relative_path=True
        )

        self.logger.info(f"✅ dbinfos 重新生成完成: {self.dst_dir}/tyjt_dbinfos_train.pkl")

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(
        description='根据场景列表重新生成PKL文件 - 简化版'
    )
    
    parser.add_argument('--src_nuscenes_dir', type=str, required=True,
                       help='nuscenes_tyjt数据集目录')
    parser.add_argument('--src_pkl_dir', type=str, required=True,
                       help='原始PKL文件目录')
    parser.add_argument('--src_train_scenes', type=str, required=True,
                       help='训练集场景列表文件')
    parser.add_argument('--src_val_scenes', type=str, required=True,
                       help='验证集场景列表文件')
    
    parser.add_argument('--dst_root', type=str, default='./output',
                       help='输出根目录')
    parser.add_argument('--dst_version', type=str, default='v02-tyjt-trainval',
                       help='输出版本名称')
    
    args = parser.parse_args()
    
    log_dir = Path(args.dst_root) / "logs"
    logger, log_file = setup_logging(log_dir)
    
    logger.info("=" * 60)
    logger.info("PKL重新生成工具 - 简化版")
    logger.info(f"源PKL目录: {args.src_pkl_dir}")
    logger.info(f"输出版本: {args.dst_version}")
    logger.info("=" * 60)
    
    try:
        regenerator = PklRegenerator(args, logger)
        train_path, val_path = regenerator.run()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ 处理完成!")
        logger.info(f"输出目录: {regenerator.dst_dir}")
        logger.info(f"日志文件: {log_file}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ 处理失败: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()