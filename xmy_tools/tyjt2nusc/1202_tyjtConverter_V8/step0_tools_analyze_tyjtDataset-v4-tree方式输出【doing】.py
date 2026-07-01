#!/usr/bin/env python3
"""
TYJT数据集分析脚本 - v4版本（修复目录树显示）
多线程分析，改进的日志机制，效率优化
"""

import json
import logging
from pathlib import Path
from collections import defaultdict
import concurrent.futures
import threading
from datetime import datetime
import sys
from typing import Dict, List, Set, Any

# ==================== 配置参数 ====================
Mode = "A100"

if Mode == "Local":
    TYJT_ROOT = "/mnt/dataset/tyjt_RawData_all/"
    DATA_PACKAGES_TO_PROCESS = [
        "2d3d4d_20250728_weiyuan", 
        "2d3d4d_20250913_weiyuan"
    ]
elif Mode == "A100":
    TYJT_ROOT = "/cephfsdata/users/lishan/00_Data/00_RawData"
    DATA_PACKAGES_TO_PROCESS = [
        "2d3d_20250114", "2d3d4d_20241218", "2d3d4d_20250618_weiyuan",
        "2d3d4d_20250728_weiyuan", "2d3d_20250218", "2d3d4d_20250117",
        "2d3d4d_20250218", "2d3d4d_20250403", "2d3d4d_20250708_weiyuan",
        "2d3d_20250221", "2d3d4d_20250213", "2d3d4d_20241122_wuxi",
        "2d3d4d_20250408",
    ]
else:
    print(f"❌【Error】: TYJT_ROOT & DATA_PACKAGES_TO_PROCESS: None")

# ==================== 全局配置 ====================
output_dir = Path("output/step1_v4/")
output_dir.mkdir(parents=True, exist_ok=True)

# 文件类型定义
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
LIDAR_EXTENSIONS = {'.npy', '.pcd', '.bin', '.ply'}
RADAR_EXTENSIONS = {'.npy', '.pcd', '.bin'}

# ==================== 日志配置 ====================
def setup_logging():
    """配置多日志系统"""
    # 主日志
    main_logger = logging.getLogger('main')
    main_logger.setLevel(logging.INFO)
    main_handler = logging.FileHandler(output_dir / 'main.log', encoding='utf-8')
    main_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    main_logger.addHandler(main_handler)
    
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    main_logger.addHandler(console_handler)
    
    return main_logger

def create_package_logger(package_name: str):
    """为每个数据包创建独立的日志器"""
    safe_name = package_name.replace('/', '_')
    logger = logging.getLogger(f'package_{safe_name}')
    logger.setLevel(logging.INFO)
    
    # 扁平清单日志
    flat_handler = logging.FileHandler(output_dir / f'{safe_name}-flat_list.log', encoding='utf-8')
    flat_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(flat_handler)
    
    # 目录结构日志
    tree_handler = logging.FileHandler(output_dir / f'{safe_name}-directory_tree.log', encoding='utf-8')
    tree_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(tree_handler)
    
    return logger, flat_handler, tree_handler

# ==================== 数据结构定义 ====================
class PackageAnalysisResult:
    """数据包分析结果"""
    def __init__(self, package_name: str):
        self.package_name = package_name
        self.calib_info = {}  # 标定文件信息
        self.sub_packets = []  # 子数据包信息
        self.top_level_folders = set()  # 顶层数据文件夹
        self.sensor_dirs = set()  # 传感器目录
        self.file_stats = defaultdict(lambda: defaultdict(int))  # 文件统计
        
    def to_dict(self):
        """转换为字典格式"""
        return {
            'package_name': self.package_name,
            'calib_info': self.calib_info,
            'sub_packet_count': len(self.sub_packets),
            'top_level_folders': list(self.top_level_folders),
            'sensor_dirs': list(self.sensor_dirs),
            'file_stats': dict(self.file_stats)
        }

class GlobalStats:
    """全局统计"""
    def __init__(self):
        self.total_packages = 0
        self.total_sub_packets = 0
        self.total_lidar_files = 0
        self.total_image_files = 0
        self.package_results = {}
        self.combinations = defaultdict(list)

# ==================== 工具函数 ====================
def safe_iterdir(path: Path):
    """安全遍历目录"""
    try:
        return list(path.iterdir())
    except (PermissionError, OSError) as e:
        return []

def safe_exists(path: Path) -> bool:
    """安全检查路径是否存在"""
    try:
        return path.exists()
    except (PermissionError, OSError):
        return False

def safe_json_load(path: Path) -> Any:
    """安全加载JSON文件"""
    try:
        if not safe_exists(path) or not path.is_file():
            return None
            
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return None
            return json.loads(content)
    except (json.JSONDecodeError, PermissionError, OSError) as e:
        return None

def count_files_in_dir(directory: Path) -> Dict[str, int]:
    """统计目录中的文件类型"""
    file_counts = defaultdict(int)
    try:
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                if suffix:
                    file_counts[suffix] += 1
    except (PermissionError, OSError):
        pass
    return file_counts

# ==================== 分析核心函数 ====================
def analyze_calibration(package_path: Path, flat_handler) -> Dict[str, Any]:
    """分析标定文件结构"""
    calib_info = {}
    calib_dir = package_path / "calib"
    
    if not safe_exists(calib_dir):
        flat_handler.stream.write("  ⚠️ 标定目录不存在\n")
        return calib_info
    
    calib_sub_dirs = safe_iterdir(calib_dir)
    total_json_files = 0
    
    flat_handler.stream.write("📋 标定文件分析:\n")
    flat_handler.stream.write("-" * 40 + "\n")
    
    for calib_sub_dir in calib_sub_dirs:
        if calib_sub_dir.is_dir():
            calib_files_info = {}
            json_files = list(calib_sub_dir.glob("*.json"))
            total_json_files += len(json_files)
            
            flat_handler.stream.write(f"  📁 标定子目录: {calib_sub_dir.name}\n")
            flat_handler.stream.write(f"     JSON文件数量: {len(json_files)}个\n")
            
            for json_file in json_files:
                calib_data = safe_json_load(json_file)
                if calib_data and isinstance(calib_data, dict):
                    keys = list(calib_data.keys())
                    calib_files_info[json_file.name] = {
                        'keys': keys,
                        'key_count': len(keys)
                    }
                    flat_handler.stream.write(f"     ├── {json_file.name}\n")
                    flat_handler.stream.write(f"     │   ├── Keys数量: {len(keys)}个\n")
                    flat_handler.stream.write(f"     │   └── Keys列表:\n")
                    
                    # 将keys按类别分组显示
                    sorted_keys = sorted(keys)
                    
                    # 分组显示keys
                    camera_keys = [k for k in sorted_keys if k.startswith('SC_')]
                    radar_keys = [k for k in sorted_keys if k.startswith('R') and not k.startswith('R0_')]
                    lidar_keys = [k for k in sorted_keys if k.startswith('R0_')]
                    other_keys = [k for k in sorted_keys if k not in camera_keys + radar_keys + lidar_keys]
                    
                    # 显示相机keys
                    if camera_keys:
                        flat_handler.stream.write(f"     │       📷 相机:\n")
                        for i, key in enumerate(camera_keys):
                            connector = "└── " if i == len(camera_keys) - 1 and not (radar_keys or lidar_keys or other_keys) else "├── "
                            flat_handler.stream.write(f"     │           {connector}{key}\n")
                    
                    # 显示雷达keys
                    if radar_keys:
                        flat_handler.stream.write(f"     │       📡 雷达:\n")
                        for i, key in enumerate(radar_keys):
                            connector = "└── " if i == len(radar_keys) - 1 and not (lidar_keys or other_keys) else "├── "
                            flat_handler.stream.write(f"     │           {connector}{key}\n")
                    
                    # 显示激光雷达keys
                    if lidar_keys:
                        flat_handler.stream.write(f"     │       🎯 激光雷达:\n")
                        for i, key in enumerate(lidar_keys):
                            connector = "└── " if i == len(lidar_keys) - 1 and not other_keys else "├── "
                            flat_handler.stream.write(f"     │           {connector}{key}\n")
                    
                    # 显示其他keys
                    if other_keys:
                        flat_handler.stream.write(f"     │       🔧 其他:\n")
                        for i, key in enumerate(other_keys):
                            connector = "└── " if i == len(other_keys) - 1 else "├── "
                            flat_handler.stream.write(f"     │           {connector}{key}\n")
                            
                else:
                    flat_handler.stream.write(f"     ├── {json_file.name} ❌ 读取失败\n")
            
            calib_info[calib_sub_dir.name] = calib_files_info
            flat_handler.stream.write("\n")
    
    flat_handler.stream.write(f"📊 标定文件统计:\n")
    flat_handler.stream.write(f"  • 标定子目录: {len(calib_sub_dirs)}个\n")
    flat_handler.stream.write(f"  • JSON文件总数: {total_json_files}个\n")
    flat_handler.stream.write("\n")
    
    return calib_info


def analyze_sub_packet_structure(sub_packet: Path) -> Dict[str, Any]:
    """分析子数据包结构"""
    structure = {
        'name': sub_packet.name,
        'top_level_folders': set(),
        'sensor_dirs': set(),
        'file_stats': defaultdict(lambda: defaultdict(int)),
        'total_files': 0
    }
    
    # 只分析第一层目录
    for item in safe_iterdir(sub_packet):
        if item.is_dir():
            dir_name = item.name
            structure['top_level_folders'].add(dir_name)
            
            # 识别传感器目录
            if any(dir_name.startswith(prefix) for prefix in ['SC_', 'CAM_', 'Camera_', 'camera_']):
                structure['sensor_dirs'].add(dir_name)
            
            # 统计文件
            file_counts = count_files_in_dir(item)
            for suffix, count in file_counts.items():
                structure['file_stats'][dir_name][suffix] += count
                structure['total_files'] += count
    
    return structure

def print_sub_packet_tree(sub_packet_structure: Dict[str, Any], tree_handler, is_last: bool = False):
    """打印子数据包树形结构（修复版）"""
    connector = "└── " if is_last else "├── "
    tree_handler.stream.write(f"      {connector}{sub_packet_structure['name']}\n")
    
    folders = sorted(sub_packet_structure['top_level_folders'])
    for i, folder in enumerate(folders):
        is_last_folder = (i == len(folders) - 1)
        folder_connector = "    └── " if is_last_folder else "    ├── "
        
        if folder in sub_packet_structure['file_stats']:
            file_info = []
            for suffix, count in sorted(sub_packet_structure['file_stats'][folder].items()):
                file_info.append(f"{suffix}:{count}")
            
            if file_info:
                file_str = ', '.join(file_info)
                tree_handler.stream.write(f"      {folder_connector}{folder} (文件: {file_str})\n")
            else:
                tree_handler.stream.write(f"      {folder_connector}{folder} (无文件)\n")
    
    # 文件总数统计
    total_connector = "    └── " if not folders else "        "
    tree_handler.stream.write(f"      {total_connector}📊 总计: {sub_packet_structure['total_files']}个文件\n")

def analyze_single_package(package_path: Path, package_name: str, main_logger, global_stats) -> PackageAnalysisResult:
    """分析单个数据包（线程安全）"""
    result = PackageAnalysisResult(package_name)
    package_logger, flat_handler, tree_handler = create_package_logger(package_name)
    
    main_logger.info(f"🔍 开始分析数据包: {package_name}")
    
    # 写入flat_list头部
    flat_handler.stream.write(f"📦 数据包: {package_name}\n")
    flat_handler.stream.write("=" * 60 + "\n\n")
    
    # 分析标定文件
    result.calib_info = analyze_calibration(package_path, flat_handler)
    
    # 分析数据文件
    datasets_dir = package_path / "datasets"
    if safe_exists(datasets_dir):
        sub_packets = [d for d in safe_iterdir(datasets_dir) if d.is_dir()]
        result.sub_packets = sub_packets
        
        flat_handler.stream.write("📁 数据集分析\n")
        flat_handler.stream.write("-" * 40 + "\n")
        flat_handler.stream.write(f"子数据包数量: {len(sub_packets)}个\n\n")
        
        # 收集所有顶层文件夹
        all_top_folders = set()
        
        # 写入目录结构日志头部
        tree_handler.stream.write(f"📦 数据包: {package_name}\n")
        tree_handler.stream.write("=" * 60 + "\n")
        tree_handler.stream.write(f"  └── calib/\n")
        
        # 标定文件结构
        for calib_sub_dir, calib_files in result.calib_info.items():
            tree_handler.stream.write(f"      └── {calib_sub_dir}/\n")
            for json_file, info in calib_files.items():
                status = "✅" if info['key_count'] > 0 else "❌"
                tree_handler.stream.write(f"          ├── {json_file} {status} ({info['key_count']}个keys)\n")
        
        tree_handler.stream.write(f"  └── datasets/ (共{len(sub_packets)}个子数据包)\n")
        
        for i, sub_packet in enumerate(sub_packets, 1):
            main_logger.info(f"    📂 处理子数据包 {i}/{len(sub_packets)}: {sub_packet.name}")
            
            # 分析子数据包结构
            sub_structure = analyze_sub_packet_structure(sub_packet)
            
            # 更新统计
            result.top_level_folders.update(sub_structure['top_level_folders'])
            result.sensor_dirs.update(sub_structure['sensor_dirs'])
            all_top_folders.update(sub_structure['top_level_folders'])
            
            # 文件统计
            for folder, stats in sub_structure['file_stats'].items():
                for suffix, count in stats.items():
                    result.file_stats[folder][suffix] += count
                    
                    # 全局文件统计
                    if suffix in IMAGE_EXTENSIONS:
                        with threading.Lock():
                            global_stats.total_image_files += count
                    elif suffix in LIDAR_EXTENSIONS or suffix in RADAR_EXTENSIONS:
                        with threading.Lock():
                            global_stats.total_lidar_files += count
            
            # 记录到目录结构
            is_last_subpacket = (i == len(sub_packets))
            tree_handler.stream.write(f"      ├── {sub_packet.name}【子数据包{i}/{len(sub_packets)}】\n")
            print_sub_packet_tree(sub_structure, tree_handler, is_last_subpacket)
        
        tree_handler.stream.write(f"      └── [本数据包结束]\n\n")
        
        # 在flat_list中记录顶层文件夹信息
        flat_handler.stream.write("📊 顶层文件夹统计:\n")
        flat_handler.stream.write(f"  • 顶层文件夹总数: {len(all_top_folders)}个\n")
        flat_handler.stream.write(f"  • 顶层文件夹列表:\n")
        for folder in sorted(all_top_folders):
            flat_handler.stream.write(f"     ├── {folder}\n")
        
        flat_handler.stream.write("\n")
        
        # 传感器目录统计
        flat_handler.stream.write("📷 传感器目录统计:\n")
        flat_handler.stream.write(f"  • 传感器目录总数: {len(result.sensor_dirs)}个\n")
        if result.sensor_dirs:
            flat_handler.stream.write(f"  • 传感器目录列表:\n")
            for sensor in sorted(result.sensor_dirs):
                flat_handler.stream.write(f"     ├── {sensor}\n")
        else:
            flat_handler.stream.write(f"  • 无传感器目录\n")
        
    else:
        flat_handler.stream.write("📁 数据集分析\n")
        flat_handler.stream.write("-" * 40 + "\n")
        flat_handler.stream.write("  ⚠️ datasets目录不存在\n")
        tree_handler.stream.write(f"📦 数据包: {package_name}\n")
        tree_handler.stream.write("=" * 60 + "\n")
        tree_handler.stream.write("  └── datasets/ ⚠️ 不存在\n")
    
    # 清理日志处理器
    for handler in package_logger.handlers[:]:
        handler.close()
        package_logger.removeHandler(handler)
    
    main_logger.info(f"✅ 完成分析数据包: {package_name}")
    return result

def analyze_combinations(global_stats: GlobalStats, main_logger):
    """分析组合方式"""
    main_logger.info("🎯 开始分析组合方式...")
    
    for package_name, result in global_stats.package_results.items():
        # 获取标定键组合
        calib_keys = set()
        for calib_info in result.calib_info.values():
            for json_info in calib_info.values():
                calib_keys.update(json_info.get('keys', []))
        
        # 组合键
        combination_key = (
            tuple(sorted(calib_keys)),
            tuple(sorted(result.top_level_folders)),
            tuple(sorted(result.sensor_dirs))
        )
        
        global_stats.combinations[combination_key].append(package_name)
    
    main_logger.info(f"📊 发现 {len(global_stats.combinations)} 种不同的组合方式")

# ==================== 主函数 ====================
def main():
    """主分析函数"""
    start_time = datetime.now()
    main_logger = setup_logging()
    global_stats = GlobalStats()
    
    main_logger.info("🚀 TYJT数据集分析工具 v4")
    main_logger.info(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    main_logger.info(f"数据根目录: {TYJT_ROOT}")
    main_logger.info(f"模式: {Mode}")
    main_logger.info(f"输出目录: {output_dir}")
    main_logger.info("")
    
    root_path = Path(TYJT_ROOT)
    if not safe_exists(root_path):
        main_logger.error(f"❌ 数据集根目录不可访问: {TYJT_ROOT}")
        return
    
    # 获取目标数据包
    all_packages = [d for d in safe_iterdir(root_path) if d.is_dir() and not d.name.startswith('.')]
    target_packages = [pkg for pkg in all_packages if pkg.name in DATA_PACKAGES_TO_PROCESS]
    
    main_logger.info(f"📁 发现 {len(all_packages)} 个数据包，指定分析 {len(target_packages)} 个")
    main_logger.info("")
    
    # 多线程分析数据包
    main_logger.info("🔄 开始多线程分析数据包...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_package = {
            executor.submit(analyze_single_package, package, package.name, main_logger, global_stats): package
            for package in target_packages
        }
        
        for future in concurrent.futures.as_completed(future_to_package):
            package = future_to_package[future]
            try:
                result = future.result()
                global_stats.package_results[result.package_name] = result
                global_stats.total_packages += 1
                global_stats.total_sub_packets += len(result.sub_packets)
                
                main_logger.info(f"✅ 已完成: {result.package_name} "
                               f"(子数据包: {len(result.sub_packets)}, "
                               f"传感器: {len(result.sensor_dirs)})")
            except Exception as e:
                main_logger.error(f"❌ 分析失败 {package.name}: {e}")
    
    # 分析组合方式
    analyze_combinations(global_stats, main_logger)
    
    # 生成统计摘要
    main_logger.info("\n📊 总体统计摘要")
    main_logger.info("=" * 50)
    main_logger.info(f"处理的数据包总数: {global_stats.total_packages}")
    main_logger.info(f"处理的子数据包总数: {global_stats.total_sub_packets}")
    main_logger.info(f"发现的LiDAR文件总数: {global_stats.total_lidar_files}")
    main_logger.info(f"发现的图像文件总数: {global_stats.total_image_files}")
    main_logger.info(f"组合方式总数: {len(global_stats.combinations)}")
    
    # 保存组合方式结果
    with open(output_dir / 'combinations.json', 'w', encoding='utf-8') as f:
        json.dump({
            'combinations': {
                str(k): v for k, v in global_stats.combinations.items()
            },
            'summary': {
                'total_packages': global_stats.total_packages,
                'total_sub_packets': global_stats.total_sub_packets,
                'total_lidar_files': global_stats.total_lidar_files,
                'total_image_files': global_stats.total_image_files,
                'combination_count': len(global_stats.combinations)
            }
        }, f, ensure_ascii=False, indent=2)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    main_logger.info("")
    main_logger.info("✅ 分析完成!")
    main_logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    main_logger.info(f"总耗时: {duration}")
    main_logger.info(f"输出文件保存在: {output_dir}")

if __name__ == "__main__":
    main()