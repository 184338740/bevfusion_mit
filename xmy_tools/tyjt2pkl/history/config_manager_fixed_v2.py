#!/usr/bin/env python3
import os
import shutil
import yaml
from datetime import datetime
from pathlib import Path

class ConfigManager:
    def __init__(self):
        self.base_path = Path("/mnt/bevfusion_mit_xmy").absolute()
        self.config_file = self.base_path / "configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/tyjt_convfuser.yaml"
        self.history_dir = self.base_path / "configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/history"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 工作目录: {os.getcwd()}")
        print(f"📁 根目录: {self.base_path}")
        print(f"📁 配置文件: {self.config_file}")
        print(f"📁 备份目录: {self.history_dir}")
        
    def check_config_exists(self):
        """检查配置文件是否存在"""
        if self.config_file.exists():
            print(f"✅ 配置文件存在: {self.config_file}")
            return True
        else:
            print(f"❌ 配置文件不存在: {self.config_file}")
            return False
        
    def backup_config(self, suffix):
        """备份配置文件，只添加后缀"""
        if not self.config_file.exists():
            print(f"❌ 配置文件不存在: {self.config_file}")
            return None
            
        # 创建备份文件名（只添加后缀）
        backup_file = self.history_dir / f"{self.config_file.name}.{suffix}"
        
        # 复制文件
        shutil.copy2(self.config_file, backup_file)
        print(f"✅ 备份完成: {backup_file}")
        return backup_file
    
    def load_config(self):
        """加载YAML配置"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            return None
    
    def save_config(self, data):
        """保存YAML配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2, width=120)
            return True
        except Exception as e:
            print(f"❌ 保存配置文件失败: {e}")
            return False
    
    def apply_memory_optimizations(self):
        """应用内存优化修改 - 修复列表索引访问问题"""
        if not self.check_config_exists():
            return False
            
        # 备份原配置
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.backup_config(f"before_memory_optimization.{timestamp}")
        
        # 加载配置
        config_data = self.load_config()
        if not config_data:
            print("❌ 无法加载配置文件")
            return False
        
        print("🔧 应用内存优化修改...")
        
        # 应用内存优化修改
        modifications = [
            # 图像尺寸优化 - 修复CUDA内存不足问题
            (['image_size', 0], 224, "内存优化：图像高度从256减小到224，降低显存占用"),
            (['image_size', 1], 640, "内存优化：图像宽度从704减小到640，降低显存占用"),
            
            # BEV范围优化 - 减小BEV特征图尺寸，降低内存消耗
            (['model', 'encoders', 'camera', 'vtransform', 'xbound', 0], -40.0, "内存优化：X范围从-54减小到-40，减小BEV网格"),
            (['model', 'encoders', 'camera', 'vtransform', 'xbound', 1], 40.0, "内存优化：X范围从54减小到40，减小BEV网格"),
            (['model', 'encoders', 'camera', 'vtransform', 'xbound', 2], 0.5, "内存优化：X步长从0.3增加到0.5，减少网格数量"),
            
            (['model', 'encoders', 'camera', 'vtransform', 'ybound', 0], -40.0, "内存优化：Y范围从-54减小到-40，减小BEV网格"),
            (['model', 'encoders', 'camera', 'vtransform', 'ybound', 1], 40.0, "内存优化：Y范围从54减小到40，减小BEV网格"),
            (['model', 'encoders', 'camera', 'vtransform', 'ybound', 2], 0.5, "内存优化：Y步长从0.3增加到0.5，减少网格数量"),
            
            # 深度范围优化 - 大幅减少深度bin数量，降低LSS变换内存
            (['model', 'encoders', 'camera', 'vtransform', 'dbound', 0], 2.0, "内存优化：起始深度从1.0增加到2.0，减少深度bin"),
            (['model', 'encoders', 'camera', 'vtransform', 'dbound', 1], 50.0, "内存优化：最大深度从60.0减小到50.0，减少深度bin"),
            (['model', 'encoders', 'camera', 'vtransform', 'dbound', 2], 2.0, "内存优化：深度步长从0.5增加到2.0，大幅减少深度bin数量"),
            
            # 输出通道优化 - 减少特征图通道数，降低计算量
            (['model', 'encoders', 'camera', 'vtransform', 'out_channels'], 64, "内存优化：输出通道从80减少到64，降低特征图通道数"),
            
            # 下采样优化 - 增加下采样率，减小特征图空间尺寸
            (['model', 'encoders', 'camera', 'vtransform', 'downsample'], 4, "内存优化：下采样从2增加到4，减小特征图尺寸"),
            
            # 特征图尺寸优化 - 适配新的图像尺寸
            (['model', 'encoders', 'camera', 'vtransform', 'feature_size', 0], 28, "内存优化：特征图高度从32减小到28"),
            (['model', 'encoders', 'camera', 'vtransform', 'feature_size', 1], 80, "内存优化：特征图宽度从88减小到80"),
            
            # Voxel数量优化 - 减少点云体素化内存占用
            (['model', 'encoders', 'lidar', 'voxelize', 'max_voxels', 0], 80000, "内存优化：最大voxels从120000减少到80000"),
            (['model', 'encoders', 'lidar', 'voxelize', 'max_voxels', 1], 120000, "内存优化：最大voxels从160000减少到120000"),
            
            # 工作进程优化 - 减少数据加载进程，降低CPU内存
            (['data', 'workers_per_gpu'], 1, "内存优化：工作进程从2减少到1，降低数据加载内存"),
        ]
        
        # 应用修改
        success_count = 0
        for path, new_value, comment in modifications:
            if self._set_nested_value(config_data, path, new_value, comment):
                success_count += 1
        
        # 添加梯度累积 - 模拟更大batch size而不增加显存
        if 'optimizer_config' in config_data:
            config_data['optimizer_config']['cumulative_iters'] = 2
            print("✅ 添加梯度累积: 2次迭代 - 内存优化：模拟更大batch size而不增加显存")
            success_count += 1
        
        # 保存修改后的配置
        if self.save_config(config_data):
            # 备份修改后的配置
            self.backup_config(f"after_memory_optimization.{timestamp}")
            print(f"🎉 内存优化配置修改完成! 成功应用 {success_count} 项优化")
            return True
        else:
            return False
    
    def _set_nested_value(self, data, path, value, comment):
        """设置嵌套值 - 修复列表索引访问问题"""
        current = data
        path_str = '.'.join(map(str, path))
        
        try:
            # 导航到目标位置
            for i, key in enumerate(path[:-1]):
                if isinstance(current, dict) and key in current:
                    current = current[key]
                elif isinstance(current, list) and isinstance(key, int) and 0 <= key < len(current):
                    current = current[key]
                else:
                    print(f"❌ 路径不存在: {path_str} (在 {key} 处)")
                    return False
            
            # 设置最终值
            final_key = path[-1]
            if isinstance(current, dict):
                old_value = current.get(final_key)
                current[final_key] = value
            elif isinstance(current, list) and isinstance(final_key, int) and 0 <= final_key < len(current):
                old_value = current[final_key]
                current[final_key] = value
            else:
                print(f"❌ 无法设置值: {path_str}")
                return False
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"✅ 修改 {path_str}: {old_value} → {value} # {timestamp} - {comment}")
            return True
            
        except Exception as e:
            print(f"❌ 修改失败 {path_str}: {e}")
            return False
    
    def list_backups(self):
        """列出所有备份文件"""
        print("📁 备份文件列表:")
        for backup_file in sorted(self.history_dir.glob("*.yaml.*")):
            print(f"  {backup_file.name}")

if __name__ == "__main__":
    manager = ConfigManager()
    
    print("🚀 开始配置内存优化...")
    print("📝 修改目的: 解决CUDA内存不足问题，使BEVFusion能在8GB GPU上运行")
    print("📝 修改原因: 原配置针对A100等大显存卡，TYJT环境需要内存优化")
    
    if manager.apply_memory_optimizations():
        print("\n📋 修改摘要:")
        print("   - 图像尺寸: 256×704 → 224×640 (降低图像处理内存)")
        print("   - BEV范围: [-54,54,0.3] → [-40,40,0.5] (减小BEV特征图)") 
        print("   - 深度范围: [1,60,0.5] → [2,50,2.0] (大幅减少深度bin)")
        print("   - 输出通道: 80 → 64 (减少特征通道)")
        print("   - 下采样: 2 → 4 (增加特征图下采样)")
        print("   - 特征图: 32×88 → 28×80 (适配新图像尺寸)")
        print("   - Voxel数量: 120000/160000 → 80000/120000 (减少点云内存)")
        print("   - 工作进程: 2 → 1 (减少数据加载内存)")
        print("   - 梯度累积: 添加2次迭代 (模拟大batch训练)")
        
        print("\n📊 备份文件:")
        manager.list_backups()
    else:
        print("❌ 配置优化失败")
