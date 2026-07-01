#!/usr/bin/env python3
import os
import shutil
import yaml
from datetime import datetime
from pathlib import Path

def restore_official_with_safe_optimizations():
    """恢复官方配置，只应用最安全的内存优化"""
    base_path = Path("/mnt/bevfusion_mit_xmy")
    config_file = base_path / "configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/tyjt_convfuser.yaml"
    history_dir = base_path / "configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/history"
    official_config = base_path / "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
    
    print("🔄 恢复官方配置并应用安全优化...")
    print("📝 策略: 使用官方BEV参数，只修改不影响形状的参数")
    
    if not official_config.exists():
        print("❌ 官方配置文件不存在，使用当前配置")
        official_config = config_file
    
    # 备份当前配置
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_file = history_dir / f"tyjt_convfuser_restore_official_{timestamp}.yaml"
    shutil.copy2(config_file, backup_file)
    print(f"✅ 备份当前配置: {backup_file}")
    
    # 加载官方配置
    with open(official_config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    print("✅ 已加载官方配置")
    
    # 只应用最安全的内存优化
    safe_optimizations = [
        # 数据路径 - 必须修改
        (['data', 'train', 'dataset', 'ann_file'], '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_train_fixed.pkl', "数据集：TYJT训练数据路径"),
        (['data', 'train', 'dataset', 'data_root'], '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/', "数据集：TYJT数据根目录"),
        (['data', 'val', 'ann_file'], '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_val_fixed.pkl', "数据集：TYJT验证数据路径"),
        (['data', 'val', 'data_root'], '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/', "数据集：TYJT验证数据根目录"),
        (['data', 'test', 'ann_file'], '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_val_fixed.pkl', "数据集：TYJT测试数据路径"),
        (['data', 'test', 'data_root'], '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/', "数据集：TYJT测试数据根目录"),
        
        # 数据集类型
        (['data', 'train', 'dataset', 'type'], 'TYJTDataset', "数据集：TYJT数据集类型"),
        (['data', 'val', 'type'], 'TYJTDataset', "数据集：TYJT验证集类型"),
        (['data', 'test', 'type'], 'TYJTDataset', "数据集：TYJT测试集类型"),
        (['dataset_type'], 'TYJTDataset', "数据集：TYJT数据集类型"),
        
        # 最安全的内存优化
        (['data', 'workers_per_gpu'], 1, "内存优化：工作进程从2减少到1"),
        (['optimizer_config', 'cumulative_iters'], 2, "内存优化：梯度累积2次迭代"),
        
        # 运行目录
        (['run_dir'], 'runs/train-tyjt-official-safe', "训练：运行目录"),
    ]
    
    # 应用安全优化
    for path, new_value, comment in safe_optimizations:
        current = config_data
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        old_value = current.get(path[-1])
        current[path[-1]] = new_value
        print(f"✅ {'.'.join(path)}: {old_value} → {new_value} - {comment}")
    
    # 保存配置
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, indent=2, width=120)
    
    print("🎉 官方配置恢复完成!")
    print("📋 应用的安全优化:")
    print("   - TYJT数据路径配置")
    print("   - 工作进程: 2 → 1")
    print("   - 梯度累积: 2次迭代")
    print("   - 保持所有BEV参数不变")

if __name__ == "__main__":
    restore_official_with_safe_optimizations()
