#!/usr/bin/env python3
import os
import shutil
import yaml
from datetime import datetime
from pathlib import Path

def fix_sweeps_in_pipeline():
    """修复所有pipeline中的sweeps_num参数 - 减少时序帧数降低内存"""
    base_path = Path("/mnt/bevfusion_mit_xmy")
    config_file = base_path / "configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/tyjt_convfuser.yaml"
    history_dir = base_path / "configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/history"
    
    print(f"📁 配置文件: {config_file}")
    print("📝 修改目的: 减少时序sweep数量，降低点云数据内存占用")
    print("📝 修改原因: 原配置使用9帧时序数据，内存消耗过大")
    
    if not config_file.exists():
        print("❌ 配置文件不存在")
        return
    
    # 备份
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_file = history_dir / f"{config_file.name}.before_sweeps_fix.{timestamp}"
    shutil.copy2(config_file, backup_file)
    print(f"✅ 备份完成: {backup_file}")
    
    # 加载配置
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # 修改所有pipeline中的sweeps_num
    pipelines_to_fix = [
        ['data', 'test', 'pipeline'],
        ['data', 'train', 'dataset', 'pipeline'], 
        ['data', 'val', 'pipeline'],
        ['evaluation', 'pipeline'],
        ['test_pipeline'],
        ['train_pipeline']
    ]
    
    modified_count = 0
    for pipeline_path in pipelines_to_fix:
        current = config_data
        valid_path = True
        
        # 导航到pipeline位置
        for key in pipeline_path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                valid_path = False
                break
        
        if valid_path and isinstance(current, list):
            for step in current:
                if isinstance(step, dict) and 'sweeps_num' in step:
                    if step['sweeps_num'] == 9:
                        step['sweeps_num'] = 3
                        modified_count += 1
                        print(f"✅ 修改 {'.'.join(pipeline_path)} sweeps_num: 9 → 3 # {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 内存优化：减少时序帧数从9到3")
    
    # 保存修改
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, indent=2, width=120)
    
    print(f"🎉 Sweep数量优化完成，修改了 {modified_count} 个位置")

if __name__ == "__main__":
    fix_sweeps_in_pipeline()
