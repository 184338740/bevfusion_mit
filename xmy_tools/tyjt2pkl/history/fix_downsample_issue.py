#!/usr/bin/env python3
import os
import shutil
import yaml
from datetime import datetime
from pathlib import Path

def fix_downsample_issue():
    """修复downsample参数问题 - 恢复为2，使用其他方式优化内存"""
    base_path = Path("/mnt/bevfusion_mit_xmy")
    config_file = base_path / "configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/tyjt_convfuser.yaml"
    history_dir = base_path / "configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/history"
    
    print("🔧 修复downsample参数问题...")
    print("📝 问题: DepthLSSTransform只支持downsample=2")
    print("📝 解决方案: 恢复downsample=2，通过其他参数优化内存")
    
    if not config_file.exists():
        print("❌ 配置文件不存在")
        return
    
    # 备份当前配置
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_file = history_dir / f"tyjt_convfuser_fix_downsample_{timestamp}.yaml"
    shutil.copy2(config_file, backup_file)
    print(f"✅ 备份完成: {backup_file}")
    
    # 加载配置
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # 修复downsample参数
    if 'model' in config_data and 'encoders' in config_data['model']:
        camera_vtransform = config_data['model']['encoders']['camera']['vtransform']
        if camera_vtransform.get('downsample') == 4:
            camera_vtransform['downsample'] = 2
            print("✅ 修复downsample: 4 → 2")
    
    # 进一步优化其他参数来补偿内存
    optimizations = [
        # 进一步减小图像尺寸
        (['image_size', 0], 192, "内存优化：进一步减小图像高度到192"),
        (['image_size', 1], 512, "内存优化：进一步减小图像宽度到512"),
        
        # 进一步减小BEV范围
        (['model', 'encoders', 'camera', 'vtransform', 'xbound', 0], -32.0, "内存优化：X范围进一步减小到-32"),
        (['model', 'encoders', 'camera', 'vtransform', 'xbound', 1], 32.0, "内存优化：X范围进一步减小到32"),
        (['model', 'encoders', 'camera', 'vtransform', 'xbound', 2], 0.8, "内存优化：X步长增加到0.8"),
        
        (['model', 'encoders', 'camera', 'vtransform', 'ybound', 0], -32.0, "内存优化：Y范围进一步减小到-32"),
        (['model', 'encoders', 'camera', 'vtransform', 'ybound', 1], 32.0, "内存优化：Y范围进一步减小到32"),
        (['model', 'encoders', 'camera', 'vtransform', 'ybound', 2], 0.8, "内存优化：Y步长增加到0.8"),
        
        # 进一步减小深度范围
        (['model', 'encoders', 'camera', 'vtransform', 'dbound', 0], 3.0, "内存优化：起始深度增加到3.0"),
        (['model', 'encoders', 'camera', 'vtransform', 'dbound', 1], 45.0, "内存优化：最大深度减小到45.0"),
        (['model', 'encoders', 'camera', 'vtransform', 'dbound', 2], 3.0, "内存优化：深度步长增加到3.0"),
        
        # 进一步减小输出通道
        (['model', 'encoders', 'camera', 'vtransform', 'out_channels'], 48, "内存优化：输出通道进一步减小到48"),
        
        # 进一步减小特征图尺寸
        (['model', 'encoders', 'camera', 'vtransform', 'feature_size', 0], 24, "内存优化：特征图高度减小到24"),
        (['model', 'encoders', 'camera', 'vtransform', 'feature_size', 1], 64, "内存优化：特征图宽度减小到64"),
        
        # 进一步减小voxel数量
        (['model', 'encoders', 'lidar', 'voxelize', 'max_voxels', 0], 60000, "内存优化：最大voxels减小到60000"),
        (['model', 'encoders', 'lidar', 'voxelize', 'max_voxels', 1], 80000, "内存优化：最大voxels减小到80000"),
    ]
    
    # 应用优化
    for path, new_value, comment in optimizations:
        current = config_data
        for key in path[:-1]:
            if key in current:
                current = current[key]
            else:
                break
        if isinstance(current, dict) and path[-1] in current:
            old_value = current[path[-1]]
            current[path[-1]] = new_value
            print(f"✅ 优化 {'.'.join(path)}: {old_value} → {new_value} - {comment}")
    
    # 保存修复后的配置
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, indent=2, width=120)
    
    print("🎉 downsample问题修复完成!")
    print("📋 新的优化方案:")
    print("   - 图像尺寸: 192×512 (进一步减小)")
    print("   - BEV范围: [-32,32,0.8] (进一步减小)")
    print("   - 深度范围: [3,45,3.0] (进一步优化)")
    print("   - 输出通道: 48 (进一步减小)")
    print("   - 特征图: 24×64 (进一步减小)")
    print("   - Voxel数量: 60000/80000 (进一步减小)")

if __name__ == "__main__":
    fix_downsample_issue()
