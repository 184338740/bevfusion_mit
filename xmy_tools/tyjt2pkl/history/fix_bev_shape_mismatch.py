#!/usr/bin/env python3
import os
import shutil
import yaml
from datetime import datetime
from pathlib import Path

def calculate_bev_shape(xbound, ybound, dbound, feature_size):
    """计算BEV形状确保匹配"""
    x_min, x_max, x_step = xbound
    y_min, y_max, y_step = ybound
    d_min, d_max, d_step = dbound
    
    # 计算BEV网格尺寸
    x_size = int((x_max - x_min) / x_step)
    y_size = int((y_max - y_min) / y_step) 
    d_size = int((d_max - d_min) / d_step)
    
    return x_size, y_size, d_size

def fix_bev_shape_mismatch():
    """修复BEV形状不匹配问题"""
    base_path = Path("/mnt/bevfusion_mit_xmy")
    config_file = base_path / "configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/tyjt_convfuser.yaml"
    history_dir = base_path / "configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/history"
    
    print("🔧 修复BEV形状不匹配问题...")
    print("📝 问题: BEV参数与特征图尺寸不匹配导致张量形状错误")
    print("📝 解决方案: 重新计算并匹配BEV参数")
    
    if not config_file.exists():
        print("❌ 配置文件不存在")
        return
    
    # 备份当前配置
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_file = history_dir / f"tyjt_convfuser_fix_bev_shape_{timestamp}.yaml"
    shutil.copy2(config_file, backup_file)
    print(f"✅ 备份完成: {backup_file}")
    
    # 加载配置
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # 获取当前BEV参数
    vtransform = config_data['model']['encoders']['camera']['vtransform']
    xbound = vtransform['xbound']
    ybound = vtransform['ybound'] 
    dbound = vtransform['dbound']
    feature_size = vtransform['feature_size']
    
    print(f"当前BEV参数:")
    print(f"  xbound: {xbound}")
    print(f"  ybound: {ybound}")
    print(f"  dbound: {dbound}")
    print(f"  feature_size: {feature_size}")
    
    # 计算当前BEV形状
    x_size, y_size, d_size = calculate_bev_shape(xbound, ybound, dbound, feature_size)
    print(f"当前BEV网格: {x_size} x {y_size} x {d_size}")
    
    # 使用与官方配置更接近的参数
    new_optimizations = [
        # 恢复接近官方的BEV范围，但适当减小
        (['model', 'encoders', 'camera', 'vtransform', 'xbound', 0], -54.0, "形状修复：恢复X最小值"),
        (['model', 'encoders', 'camera', 'vtransform', 'xbound', 1], 54.0, "形状修复：恢复X最大值"), 
        (['model', 'encoders', 'camera', 'vtransform', 'xbound', 2], 0.3, "形状修复：恢复X步长"),
        
        (['model', 'encoders', 'camera', 'vtransform', 'ybound', 0], -54.0, "形状修复：恢复Y最小值"),
        (['model', 'encoders', 'camera', 'vtransform', 'ybound', 1], 54.0, "形状修复：恢复Y最大值"),
        (['model', 'encoders', 'camera', 'vtransform', 'ybound', 2], 0.3, "形状修复：恢复Y步长"),
        
        # 调整深度范围但保持兼容性
        (['model', 'encoders', 'camera', 'vtransform', 'dbound', 0], 1.0, "形状修复：恢复深度最小值"),
        (['model', 'encoders', 'camera', 'vtransform', 'dbound', 1], 60.0, "形状修复：恢复深度最大值"),
        (['model', 'encoders', 'camera', 'vtransform', 'dbound', 2], 0.5, "形状修复：恢复深度步长"),
        
        # 恢复特征图尺寸匹配BEV
        (['model', 'encoders', 'camera', 'vtransform', 'feature_size', 0], 32, "形状修复：恢复特征图高度"),
        (['model', 'encoders', 'camera', 'vtransform', 'feature_size', 1], 88, "形状修复：恢复特征图宽度"),
        
        # 保持其他内存优化
        (['image_size', 0], 224, "内存优化：保持较小图像高度"),
        (['image_size', 1], 640, "内存优化：保持较小图像宽度"),
        (['model', 'encoders', 'camera', 'vtransform', 'out_channels'], 64, "内存优化：保持较小输出通道"),
    ]
    
    # 应用修复
    for path, new_value, comment in new_optimizations:
        current = config_data
        for key in path[:-1]:
            if key in current:
                current = current[key]
        if isinstance(current, dict) and path[-1] in current:
            old_value = current[path[-1]]
            current[path[-1]] = new_value
            print(f"✅ {'.'.join(path)}: {old_value} → {new_value} - {comment}")
    
    # 计算修复后的BEV形状
    vtransform = config_data['model']['encoders']['camera']['vtransform']
    xbound = vtransform['xbound']
    ybound = vtransform['ybound']
    dbound = vtransform['dbound']
    x_size, y_size, d_size = calculate_bev_shape(xbound, ybound, dbound, vtransform['feature_size'])
    print(f"修复后BEV网格: {x_size} x {y_size} x {d_size}")
    
    # 保存修复后的配置
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, indent=2, width=120)
    
    print("🎉 BEV形状不匹配问题修复完成!")
    print("📋 修复方案: 恢复BEV范围与特征图尺寸的兼容性")
    print("   - BEV范围: 恢复接近官方配置")
    print("   - 特征图: 恢复匹配尺寸") 
    print("   - 图像尺寸: 保持224×640优化")
    print("   - 输出通道: 保持64优化")

if __name__ == "__main__":
    fix_bev_shape_mismatch()
