#!/usr/bin/env python3
import yaml
from pathlib import Path

def verify_all_optimizations():
    """验证所有内存优化配置是否生效"""
    base_path = Path("/mnt/bevfusion_mit_xmy")
    config_file = base_path / "configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/tyjt_convfuser.yaml"
    
    print(f"🔍 验证配置文件: {config_file}")
    print("📝 验证目的: 确认所有内存优化参数已正确设置")
    print("📝 验证原因: 确保配置修改生效，避免训练时内存不足")
    
    if not config_file.exists():
        print("❌ 配置文件不存在")
        return
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n内存优化配置验证结果:")
    
    # 定义期望值和验证说明
    expected_values = {
        'image_size': ([224, 640], "图像尺寸优化"),
        'model.encoders.camera.vtransform.xbound': ([-40.0, 40.0, 0.5], "BEV X范围优化"),
        'model.encoders.camera.vtransform.ybound': ([-40.0, 40.0, 0.5], "BEV Y范围优化"), 
        'model.encoders.camera.vtransform.dbound': ([2.0, 50.0, 2.0], "深度范围优化"),
        'model.encoders.camera.vtransform.out_channels': (64, "输出通道优化"),
        'model.encoders.camera.vtransform.downsample': (4, "下采样优化"),
        'model.encoders.camera.vtransform.feature_size': ([28, 80], "特征图尺寸优化"),
        'model.encoders.lidar.voxelize.max_voxels': ([80000, 120000], "Voxel数量优化"),
        'data.workers_per_gpu': (1, "工作进程优化"),
        'optimizer_config.cumulative_iters': (2, "梯度累积优化")
    }
    
    all_passed = True
    
    for path, (expected, description) in expected_values.items():
        keys = path.split('.')
        current = config
        
        # 导航到目标位置
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                current = None
                break
        
        if current == expected:
            print(f"✅ {path}: {current} - {description}")
        else:
            print(f"❌ {path}: 期望 {expected}, 实际 {current} - {description}")
            all_passed = False
    
    # 验证sweeps_num修改
    print("\nSweep数量验证:")
    sweep_locations = [
        config.get('data', {}).get('test', {}).get('pipeline', []),
        config.get('data', {}).get('train', {}).get('dataset', {}).get('pipeline', []),
        config.get('data', {}).get('val', {}).get('pipeline', []),
        config.get('evaluation', {}).get('pipeline', []),
        config.get('test_pipeline', []),
        config.get('train_pipeline', [])
    ]
    
    sweep_modified = True
    for pipeline in sweep_locations:
        for step in pipeline:
            if isinstance(step, dict) and 'sweeps_num' in step:
                if step['sweeps_num'] != 3:
                    sweep_modified = False
                    print(f"❌ 发现未修改的sweeps_num: {step['sweeps_num']}")
    
    if sweep_modified:
        print("✅ 所有sweeps_num已修改为3")
    else:
        print("❌ 部分sweeps_num未正确修改")
        all_passed = False
    
    if all_passed:
        print("\n🎉 所有内存优化配置验证通过! 可以开始训练")
    else:
        print("\n⚠️ 部分配置未正确设置! 请检查修改")

if __name__ == "__main__":
    verify_all_optimizations()
