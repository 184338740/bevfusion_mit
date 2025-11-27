import os
import json
import mmcv
import tempfile
from pathlib import Path

print("=== 验证问题深度追踪 ===")

# 1. 检查所有可能的临时目录
print("\n1. 检查所有临时目录:")
temp_dirs = list(Path("/tmp").glob("tmp*"))
print(f"找到 {len(temp_dirs)} 个临时目录")

for temp_dir in temp_dirs[:10]:  # 只检查前10个
    results_dir = temp_dir / "results"
    if results_dir.exists():
        result_files = list(results_dir.glob("*.json"))
        if result_files:
            print(f"在 {temp_dir} 中找到结果文件:")
            for f in result_files:
                print(f"  {f}")

# 2. 检查验证集样本完整性
print("\n2. 验证集样本检查:")
val_info_path = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/tyjt_infos_val.pkl"
val_infos = mmcv.load(val_info_path)['infos']

print(f"验证集样本数: {len(val_infos)}")

# 检查样本的Lidar文件是否存在
missing_files = []
for i, info in enumerate(val_infos[:5]):  # 只检查前5个
    lidar_path = info['lidar_path']
    full_path = f"/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/{lidar_path}"
    exists = os.path.exists(full_path)
    print(f"样本 {i}: {info['token'][:8]}... - Lidar文件: {exists}")
    if not exists:
        missing_files.append(full_path)

if missing_files:
    print(f"❌ 缺失 {len(missing_files)} 个Lidar文件")

# 3. 检查训练配置
print("\n3. 训练配置检查:")
config_path = "configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/tyjt_convfuser_tmp.yaml"
if os.path.exists(config_path):
    config = mmcv.load(config_path)
    if 'evaluation' in config:
        print(f"验证间隔: {config['evaluation'].get('interval', 'N/A')}")
    if 'data' in config and 'val' in config['data']:
        val_config = config['data']['val']
        print(f"验证集路径: {val_config.get('ann_file', 'N/A')}")
        print(f"验证集根目录: {val_config.get('dataset_root', 'N/A')}")

# 4. 实时监控临时文件创建
print("\n4. 实时监控临时目录创建:")
print("在当前终端运行以下命令来监控:")
print("  watch -n 1 'ls -la /tmp/tmp* 2>/dev/null | head -20'")
print("然后在另一个终端重新运行训练命令")

# 5. 检查是否有权限问题
print("\n5. 权限检查:")
tmp_dir = Path("/tmp")
if tmp_dir.exists():
    print(f"/tmp 权限: {oct(tmp_dir.stat().st_mode)[-3:]}")
    
    # 测试创建临时文件
    try:
        test_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        test_file.write(b'test')
        test_file.close()
        os.unlink(test_file.name)
        print("✅ 可以创建临时文件")
    except Exception as e:
        print(f"❌ 无法创建临时文件: {e}")
