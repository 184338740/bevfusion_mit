#!/usr/bin/env python3
"""
快速修复 PKL 文件中的 version 字段，使其符合 nuscenes_eval 要求（必须以 'trainval' 结尾）。
用法：
    python fix_pkl_version.py <pkl_file_path> [new_version]
示例：
    python fix_pkl_version.py ./tyjt_infos_val.pkl v1.0-tyjt-trainval
"""

import pickle
import sys
from pathlib import Path

def fix_pkl_version(pkl_path: Path, new_version: str = None):
    """修改 PKL 文件中的 metadata.version"""
    if not pkl_path.exists():
        print(f"❌ 文件不存在: {pkl_path}")
        return False

    # 1. 读取原文件
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 2. 检查格式
    if not isinstance(data, dict) or 'metadata' not in data:
        print(f"❌ 不是标准的 BEVFusion PKL 格式（缺少 metadata）")
        return False

    old_version = data['metadata'].get('version', 'unknown')
    print(f"当前 version: {old_version}")

    # 3. 确定新版本名
    if new_version is None:
        # 自动生成：在原版本后添加 '-trainval'（如果还没有）
        if not old_version.endswith('trainval'):
            new_version = old_version + '-trainval'
        else:
            new_version = old_version  # 已经是合法版本
    else:
        # 确保以 trainval 结尾（用于评测）
        if not new_version.endswith('trainval'):
            new_version = new_version + '-trainval'
            print(f"自动添加 -trainval 后缀: {new_version}")

    # 4. 修改
    data['metadata']['version'] = new_version
    data['metadata']['fixed_by'] = 'fix_pkl_version.py'
    data['metadata']['fixed_date'] = str(Path(pkl_path).stat().st_mtime)  # 简单记录

    # 5. 写回原文件（建议先备份）
    backup_path = pkl_path.with_suffix(pkl_path.suffix + '.bak')
    if not backup_path.exists():
        pkl_path.rename(backup_path)
        print(f"✅ 已备份原文件至: {backup_path}")

    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"✅ 已修改 version: {old_version} -> {new_version}")
    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pkl_file = Path(sys.argv[1])
    new_ver = sys.argv[2] if len(sys.argv) > 2 else None
    fix_pkl_version(pkl_file, new_ver)