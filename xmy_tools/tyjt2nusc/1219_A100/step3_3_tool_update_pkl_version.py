#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
========================================================================
File: step3_3_update_pkl_version.py
Version: v1.0
Date: 2026-02-13
Author: xmy
========================================================================
Description:
    纯粹更新 nuScenes info pkl 文件的 metadata.version 字段。
    直接修改脚本开头的参数即可运行，不添加任何额外逻辑。

Usage:
    1. 修改下方 src_pkl_file, dst_pkl_file, new_version 三个变量
    2. 运行: python step3_3_update_pkl_version.py
========================================================================
"""

import pickle

# ========== 在这里修改配置 ==========
src_pkl_file = "./output-1226-v9.2.3-all/step1/nuscenes_tyjt/V02-TYJT/pkl_tmp/tyjt_infos_val_a.pkl"
dst_pkl_file = "./output-1226-v9.2.3-all/step1/nuscenes_tyjt/V02-TYJT/pkl_tmp/tyjt_infos_val_b.pkl"
new_version = "v02.1-tyjt-trainval"
# ===================================

def main():
    print(f"🔧 读取源文件: {src_pkl_file}")
    with open(src_pkl_file, 'rb') as f:
        data = pickle.load(f)

    # 直接修改版本
    data['metadata']['version'] = new_version

    print(f"📝 写入目标文件: {dst_pkl_file}")
    with open(dst_pkl_file, 'wb') as f:
        pickle.dump(data, f)

    print("✅ 更新完成")

if __name__ == "__main__":
    main()