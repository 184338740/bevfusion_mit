#!/usr/bin/env python3
import pickle

print('=== 验证修复结果 ===')
files = [
    '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_train_fixed.pkl',
    '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_val_fixed.pkl',
    '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_dbinfos_train_fixed.pkl'
]

for file in files:
    print(f'检查: {file}')
    with open(file, 'rb') as f:
        data = pickle.load(f)
    print(f'  类型: {type(data)}')
    if isinstance(data, dict):
        print(f'  keys: {list(data.keys())}')
        if 'infos' in data:
            print(f'  infos长度: {len(data["infos"])}')
    print()
