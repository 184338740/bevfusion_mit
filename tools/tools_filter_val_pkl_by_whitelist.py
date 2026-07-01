#!/usr/bin/env python3
"""
功能：基于白名单（标注JSON绝对路径列表）过滤已生成的 tyjt_infos_val.pkl

版本号 V1.1 (0423)
    - 改进：改用绝对路径直接匹配，消除文件名重名风险
    - 移除 --data-root 参数，简化使用

版本号 v1.0 (0422)
    - 新增：从白名单文件中读取标注JSON绝对路径，转换为相对路径后匹配pkl中的label_path字段
    - 新增：自动清除过滤后样本的 prev/next/sweeps 字段，避免评测时因时序断裂导致索引错误
    - 支持：自定义数据集根目录，适配不同挂载点

参数说明：
    --input-pkl       : 原始全量验证集 pkl 文件路径（必填）
    --whitelist       : 白名单文件路径，每行一个标注JSON绝对路径（必填）
    --data-root       : 数据集根目录，必须与生成原始pkl时使用的 --data-root 保持一致（必填）
    --output-pkl      : 输出过滤后的 pkl 文件路径（必填）
    --clear-sweeps    : 是否清除 prev/next/sweeps 字段，默认为 True（推荐），若设为 False 需确保保留样本时序连续

使用说明：
    1. 确保已通过 tyjt_converter_A100.py 生成了全量 tyjt_infos_val.pkl
    2. 准备白名单文件，每行一个标注JSON绝对路径（如 total_useful_labeled_only.list）
    3. 运行本脚本：
       python tools_filter_val_pkl_by_whitelist.py \\
           --input-pkl /path/to/tyjt_infos_val.pkl \\
           --whitelist /path/to/total_useful_labeled_only.list \\
           --data-root /cephfsdata/users/lishan/00_Data/00_RawData \\
           --output-pkl /path/to/tyjt_infos_val_clean.pkl
    4. 将 BEVFusion 配置文件中的验证集路径指向新生成的 pkl 即可用于评测

注意事项：
    - --data-root 必须与原始 pkl 生成时一致，否则相对路径匹配失败会导致样本数为0
    - 过滤后样本数可能少于白名单行数（部分样本可能因相机/标定缺失未被原始 pkl 收录）
    - 默认清除时序字段，不影响 BEVFusion 单帧评测；若需要多帧融合，请谨慎使用并确保时序连续
"""

import pickle
import argparse
from pathlib import Path

def parse_whitelist_paths(whitelist_path: str) -> set:
    """返回白名单中所有标注JSON的绝对路径集合（保持原样）"""
    paths = set()
    with open(whitelist_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                paths.add(line)
    return paths

def main():
    parser = argparse.ArgumentParser(description="过滤验证集 pkl，只保留白名单中的样本")
    parser.add_argument("--input-pkl", required=True, help="原始全量验证集 pkl 路径")
    parser.add_argument("--whitelist", required=True, help="白名单文件（每行一个标注JSON绝对路径）")
    parser.add_argument("--output-pkl", required=True, help="输出过滤后的 pkl 路径")
    parser.add_argument("--clear-sweeps", action="store_true", default=True,
                        help="是否清除 prev/next/sweeps（默认清除）")
    args = parser.parse_args()

    # 加载原始 pkl
    print(f"加载原始 pkl: {args.input_pkl}")
    with open(args.input_pkl, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos']
    metadata = data['metadata']
    print(f"原始样本数: {len(infos)}")

    # 解析白名单绝对路径集合
    allowed_paths = parse_whitelist_paths(args.whitelist)
    print(f"白名单中的路径数: {len(allowed_paths)}")

    # 过滤：直接比较 label_path 是否在白名单中
    filtered_infos = []
    for info in infos:
        label_path = info.get('label_path', '')
        if label_path in allowed_paths:
            filtered_infos.append(info)

    print(f"白名单命中样本数: {len(filtered_infos)}")
    print(f"未命中样本数: {len(infos) - len(filtered_infos)}")

    if args.clear_sweeps:
        for info in filtered_infos:
            info['prev'] = -1
            info['next'] = -1
            info['sweeps'] = []
        print("已清空 prev/next/sweeps 字段")

    # 保存
    new_data = {'infos': filtered_infos, 'metadata': metadata}
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(new_data, f)
    print(f"过滤后 pkl 已保存至: {args.output_pkl}")

if __name__ == '__main__':
    main()
