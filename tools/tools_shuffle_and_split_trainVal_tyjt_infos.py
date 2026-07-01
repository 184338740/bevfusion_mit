#!/usr/bin/env python3
"""
TYJT 数据集 info 拆分脚本

功能：
    读取合并的 tyjt_infos_train.pkl（包含 train+val 全部样本），
    按指定比例随机拆分为新的训练集和验证集 pkl 文件。
    支持任意比例（如 8:2、1.3:1.5、8:0 等），固定随机种子确保可复现。

版本: v1.0 (0423)
    
输出：
    - tyjt_infos_train_split.pkl
    - tyjt_infos_val_split.pkl
    - 拆分日志（打印到控制台，也可重定向到文件）

使用说明：
python tools_shuffle_and_split_trainVal_tyjt_infos.py \
    --input-pkl ../datasets/tyjt2pkl/A100_V032d_sub_0423/tyjt_infos_train.pkl \
    --output-dir ../datasets/tyjt2pkl/A100_V032d_sub_0423/ \
    --ratio 8:2 \
    --seed 42
"""

import argparse
import pickle
import random
import time
from pathlib import Path
from typing import Tuple, Dict, Any, List

def parse_ratio(ratio_str: str) -> Tuple[float, float]:
    """
    解析比例字符串，如 "8:2" -> (8.0, 2.0)，返回两个浮点数。
    支持小数如 "1.3:1.5"。
    """
    parts = ratio_str.strip().split(':')
    if len(parts) != 2:
        raise ValueError(f"比例格式错误，应为 'a:b'，实际为 '{ratio_str}'")
    try:
        a = float(parts[0])
        b = float(parts[1])
    except ValueError:
        raise ValueError(f"比例中的数字必须为数值，实际为 '{parts[0]}' 和 '{parts[1]}'")
    if a < 0 or b < 0:
        raise ValueError(f"比例不能为负数，实际为 a={a}, b={b}")
    if a == 0 and b == 0:
        raise ValueError("比例不能同时为0")
    return a, b

def split_infos(
    infos: List[Dict],
    train_ratio: float,
    val_ratio: float,
    seed: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    根据 train_ratio 和 val_ratio（两者之和不必为1）对样本进行拆分。
    实际训练集占比 = train_ratio / (train_ratio + val_ratio)
    """
    total = len(infos)
    if total == 0:
        return [], []
    
    # 计算训练集目标数量
    train_frac = train_ratio / (train_ratio + val_ratio)
    train_size = int(round(train_frac * total))
    # 确保边界情况：如果 train_ratio=0，train_size=0；val_ratio=0，train_size=total
    if train_ratio == 0:
        train_size = 0
    elif val_ratio == 0:
        train_size = total
    
    # 打乱（固定种子）
    rng = random.Random(seed)
    shuffled = infos.copy()
    rng.shuffle(shuffled)
    
    train_infos = shuffled[:train_size]
    val_infos = shuffled[train_size:]
    return train_infos, val_infos

def main():
    parser = argparse.ArgumentParser(
        description="将合并的 TYJT info pkl 按比例拆分为训练集和验证集"
    )
    parser.add_argument(
        "--input-pkl", type=str, required=True,
        help="输入的合并 info pkl 路径（例如 tyjt_infos_train.pkl）"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="输出目录（将生成 tyjt_infos_train_split.pkl 和 tyjt_infos_val_split.pkl）"
    )
    parser.add_argument(
        "--ratio", type=str, default="8:2",
        help="拆分比例，格式 'a:b'，例如 '8:2'、'1.3:1.5'、'8:0'，默认为 '8:2'"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子，用于可复现打乱，默认为 42"
    )
    parser.add_argument(
        "--train-pkl-name", type=str, default="tyjt_infos_train_split.pkl",
        help="输出的训练集 pkl 文件名，默认为 tyjt_infos_train_split.pkl"
    )
    parser.add_argument(
        "--val-pkl-name", type=str, default="tyjt_infos_val_split.pkl",
        help="输出的验证集 pkl 文件名，默认为 tyjt_infos_val_split.pkl"
    )
    args = parser.parse_args()

    # 解析比例
    train_ratio, val_ratio = parse_ratio(args.ratio)
    print(f"[INFO] 拆分比例: {args.ratio} -> 训练样本占比={train_ratio}, 验证样本占比={val_ratio}")

    # 读取输入 pkl
    input_path = Path(args.input_pkl)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # 兼容两种可能格式：直接是列表，或包含 "infos" 键的字典
    if isinstance(data, dict) and "infos" in data:
        infos = data["infos"]
        metadata = data.get("metadata", {})
        original_format = "dict"
    elif isinstance(data, list):
        infos = data
        metadata = {}
        original_format = "list"
    else:
        raise TypeError(f"不支持的 pkl 格式，期望 dict 或 list，实际为 {type(data)}")
    
    total_samples = len(infos)
    print(f"[INFO] 总样本数: {total_samples}")
    print(f"[INFO] 随机种子: {args.seed}")

    # 拆分
    train_infos, val_infos = split_infos(infos, train_ratio, val_ratio, args.seed)
    train_size = len(train_infos)
    val_size = len(val_infos)
    
    # 计算实际比例
    actual_train_ratio = train_size / total_samples if total_samples > 0 else 0
    actual_val_ratio = val_size / total_samples if total_samples > 0 else 0
    
    print(f"[INFO] 拆分结果: 训练集 {train_size} 样本 ({actual_train_ratio:.2%}), "
          f"验证集 {val_size} 样本 ({actual_val_ratio:.2%})")
    
    # 构建输出数据结构（保持与原格式一致）
    if original_format == "dict":
        # 更新 metadata，添加拆分信息
        new_metadata = metadata.copy()
        new_metadata.update({
            "split_info": {
                "original_file": str(input_path),
                "ratio": args.ratio,
                "seed": args.seed,
                "train_samples": train_size,
                "val_samples": val_size,
                "split_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        })
        train_out = {"infos": train_infos, "metadata": new_metadata}
        val_out = {"infos": val_infos, "metadata": new_metadata}
    else:
        train_out = train_infos
        val_out = val_infos
    
    # 确保输出目录存在
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_pkl = out_dir / args.train_pkl_name
    val_pkl = out_dir / args.val_pkl_name
    
    with open(train_pkl, 'wb') as f:
        pickle.dump(train_out, f)
    with open(val_pkl, 'wb') as f:
        pickle.dump(val_out, f)
    
    print(f"[INFO] 训练集已保存至: {train_pkl}")
    print(f"[INFO] 验证集已保存至: {val_pkl}")
    
    # 可选：同时保存拆分日志到文本文件
    log_path = out_dir / "split_log.txt"
    with open(log_path, 'w') as log_f:
        log_f.write(f"Split log for {input_path}\n")
        log_f.write(f"Ratio: {args.ratio} (train_weight={train_ratio}, val_weight={val_ratio})\n")
        log_f.write(f"Seed: {args.seed}\n")
        log_f.write(f"Total samples: {total_samples}\n")
        log_f.write(f"Train samples: {train_size} ({actual_train_ratio:.2%})\n")
        log_f.write(f"Val samples: {val_size} ({actual_val_ratio:.2%})\n")
        log_f.write(f"Output files: {train_pkl}, {val_pkl}\n")
        log_f.write(f"Split time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"[INFO] 拆分日志已保存至: {log_path}")

if __name__ == "__main__":
    main()