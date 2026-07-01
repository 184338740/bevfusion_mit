#!/usr/bin/env python3
"""
step1_analyze_bevfusion_pkl_format-v2.py
分析BEVFusion PKL文件格式 - 简化调试版本
"""

import pickle
import json
import os
import numpy as np
from pathlib import Path
import traceback

def safe_analyze_data(data, max_depth=3, current_depth=0):
    """安全地分析数据结构"""
    if current_depth > max_depth:
        return {"type": str(type(data)), "depth_limit": True}
    
    try:
        # 基本类型
        if data is None or isinstance(data, (int, float, str, bool)):
            return {
                "type": type(data).__name__,
                "value": data
            }
        
        # numpy数组
        elif isinstance(data, np.ndarray):
            return {
                "type": "numpy.ndarray",
                "dtype": str(data.dtype),
                "shape": data.shape,
                "sample": data.flatten()[:3].tolist() if data.size > 0 else []
            }
        
        # 字典
        elif isinstance(data, dict):
            result = {
                "type": "dict",
                "length": len(data),
                "keys": list(data.keys())[:10],  # 只显示前10个key
                "items": {}
            }
            # 只分析前5个键值对
            for i, (key, value) in enumerate(list(data.items())[:5]):
                result["items"][str(key)] = safe_analyze_data(value, max_depth, current_depth + 1)
            return result
        
        # 列表/元组
        elif isinstance(data, (list, tuple)):
            result = {
                "type": "list" if isinstance(data, list) else "tuple",
                "length": len(data),
                "sample_items": []
            }
            # 只分析前3个元素
            for i in range(min(3, len(data))):
                result["sample_items"].append(safe_analyze_data(data[i], max_depth, current_depth + 1))
            return result
        
        # 其他类型
        else:
            return {
                "type": str(type(data)),
                "methods": [method for method in dir(data) if not method.startswith('_')][:10]
            }
            
    except Exception as e:
        return {
            "type": str(type(data)),
            "error": f"分析失败: {str(e)}"
        }

def analyze_pkl_file(pkl_path, output_dir):
    """分析单个PKL文件"""
    print(f"\n分析文件: {pkl_path}")
    
    if not os.path.exists(pkl_path):
        print(f"错误: 文件不存在")
        return False
    
    try:
        # 读取PKL文件
        print("正在读取PKL文件...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        print(f"读取成功 - 类型: {type(data)}, 长度: {len(data) if hasattr(data, '__len__') else 'N/A'}")
        
        # 创建输出文件名
        filename = os.path.basename(pkl_path).replace('.pkl', '')
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成基本信息
        basic_info = {
            "file": pkl_path,
            "data_type": str(type(data)),
            "data_length": len(data) if hasattr(data, '__len__') else "N/A",
            "is_dict": isinstance(data, dict),
            "is_list": isinstance(data, list)
        }
        
        # 分析数据结构
        print("分析数据结构...")
        structure_info = safe_analyze_data(data)
        
        # 组合报告
        report = {
            "basic_info": basic_info,
            "structure": structure_info
        }
        
        # 保存JSON报告
        json_output = output_path / f"{filename}_analysis.json"
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"JSON报告已保存: {json_output}")
        
        # 保存简化的TXT报告
        txt_output = output_path / f"{filename}_overview.txt"
        with open(txt_output, 'w', encoding='utf-8') as f:
            f.write(f"PKL文件分析报告: {filename}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("基本信息:\n")
            for key, value in basic_info.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            if isinstance(data, dict):
                f.write("字典键列表:\n")
                for key in data.keys():
                    f.write(f"  - {key}\n")
                f.write("\n")
                
                # 显示每个键的基本信息
                f.write("键值类型信息:\n")
                for key, value in list(data.items())[:10]:  # 只显示前10个
                    f.write(f"  {key}: {type(value).__name__}")
                    if hasattr(value, '__len__'):
                        f.write(f" (长度: {len(value)})")
                    f.write("\n")
                    
        print(f"TXT报告已保存: {txt_output}")
        
        # 如果是字典，额外保存键的详细信息
        if isinstance(data, dict):
            keys_output = output_path / f"{filename}_keys_detail.txt"
            with open(keys_output, 'w', encoding='utf-8') as f:
                f.write("详细键值分析:\n")
                f.write("=" * 30 + "\n\n")
                
                for key, value in data.items():
                    f.write(f"键: {key}\n")
                    f.write(f"  类型: {type(value)}\n")
                    f.write(f"  长度: {len(value) if hasattr(value, '__len__') else 'N/A'}\n")
                    
                    if isinstance(value, (list, tuple)) and len(value) > 0:
                        f.write(f"  元素类型: {type(value[0])}\n")
                        if len(value) > 1:
                            f.write(f"  前3个元素类型: {[type(x) for x in value[:3]]}\n")
                    
                    f.write("\n")
            
            print(f"键值详情已保存: {keys_output}")
        
        return True
        
    except Exception as e:
        print(f"分析文件时出错: {str(e)}")
        print("详细错误信息:")
        traceback.print_exc()
        return False

def main():
    # 配置路径
    base_pkl_dir = "/mnt/bevfusion_mit_xmy/data/nuscenes"
    output_base_dir = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step1"
    
    # 要分析的PKL文件
    pkl_files = [
        "nuscenes_infos_train.pkl",
        "nuscenes_infos_val.pkl", 
        "nuscenes_dbinfos_train.pkl"
    ]
    
    print("开始分析BEVFusion PKL文件格式...")
    print(f"输入目录: {base_pkl_dir}")
    print(f"输出目录: {output_base_dir}")
    
    # 分析每个文件
    success_count = 0
    for pkl_file in pkl_files:
        pkl_path = os.path.join(base_pkl_dir, pkl_file)
        
        if analyze_pkl_file(pkl_path, output_base_dir):
            success_count += 1
            print(f"✓ 成功分析: {pkl_file}")
        else:
            print(f"✗ 分析失败: {pkl_file}")
    
    print(f"\n分析完成! 成功处理 {success_count}/{len(pkl_files)} 个文件")
    print(f"详细报告保存在: {output_base_dir}")

if __name__ == "__main__":
    main()