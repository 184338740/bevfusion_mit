#!/usr/bin/env python3
"""
20251030_smart_compare.py
智能对比 - 使用配置1的顺序，对比配置2的值
"""

import yaml
import re

def smart_compare():
    """智能对比方法"""
    
    config1_path = "/mnt/bevfusion_mit_xmy/configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
    config2_path = "/mnt/bevfusion_mit_xmy/configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/tyjt_convfuser.yaml"
    output_path = "/mnt/bevfusion_mit_xmy/configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/compare_smart.yaml"
    
    print("读取原始配置文件...")
    
    # 读取配置1的原始行
    with open(config1_path, 'r', encoding='utf-8') as f:
        config1_lines = [line.rstrip() for line in f]
    
    # 读取配置2的数据
    with open(config2_path, 'r', encoding='utf-8') as f:
        config2_data = yaml.safe_load(f)
    
    print("构建配置2的完整路径映射...")
    
    # 构建配置2的完整路径映射
    config2_map = {}
    
    def build_map(data, path=""):
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                if isinstance(value, (dict, list)):
                    build_map(value, new_path)
                else:
                    config2_map[new_path] = value
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]"
                if isinstance(item, (dict, list)):
                    build_map(item, new_path)
                else:
                    config2_map[new_path] = item
    
    build_map(config2_data)
    
    print(f"配置2键数量: {len(config2_map)}")
    
    # 处理配置1的每一行
    output_lines = []
    current_path = []
    
    for line in config1_lines:
        if not line.strip() or line.strip().startswith('#'):
            # 空行或注释行
            output_lines.append(line)
            continue
        
        # 分析缩进和键
        indent = len(line) - len(line.lstrip())
        level = indent // 2
        
        if ':' in line:
            parts = line.split(':', 1)
            key = parts[0].strip()
            value1 = parts[1].strip() if len(parts) > 1 else ''
            
            # 更新当前路径
            current_path = current_path[:level]
            current_path.append(key)
            full_path = '.'.join(current_path)
            
            # 在配置2中查找
            if full_path in config2_map:
                value2 = config2_map[full_path]
                if value1 == str(value2):
                    output_lines.append(line + f"  # 未修改 - 原数值: {value1}, 当前数值: {value1}")
                else:
                    output_lines.append(line + f"  # 修改 - 配置1: {value1}, 配置2: {value2}")
            else:
                output_lines.append(line + f"  # 配置2中不存在此配置")
        else:
            # 非键值行
            output_lines.append(line)
    
    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')
    
    print(f"输出文件: {output_path}")
    print(f"输出行数: {len(output_lines)}")
    print(f"配置1原行数: {len(config1_lines)}")
    
    return output_path

if __name__ == "__main__":
    output_file = smart_compare()
    print(f"\n完成! 文件已保存到: {output_file}")