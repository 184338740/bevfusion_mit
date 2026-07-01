#!/usr/bin/env python3
import os
import yaml

def compress_with_newline_markers(content, lines_per_block=10):
    """使用换行标记压缩内容"""
    lines = content.splitlines()
    compressed_lines = []
    
    i = 0
    while i < len(lines):
        if i + lines_per_block < len(lines):
            # 将多行合并为一行，用"换行"分隔
            block = "换行".join(lines[i:i+lines_per_block])
            compressed_lines.append(block)
            i += lines_per_block
        else:
            # 剩余行数不足一个块，直接合并
            block = "换行".join(lines[i:])
            compressed_lines.append(block)
            break
    
    return compressed_lines

def analyze_nuscenes_configs():
    base_path = "/mnt/bevfusion_mit_xmy/configs"
    output_file = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/nuscenes_config_analysis_compressed.txt"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 分析NuScenes的完整配置层级
    config_files = {
        "GLOBAL_DEFAULT": "default.yaml",
        "NUSCENES_TOP": "nuscenes/default.yaml",
        "NUSCENES_DET": "nuscenes/det/default.yaml",
        "NUSCENES_TRANSFUSION": "nuscenes/det/transfusion/default.yaml", 
        "NUSCENES_SECFPN": "nuscenes/det/transfusion/secfpn/default.yaml",
        "NUSCENES_CAMERA_LIDAR": "nuscenes/det/transfusion/secfpn/camera+lidar/default.yaml",
        "NUSCENES_SWINT": "nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/default.yaml",
        "NUSCENES_MAIN": "nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
    }
    
    analysis = []
    analysis.append("=== NuScenes配置完整分析（换行标记压缩版） ===")
    analysis.append("说明：所有'换行'标记处代表原始配置文件中的换行符")
    analysis.append("=" * 80)
    
    total_original_lines = 0
    total_compressed_lines = 0
    
    for level_name, rel_path in config_files.items():
        full_path = os.path.join(base_path, rel_path)
        analysis.append(f"\n---【{level_name}】---")
        analysis.append(f"路径: {rel_path}")
        
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 记录原始信息
                original_lines = content.splitlines()
                original_line_count = len(original_lines)
                total_original_lines += original_line_count
                
                analysis.append(f"原始行数: {original_line_count}")
                analysis.append("配置内容开始 >>>")
                
                # 使用换行标记压缩
                compressed_content = compress_with_newline_markers(content, lines_per_block=8)
                total_compressed_lines += len(compressed_content)
                
                for block in compressed_content:
                    analysis.append(block)
                
                analysis.append("<<< 配置内容结束")
                
                # 提取关键信息用于参考
                try:
                    config_data = yaml.safe_load(content)
                    if config_data:
                        analysis.append("关键参数摘要:")
                        key_params = [
                            ('dataset', '数据集'),
                            ('dataset_type', '数据集类型'),
                            ('class_names', '类别名称'), 
                            ('data_root', '数据根路径'),
                            ('info_path', '信息文件路径')
                        ]
                        
                        for key, desc in key_params:
                            if key in config_data:
                                value = config_data[key]
                                if key == 'class_names' and isinstance(value, list):
                                    analysis.append(f"  {desc}: {len(value)}个类别")
                                else:
                                    analysis.append(f"  {desc}: {value}")
                        
                        # 检查模型配置
                        if 'model' in config_data and 'bbox_head' in config_data['model']:
                            bbox_head = config_data['model']['bbox_head']
                            if 'num_classes' in bbox_head:
                                analysis.append(f"  模型类别数: {bbox_head['num_classes']}")
                            
                except Exception as e:
                    analysis.append(f"配置解析备注: {e}")
                    
            except Exception as e:
                analysis.append(f"文件读取错误: {e}")
        else:
            analysis.append("文件不存在")
    
    # 统计信息
    analysis.append(f"\n" + "="*80)
    analysis.append("压缩统计:")
    analysis.append(f"总原始行数: {total_original_lines}")
    analysis.append(f"压缩后行数: {total_compressed_lines}") 
    analysis.append(f"压缩率: {total_compressed_lines/(total_original_lines+1)*100:.1f}%")
    analysis.append(f"分析文件总行数: {len(analysis)}")
    
    # 添加恢复说明
    analysis.append(f"\n" + "="*80)
    analysis.append("内容恢复说明:")
    analysis.append("将所有'换行'标记替换为实际的换行符即可恢复原始配置")
    analysis.append("示例Python代码:")
    analysis.append("""
# 恢复原始配置内容
compressed_text = \"\"\"[粘贴压缩内容 here]\"\"\"
original_content = compressed_text.replace('换行', '\\n')
print(original_content)
""")
    
    # 写入分析结果
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(analysis))
    
    print(f"NuScenes配置分析完成！")
    print(f"输出文件: {output_file}")
    print(f"原始配置: {total_original_lines}行 → 压缩后: {total_compressed_lines}行")
    print(f"分析文件: {len(analysis)}行")

if __name__ == "__main__":
    analyze_nuscenes_configs()
