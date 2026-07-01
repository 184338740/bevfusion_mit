import pickle
import os
import numpy as np
import json
from pathlib import Path
from datetime import datetime

class TruncatedNumpyJSONEncoder(json.JSONEncoder):
    def __init__(self, *args, max_lines=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_lines = max_lines
        self.current_lines = 0
        self.truncated = False

    def default(self, obj):
        if self.truncated:
            return None
        
        self.current_lines += 1
        if self.current_lines > self.max_lines:
            self.truncated = True
            return "⚠️ 数据已截断（超过最大行数限制）"
        
        if isinstance(obj, np.ndarray):
            data = obj.tolist()
            if isinstance(data, list) and len(data) > 5:
                data = data[:5] + ["..."]
            return {
                "type": "ndarray",
                "shape": obj.shape,
                "dtype": str(obj.dtype),
                "data": data
            }
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (list, tuple)) and len(obj) > 5:
            return list(obj[:5]) + ["..."]
        elif isinstance(obj, dict) and len(obj) > 5:
            first_5 = {k: obj[k] for k in list(obj.keys())[:5]}
            first_5["..."] = f"省略{len(obj)-5}个键"
            return first_5
        return super().default(obj)

def _get_type_details(obj):
    if isinstance(obj, dict):
        return f"dict (键数量: {len(obj)})"
    elif isinstance(obj, (list, tuple)):
        elem_type = type(obj[0]).__name__ if len(obj) > 0 else "empty"
        return f"{type(obj).__name__} (长度: {len(obj)}, 元素类型: {elem_type})"
    elif isinstance(obj, np.ndarray):
        return f"ndarray (形状: {obj.shape}, 数据类型: {obj.dtype})"
    elif isinstance(obj, str):
        return f"str (长度: {len(obj)})"
    else:
        return f"{type(obj).__name__}"

def _format_value(obj):
    if isinstance(obj, (dict, list, tuple, np.ndarray)):
        return f"({_get_type_details(obj)})"
    else:
        val_str = str(obj)[:50] + ("..." if len(str(obj)) > 50 else "")
        return val_str

def _print_recursive(obj, prefix="", is_last=False, max_depth=3, current_depth=0, lines=None):
    if lines is None:
        lines = []
    
    # 层级前缀（核心修复：确保顶层键的连接线正确）
    line_prefix = prefix + ("└─ " if is_last else "├─ ") if current_depth > 0 else ""
    
    # 处理字典（顶层键如car、truck在此处展示）
    if isinstance(obj, dict):
        # 顶层字典不单独显示类型行，直接展开键值对（解决最浅层缺失问题）
        keys = list(obj.keys())
        for i, key in enumerate(keys):
            child_is_last = (i == len(keys) - 1)
            value = obj[key]
            # 顶层键行：明确显示键名和类型
            line = f"{line_prefix}{key}: {_get_type_details(value)} → {_format_value(value)}"
            lines.append(line)
            # 递归处理子对象（进入下一层级）
            if isinstance(value, (dict, list, tuple, np.ndarray)) and current_depth < max_depth:
                _print_recursive(
                    value,
                    prefix=prefix + ("   " if is_last else "│  "),
                    is_last=child_is_last,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                    lines=lines
                )
        return lines
    
    # 处理列表/元组（修复子元素前缀对齐）
    elif isinstance(obj, (list, tuple)):
        # 显示列表本身的类型信息
        lines.append(f"{line_prefix}{_get_type_details(obj)}")
        if len(obj) == 0 or current_depth >= max_depth:
            return lines
        
        # 子元素前缀（与父级严格对齐）
        child_prefix = prefix + ("   " if is_last else "│  ")
        # 显示第0个元素
        _print_recursive(
            obj[0],
            prefix=child_prefix,
            is_last=True,
            max_depth=max_depth,
            current_depth=current_depth + 1,
            lines=lines
        )
        # 剩余元素提示（修复省略提示的层级）
        if len(obj) > 1:
            lines.append(f"{child_prefix}└─ 其余{len(obj)-1}个元素省略...")
        return lines
    
    # 处理ndarray
    elif isinstance(obj, np.ndarray):
        lines.append(f"{line_prefix}{_get_type_details(obj)}")
        return lines
    
    # 基础类型
    else:
        lines.append(f"{line_prefix}{_get_type_details(obj)} → {_format_value(obj)}")
        return lines

def pkl_to_json(pkl_path, json_path, max_lines=1000):
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        encoder = TruncatedNumpyJSONEncoder(max_lines=max_lines, indent=2, ensure_ascii=False)
        json_str = encoder.encode(data)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        
        if encoder.truncated:
            print(f"⚠️ JSON文件已截断（超过{max_lines}行限制）")
        return True
    except Exception as e:
        print(f"❌ JSON转换失败: {str(e)}")
        return False

def visualize_pkl(pkl_path, max_depth=3, save_dir="output/debug", json_max_lines=1000):
    if not os.path.exists(pkl_path):
        print(f"❌ 跳过不存在的文件: {pkl_path}")
        return
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filename = os.path.basename(pkl_path)
    structure_save_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_structure.txt")
    json_save_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}.json")
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        lines = [f"===== PKL文件结构 ({filename}) ====="]
        lines.append(f"路径: {pkl_path} | 解析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        _print_recursive(data, max_depth=max_depth, lines=lines)
        
        with open(structure_save_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        json_success = pkl_to_json(pkl_path, json_save_path, max_lines=json_max_lines)
        
        print(f"✅ 处理完成: {filename} (顶层类型: {_get_type_details(data)})")
        print(f"   结构文件: {structure_save_path}")
        if json_success:
            print(f"   JSON文件: {json_save_path}")
    
    except Exception as e:
        error_msg = f"解析失败: {str(e)}"
        print(f"❌ {filename}: {error_msg}")
        with open(os.path.join(save_dir, "error.log"), 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now()}] {filename}: {error_msg}\n")

def batch_visualize(pkl_paths, max_depth=3, json_max_lines=1000):
    total = len(pkl_paths)
    print(f"===== 开始批量处理 ({total}个文件) =====")
    print(f"⚠️ JSON文件最大行数限制: {json_max_lines}行\n")
    for i, path in enumerate(pkl_paths, 1):
        print(f"[{i}/{total}] 处理文件: {os.path.basename(path)}")
        visualize_pkl(path, max_depth=max_depth, json_max_lines=json_max_lines)
    print(f"\n===== 所有文件处理完成，结果保存至: {os.path.abspath('output/debug')} =====")

if __name__ == "__main__":
    pkl_paths = [
        "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_dbinfos_train_fixed.pkl",  # 优先修复此文件的显示
        "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_train_fixed.pkl",
        "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/output/step3_1/tyjt_infos_val_fixed.pkl",
        "/mnt/bevfusion_mit_xmy/data/nuscenes/nuscenes_dbinfos_train.pkl",
        "/mnt/bevfusion_mit_xmy/data/nuscenes/nuscenes_infos_train.pkl",
        "/mnt/bevfusion_mit_xmy/data/nuscenes/nuscenes_infos_val.pkl"
    ]
    batch_visualize(pkl_paths, max_depth=4, json_max_lines=2000)