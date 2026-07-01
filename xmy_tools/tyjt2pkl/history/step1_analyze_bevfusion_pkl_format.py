# step1_diagnose_annotation_issue.py
import pickle
import numpy as np
from pathlib import Path
import json

class AnnotationDiagnoser:
    def __init__(self, output_dir: str = "./step1_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def diagnose_annotation_issue(self):
        """诊断标注信息问题"""
        base_path = Path("/mnt/bevfusion_mit_xmy/data/nuscenes")
        pkl_files = {
            "train": "nuscenes_infos_train.pkl",
            "val": "nuscenes_infos_val.pkl", 
        }
        
        diagnosis_results = {}
        
        for name, filename in pkl_files.items():
            pkl_path = base_path / filename
            if pkl_path.exists():
                print(f"诊断文件: {pkl_path}")
                diagnosis_results[name] = self.diagnose_single_file(pkl_path, name)
        
        self.save_diagnosis_results(diagnosis_results)
        return diagnosis_results
    
    def diagnose_single_file(self, pkl_path: Path, file_name: str):
        """诊断单个文件"""
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        diagnosis = {
            "file_info": {
                "file_name": pkl_path.name,
                "data_type": type(data).__name__,
                "total_samples": len(data['infos']) if 'infos' in data else 0
            },
            "sample_analysis": {},
            "annotation_search": {},
            "recommendations": []
        }
        
        if 'infos' in data and data['infos']:
            # 分析前5个样本
            samples_to_analyze = data['infos'][:5]
            
            for i, sample in enumerate(samples_to_analyze):
                sample_diagnosis = self.analyze_sample_structure(sample, i)
                diagnosis["sample_analysis"][f"sample_{i}"] = sample_diagnosis
            
            # 搜索可能的标注字段
            diagnosis["annotation_search"] = self.search_annotation_fields(data['infos'][:10])
        
        return diagnosis
    
    def analyze_sample_structure(self, sample, sample_index):
        """分析样本结构"""
        analysis = {
            "all_keys": list(sample.keys()),
            "keys_with_ann_like_names": [],
            "camera_info": {},
            "possible_annotation_fields": {}
        }
        
        # 查找包含标注相关关键词的字段
        annotation_keywords = ['ann', 'gt', 'label', 'box', 'bbox', 'object', 'detection']
        for key in sample.keys():
            key_lower = key.lower()
            if any(keyword in key_lower for keyword in annotation_keywords):
                analysis["keys_with_ann_like_names"].append(key)
                
                # 分析这个字段的内容
                value = sample[key]
                analysis["possible_annotation_fields"][key] = {
                    "type": type(value).__name__,
                    "details": self.get_value_details(value)
                }
        
        # 检查相机信息
        if 'cams' in sample:
            analysis["camera_info"] = {
                "num_cameras": len(sample['cams']),
                "camera_names": list(sample['cams'].keys())
            }
        
        return analysis
    
    def search_annotation_fields(self, samples):
        """在所有样本中搜索标注字段"""
        annotation_fields_found = {}
        field_frequency = {}
        
        for sample in samples:
            for key in sample.keys():
                if key not in field_frequency:
                    field_frequency[key] = 0
                field_frequency[key] += 1
                
                # 检查字段内容是否像标注数据
                if self.looks_like_annotation(sample[key]):
                    if key not in annotation_fields_found:
                        annotation_fields_found[key] = {
                            "type": type(sample[key]).__name__,
                            "sample_content": self.get_value_details(sample[key]),
                            "frequency": 0
                        }
                    annotation_fields_found[key]["frequency"] += 1
        
        return {
            "all_fields": field_frequency,
            "possible_annotation_fields": annotation_fields_found
        }
    
    def looks_like_annotation(self, value):
        """判断值是否像标注数据"""
        if value is None:
            return False
        
        # 如果是numpy数组，检查形状和数据类型
        if isinstance(value, np.ndarray):
            if len(value.shape) == 2 and value.shape[1] in [7, 8, 9]:  # 可能的3D边界框
                return True
            if len(value.shape) == 1:  # 可能的标签
                return True
        
        # 如果是列表，检查内容
        if isinstance(value, list) and len(value) > 0:
            first_element = value[0]
            if isinstance(first_element, str):  # 类别名称
                return True
            if isinstance(first_element, (list, np.ndarray)):  # 边界框列表
                return True
        
        # 如果是字典，检查键
        if isinstance(value, dict):
            dict_keys = list(value.keys())
            annotation_keywords = ['boxes', 'labels', 'scores', 'classes']
            if any(keyword in str(dict_keys).lower() for keyword in annotation_keywords):
                return True
        
        return False
    
    def get_value_details(self, value):
        """获取值的详细信息"""
        if value is None:
            return "None"
        elif isinstance(value, (str, int, float, bool)):
            return str(value)[:100]
        elif isinstance(value, np.ndarray):
            return {
                "shape": value.shape,
                "dtype": str(value.dtype),
                "sample": value.flatten()[:3].tolist() if value.size > 0 else []
            }
        elif isinstance(value, list):
            return {
                "length": len(value),
                "first_element_type": type(value[0]).__name__ if len(value) > 0 else "empty",
                "sample": value[:2] if len(value) > 0 and not isinstance(value[0], (np.ndarray, dict)) else "complex_data"
            }
        elif isinstance(value, dict):
            return {
                "keys": list(value.keys())[:5],
                "num_keys": len(value)
            }
        else:
            return str(type(value))
    
    def save_diagnosis_results(self, diagnosis_results):
        """保存诊断结果"""
        # 保存JSON格式的诊断结果
        with open(self.output_dir / "annotation_diagnosis.json", 'w', encoding='utf-8') as f:
            json.dump(diagnosis_results, f, indent=2, ensure_ascii=False)
        
        # 生成诊断报告
        self.generate_diagnosis_report(diagnosis_results)
        
        print("标注问题诊断完成！")
    
    def generate_diagnosis_report(self, diagnosis_results):
        """生成诊断报告"""
        report_lines = []
        
        report_lines.append("BEVFusion PKL文件标注问题诊断报告")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        for file_name, diagnosis in diagnosis_results.items():
            report_lines.append(f"文件: {file_name.upper()}")
            report_lines.append("-" * 40)
            
            file_info = diagnosis["file_info"]
            report_lines.append(f"  文件名: {file_info['file_name']}")
            report_lines.append(f"  数据类型: {file_info['data_type']}")
            report_lines.append(f"  总样本数: {file_info['total_samples']}")
            report_lines.append("")
            
            # 样本分析
            if diagnosis["sample_analysis"]:
                report_lines.append("  样本结构分析:")
                for sample_key, sample_analysis in list(diagnosis["sample_analysis"].items())[:3]:  # 只显示前3个样本
                    report_lines.append(f"    {sample_key}:")
                    report_lines.append(f"      所有字段: {sample_analysis['all_keys']}")
                    if sample_analysis['keys_with_ann_like_names']:
                        report_lines.append(f"      标注相关字段: {sample_analysis['keys_with_ann_like_names']}")
                    else:
                        report_lines.append(f"      标注相关字段: 未找到")
                    
                    if sample_analysis['camera_info']:
                        report_lines.append(f"      相机数量: {sample_analysis['camera_info']['num_cameras']}")
                        report_lines.append(f"      相机名称: {sample_analysis['camera_info']['camera_names']}")
                report_lines.append("")
            
            # 标注字段搜索
            annotation_search = diagnosis["annotation_search"]
            if annotation_search["possible_annotation_fields"]:
                report_lines.append("  可能的标注字段:")
                for field, info in annotation_search["possible_annotation_fields"].items():
                    report_lines.append(f"    {field}: {info['type']} (出现{info['frequency']}次)")
            else:
                report_lines.append("  可能的标注字段: 未找到")
            
            report_lines.append("")
            
            # 生成建议
            report_lines.append("  建议:")
            if file_info['total_samples'] == 0:
                report_lines.append("    - 文件为空，请检查数据集")
            elif not any(diagnosis["sample_analysis"][key]["keys_with_ann_like_names"] 
                        for key in diagnosis["sample_analysis"]):
                report_lines.append("    - 未找到标注字段，这可能是一个测试集")
                report_lines.append("    - 标注信息可能在单独的文件中")
                report_lines.append("    - 检查是否有其他PKL文件包含标注信息")
            else:
                report_lines.append("    - 找到可能的标注字段，但需要进一步验证")
            
            report_lines.append("")
        
        # 检查数据库信息文件
        db_path = Path("/mnt/bevfusion_mit_xmy/data/nuscenes/nuscenes_dbinfos_train.pkl")
        if db_path.exists():
            report_lines.append("数据库信息文件分析:")
            report_lines.append("-" * 40)
            with open(db_path, 'rb') as f:
                db_data = pickle.load(f)
            report_lines.append(f"  包含的类别: {list(db_data.keys())}")
            for class_name in list(db_data.keys())[:5]:  # 显示前5个类别
                if db_data[class_name]:
                    report_lines.append(f"  {class_name}: {len(db_data[class_name])}个样本")
        
        # 写入报告文件
        with open(self.output_dir / "annotation_diagnosis_report.txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))

def main():
    diagnoser = AnnotationDiagnoser()
    results = diagnoser.diagnose_annotation_issue()
    
    print("\n诊断总结:")
    for file_name, diagnosis in results.items():
        file_info = diagnosis["file_info"]
        print(f"{file_name}: {file_info['total_samples']}个样本")
        
        # 检查是否有标注字段
        has_annotation = False
        for sample_key, sample_analysis in diagnosis["sample_analysis"].items():
            if sample_analysis["keys_with_ann_like_names"]:
                has_annotation = True
                break
        
        if has_annotation:
            print(f"  ✅ 找到可能的标注字段")
        else:
            print(f"  ❌ 未找到标注字段")

if __name__ == "__main__":
    main()