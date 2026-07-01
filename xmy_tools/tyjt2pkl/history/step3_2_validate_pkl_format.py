# step3_2_validate_pkl_format.py
import pickle
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any

class FixedPklValidator:
    def __init__(self, original_pkl_dir: str, generated_pkl_dir: str, output_dir: str = "./step3_2_output_fixed"):
        self.original_pkl_dir = Path(original_pkl_dir)
        self.generated_pkl_dir = Path(generated_pkl_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_original_pkl(self) -> Dict:
        """加载原始BEVFusion PKL文件"""
        original_train_pkl = self.original_pkl_dir / "nuscenes_infos_train.pkl"
        with open(original_train_pkl, 'rb') as f:
            return pickle.load(f)
    
    def load_generated_pkl(self) -> Dict:
        """加载生成的修复版PKL文件"""
        generated_train_pkl = self.generated_pkl_dir / "tyjt_infos_train.pkl"
        with open(generated_train_pkl, 'rb') as f:
            return pickle.load(f)
    
    def compare_structure(self, original: Dict, generated: Dict) -> Dict:
        """比较整体结构差异"""
        comparison = {
            'overall_structure': {},
            'key_differences': [],
            'improvements': []
        }
        
        # 顶层结构比较
        original_keys = set(original.keys())
        generated_keys = set(generated.keys())
        
        comparison['overall_structure'] = {
            'original_keys': list(original_keys),
            'generated_keys': list(generated_keys),
            'common_keys': list(original_keys & generated_keys),
            'missing_in_generated': list(original_keys - generated_keys),
            'extra_in_generated': list(generated_keys - original_keys)
        }
        
        # 检查关键差异
        if 'infos' not in generated:
            comparison['key_differences'].append("❌ 生成的PKL缺少'infos'字段")
        else:
            comparison['improvements'].append("✅ 'infos'字段存在")
            
        if 'metadata' not in generated:
            comparison['key_differences'].append("❌ 生成的PKL缺少'metadata'字段")
        else:
            comparison['improvements'].append("✅ 'metadata'字段存在")
        
        return comparison
    
    def compare_sample_structure(self, original_sample: Dict, generated_sample: Dict) -> Dict:
        """比较样本结构差异"""
        comparison = {
            'basic_fields': {},
            'camera_fields': {},
            'annotation_fields': {},
            'coordinate_fields': {},
            'differences': [],
            'improvements': []
        }
        
        # 基本字段比较
        original_basic_keys = set(original_sample.keys())
        generated_basic_keys = set(generated_sample.keys())
        
        comparison['basic_fields'] = {
            'original_keys_count': len(original_basic_keys),
            'generated_keys_count': len(generated_basic_keys),
            'common_keys': list(original_basic_keys & generated_basic_keys),
            'missing_in_generated': list(original_basic_keys - generated_basic_keys),
            'extra_in_generated': list(generated_basic_keys - original_basic_keys)
        }
        
        # 检查之前缺失的字段是否已修复
        previously_missing = [
            'prev', 'lidar2ego_rotation', 'radars', 'gt_boxes', 'num_lidar_pts', 
            'ego2global_rotation', 'ego2global_translation', 'num_radar_pts', 
            'prev_token', 'lidar2ego_translation', 'gt_velocity', 'gt_names', 'valid_flag'
        ]
        
        fixed_fields = []
        still_missing = []
        for field in previously_missing:
            if field in generated_sample:
                fixed_fields.append(field)
                comparison['improvements'].append(f"✅ 已修复字段: {field}")
            else:
                still_missing.append(field)
                comparison['differences'].append(f"❌ 仍然缺失字段: {field}")
        
        # 相机数据比较
        if 'cams' in original_sample and 'cams' in generated_sample:
            comparison['camera_fields'] = self.compare_camera_data(
                original_sample['cams'], generated_sample['cams']
            )
        else:
            if 'cams' not in generated_sample:
                comparison['differences'].append("❌ 生成的样本缺少'cams'字段")
            else:
                comparison['improvements'].append("✅ 'cams'字段存在")
        
        # 标注数据比较
        comparison['annotation_fields'] = self.compare_annotation_data(original_sample, generated_sample)
        
        # 坐标系字段比较
        comparison['coordinate_fields'] = self.compare_coordinate_fields(original_sample, generated_sample)
        
        return comparison
    
    def compare_camera_data(self, original_cams: Dict, generated_cams: Dict) -> Dict:
        """比较相机数据差异"""
        comparison = {
            'camera_count': {
                'original': len(original_cams),
                'generated': len(generated_cams),
                'note': 'TYJT只有4个相机，NuScenes有6个，这是硬件差异'
            },
            'camera_names': {
                'original': list(original_cams.keys()),
                'generated': list(generated_cams.keys())
            },
            'camera_structure': {},
            'field_analysis': {}
        }
        
        # 相机名称差异
        original_cam_names = set(original_cams.keys())
        generated_cam_names = set(generated_cams.keys())
        
        comparison['camera_names'].update({
            'common_cameras': list(original_cam_names & generated_cam_names),
            'missing_cameras': list(original_cam_names - generated_cam_names),
            'extra_cameras': list(generated_cam_names - original_cam_names)
        })
        
        # 相机字段结构比较（使用第一个相机）
        if original_cams and generated_cams:
            first_original_cam = list(original_cams.values())[0]
            first_generated_cam = list(generated_cams.values())[0]
            
            comparison['camera_structure'] = self.compare_camera_structure(
                first_original_cam, first_generated_cam
            )
        
        return comparison
    
    def compare_camera_structure(self, original_cam: Dict, generated_cam: Dict) -> Dict:
        """比较单个相机的字段结构"""
        comparison = {
            'required_fields_analysis': {},
            'data_type_analysis': {},
            'shape_analysis': {},
            'differences': [],
            'improvements': []
        }
        
        # 必需字段检查
        required_fields = [
            'data_path', 'cam_intrinsic', 'sensor2lidar_rotation', 
            'sensor2lidar_translation', 'sensor2ego_rotation',
            'sensor2ego_translation', 'ego2global_rotation',
            'ego2global_translation', 'timestamp'
        ]
        
        missing_required = []
        present_required = []
        for field in required_fields:
            if field not in generated_cam:
                missing_required.append(field)
                comparison['differences'].append(f"❌ 相机缺少必需字段: {field}")
            else:
                present_required.append(field)
                comparison['improvements'].append(f"✅ 相机必需字段存在: {field}")
        
        comparison['required_fields_analysis'] = {
            'required_fields': required_fields,
            'missing_required': missing_required,
            'present_required': present_required
        }
        
        # 数据类型和形状检查
        for field in required_fields:
            if field in original_cam and field in generated_cam:
                orig_val = original_cam[field]
                gen_val = generated_cam[field]
                
                # 数据类型比较
                orig_type = type(orig_val).__name__
                gen_type = type(gen_val).__name__
                
                if orig_type != gen_type:
                    comparison['data_type_analysis'][field] = {
                        'original': orig_type,
                        'generated': gen_type,
                        'match': False
                    }
                    comparison['differences'].append(f"❌ {field} 数据类型不匹配: {orig_type} vs {gen_type}")
                else:
                    comparison['data_type_analysis'][field] = {
                        'original': orig_type,
                        'generated': gen_type,
                        'match': True
                    }
                    comparison['improvements'].append(f"✅ {field} 数据类型匹配: {gen_type}")
                
                # 数组形状比较（如果是numpy数组）
                if isinstance(orig_val, np.ndarray) and isinstance(gen_val, np.ndarray):
                    if orig_val.shape != gen_val.shape:
                        comparison['shape_analysis'][field] = {
                            'original_shape': orig_val.shape,
                            'generated_shape': gen_val.shape,
                            'match': False
                        }
                        comparison['differences'].append(f"❌ {field} 形状不匹配: {orig_val.shape} vs {gen_val.shape}")
                    else:
                        comparison['shape_analysis'][field] = {
                            'original_shape': orig_val.shape,
                            'generated_shape': gen_val.shape,
                            'match': True
                        }
                        comparison['improvements'].append(f"✅ {field} 形状匹配: {gen_val.shape}")
        
        return comparison
    
    def compare_annotation_data(self, original_sample: Dict, generated_sample: Dict) -> Dict:
        """比较标注数据差异"""
        comparison = {
            'fields_present': {
                'original_annotation_fields': [k for k in original_sample.keys() if k.startswith('gt_') or k in ['num_lidar_pts', 'num_radar_pts', 'valid_flag']],
                'generated_annotation_fields': [k for k in generated_sample.keys() if k.startswith('gt_') or k in ['num_lidar_pts', 'num_radar_pts', 'valid_flag']]
            },
            'data_analysis': {},
            'differences': [],
            'improvements': []
        }
        
        # 检查必需标注字段
        required_ann_fields = ['gt_boxes', 'gt_names']
        for field in required_ann_fields:
            if field in original_sample and field not in generated_sample:
                comparison['differences'].append(f"❌ 标注字段缺失: {field}")
            elif field in original_sample and field in generated_sample:
                comparison['improvements'].append(f"✅ 标注字段存在: {field}")
                # 检查形状和数据类型
                orig_val = original_sample[field]
                gen_val = generated_sample[field]
                
                if isinstance(orig_val, np.ndarray) and isinstance(gen_val, np.ndarray):
                    if field == 'gt_boxes':
                        if orig_val.shape[1] == gen_val.shape[1]:
                            comparison['improvements'].append(f"✅ gt_boxes维度匹配: {gen_val.shape[1]}维")
                        else:
                            comparison['differences'].append(
                                f"❌ gt_boxes维度不匹配: {orig_val.shape[1]} vs {gen_val.shape[1]}"
                            )
        
        return comparison
    
    def compare_coordinate_fields(self, original_sample: Dict, generated_sample: Dict) -> Dict:
        """比较坐标系字段差异"""
        comparison = {
            'transform_fields': {},
            'differences': [],
            'improvements': []
        }
        
        # 检查坐标系变换字段
        coord_fields = ['lidar2global', 'ego2global', 'lidar2ego_translation', 'lidar2ego_rotation', 
                       'ego2global_translation', 'ego2global_rotation']
        
        for field in coord_fields:
            if field in original_sample and field not in generated_sample:
                comparison['differences'].append(f"❌ 坐标系字段缺失: {field}")
            elif field in original_sample and field in generated_sample:
                comparison['improvements'].append(f"✅ 坐标系字段存在: {field}")
                orig_val = original_sample[field]
                gen_val = generated_sample[field]
                
                if isinstance(orig_val, np.ndarray) and isinstance(gen_val, np.ndarray):
                    if orig_val.shape == gen_val.shape:
                        comparison['improvements'].append(f"✅ {field} 形状匹配: {gen_val.shape}")
                    else:
                        comparison['differences'].append(
                            f"❌ {field} 形状不匹配: {orig_val.shape} vs {gen_val.shape}"
                        )
        
        return comparison
    
    def generate_comprehensive_report(self, comparison_results: Dict):
        """生成综合对比报告"""
        report_lines = []
        
        report_lines.append("BEVFusion PKL格式修复验证报告")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 整体结构对比
        report_lines.append("1. 整体结构对比")
        report_lines.append("-" * 40)
        overall = comparison_results['overall_structure']
        report_lines.append(f"  原始PKL顶层字段: {overall['original_keys']}")
        report_lines.append(f"  生成PKL顶层字段: {overall['generated_keys']}")
        report_lines.append(f"  共同字段: {overall['common_keys']}")
        report_lines.append(f"  生成PKL缺少字段: {overall['missing_in_generated']}")
        report_lines.append(f"  生成PKL额外字段: {overall['extra_in_generated']}")
        
        # 改进总结
        if comparison_results['improvements']:
            report_lines.append("")
            report_lines.append("2. 修复改进总结")
            report_lines.append("-" * 40)
            for improvement in comparison_results['improvements'][:10]:  # 显示前10个改进
                report_lines.append(f"  {improvement}")
        
        # 样本结构对比
        if 'sample_comparison' in comparison_results:
            sample_comp = comparison_results['sample_comparison']
            report_lines.append("")
            report_lines.append("3. 样本结构对比")
            report_lines.append("-" * 40)
            
            # 基本字段
            basic_fields = sample_comp['basic_fields']
            report_lines.append("  基本字段:")
            report_lines.append(f"    原始样本字段数: {basic_fields['original_keys_count']}")
            report_lines.append(f"    生成样本字段数: {basic_fields['generated_keys_count']}")
            report_lines.append(f"    共同字段数: {len(basic_fields['common_keys'])}")
            
            if basic_fields['missing_in_generated']:
                report_lines.append(f"    ⚠️  仍然缺失字段: {basic_fields['missing_in_generated']}")
            else:
                report_lines.append("    ✅ 无缺失字段")
            
            if basic_fields['extra_in_generated']:
                report_lines.append(f"    📝 额外字段: {basic_fields['extra_in_generated']}")
            
            # 相机数据
            if sample_comp['camera_fields']:
                cam_fields = sample_comp['camera_fields']
                report_lines.append("")
                report_lines.append("  相机数据:")
                report_lines.append(f"    原始相机数量: {cam_fields['camera_count']['original']}")
                report_lines.append(f"    生成相机数量: {cam_fields['camera_count']['generated']}")
                report_lines.append(f"    📝 {cam_fields['camera_count']['note']}")
                report_lines.append(f"    共同相机: {cam_fields['camera_names']['common_cameras']}")
                report_lines.append(f"    缺失相机: {cam_fields['camera_names']['missing_cameras']}")
                report_lines.append(f"    额外相机: {cam_fields['camera_names']['extra_cameras']}")
        
        # 差异总结
        all_differences = []
        all_differences.extend(comparison_results.get('key_differences', []))
        if 'sample_comparison' in comparison_results:
            sample_comp = comparison_results['sample_comparison']
            all_differences.extend(sample_comp.get('differences', []))
            if 'camera_fields' in sample_comp and 'camera_structure' in sample_comp['camera_fields']:
                all_differences.extend(sample_comp['camera_fields']['camera_structure'].get('differences', []))
        
        # 总结
        report_lines.append("")
        report_lines.append("4. 格式兼容性总结")
        report_lines.append("-" * 40)
        
        if not all_differences:
            report_lines.append("🎉 格式完全兼容！修复版PKL文件可以直接用于BEVFusion训练")
            report_lines.append("")
            report_lines.append("✅ 主要成就:")
            report_lines.append("   - 所有必需字段都已添加")
            report_lines.append("   - 相机数据结构正确")
            report_lines.append("   - 标注字段格式匹配")
            report_lines.append("   - 坐标系变换字段完整")
        else:
            report_lines.append(f"⚠️  发现 {len(all_differences)} 个兼容性问题，需要进一步修复")
            report_lines.append("")
            report_lines.append("❌ 剩余问题:")
            for diff in all_differences:
                report_lines.append(f"   - {diff}")
        
        # 写入报告文件
        report_path = self.output_dir / "fixed_pkl_validation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        
        print(f"修复验证报告已保存: {report_path}")
        
        return len(all_differences) == 0
    
    def validate_fixed_pkl_format(self):
        """执行修复版PKL格式验证"""
        print("开始修复版PKL格式验证...")
        
        try:
            # 加载文件
            original_data = self.load_original_pkl()
            generated_data = self.load_generated_pkl()
            
            print("✅ 文件加载成功")
            
            # 整体结构对比
            comparison_results = self.compare_structure(original_data, generated_data)
            
            # 样本结构对比（使用第一个样本）
            if original_data['infos'] and generated_data['infos']:
                original_sample = original_data['infos'][0]
                generated_sample = generated_data['infos'][0]
                
                sample_comparison = self.compare_sample_structure(original_sample, generated_sample)
                comparison_results['sample_comparison'] = sample_comparison
            
            # 保存详细对比结果
            comparison_path = self.output_dir / "fixed_detailed_comparison.json"
            with open(comparison_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, indent=2, ensure_ascii=False)
            
            print(f"详细对比结果已保存: {comparison_path}")
            
            # 生成综合报告并检查兼容性
            is_fully_compatible = self.generate_comprehensive_report(comparison_results)
            
            return is_fully_compatible, comparison_results
            
        except Exception as e:
            print(f"❌ 格式验证失败: {e}")
            raise

def main():
    """主函数"""
    
    # 配置路径
    original_pkl_dir = "/mnt/bevfusion_mit_xmy/data/nuscenes"  # 原始BEVFusion PKL文件目录
    generated_pkl_dir = "./step3_1_output"  # 生成的修复版PKL文件目录
    output_dir = "./step3_2_output"
    
    print("🚀 开始修复版PKL格式验证...")
    print(f"原始PKL目录: {original_pkl_dir}")
    print(f"生成PKL目录: {generated_pkl_dir}")
    print(f"输出目录: {output_dir}")
    print()
    
    try:
        validator = FixedPklValidator(original_pkl_dir, generated_pkl_dir, output_dir)
        is_compatible, results = validator.validate_fixed_pkl_format()
        
        if is_compatible:
            print("\n🎉 恭喜！修复版PKL文件格式完全兼容BEVFusion！")
            print("   现在可以开始进行BEVFusion训练了！")
        else:
            print("\n⚠️  修复版PKL文件仍有兼容性问题，请查看报告进行进一步修复")
            
    except Exception as e:
        print(f"❌ PKL格式验证失败: {e}")
        raise

if __name__ == "__main__":
    main()