#!/usr/bin/env python3
"""
Step2-4: nuscenes数据一致性校验工具 - 修复标定输出
版本: v4.9-consistency-check-fixed
功能: 对比转换后nusc_tyjt数据与原始tyjt数据的一致性
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from nuscenes import NuScenes

# ==================== 配置参数 ====================
TYJT_ROOT = "/mnt/dataset/tyjt_RawData"
NUSC_ROOT = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt"
OUTPUT_DIR = "/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step2_4_checkConsistency"
NUSC_VERSION = "v1.0-tyjt"

# 类别映射 (与转换工具保持一致)
CATEGORY_MAPPING = {
    "car": "vehicle.car",
    "truck": "vehicle.truck",
    "construction_truck": "vehicle.construction",
    "van": "vehicle.van",
    "bus": "vehicle.bus",
    "robot": "vehicle.emergency.vehicle",
    "pedestrian": "human.pedestrian.adult",
    "cyclist": "human.pedestrian.cyclist",
    "bicycle": "vehicle.bicycle",
    "tricycle": "vehicle.tricycle",
    "tricyclist": "human.pedestrian.police_officer",
    "trolley": "vehicle.trailer",
    "cone": "movable_object.trafficcone",
    "barrier": "movable_object.barrier",
    "other": "movable_object.debris"
}

def create_output_dirs():
    """创建输出目录"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subdirs = ['category_comparison', 'position_validation', 'calibration_check', 'summary']
    for subdir in subdirs:
        os.makedirs(Path(OUTPUT_DIR) / subdir, exist_ok=True)
    return subdirs

def find_original_annotation_by_timestamp(timestamp):
    """根据时间戳查找原始标注文件"""
    tyjt_root = Path(TYJT_ROOT)
    
    for package in tyjt_root.iterdir():
        if not package.is_dir() or package.name.startswith('.'):
            continue
            
        datasets_dir = package / "datasets"
        if not datasets_dir.exists():
            continue
            
        for sub_packet in datasets_dir.iterdir():
            if not sub_packet.is_dir():
                continue
                
            label_file = sub_packet / "lidar" / "label" / f"{timestamp}.json"
            if label_file.exists():
                try:
                    with open(label_file, 'r') as f:
                        label_data = json.load(f)
                    
                    if isinstance(label_data, dict) and 'objects' in label_data:
                        return label_data['objects'], str(label_file)
                    elif isinstance(label_data, list):
                        return label_data, str(label_file)
                except Exception as e:
                    print(f"  读取原始标注失败 {label_file}: {e}")
    
    return None, None

def load_original_calibration(package_name):
    """加载原始标定文件"""
    calib_path = Path(TYJT_ROOT) / package_name / "calib" / "sensor2map_calib.json"
    if calib_path.exists():
        try:
            with open(calib_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  读取原始标定失败 {calib_path}: {e}")
    return None

def compare_annotation_consistency(nusc, sample_indices=None):
    """对比标注一致性"""
    if sample_indices is None:
        sample_indices = range(min(10, len(nusc.sample)))
    
    comparison_results = {}
    
    for i in sample_indices:
        sample = nusc.sample[i]
        sample_token = sample['token']
        
        print(f"\n📊 样本 {i} 一致性校验:")
        print(f"  Token: {sample_token[:8]}")
        
        # 获取时间戳
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        timestamp = Path(lidar_data['filename']).stem
        
        # 查找原始标注
        original_objects, original_file = find_original_annotation_by_timestamp(timestamp)
        
        comparison_results[i] = {
            'sample_token': sample_token,
            'timestamp': timestamp,
            'nusc_ann_count': len(sample['anns']),
            'original_ann_count': len(original_objects) if original_objects else 0,
            'original_file': original_file,
            'category_matches': [],
            'position_matches': [],
            'issues': []
        }
        
        if not original_objects:
            print(f"  ⚠️  未找到原始标注文件")
            comparison_results[i]['issues'].append("未找到原始标注文件")
            continue
        
        print(f"  原始标注文件: {Path(original_file).name}")
        print(f"  标注数量对比: nusc={len(sample['anns'])}, 原始={len(original_objects)}")
        
        # 详细对比每个标注
        for ann_idx, ann_token in enumerate(sample['anns']):
            ann = nusc.get('sample_annotation', ann_token)
            
            # 在原始标注中寻找匹配
            matched_obj = find_matching_original_object(ann, original_objects)
            
            if matched_obj:
                position_diff = calculate_position_difference(ann, matched_obj)
                category_match = check_category_mapping(ann, matched_obj)
                
                match_info = {
                    'nusc_category': ann['category_name'],
                    'original_category': matched_obj['type'],
                    'position_diff': position_diff,
                    'category_correct': category_match
                }
                
                comparison_results[i]['category_matches'].append(match_info)
                comparison_results[i]['position_matches'].append(position_diff)
                
                status = "✅" if category_match and position_diff['distance'] < 1.0 else "⚠️"
                print(f"    {status} 标注{ann_idx}: {ann['category_name']} <- {matched_obj['type']}")
                print(f"      位置差异: {position_diff['distance']:.3f}m")
                
                if not category_match:
                    comparison_results[i]['issues'].append(
                        f"类别映射错误: {matched_obj['type']} -> {ann['category_name']}"
                    )
                if position_diff['distance'] > 2.0:
                    comparison_results[i]['issues'].append(
                        f"位置差异过大: {position_diff['distance']:.3f}m"
                    )
            else:
                comparison_results[i]['issues'].append(f"标注{ann_idx} 未找到原始对应")
                print(f"    ❌ 标注{ann_idx}: {ann['category_name']} - 未找到原始对应")
    
    return comparison_results

def find_matching_original_object(nusc_ann, original_objects, position_threshold=3.0):
    """在原始标注中寻找匹配的对象"""
    nusc_pos = np.array(nusc_ann['translation'])
    
    best_match = None
    min_distance = float('inf')
    
    for obj in original_objects:
        if 'box3d' in obj and len(obj['box3d']) >= 3:
            original_pos = np.array(obj['box3d'][:3])
            distance = np.linalg.norm(nusc_pos - original_pos)
            
            if distance < min_distance and distance < position_threshold:
                min_distance = distance
                best_match = obj
    
    return best_match

def calculate_position_difference(nusc_ann, original_obj):
    """计算位置差异"""
    nusc_pos = np.array(nusc_ann['translation'])
    original_pos = np.array(original_obj['box3d'][:3])
    
    diff_vector = nusc_pos - original_pos
    distance = np.linalg.norm(diff_vector)
    
    return {
        'distance': float(distance),
        'diff_vector': diff_vector.tolist(),
        'nusc_position': nusc_pos.tolist(),
        'original_position': original_pos.tolist()
    }

def check_category_mapping(nusc_ann, original_obj):
    """检查类别映射是否正确"""
    original_type = original_obj.get('type', 'other')
    expected_nusc_category = CATEGORY_MAPPING.get(original_type, 'movable_object.debris')
    
    return nusc_ann['category_name'] == expected_nusc_category

def validate_calibration_consistency(nusc):
    """验证标定一致性 - 修复版本"""
    print(f"\n🔧 验证标定一致性")
    print("=" * 50)
    
    calibration_results = {}
    
    # 获取场景信息以找到对应的原始标定
    scene = nusc.scene[0]
    scene_name = scene['name']
    print(f"场景: {scene_name}")
    
    # 加载原始标定
    original_calib = load_original_calibration(scene_name)
    if not original_calib:
        print("  ⚠️  未找到原始标定文件")
        return calibration_results
    
    # 检查第一个样本的标定信息
    sample = nusc.sample[0]
    sample_token = sample['token']
    
    print(f"样本: {sample_token[:8]}")
    
    # 标定名称映射
    calib_name_mapping = {
        'CAM_FRONT': 'SC_R1_Aw_CpcS_CAMR_new',
        'CAM_FRONT_RIGHT': 'SC_R1_Bn_CpcW_CAMR_new',
        'CAM_BACK': 'SC_R1_Ce_CpcN_CAMR_new', 
        'CAM_FRONT_LEFT': 'SC_R1_Ds_CpcE_CAMR_new'
    }
    
    for sensor_type, sensor_token in sample['data'].items():
        if not sensor_type.startswith('CAM'):
            continue
            
        try:
            sensor_data = nusc.get('sample_data', sensor_token)
            calib = nusc.get('calibrated_sensor', sensor_data['calibrated_sensor_token'])
            
            # 获取原始标定参数
            original_calib_name = calib_name_mapping.get(sensor_type)
            original_params = original_calib.get(original_calib_name) if original_calib_name else None
            
            calibration_results[sensor_type] = {
                'nusc_translation': calib['translation'],
                'nusc_rotation': calib['rotation'],
                'has_intrinsic': 'camera_intrinsic' in calib and bool(calib['camera_intrinsic']),
                'original_calib_available': original_params is not None,
                'original_translation': [original_params['tx'], original_params['ty'], original_params['tz']] if original_params else None,
                'original_rotation': [original_params['rx'], original_params['ry'], original_params['rz'], original_params['rw']] if original_params else None
            }
            
            print(f"  📷 {sensor_type}:")
            print(f"    NuScenes平移: {calib['translation']}")
            print(f"    NuScenes旋转: {calib['rotation'][:2]}...")
            
            if original_params:
                print(f"    原始标定平移: [{original_params['tx']:.2f}, {original_params['ty']:.2f}, {original_params['tz']:.2f}]")
                print(f"    原始标定旋转: [{original_params['rx']:.3f}, {original_params['ry']:.3f}, {original_params['rz']:.3f}, {original_params['rw']:.3f}]")
                
                # 计算平移差异
                nusc_trans = np.array(calib['translation'])
                original_trans = np.array([original_params['tx'], original_params['ty'], original_params['tz']])
                trans_diff = np.linalg.norm(nusc_trans - original_trans)
                print(f"    平移差异: {trans_diff:.3f}m")
                
                if trans_diff > 1.0:
                    print(f"    ⚠️  平移差异较大")
            else:
                print(f"    ❌ 未找到原始标定参数")
            
            print(f"    有内参: {calibration_results[sensor_type]['has_intrinsic']}")
            
        except Exception as e:
            print(f"  ❌ 加载{sensor_type}标定失败: {e}")
    
    # 保存详细的标定对比结果
    calib_output_path = Path(OUTPUT_DIR) / "calibration_check" / "calibration_comparison.json"
    with open(calib_output_path, 'w') as f:
        json.dump(calibration_results, f, indent=2, ensure_ascii=False)
    print(f"  💾 标定对比结果保存至: {calib_output_path}")
    
    return calibration_results

def generate_consistency_report(comparison_results, calibration_results):
    """生成一致性报告"""
    print(f"\n📋 数据一致性报告")
    print("=" * 50)
    
    total_samples = len(comparison_results)
    total_issues = sum(len(result['issues']) for result in comparison_results.values())
    category_errors = 0
    position_errors = 0
    
    for sample_idx, result in comparison_results.items():
        for match in result['category_matches']:
            if not match['category_correct']:
                category_errors += 1
            if match['position_diff']['distance'] > 1.0:
                position_errors += 1
    
    # 标定统计
    calib_sensors = len(calibration_results)
    calib_with_original = sum(1 for calib in calibration_results.values() if calib['original_calib_available'])
    
    report = {
        'summary': {
            'total_samples_checked': total_samples,
            'total_issues_found': total_issues,
            'category_mapping_errors': category_errors,
            'position_difference_errors': position_errors,
            'calibration_sensors_checked': calib_sensors,
            'calibration_with_original_data': calib_with_original
        },
        'sample_details': {
            f'sample_{i}': {
                'nusc_annotations': result['nusc_ann_count'],
                'original_annotations': result['original_ann_count'],
                'match_count': len(result['category_matches']),
                'issues': result['issues'][:3]  # 只显示前3个问题
            } for i, result in comparison_results.items()
        },
        'calibration_summary': {
            sensor: {
                'has_original': calib['original_calib_available'],
                'has_intrinsic': calib['has_intrinsic']
            } for sensor, calib in calibration_results.items()
        },
        'recommendations': []
    }
    
    # 生成建议
    if category_errors > 0:
        report['recommendations'].append("修复类别映射表中的错误映射")
    if position_errors > 0:
        report['recommendations'].append("检查坐标转换逻辑，确保位置一致性")
    if calib_with_original < calib_sensors:
        report['recommendations'].append("检查标定名称映射，确保所有传感器都能找到原始标定")
    
    print(f"📊 校验统计:")
    print(f"  检查样本数: {total_samples}")
    print(f"  发现问题数: {total_issues}")
    print(f"  类别映射错误: {category_errors}")
    print(f"  位置差异过大: {position_errors}")
    print(f"  标定传感器: {calib_sensors}")
    print(f"  有原始标定: {calib_with_original}")
    
    print(f"\n💡 建议:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    return report

def main():
    """主函数"""
    print("🚀 Step2-4: nuscenes数据一致性校验工具 v4.9-fixed")
    print(f"原始数据: {TYJT_ROOT}")
    print(f"转换数据: {NUSC_ROOT}")
    print(f"输出路径: {OUTPUT_DIR}")
    
    # 创建输出目录
    create_output_dirs()
    
    # 加载nuscenes数据
    try:
        nusc = NuScenes(version=NUSC_VERSION, dataroot=NUSC_ROOT, verbose=False)
        print(f"✅ NuScenes加载成功")
        print(f"   样本: {len(nusc.sample)}, 标注: {len(nusc.sample_annotation)}")
        print(f"   场景: {len(nusc.scene)}")
    except Exception as e:
        print(f"❌ NuScenes加载失败: {e}")
        return
    
    # 1. 对比标注一致性
    comparison_results = compare_annotation_consistency(nusc, range(min(5, len(nusc.sample))))
    
    # 2. 验证标定一致性
    calibration_results = validate_calibration_consistency(nusc)
    
    # 3. 生成报告
    report = generate_consistency_report(comparison_results, calibration_results)
    
    # 保存结果
    report_path = Path(OUTPUT_DIR) / "summary" / "consistency_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    comparison_path = Path(OUTPUT_DIR) / "category_comparison" / "detailed_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Step2-4 完成!")
    print(f"💾 报告保存至: {report_path}")

if __name__ == "__main__":
    main()