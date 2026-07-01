# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.

from typing import Callable

import numpy as np

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
from nuscenes.eval.detection.data_classes import DetectionMetricData


Debug = True
from collections import defaultdict
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

if Debug:
    import time
    print(f">>>[xmy]🟤[/opt/venv/lib/python3.8/site-packages/nuscenes/eval/detection/algo.py] >>> [Debug = True] ")

# 🟤 全局数据收集器和配置
if Debug:
    # 配置选项
    DEBUG_CONFIG = {
        'output_mode': 'both',  # 'print', 'file', 'both'
        'file_path': '/tmp/nuscenes_eval_debug',  # 文件保存路径
        'file_format': 'json',  # 'json', 'txt', 'both'
        'min_pred_count_to_analyze': 1,  # 最小预测数才进行分析
        'analyze_all_classes': True,  # 分析所有类别
        'last_class_name': 'barrier'  # 假设的最后一个类别
    }
    
    # 全局数据收集器
    global_analysis_data = {
        'all_classes': defaultdict(list),
        'class_details': defaultdict(dict),
        'error_reasons': defaultdict(lambda: defaultdict(int)),
        'distance_distribution': defaultdict(lambda: defaultdict(int)),
        'confidence_distribution': defaultdict(list),
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'evaluation_info': {}
    }
    
    # 输出缓冲区
    output_buffer = []
    
    # 定义距离阈值分类
    DISTANCE_CATEGORIES = {
        'very_close': 0.5,    # 0-0.5m
        'close': 1.0,         # 0.5-1.0m
        'medium': 2.0,        # 1.0-2.0m
        'far': 4.0,           # 2.0-4.0m
        'very_far': 10.0      # >4.0m
    }
    # 全局数据收集器
    global_analysis_data = {
        'all_classes': defaultdict(list),
        'class_details': defaultdict(dict),
        'error_reasons': defaultdict(lambda: defaultdict(int)),
        'distance_distribution': defaultdict(lambda: defaultdict(int)),
        'confidence_distribution': defaultdict(list),
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'evaluation_info': {},
        'ap_info': {},  # 🔴 新增：存储每个类别的AP信息
        'processed_classes': []  # 🔴 新增：记录已处理的类别
    }


def xmy_set_debug_config(**kwargs):
    """设置调试配置"""
    if not Debug:
        return
    DEBUG_CONFIG.update(kwargs)
    print(f">>>[xmy]🟤[algo.py] [XMY] 调试配置已更新: {kwargs}")


def accumulate(gt_boxes: EvalBoxes,
               pred_boxes: EvalBoxes,
               class_name: str,
               dist_fcn: Callable,
               dist_th: float,
               verbose: bool = False) -> DetectionMetricData:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])
    if verbose:
        print("Found {} GT of class {} out of {} total across {} samples.".
              format(npos, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    # 🔴 添加简洁的调试
    if Debug:
        print(f"\n>>>[xmy]🔵 [algo.py]  [匹配统计] {class_name}匹配开始:")
        print(f"   距离阈值: {dist_th}m")
        print(f"   GT数量: {npos}")
        
        # 快速统计预测数量
        pred_count = len([1 for box in pred_boxes.all if box.detection_name == class_name])
        print(f"   预测数量: {pred_count}")
        
        # 如果有预测，分析前几个
        if pred_count > 0:
            pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
            confidences = [p.detection_score for p in pred_boxes_list]
            print(f"   预测置信度范围: {min(confidences):.3f} ~ {max(confidences):.3f}")
            
    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        if Debug:
            # 即使没有GT，也收集预测信息
            xmy_collect_class_data(class_name, pred_boxes, gt_boxes, dist_th, npos)
            # 记录AP信息
            global_analysis_data['ap_info'][class_name] = {
                'distance_threshold': dist_th,
                'gt_count': 0,
                'pred_count': pred_count,
                'ap': 0.0,
                'tp': 0,
                'fp': pred_count
            }
        return DetectionMetricData.no_predictions()

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    if verbose:
        print("Found {} PRED of class {} out of {} total across {} samples.".
              format(len(pred_confs), class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': []}

    # 🔴 初始化当前类别的分析数据收集器
    if Debug:
        current_class_data = xmy_init_class_data(class_name, dist_th, npos, len(pred_boxes_list))

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    if Debug:
        match_stats = {'matches': 0, 'no_match': 0, 'no_gt': 0}

    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):

            # Find closest match among ground truth boxes
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        # 🔴 收集匹配数据
        if Debug:
            if is_match:
                match_stats['matches'] += 1
            else:
                if min_dist < float('inf'):
                    match_stats['no_match'] += 1
                else:
                    match_stats['no_gt'] += 1
            
            # 收集详细匹配信息
            match_info = {
                'sample_token': pred_box.sample_token[:8],
                'confidence': pred_box.detection_score,
                'min_distance': min_dist,
                'is_match': is_match
            }
            
            if is_match:
                current_class_data['tp_details'].append(match_info)
                current_class_data['match_distances'].append(min_dist)
                current_class_data['confidence_tp'].append(pred_box.detection_score)
            else:
                current_class_data['fp_details'].append(match_info)
                current_class_data['confidence_fp'].append(pred_box.detection_score)

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]

            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box.detection_score)

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        if Debug:
            # 保存当前类别的数据
            xmy_save_class_data(current_class_data, tp, fp, conf)
            
            print(f"\n>>>[xmy]🟤 [algo.py] [{class_name}匹配统计] 距离阈值={dist_th}m:")
            print(f"   总预测数: {len(pred_boxes_list)}")
            print(f"   成功匹配: 0")
            print(f"   未匹配(距离>阈值): 0")
            print(f"   无GT(找不到对应GT): 0")
            print(f"   📊 AP信息: 没有匹配，AP=0.0")
            
            # 保存AP信息
            global_analysis_data['ap_info'][class_name] = {
                'distance_threshold': dist_th,
                'gt_count': npos,
                'pred_count': len(pred_boxes_list),
                'ap': 0.0,
                'tp': 0,
                'fp': len(pred_boxes_list),
                'precision': 0.0,
                'recall': 0.0
            }
        
        # 🔴 修复：在这里直接返回，不要继续执行后面的代码
        return DetectionMetricData.no_predictions()

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    # 🔴 输出匹配统计摘要
    if Debug:
        print(f"\n>>>[xmy]🟤 [algo.py] [{class_name}匹配统计] 距离阈值={dist_th}m:")
        print(f"   总预测数: {len(pred_boxes_list)}")
        print(f"   成功匹配: {match_stats['matches']}")
        print(f"   未匹配(距离>阈值): {match_stats['no_match']}")
        print(f"   无GT(找不到对应GT): {match_stats['no_gt']}")
        
        # 计算AP信息 - 🔴 修复：现在prec已经定义，不会出现未定义错误
        if len(tp) > 0:
            tp_cumulative = int(tp[-1]) if len(tp) > 0 else 0
            fp_cumulative = int(fp[-1]) if len(fp) > 0 else 0
            
            # 计算简单AP（曲线下面积）
            valid_indices = ~np.isnan(prec)
            simple_ap = 0.0
            if np.sum(valid_indices) > 0:
                simple_ap = np.trapz(prec[valid_indices], rec[valid_indices])
            
            print(f"   📊 AP信息:")
            print(f"     匹配数: {len(match_data['trans_err'])}")
            print(f"     TP累计: {tp_cumulative}")
            print(f"     FP累计: {fp_cumulative}")
            print(f"     简单AP(曲线面积): {simple_ap:.4f}")
            print(f"     最终precision: {prec[-1]:.4f}")
            print(f"     最终recall: {rec[-1]:.4f}")
            
            # 保存AP信息
            global_analysis_data['ap_info'][class_name] = {
                'distance_threshold': dist_th,
                'gt_count': npos,
                'pred_count': len(pred_boxes_list),
                'ap': float(simple_ap),
                'tp': tp_cumulative,
                'fp': fp_cumulative,
                'precision': float(prec[-1]),
                'recall': float(rec[-1])
            }
        
        if match_stats['matches'] == 0 and len(pred_boxes_list) > 0:
            print(f"   🔴 问题诊断: 所有预测都没有匹配!")
            print(f"      1. 预测位置与GT相差太大")
            print(f"      2. 距离阈值 {dist_th}m 太严格")
            print(f"      3. GT和预测在不同样本中")
        elif len(pred_boxes_list) == 0:
            print(f"   ⚠️  警告: 该类别没有预测框!")

    # 🔴 保存当前类别的数据到全局收集器
    if Debug:
        xmy_save_class_data(current_class_data, tp, fp, conf)
        
        # 🔴 在所有类别处理完后执行分析
        if 'processed_classes' not in global_analysis_data:
            global_analysis_data['processed_classes'] = []
        
        if class_name not in global_analysis_data['processed_classes']:
            global_analysis_data['processed_classes'].append(class_name)
        
        # 假设总共有10个类别
        total_classes = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 
                        'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
        
        processed_set = set(global_analysis_data['processed_classes'])
        total_set = set(total_classes)
        
        if processed_set == total_set:
            # 收集评估信息
            global_analysis_data['evaluation_info'] = {
                'total_samples': len(gt_boxes.sample_tokens),
                'total_gt_boxes': len(gt_boxes.all),
                'total_pred_boxes': len(pred_boxes.all),
                'distance_threshold': dist_th,
                'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'processed_classes': list(processed_set)
            }
            
            # 执行分析
            analysis_results = xmy_analyze_all_classes()
            
            # 🔴 输出所有类别的AP汇总
            xmy_print_ap_summary()
            
            # 根据配置输出结果
            if DEBUG_CONFIG['output_mode'] in ['print', 'both']:
                xmy_print_analysis(analysis_results)
            
            if DEBUG_CONFIG['output_mode'] in ['file', 'both']:
                xmy_save_analysis_to_file(analysis_results)

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(recall=rec,
                               precision=prec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err'])

def xmy_print_ap_summary():
    """打印所有类别的AP汇总信息"""
    if not Debug or 'ap_info' not in global_analysis_data:
        return
    
    print("\n" + "="*80)
    print("XMY AP汇总报告 - 所有类别")
    print("="*80)
    
    ap_info = global_analysis_data['ap_info']
    
    print(f"\n{'类别':<20} {'距离阈值':<10} {'GT数':<8} {'预测数':<8} {'TP数':<8} {'FP数':<8} {'AP':<8} {'Precision':<10} {'Recall':<10}")
    print("-" * 100)
    
    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_ap = 0.0
    class_count = 0
    
    for class_name in sorted(ap_info.keys()):
        info = ap_info[class_name]
        gt_count = info.get('gt_count', 0)
        pred_count = info.get('pred_count', 0)
        tp_count = info.get('tp', 0)
        fp_count = info.get('fp', 0)
        ap_value = info.get('ap', 0.0)
        precision = info.get('precision', 0.0)
        recall = info.get('recall', 0.0)
        dist_th = info.get('distance_threshold', 0.0)
        
        total_gt += gt_count
        total_pred += pred_count
        total_tp += tp_count
        total_fp += fp_count
        total_ap += ap_value
        class_count += 1
        
        print(f"{class_name:<20} {dist_th:<10.1f} {gt_count:<8} {pred_count:<8} "
              f"{tp_count:<8} {fp_count:<8} {ap_value:<8.4f} {precision:<10.4f} {recall:<10.4f}")
    
    print("-" * 100)
    
    # 计算平均值
    avg_ap = total_ap / max(1, class_count)
    overall_precision = total_tp / max(1, total_tp + total_fp)
    overall_recall = total_tp / max(1, total_gt)
    
    print(f"{'平均/总计':<20} {'-':<10} {total_gt:<8} {total_pred:<8} "
          f"{total_tp:<8} {total_fp:<8} {avg_ap:<8.4f} {overall_precision:<10.4f} {overall_recall:<10.4f}")
    
    # 问题诊断
    print(f"\n🔍 问题诊断:")
    
    # 检查哪些类别没有预测
    no_pred_classes = []
    for class_name, info in ap_info.items():
        if info.get('pred_count', 0) == 0 and info.get('gt_count', 0) > 0:
            no_pred_classes.append(class_name)
    
    if no_pred_classes:
        print(f"   1. 以下类别有GT但没有预测: {', '.join(no_pred_classes)}")
    
    # 检查哪些类别AP为0
    zero_ap_classes = []
    for class_name, info in ap_info.items():
        if info.get('ap', 0.0) == 0 and info.get('pred_count', 0) > 0:
            zero_ap_classes.append(class_name)
    
    if zero_ap_classes:
        print(f"   2. 以下类别AP为0但有预测: {', '.join(zero_ap_classes)}")
    
    print("="*80 + "\n")


def calc_ap(md: DetectionMetricData, min_recall: float, min_precision: float) -> float:
    """ Calculated average precision. """

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(md.precision)
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def calc_tp(md: DetectionMetricData, min_recall: float, metric_name: str) -> float:
    """ Calculates true positive errors. """

    first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.
    last_ind = md.max_recall_ind  # First instance of confidence = 0 is index of max achieved recall.
    if last_ind < first_ind:
        return 1.0  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
    else:
        return float(np.mean(getattr(md, metric_name)[first_ind: last_ind + 1]))  # +1 to include error at max recall.



# ==================================================================================================================
# ============================================================================
# 🟤 新增的调试分析函数
# ============================================================================
def xmy_init_class_data(class_name, dist_th, gt_count, pred_count):
    """初始化当前类别的数据收集器"""
    return {
        'class_name': class_name,
        'dist_th': dist_th,
        'gt_count': gt_count,
        'pred_count': pred_count,
        'tp_details': [],
        'fp_details': [],
        'match_distances': [],
        'error_reasons': defaultdict(int),
        'distance_distribution': defaultdict(int),
        'confidence_tp': [],
        'confidence_fp': [],
        'cumulative_tp': [],
        'cumulative_fp': [],
        'confidences': []
    }

def xmy_collect_match_data(class_data, pred_box, min_dist, is_match, dist_th, match_gt_idx, class_name):
    """收集单个预测的匹配数据"""
    
    match_info = {
        'sample_token': pred_box.sample_token[:8],
        'confidence': pred_box.detection_score,
        'min_distance': min_dist,
        'is_match': is_match
    }
    
    # 🔴 添加错误原因分类
    if not is_match:
        if min_dist >= float('inf'):
            match_info['error_reason'] = 'no_gt_in_sample'
        elif min_dist > dist_th:
            match_info['error_reason'] = 'distance_exceeds_threshold'
        else:
            match_info['error_reason'] = 'other'
    
    # 🔴 添加距离分类
    if min_dist < float('inf'):
        if min_dist <= 0.5:
            match_info['distance_category'] = 'very_close'
        elif min_dist <= 1.0:
            match_info['distance_category'] = 'close'
        elif min_dist <= 2.0:
            match_info['distance_category'] = 'medium'
        elif min_dist <= 4.0:
            match_info['distance_category'] = 'far'
        else:
            match_info['distance_category'] = 'very_far'
    
    if is_match:
        class_data['tp_details'].append(match_info)
        class_data['match_distances'].append(min_dist)
        class_data['confidence_tp'].append(pred_box.detection_score)
    else:
        class_data['fp_details'].append(match_info)
        class_data['confidence_fp'].append(pred_box.detection_score)


def xmy_save_class_data(class_data, tp, fp, conf):
    """保存当前类别的数据到全局收集器"""
    if not Debug:
        return
    
    class_name = class_data['class_name']
    
    # 保存基本统计
    global_analysis_data['class_details'][class_name] = {
        'gt_count': class_data['gt_count'],
        'pred_count': class_data['pred_count'],
        'tp_count': len(class_data['tp_details']),
        'fp_count': len(class_data['fp_details']),
        'match_ratio': len(class_data['tp_details']) / max(1, class_data['pred_count'])
    }
    
    # 收集到全局数据
    for tp_detail in class_data['tp_details']:
        global_analysis_data['all_classes']['tp'].append({
            'class': class_name,
            'confidence': tp_detail['confidence'],
            'distance': tp_detail['min_distance']
        })
    
    for fp_detail in class_data['fp_details']:
        # 🔴 修复：检查error_reason是否存在
        fp_item = {
            'class': class_name,
            'confidence': fp_detail['confidence'],
            'distance': fp_detail['min_distance']
        }
        
        # 只有存在error_reason时才添加
        if 'error_reason' in fp_detail and fp_detail['error_reason'] is not None:
            fp_item['error_reason'] = fp_detail['error_reason']
        
        global_analysis_data['all_classes']['fp'].append(fp_item)
    
    # 记录错误原因 - 🔴 修复：只有存在error_reason时才记录
    for fp_detail in class_data['fp_details']:
        if 'error_reason' in fp_detail and fp_detail['error_reason'] is not None:
            error_reason = fp_detail['error_reason']
            global_analysis_data['error_reasons'][class_name][error_reason] = \
                global_analysis_data['error_reasons'][class_name].get(error_reason, 0) + 1
    
    # 记录距离分布
    for fp_detail in class_data['fp_details']:
        if 'distance_category' in fp_detail:
            distance_cat = fp_detail['distance_category']
            global_analysis_data['distance_distribution'][class_name][distance_cat] = \
                global_analysis_data['distance_distribution'][class_name].get(distance_cat, 0) + 1
    
    # 记录置信度分布
    if class_data['confidence_tp']:
        global_analysis_data['confidence_distribution'][class_name].extend([
            {'type': 'tp', 'confidence': c} for c in class_data['confidence_tp']
        ])
    if class_data['confidence_fp']:
        global_analysis_data['confidence_distribution'][class_name].extend([
            {'type': 'fp', 'confidence': c} for c in class_data['confidence_fp']
        ])


def xmy_collect_class_data(class_name, pred_boxes, gt_boxes, dist_th, npos):
    """收集类别的数据（即使没有GT）"""
    if not Debug:
        return
    
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    if not pred_boxes_list:
        return
    
    # 初始化数据
    class_data = xmy_init_class_data(class_name, dist_th, npos, len(pred_boxes_list))
    
    # 收集所有预测作为FP
    for pred_box in pred_boxes_list:
        min_dist = float('inf')
        
        # 尝试找到最近的GT
        if pred_box.sample_token in gt_boxes.boxes:
            for gt_box in gt_boxes[pred_box.sample_token]:
                if gt_box.detection_name == class_name:
                    dist = center_distance(gt_box, pred_box)
                    if dist < min_dist:
                        min_dist = dist
        
        xmy_collect_match_data(
            class_data, pred_box, min_dist, False, dist_th, None, class_name
        )
    
    # 保存数据
    xmy_save_class_data(class_data, [], [], [])

def xmy_analyze_all_classes() -> Dict:
    """分析所有类别的TP/FP分布和错误原因"""
    if not Debug:
        return {}
    
    # 准备分析结果
    analysis_results = {
        'class_details': dict(global_analysis_data['class_details']),
        'all_tp': global_analysis_data['all_classes']['tp'],
        'all_fp': global_analysis_data['all_classes']['fp'],
        'error_reasons': dict(global_analysis_data['error_reasons']),
        'distance_distribution': dict(global_analysis_data['distance_distribution']),
        'confidence_distribution': dict(global_analysis_data['confidence_distribution'])
    }
    
    return analysis_results

def xmy_print_analysis(analysis_results: Dict):
    """打印分析结果到控制台"""
    if not Debug:
        return
    
    print("\n" + "="*80)
    print("XMY 调试分析报告 - 所有类别汇总")
    print(f"生成时间: {global_analysis_data['timestamp']}")
    print("="*80)
    
    # 输出评估信息
    eval_info = global_analysis_data.get('evaluation_info', {})
    if eval_info:
        print(f"\n评估信息:")
        print(f"  总样本数: {eval_info.get('total_samples', 'N/A')}")
        print(f"  GT框总数: {eval_info.get('total_gt_boxes', 'N/A')}")
        print(f"  预测框总数: {eval_info.get('total_pred_boxes', 'N/A')}")
        print(f"  距离阈值: {eval_info.get('distance_threshold', 'N/A')}m")
        print(f"  评估时间: {eval_info.get('evaluation_time', 'N/A')}")
    
    # 1. 分类型的TP/FP分布
    print("\n1. 分类型的TP/FP分布:")
    print("-" * 60)
    print(f"{'类别':<20} {'GT数':<8} {'预测数':<8} {'TP数':<8} {'FP数':<8} {'匹配率':<8}")
    print("-" * 60)
    
    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    
    for class_name, details in sorted(analysis_results['class_details'].items()):
        gt_count = details['gt_count']
        pred_count = details['pred_count']
        tp_count = details['tp_count']
        fp_count = details['fp_count']
        match_ratio = details['match_ratio']
        
        total_gt += gt_count
        total_pred += pred_count
        total_tp += tp_count
        total_fp += fp_count
        
        print(f"{class_name:<20} {gt_count:<8} {pred_count:<8} {tp_count:<8} {fp_count:<8} {match_ratio:.3f}")
    
    print("-" * 60)
    overall_match_ratio = total_tp / max(1, total_pred)
    print(f"{'总计':<20} {total_gt:<8} {total_pred:<8} {total_tp:<8} {total_fp:<8} {overall_match_ratio:.3f}")
    
    # 2. 整体的TP/FP分布
    print("\n2. 整体的TP/FP分布:")
    print("-" * 40)
    
    all_tp = analysis_results['all_tp']
    all_fp = analysis_results['all_fp']
    
    total_tp = len(all_tp)
    total_fp = len(all_fp)
    total_predictions = total_tp + total_fp
    
    print(f"总预测数: {total_predictions}")
    print(f"TP总数: {total_tp} ({total_tp/max(1,total_predictions)*100:.1f}%)")
    print(f"FP总数: {total_fp} ({total_fp/max(1,total_predictions)*100:.1f}%)")
    
    if all_tp:
        tp_confidences = [item['confidence'] for item in all_tp]
        tp_distances = [item['distance'] for item in all_tp if item['distance'] < float('inf')]
        
        if tp_confidences:
            print(f"TP置信度范围: {min(tp_confidences):.3f} - {max(tp_confidences):.3f}")
            print(f"TP平均置信度: {np.mean(tp_confidences):.3f}")
        if tp_distances:
            print(f"TP平均距离: {np.mean(tp_distances):.2f}m")
    
    if all_fp:
        fp_confidences = [item['confidence'] for item in all_fp]
        if fp_confidences:
            print(f"FP置信度范围: {min(fp_confidences):.3f} - {max(fp_confidences):.3f}")
            print(f"FP平均置信度: {np.mean(fp_confidences):.3f}")
    
    # 3. 错误原因分析
    print("\n3. 错误原因分布:")
    print("-" * 40)
    
    has_errors = False
    for class_name, error_dict in analysis_results['error_reasons'].items():
        if error_dict:
            has_errors = True
            print(f"\n{class_name}:")
            total_errors = sum(error_dict.values())
            for error_reason, count in sorted(error_dict.items(), key=lambda x: x[1], reverse=True):
                percentage = count / max(1, total_errors) * 100
                print(f"  {error_reason:<25} {count:<5} ({percentage:.1f}%)")
    
    if not has_errors:
        print("  所有预测都成功匹配！")
    
    # 4. 距离分布分析
    print("\n4. 距离分布:")
    print("-" * 40)
    
    has_distance_data = False
    for class_name, dist_dict in analysis_results['distance_distribution'].items():
        if dist_dict:
            has_distance_data = True
            print(f"\n{class_name}:")
            total = sum(dist_dict.values())
            for dist_cat in ['very_close', 'close', 'medium', 'far', 'very_far']:
                count = dist_dict.get(dist_cat, 0)
                if count > 0:
                    percentage = count / max(1, total) * 100
                    print(f"  {dist_cat:<15} {count:<5} ({percentage:.1f}%)")
    
    if not has_distance_data:
        print("  无距离分布数据")
    
    print("\n" + "="*80)

def xmy_save_analysis_to_file(analysis_results: Dict):
    """保存分析结果到文件"""
    if not Debug:
        return
    
    # 确保目录存在
    os.makedirs(DEBUG_CONFIG['file_path'], exist_ok=True)
    
    timestamp = global_analysis_data['timestamp']
    base_filename = f"eval_analysis_{timestamp}"
    
    # 保存为JSON格式（结构化数据）
    if DEBUG_CONFIG['file_format'] in ['json', 'both']:
        json_data = {
            'metadata': {
                'timestamp': timestamp,
                'evaluation_info': global_analysis_data.get('evaluation_info', {}),
                'debug_config': DEBUG_CONFIG
            },
            'analysis_results': analysis_results,
            'summary': xmy_generate_summary(analysis_results)
        }
        
        json_filepath = os.path.join(DEBUG_CONFIG['file_path'], f"{base_filename}.json")
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"[XMY] 分析结果已保存为JSON文件: {json_filepath}")
    
    # 保存为文本格式（易读）
    if DEBUG_CONFIG['file_format'] in ['txt', 'both']:
        txt_filepath = os.path.join(DEBUG_CONFIG['file_path'], f"{base_filename}.txt")
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            # 写入标题
            f.write("="*80 + "\n")
            f.write("XMY NuScenes评估调试分析报告\n")
            f.write(f"生成时间: {timestamp}\n")
            f.write("="*80 + "\n\n")
            
            # 写入评估信息
            eval_info = global_analysis_data.get('evaluation_info', {})
            if eval_info:
                f.write("评估信息:\n")
                for key, value in eval_info.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # 写入分类型的TP/FP分布
            f.write("1. 分类型的TP/FP分布:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'类别':<20} {'GT数':<8} {'预测数':<8} {'TP数':<8} {'FP数':<8} {'匹配率':<8}\n")
            f.write("-" * 60 + "\n")
            
            for class_name, details in sorted(analysis_results['class_details'].items()):
                f.write(f"{class_name:<20} {details['gt_count']:<8} {details['pred_count']:<8} "
                       f"{details['tp_count']:<8} {details['fp_count']:<8} {details['match_ratio']:.3f}\n")
            
            f.write("\n")
            
            # 写入错误原因分布
            f.write("2. 错误原因分布:\n")
            f.write("-" * 40 + "\n")
            
            has_errors = False
            for class_name, error_dict in analysis_results['error_reasons'].items():
                if error_dict:
                    has_errors = True
                    f.write(f"\n{class_name}:\n")
                    total_errors = sum(error_dict.values())
                    for error_reason, count in sorted(error_dict.items(), key=lambda x: x[1], reverse=True):
                        percentage = count / max(1, total_errors) * 100
                        f.write(f"  {error_reason:<25} {count:<5} ({percentage:.1f}%)\n")
            
            if not has_errors:
                f.write("所有预测都成功匹配！\n")
            
            f.write("\n")
            
            # 写入距离分布
            f.write("3. 距离分布:\n")
            f.write("-" * 40 + "\n")
            
            has_distance_data = False
            for class_name, dist_dict in analysis_results['distance_distribution'].items():
                if dist_dict:
                    has_distance_data = True
                    f.write(f"\n{class_name}:\n")
                    total = sum(dist_dict.values())
                    for dist_cat in ['very_close', 'close', 'medium', 'far', 'very_far']:
                        count = dist_dict.get(dist_cat, 0)
                        if count > 0:
                            percentage = count / max(1, total) * 100
                            f.write(f"  {dist_cat:<15} {count:<5} ({percentage:.1f}%)\n")
            
            if not has_distance_data:
                f.write("无距离分布数据\n")
        
        print(f"[XMY] 分析结果已保存为文本文件: {txt_filepath}")

def xmy_generate_summary(analysis_results: Dict) -> Dict:
    """生成分析摘要"""
    summary = {
        'total_classes': len(analysis_results['class_details']),
        'total_predictions': 0,
        'total_tp': 0,
        'total_fp': 0,
        'overall_match_rate': 0.0,
        'class_performance': {},
        'main_error_reasons': []
    }
    
    # 计算总计
    for class_name, details in analysis_results['class_details'].items():
        summary['total_predictions'] += details['pred_count']
        summary['total_tp'] += details['tp_count']
        summary['total_fp'] += details['fp_count']
        
        summary['class_performance'][class_name] = {
            'match_rate': details['match_ratio'],
            'tp_count': details['tp_count'],
            'fp_count': details['fp_count']
        }
    
    # 计算整体匹配率
    if summary['total_predictions'] > 0:
        summary['overall_match_rate'] = summary['total_tp'] / summary['total_predictions']
    
    # 收集主要错误原因
    all_errors = defaultdict(int)
    for class_name, error_dict in analysis_results['error_reasons'].items():
        for error_reason, count in error_dict.items():
            all_errors[error_reason] += count
    
    summary['main_error_reasons'] = sorted(
        [(reason, count) for reason, count in all_errors.items()],
        key=lambda x: x[1],
        reverse=True
    )[:5]  # 只取前5个主要错误原因
    
    return summary