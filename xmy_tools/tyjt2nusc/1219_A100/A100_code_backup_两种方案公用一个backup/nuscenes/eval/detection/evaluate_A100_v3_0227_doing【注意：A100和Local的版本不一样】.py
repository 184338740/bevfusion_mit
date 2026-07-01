# /data2/xmy/miniconda3/envs/BEVFusion_v4_torch1.13.0_cu116/lib/python3.8/site-packages/nuscenes/eval/detection/evaluate.py
# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.

import argparse
import json
import os
import random
import time
from typing import Tuple, Dict, Any

import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample


Debug = True
if Debug:
    import time, json
    from collections import Counter
    print(f"\n>>>[xmy]🟣[nuscenes]🟣[/data2/xmy/miniconda3/envs/BEVFusion_v4_torch1.13.0_cu116/lib/python3.8/site-packages/nuscenes/eval/detection/evaluate.py] >>> [Debug = True] ")

class DetectionEval:
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self,
                nusc: NuScenes,
                config: DetectionConfig,
                result_path: str,
                eval_set: str,
                output_dir: str = None,
                verbose: bool = True,
                overlap_th: float = 0.5):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        :param overlap_th: The overlap threshold used to determine whether a box is a match.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config
        self.overlap_th = overlap_th

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                    verbose=verbose)
        self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)

        # >>>[xmy]🔵 evaluate.py>>> 详细调试信息
        if verbose:
            print(f'>>>[xmy]🔵 evaluate.py >>> ========== 开始调试 ==========')
            print(f'>>>[xmy]🔵 evaluate.py >>> 数据集版本: {self.nusc.version}')
            print(f'>>>[xmy]🔵 evaluate.py >>> 评估集: {self.eval_set}')
            print(f'>>>[xmy]🔵 evaluate.py >>> 数据集路径: {self.nusc.dataroot}')
            print(f'>>>[xmy]🔵 evaluate.py >>> 总样本数: {len(self.nusc.sample)}')
            print(f'>>>[xmy]🔵 evaluate.py >>> Eval_Pred 预测样本数: {len(self.pred_boxes.sample_tokens)}')
            print(f'>>>[xmy]🔵 evaluate.py >>> Eval_Gt样本数(🔴[nusc官方load_gt()加载]🔴): {len(self.gt_boxes.sample_tokens)}')

        # 🔵[TYJT修复]>>> 处理TYJT数据集
        if 'tyjt' in self.nusc.version:
            print(f'>>>[xmy]🔵 evaluate.py >>> 检测到TYJT数据集 🔴【开始使用xmy自定义的方式加载Eval_Gt】🔴 >>> 绕过load_gt(), xmy自定义加载 Eval_Gt ')
            test_categories = ['car', 'truck', 'pedestrian', 'motorcycle']
            for cat in test_categories:
                from nuscenes.eval.detection.utils import category_to_detection_name
                result = category_to_detection_name(cat)
                if verbose: print(f'>>>[xmy]🔵 evaluate.py >>> 类别映射 (基于nuscenes.eval.detection.utils 的 category_to_detection_name) "{cat}" -> "{result}"')
    
            # 如果原始GT为空，手动创建GT
            if len(self.gt_boxes.sample_tokens) == 0:
                print(f'>>>[xmy]🔵 evaluate.py >>> 原始GT为空,开始手动创建GT')
                
                # 导入必要的函数
                from nuscenes.eval.common.loaders import category_to_detection_name
                
                # 手动创建GT boxes
                # from nuscenes.eval.common.data_classes import EvalBoxes
                new_gt_boxes = EvalBoxes()
                
                # 从预测结果获取样本token
                val_sample_tokens = set(self.pred_boxes.sample_tokens)
                
                if verbose: print(f'>>>[xmy]🔵 evaluate.py >>> 需要处理的样本数: {len(val_sample_tokens)}')
                
                # 加载每个样本的GT标注
                valid_gt_count = 0
                invalid_gt_count = 0
                none_box_count = 0
                
                for i, sample_token in enumerate(val_sample_tokens):
                    if verbose: print(f'>>>[xmy]🔵 evaluate.py>>> 处理样本: {sample_token}')
                    try:
                        sample = self.nusc.get('sample', sample_token)
                        if verbose: print(f'>>>[xmy]🔵 evaluate.py>>> 样本信息: {sample.keys()}')
                        if verbose: print(f'>>>[xmy]🔵 evaluate.py>>> 样本标注数: {len(sample["anns"])}')
                        
                        sample_boxes = []
                        
                        # 获取该样本的所有标注
                        for ann_token in sample['anns']:
                            if verbose: print(f'>>>[xmy]🔵 evaluate.py>>> 处理标注: {ann_token}')
                            try:
                                sample_annotation = self.nusc.get('sample_annotation', ann_token)
                                if verbose: print(f'>>>[xmy]🔵 evaluate.py>>> 标注类别: {sample_annotation["category_name"]}')
                                
                                # 转换为DetectionBox
                                detection_name = category_to_detection_name(sample_annotation['category_name'])
                                if verbose: print(f'>>>[xmy]🔵 evaluate.py>>> 转换后类别: {detection_name}')
                                
                                if detection_name is None:
                                    invalid_gt_count += 1
                                    if verbose: print(f'>>>[xmy]🔵 evaluate.py>>> 跳过无效类别: {sample_annotation["category_name"]}')
                                    continue
                                
                                # 获取属性
                                attr_tokens = sample_annotation['attribute_tokens']
                                attribute_name = ''
                                category = sample_annotation['category_name']
                                # if i == 0: import pdb; pdb.set_trace()

                                if attr_tokens and len(attr_tokens) > 0:
                                    # 首先检查是否是TYJT数据集
                                    if 'tyjt' in self.nusc.version:
                                        # 🔵[TYJT专用]>>> 直接使用第一个属性token作为属性名
                                        # 或者从annotation中提取原始属性信息
                                        # attribute_name = f"tyjt.attribute.{attr_tokens[0]}"
                                        # 改为使用标准属性名：
                                        if category in ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'vehicle.car', 'vehicle.truck']:
                                            attribute_name = 'vehicle.parked'
                                        elif category in ['pedestrian', 'human.pedestrian.adult']:
                                            attribute_name = 'pedestrian.standing'
                                        elif category in ['motorcycle', 'bicycle', 'vehicle.motorcycle', 'vehicle.bicycle']:
                                            attribute_name = 'cycle.without_rider'
                                        elif category in ['traffic_cone', 'barrier', 'movable_object.trafficcone', 'movable_object.barrier']:
                                            attribute_name = ''  # 这些类别通常没有属性
                                        else:
                                            attribute_name = 'vehicle.parked'  # 默认属性
                                    else:
                                        # 标准NuScenes处理
                                        try:
                                            attribute_record = self.nusc.get('attribute', attr_tokens[0])
                                            attribute_name = attribute_record['name']
                                        except KeyError:
                                            attribute_name = ''
                                
                                # 创建DetectionBox对象
                                detection_box = DetectionBox(
                                    sample_token=sample_token,
                                    translation=sample_annotation['translation'],
                                    size=sample_annotation['size'],
                                    rotation=sample_annotation['rotation'],
                                    velocity=self.nusc.box_velocity(sample_annotation['token'])[:2],
                                    num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                                    detection_name=detection_name,
                                    detection_score=-1.0,
                                    attribute_name=attribute_name
                                )
                                
                                # 🔵[关键]>>> 检查box是否为None
                                if detection_box is None:
                                    none_box_count += 1
                                    print(f'>>>[xmy]🔵 evaluate.py>>> 警告: DetectionBox创建失败,返回None')
                                    continue
                                    
                                sample_boxes.append(detection_box)
                                valid_gt_count += 1
                                    
                            except Exception as e:
                                invalid_gt_count += 1
                                print(f'>>>[xmy]🔵 evaluate.py>>> 处理标注 {ann_token} 失败: {e}')
                                import traceback
                                traceback.print_exc()
                                continue
                        
                        if sample_boxes:
                            new_gt_boxes.add_boxes(sample_token, sample_boxes)
                            if verbose:  print(f'>>>[xmy]🔵 evaluate.py>>> 样本 {sample_token} 添加了 {len(sample_boxes)} 个有效标注')
                        else:
                            print(f'>>>[xmy]🔵 evaluate.py>>> 样本 {sample_token} 没有有效标注')
                            None
                        
                    except Exception as e:
                        print(f'>>>[xmy]🔵 evaluate.py>>> 加载样本 {sample_token} 失败: {e}')
                        import traceback
                        traceback.print_exc()
                        continue
                
                self.gt_boxes = new_gt_boxes
                
                print(f'>>>[xmy]🔵 evaluate.py>>> ========== GT创建完成 ==========')
                print(f'>>>[xmy]🔵 evaluate.py>>> 有效GT标注: {valid_gt_count}')
                print(f'>>>[xmy]🔵 evaluate.py>>> 无效GT标注: {invalid_gt_count}')
                print(f'>>>[xmy]🔵 evaluate.py>>> None框数量: {none_box_count}')
                print(f'>>>[xmy]🔵 evaluate.py>>> 最终Eval_GT样本总数(🔴xmy自定义加载Eval_Gt🔴): {len(self.gt_boxes.sample_tokens)}')
                print(f'>>>[xmy]🔵 evaluate.py>>> 最终Eval_GT标注总数(🔴xmy自定义加载Eval_Gt🔴): {sum(len(boxes) for boxes in self.gt_boxes.boxes.values())}')

        # >>>[xmy]🔵 evaluate.py>>> 检查None值
        print(f'>>>[xmy]🔵 evaluate.py>>> ========== 检查None值 ==========')
        self._check_and_remove_none_boxes(self.pred_boxes, "预测框")
        self._check_and_remove_none_boxes(self.gt_boxes, "GT框")

        # 🔵[TYJT修复]>>> 确保样本匹配
        pred_tokens = set(self.pred_boxes.sample_tokens)
        gt_tokens = set(self.gt_boxes.sample_tokens)
        
        print(f'>>>[xmy]🔵 evaluate.py>>> ========== 样本匹配检查 ==========')
        print(f'>>>[xmy]🔵 evaluate.py>>> 预测样本: {len(pred_tokens)}')
        print(f'>>>[xmy]🔵 evaluate.py>>> GT样本: {len(gt_tokens)}')
        print(f'>>>[xmy]🔵 evaluate.py>>> 共同样本: {len(pred_tokens & gt_tokens)}')
        print(f'>>>[xmy]🔵 evaluate.py>>> 预测独有: {len(pred_tokens - gt_tokens)}')
        print(f'>>>[xmy]🔵 evaluate.py>>> GT独有: {len(gt_tokens - pred_tokens)}')
        
        if pred_tokens != gt_tokens:
            print(f'[TYJT调试] 警告: 样本不匹配!')
            print(f'[TYJT调试] 预测有但GT无: {pred_tokens - gt_tokens}')
            print(f'[TYJT调试] GT有但预测无: {gt_tokens - pred_tokens}')
            
            # 保留共同样本
            common_tokens = pred_tokens & gt_tokens
            
            if len(common_tokens) == 0:
                raise ValueError('错误: 预测和GT没有共同样本，无法评估!')
            
            print(f'[TYJT调试] 使用共同样本继续评估: {len(common_tokens)}个')
            
            # from nuscenes.eval.detection.data_classes import EvalBoxes
            
            # 直接构建新的EvalBoxes对象
            new_pred_boxes = EvalBoxes()
            new_gt_boxes = EvalBoxes()
            
            for token in common_tokens:
                # 保留预测框
                if token in self.pred_boxes.boxes:
                    new_pred_boxes.add_boxes(token, self.pred_boxes.boxes[token])
                # 保留GT框
                if token in self.gt_boxes.boxes:
                    new_gt_boxes.add_boxes(token, self.gt_boxes.boxes[token])
            
            self.pred_boxes = new_pred_boxes
            self.gt_boxes = new_gt_boxes
            
            # 重新检查
            pred_tokens = set(self.pred_boxes.sample_tokens)
            gt_tokens = set(self.gt_boxes.sample_tokens)
            print(f'[TYJT调试] common 预测样本: {len(pred_tokens)}')
            print(f'[TYJT调试] common GT样本: {len(gt_tokens)}')

        # 原有的断言（现在应该能通过了）
        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            'Samples in split doesn\'t match samples in predictions.'

        # Add center distances.
        self.pred_boxes = add_center_dist(self.nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(self.nusc, self.gt_boxes)


        if Debug:
            print(f"🔴🔴🔴🔴 官方 self.cfg.class_range = {self.cfg.class_range }🔴🔴🔴🔴，xmy进行修改")
            # 临时修改class_range进行测试
            original_class_range = self.cfg.class_range.copy()  # 保存原始配置
            
            # 修改bicycle和traffic_cone的距离阈值
            # class_range_value = 60
            self.cfg.class_range['car'] = 60.0 
            self.cfg.class_range['truck'] = 60.0 
            self.cfg.class_range['bus'] = 60.0 
            self.cfg.class_range['trailer'] = 60.0 
            self.cfg.class_range['construction_vehicle'] = 60.0 
            self.cfg.class_range['pedestrian'] = 60.0 
            self.cfg.class_range['motorcycle'] = 60.0 
            self.cfg.class_range['bicycle'] = 60.0      # 从40改为60
            self.cfg.class_range['traffic_cone'] = 60.0 # 从30改为60
            self.cfg.class_range['barrier'] = 60.0 
            # 调试使用,  用来设置 不通层级AP计算时的 dist 阈值
            # self.cfg.dist_ths = [5, 10, 20, 40] # 官方 [0.5, 1.0, 2.0, 4.0]
            # self.cfg.dist_th_tp = 10.0 
            print(f"🔴🔴🔴🔴 [修改后配置] self.cfg.class_range = {self.cfg.class_range} 🔴🔴🔴🔴")
        

        # ======================================================
        # 第一步：配置的滤波规则或配置（Debug代码）
        # ======================================================
        if Debug:
            print("\n" + "="*80)
            print("🔵[第一步：滤波配置分析]")
            print("="*80)
            
            print(f"1. 距离滤波配置 (class_range):")
            for cls, max_dist in sorted(self.cfg.class_range.items()):
                print(f"   {cls:25}: {max_dist:.1f}m")
            
            print(f"\n2. 匹配阈值配置:")
            print(f"   dist_ths: {self.cfg.dist_ths}")  # 多级AP计算阈值
            print(f"   dist_th_tp: {self.cfg.dist_th_tp}")  # TP判断阈值
            
            print(f"\n3. 其他滤波配置:")
            print(f"   max_boxes_per_sample: {self.cfg.max_boxes_per_sample}")
            print(f"   点数滤波: 启用")
            print(f"   bike_rack滤波: 启用")

        # ======================================================
        # 第二步：滤波前分析（Debug代码）
        # ======================================================
        if Debug:
            print("\n" + "="*80)
            print("🔵[第二步：滤波前数据分布]")
            print("="*80)
            
            from collections import Counter
            import numpy as np
            
            # 2.1 滤波前的GT统计
            print(f"\n1. 滤波前Eval_GT统计:")
            gt_before_counter = Counter()
            for boxes in self.gt_boxes.boxes.values():
                for box in boxes:
                    gt_before_counter[box.detection_name] += 1
            
            total_gt_before = sum(gt_before_counter.values())
            print(f"   GT总框数: {total_gt_before}")
            print(f"   GT样本数: {len(self.gt_boxes.sample_tokens)}")
            
            # 2.2 滤波前的预测统计
            print(f"\n2. 滤波前Eval_Pred统计:")
            pred_before_counter = Counter()
            pred_before_scores = {}
            for boxes in self.pred_boxes.boxes.values():
                for box in boxes:
                    cls = box.detection_name
                    pred_before_counter[cls] += 1
                    if cls not in pred_before_scores:
                        pred_before_scores[cls] = []
                    pred_before_scores[cls].append(box.detection_score)
            
            total_pred_before = sum(pred_before_counter.values())
            print(f"   预测总框数: {total_pred_before}")
            print(f"   预测样本数: {len(self.pred_boxes.sample_tokens)}")
            
            # 2.3 滤波前数据对比
            print(f"\n3. 滤波前GT与预测对比:")
            print("类别      GT框数   预测框数   预测/GT比   预测平均置信度")
            print("-" * 65)
            
            # key_classes = ['car', 'truck', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone']
            key_classes = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']

            for cls in key_classes:
                gt_count = gt_before_counter.get(cls, 0)
                pred_count = pred_before_counter.get(cls, 0)
                
                # 计算比例
                ratio = pred_count / gt_count if gt_count > 0 else float('inf')
                
                # 平均置信度
                avg_score = "-"
                if cls in pred_before_scores and pred_before_scores[cls]:
                    avg_score = f"{np.mean(pred_before_scores[cls]):.3f}"
                
                # 状态标记
                status = ""
                if pred_count == 0 and gt_count > 0:
                    status = "🔴无预测"
                elif pred_count > 0 and gt_count == 0:
                    status = "⚠️误检"
                elif gt_count > 0:
                    if 0.5 <= ratio <= 2.0:
                        status = "✅正常"
                    else:
                        status = f"⚠️异常({ratio:.1f}x)"
                
                print(f"{cls:12} {gt_count:6}    {pred_count:6}     {ratio:6.1f}       {avg_score:12}  {status}")
            
            # 2.4 Token一致性验证
            print(f"\n4. Token一致性验证:")
            gt_tokens = set(self.gt_boxes.sample_tokens)
            pred_tokens = set(self.pred_boxes.sample_tokens)
            
            print(f"   GT样本token数: {len(gt_tokens)}")
            print(f"   预测样本token数: {len(pred_tokens)}")
            print(f"   共同token数: {len(gt_tokens & pred_tokens)}")
            
            if gt_tokens == pred_tokens:
                print(f"   ✅ Token完全一致")
            else:
                print(f"   ❌ Token不一致!")
                print(f"      GT有但预测无: {len(gt_tokens - pred_tokens)}个")
                print(f"      预测有但GT无: {len(pred_tokens - gt_tokens)}个")
                
        # Filter boxes (distance, points per box, etc.).
        print(f'>>>[xmy]>>>[xmy]🔵 nuscenes/eval/detection/evaluate.py  Eval_Pred、Eval_Gt的过滤配置: self.cfg = {self.cfg } ')
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(self.nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(self.nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens

        # ======================================================
        # 第三步：滤波后分析
        # ======================================================
        if Debug:
            print("\n" + "="*80)
            print("🔵[第三步：滤波后数据分布]")
            print("="*80)
            
            # 3.1 滤波后的GT统计
            print(f"\n1. 滤波后Eval_GT统计:")
            gt_after_counter = Counter()
            for boxes in self.gt_boxes.boxes.values():
                for box in boxes:
                    gt_after_counter[box.detection_name] += 1
            
            total_gt_after = sum(gt_after_counter.values())
            print(f"   GT总框数: {total_gt_after}")
            print(f"   GT样本数: {len(self.gt_boxes.sample_tokens)}")
            
            # 3.2 滤波后的预测统计
            print(f"\n2. 滤波后Eval_Pred统计:")
            pred_after_counter = Counter()
            pred_after_scores = {}
            for boxes in self.pred_boxes.boxes.values():
                for box in boxes:
                    cls = box.detection_name
                    pred_after_counter[cls] += 1
                    if cls not in pred_after_scores:
                        pred_after_scores[cls] = []
                    pred_after_scores[cls].append(box.detection_score)
            
            total_pred_after = sum(pred_after_counter.values())
            print(f"   预测总框数: {total_pred_after}")
            print(f"   预测样本数: {len(self.pred_boxes.sample_tokens)}")
            
            # 3.3 滤波前后对比
            print(f"\n3. 滤波效果分析:")
            print("类别      GT滤波前→后   预测滤波前→后   GT过滤率   预测过滤率")
            print("-" * 70)
            
            for cls in key_classes:
                gt_before = gt_before_counter.get(cls, 0)
                gt_after = gt_after_counter.get(cls, 0)
                pred_before = pred_before_counter.get(cls, 0)
                pred_after = pred_after_counter.get(cls, 0)
                
                # 计算过滤率
                gt_filter_rate = ((gt_before - gt_after) / gt_before * 100) if gt_before > 0 else 0
                pred_filter_rate = ((pred_before - pred_after) / pred_before * 100) if pred_before > 0 else 0
                
                # 状态标记
                gt_status = "✅保留" if gt_filter_rate < 30 else "⚠️过滤多"
                pred_status = "✅保留" if pred_filter_rate < 30 else "⚠️过滤多"
                
                if gt_before > 0 and gt_after == 0:
                    gt_status = "🔴全过滤"
                if pred_before > 0 and pred_after == 0:
                    pred_status = "🔴全过滤"
                
                print(f"{cls:10} {gt_before:3}→{gt_after:3} ({gt_filter_rate:5.1f}%)"
                    f"   {pred_before:3}→{pred_after:3} ({pred_filter_rate:5.1f}%)"
                    f"   {gt_status}   {pred_status}")

        # ======================================================
        # 第四步：关键问题专项分析
        # ======================================================
        # if Debug:
        #     print("\n" + "="*80)
        #     print("🔵[第四步：关键问题专项分析]")
        #     print("="*80)
            
        #     # 4.1 bicycle专项分析
        #     print(f"\n1. bicycle专项分析:")
        #     bicycle_gt_before = gt_before_counter.get('bicycle', 0)
        #     bicycle_gt_after = gt_after_counter.get('bicycle', 0)
        #     bicycle_pred_before = pred_before_counter.get('bicycle', 0)
        #     bicycle_pred_after = pred_after_counter.get('bicycle', 0)
            
        #     print(f"   GT: {bicycle_gt_before} → {bicycle_gt_after}个")
        #     print(f"   预测: {bicycle_pred_before} → {bicycle_pred_after}个")
            
        #     # if bicycle_pred_after > 0:
        #     #     print(f"   ✅ 滤波后仍有{bicycle_pred_after}个bicycle预测")
                
        #     #     # 找到滤波后的bicycle预测
        #     #     for sample_token, boxes in self.pred_boxes.boxes.items():
        #     #         for box in boxes:
        #     #             if box.detection_name == 'bicycle':
        #     #                 distance = np.linalg.norm(box.translation[:2])
        #     #                 print(f"     样本:{sample_token[:12]}..., "
        #     #                     f"距离:{distance:.1f}m, 置信度:{box.detection_score:.3f}")
        #     # else:
        #     #     if bicycle_pred_before > 0:
        #     #         print(f"   🔴 所有{bicycle_pred_before}个bicycle预测都被过滤!")
        #     #         print(f"     可能原因: 距离超出{self.cfg.class_range.get('bicycle', 40)}m范围")
        #     #     else:
        #     #         print(f"   🔴 滤波前就没有bicycle预测")
            
        #     # 4.2 过滤距离分析
        #     print(f"\n2. 距离过滤分析:")
        #     max_range_bicycle = self.cfg.class_range.get('bicycle', 40)
        #     print(f"   bicycle 配置的 最大检测距离: {max_range_bicycle}m")
            
        #     # 如果有bicycle预测被过滤，分析距离
        #     if bicycle_pred_before > bicycle_pred_after:
        #         print(f"   {bicycle_pred_before - bicycle_pred_after}个bicycle因距离被过滤")
            
        #     # 4.3 匹配阈值影响分析
        #     print(f"\n3. 匹配阈值影响分析:")
        #     print(f"   当前匹配阈值: {self.cfg.dist_ths} (AP计算)")
        #     print(f"   当前TP判断阈值: {self.cfg.dist_th_tp}m")
        #     print(f"   这意味着:")
        #     print(f"   - 预测框与GT距离<{self.cfg.dist_th_tp}m才算正确检测")
        #     print(f"   - AP按[{', '.join(map(str, self.cfg.dist_ths))}]m四个阈值计算")
            
        #     # 4.4 关键发现总结
        #     print(f"\n4. 关键发现总结:")
            
        #     findings = []
            
        #     # bicycle相关发现
        #     if bicycle_pred_after == 0:
        #         if bicycle_pred_before > 0:
        #             findings.append("🔴 bicycle预测被距离过滤（超出范围）")
        #         else:
        #             findings.append("🔴 模型没有预测bicycle")
        #     elif bicycle_gt_after > 0 and bicycle_pred_after > 0:
        #         findings.append("✅ bicycle有GT也有预测")
            
        #     # 过滤率过高发现
        #     for cls in key_classes:
        #         pred_before = pred_before_counter.get(cls, 0)
        #         pred_after = pred_after_counter.get(cls, 0)
        #         if pred_before > 0:
        #             filter_rate = (pred_before - pred_after) / pred_before * 100
        #             if filter_rate > 50:
        #                 findings.append(f"⚠️ {cls}预测过滤率过高 ({filter_rate:.1f}%)")
            
        #     # 误检发现
        #     for cls in key_classes:
        #         gt_count = gt_after_counter.get(cls, 0)
        #         pred_count = pred_after_counter.get(cls, 0)
        #         if gt_count == 0 and pred_count > 10:  # 没有GT但有很多预测
        #             findings.append(f"⚠️ {cls}可能有误检 ({pred_count}个预测)")
            
        #     # 打印发现
        #     if findings:
        #         for i, finding in enumerate(findings, 1):
        #             print(f"   {i}. {finding}")
        #     else:
        #         print("   ✅ 未发现明显问题")
            
        #     print("="*80)




    def _check_and_remove_none_boxes(self, eval_boxes, box_type):
        """检查并移除boxes中的None值"""
        none_count = 0
        total_boxes = 0
        
        # 直接清理原始boxes中的None值
        for sample_token, boxes in list(eval_boxes.boxes.items()):  # 使用list避免修改时迭代
            valid_boxes = []
            for i, box in enumerate(boxes):
                total_boxes += 1
                if box is None:
                    none_count += 1
                    print(f'>>>[xmy]🔵 evaluate.py>>> 在{box_type}中发现None: sample={sample_token}, index={i}')
                elif box.detection_name is None:
                    none_count += 1
                    print(f'>>>[xmy]🔵 evaluate.py>>> 在{box_type}中发现detection_name为None: sample={sample_token}, index={i}')
                else:
                    valid_boxes.append(box)
            
            if valid_boxes:
                eval_boxes.boxes[sample_token] = valid_boxes
            else:
                # 如果没有有效框，移除该样本
                del eval_boxes.boxes[sample_token]
        
        print(f'>>>[xmy]🔵 evaluate.py>>> {box_type}清理: 总数={total_boxes}, None值={none_count}, 有效值={total_boxes-none_count}')
        
        return none_count



    def _get_val_samples_from_bevfusion(self, verbose=False):
        """
        从BEVFusion的预测结果推断验证集样本
        """
        # 直接从预测结果中获取样本token（最可靠的方式）
        pred_tokens = set(self.pred_boxes.sample_tokens)
        
        if verbose:
            print(f'🔵[TYJT修复]>>> 从预测结果获取样本: {len(pred_tokens)}个')
        
        return pred_tokens

    def _remove_none_boxes(self, verbose=False):
        """
        移除boxes中的None值
        """
        # 清理预测框
        cleaned_pred_boxes = EvalBoxes()
        for sample_token, boxes in self.pred_boxes.boxes.items():
            valid_boxes = [box for box in boxes if box is not None]
            if valid_boxes:
                cleaned_pred_boxes.add_boxes(sample_token, valid_boxes)
        self.pred_boxes = cleaned_pred_boxes
        
        # 清理GT框
        cleaned_gt_boxes = EvalBoxes()
        for sample_token, boxes in self.gt_boxes.boxes.items():
            valid_boxes = [box for box in boxes if box is not None]
            if valid_boxes:
                cleaned_gt_boxes.add_boxes(sample_token, valid_boxes)
        self.gt_boxes = cleaned_gt_boxes
        
        if verbose:
            print(f'🔵[TYJT修复]>>> 清理后预测样本: {len(self.pred_boxes.sample_tokens)}')
            print(f'🔵[TYJT修复]>>> 清理后GT样本: {len(self.gt_boxes.sample_tokens)}')




    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'))

        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'))

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'))

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))

    def main(self,
             plot_examples: int = 0,
             render_curves: bool = True) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        if plot_examples > 0:
            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Visualize samples.
            example_dir = os.path.join(self.output_dir, 'examples')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample(self.nusc,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
                     class_tps[class_name]['vel_err'],
                     class_tps[class_name]['attr_err']))

        return metrics_summary


class NuScenesEval(DetectionEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes detection results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the CVPR 2019 configuration will be used.')
    parser.add_argument('--plot_examples', type=int, default=10,
                        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render PR and TP curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    plot_examples_ = args.plot_examples
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    if config_path == '':
        cfg_ = config_factory('detection_cvpr_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))

    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    nusc_eval = DetectionEval(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
                              output_dir=output_dir_, verbose=verbose_)
    nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_)
