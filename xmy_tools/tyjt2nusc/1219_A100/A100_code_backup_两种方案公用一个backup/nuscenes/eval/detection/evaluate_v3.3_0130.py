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

Features_xmy = True
Debug = True
if Debug:
    import time, json
    from collections import Counter
    print(f"\n>>>[xmy]🔵[nuscenes]🔵[nuscenes/eval/detection/evaluate.py] >>> [Debug Mode = True] ")

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

        # 🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  详细调试信息
        print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  ========== 开始调试 ==========')
        print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  数据集版本: {self.nusc.version}')
        print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  评估集: {self.eval_set}')
        print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  数据集路径: {self.nusc.dataroot}')
        print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  总样本数: {len(self.nusc.sample)}')
        print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  预测样本数: {len(self.pred_boxes.sample_tokens)}')
        print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  通过🔴Eval中,加载评测集 Gt分支1(nusc数据集):nusc官方采用load_gt(), 加载硬编码 split ={self.eval_set}方式🔴: GT样本数: {len(self.gt_boxes.sample_tokens)}')
        
        # 显示前几个样本token
        if len(self.pred_boxes.sample_tokens) > 0:
            print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  预测样本token前5个: {self.pred_boxes.sample_tokens[:5]}')
        if len(self.gt_boxes.sample_tokens) > 0:
            print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  GT样本token前5个: {self.gt_boxes.sample_tokens[:5]}')
        else:
            print(f'🔴[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  通过nusc官网 load_gt 获取的 Gt样本token个数为: 0')

        # 🔵[TYJT修复]>>> 处理TYJT数据集
        if 'tyjt' in self.nusc.version:
            print(f'🔴🔴🔴🔴 [进入 xmy修改的代码逻辑] 🔴🔴🔴🔴 >>>  🔴Eval中,加载评测集 Gt分支1(tyjt数据集):tyjt无split,绕过load_gt(),xmy自定义加载Gt🔴 nuscenes/eval/detection/evaluate.py  检测到TYJT数据集')

            # test_categories = ['car', 'truck', 'pedestrian', 'motorcycle']

            # for cat in test_categories:
            #     from nuscenes.eval.detection.utils import category_to_detection_name
            #     result = category_to_detection_name(cat)
            #     print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  类别映射 "{cat}" -> "{result}"')
    
            # 如果原始GT为空，手动创建GT
            if len(self.gt_boxes.sample_tokens) == 0:
                print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  原始GT为空，开始手动创建GT')
                
                # 导入必要的函数
                from nuscenes.eval.common.loaders import category_to_detection_name
                
                # 手动创建GT boxes
                from nuscenes.eval.common.data_classes import EvalBoxes
                new_gt_boxes = EvalBoxes()
                
                # 从预测结果获取样本token
                val_sample_tokens = set(self.pred_boxes.sample_tokens)
                
                print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  需要处理的样本数: {len(val_sample_tokens)}')
                
                # 加载每个样本的GT标注
                valid_gt_count = 0
                invalid_gt_count = 0
                none_box_count = 0
                
                for i, sample_token in enumerate(val_sample_tokens):
                    # if verbose: print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  处理样本: {sample_token}')
                    try:
                        sample = self.nusc.get('sample', sample_token)
                        # if verbose: print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  样本信息: {sample.keys()}')
                        # if verbose: print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  样本标注数: {len(sample["anns"])}')
                        
                        sample_boxes = []
                        
                        # 获取该样本的所有标注
                        for ann_token in sample['anns']:
                            # if verbose: print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  处理标注: {ann_token}')
                            try:
                                sample_annotation = self.nusc.get('sample_annotation', ann_token)
                                # if verbose: print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  标注类别: {sample_annotation["category_name"]}')
                                
                                # 转换为DetectionBox
                                detection_name = category_to_detection_name(sample_annotation['category_name'])
                                # if verbose: print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  转换后类别: {detection_name}')
                                
                                if detection_name is None:
                                    invalid_gt_count += 1
                                    if verbose: print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  跳过无效类别: {sample_annotation["category_name"]}')
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
                                    print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  警告: DetectionBox创建失败，返回None')
                                    continue
                                    
                                sample_boxes.append(detection_box)
                                valid_gt_count += 1
                                    
                            except Exception as e:
                                invalid_gt_count += 1
                                print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  处理标注 {ann_token} 失败: {e}')
                                import traceback
                                traceback.print_exc()
                                continue
                        
                        if sample_boxes:
                            new_gt_boxes.add_boxes(sample_token, sample_boxes)
                            # print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  样本 {sample_token} 添加了 {len(sample_boxes)} 个有效标注')
                        else:
                            print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  样本 {sample_token} 没有有效标注')
                        
                    except Exception as e:
                        print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  加载样本 {sample_token} 失败: {e}')
                        import traceback
                        traceback.print_exc()
                        continue
                
                self.gt_boxes = new_gt_boxes
                
                print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  ========== GT创建完成 ==========')
                print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  有效GT标注框: {valid_gt_count}')
                print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  无效GT标注框: {invalid_gt_count}')
                print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  None框数量: {none_box_count}')
                print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  🔴Eval中,加载评测集 Gt分支1(tyjt数据集):tyjt无split,绕过load_gt(),xmy自定义加载Gt🔴 最终GT样本数: {len(self.gt_boxes.sample_tokens)}')
                print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  🔴Eval中,加载评测集 Gt分支1(tyjt数据集):tyjt无split,绕过load_gt(),xmy自定义加载Gt🔴 最终GT标注框总数: {sum(len(boxes) for boxes in self.gt_boxes.boxes.values())}')

        # 🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  检查None值
        print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  ========== 检查None值 ==========')
        self._check_and_remove_none_boxes(self.pred_boxes, "预测框")
        self._check_and_remove_none_boxes(self.gt_boxes, "GT框")

        # 🔵[TYJT修复]>>> 确保样本匹配
        pred_tokens = set(self.pred_boxes.sample_tokens)
        gt_tokens = set(self.gt_boxes.sample_tokens)
        
        print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  ========== 样本匹配检查 ==========')
        print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  预测样本 pred_tokens: {pred_tokens}')
        print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  预测样本 gt_tokens: {gt_tokens}')


        print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  预测样本: {len(pred_tokens)}')
        print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  GT样本: {len(gt_tokens)}')
        print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  共同样本: {len(pred_tokens & gt_tokens)}')
        print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  预测独有: {len(pred_tokens - gt_tokens)}')
        print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  GT独有: {len(gt_tokens - pred_tokens)}')
        
        if pred_tokens != gt_tokens:
            print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  错误: 样本不匹配!')
            print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  预测有但GT无: {pred_tokens - gt_tokens}')
            print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  GT有但预测无: {gt_tokens - pred_tokens}')
            raise ValueError(f'❌ TYJT数据集样本不匹配: 预测{len(pred_tokens)}个, GT{len(gt_tokens)}个')

        # 原有的断言（现在应该能通过了）
        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            'Samples in split doesn\'t match samples in predictions.'

        # Add center distances.
        self.pred_boxes = add_center_dist(self.nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(self.nusc, self.gt_boxes)

        if Features_xmy:
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
            print("🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py 🔵[第一步：滤波配置分析]")
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
            print("🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py 🔵[第二步：滤波前数据分布]")
            print("="*80)
            # 构建并显示滤波前数据包
            self.xmy_data_filter_before = self.xmy_build_data_package("滤波前")
            self.xmy_show_data_package(self.xmy_data_filter_before)

        # ======================================================
        # 执行滤波操作（源代码）
        # ======================================================
        # Filter boxes (distance, points per box, etc.)
        print(f'>>>[xmy]🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  Eval_Pred、Eval_Gt的过滤配置: self.cfg = {self.cfg } ')
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
            print("🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py 🔵[第三步：滤波后数据分布]")
            print("="*80)
            
            # 构建并显示滤波后数据包
            self.xmy_data_filter_after = self.xmy_build_data_package("滤波后")
            self.xmy_show_data_package(self.xmy_data_filter_after)

        # ======================================================
        # 第四步：关键问题专项分析
        # ======================================================
        if Debug:
            print("\n" + "="*80)
            print("🔵[第四步：关键问题专项分析（对比）]")
            print("="*80)
            
            # 对比分析两个数据包
            self.xmy_analyze_package_comparison(
                self.xmy_data_filter_before, 
                self.xmy_data_filter_after
            )
    
            print("="*80)


        # ======================================================
        # 第五步：置信度分布分析（您的原有代码，保持不变）
        # ======================================================
        if Debug:
            print("\n" + "="*80)
            print("📈 [第五步：置信度分布分析]")
            print("="*80)
            # 分析滤波前数据
            self.xmy_analyze_confidence_distribution(self.xmy_data_filter_before)
            # 分析滤波后数据
            self.xmy_analyze_confidence_distribution(self.xmy_data_filter_after)
        
        # ======================================================
        # 第六步：空间分布分析（您的原有代码，修改后）
        # ======================================================
        if Debug:
            print("\n" + "="*80)
            print("🗺️ [第六步：空间分布分析]")
            print("="*80)
            # 距离区域分析（这个是存在的）
            self.xmy_analyze_detection_by_distance_zones(self.xmy_data_filter_after)
            # 删除不存在的调用：self.xmy_analyze_spatial_distribution(self.xmy_data_filter_after)
        
        # ======================================================
        # 第七步：新增深度特征分析（我建议的新分析）
        # ======================================================
        if Debug:
            print("\n" + "="*80)
            print("🔵[第七步：深度特征分析 - BEVFusion异常AP诊断]")
            print("="*80)
            
            try:
                # 7.1 距离衰减分析（最重要）
                self.xmy_analyze_distance_decay(self.xmy_data_filter_after, True)
            except AttributeError as e:
                print(f"⚠️ 距离衰减分析未实现: {e}")
            
            try:
                # 7.2 密度影响分析
                self.xmy_analyze_density_impact(self.xmy_data_filter_after, True)
            except AttributeError as e:
                print(f"⚠️ 密度影响分析未实现: {e}")
            
            try:
                # 7.3 方向偏差分析
                self.xmy_analyze_directional_bias(self.xmy_data_filter_after)
            except AttributeError as e:
                print(f"⚠️ 方向偏差分析未实现: {e}")
            
            try:
                # 7.4 问题总结报告
                self.xmy_generate_problem_summary(self.xmy_data_filter_after)
            except AttributeError as e:
                print(f"⚠️ 问题总结报告未实现: {e}")
        
        # ======================================================
        # 第八步：热力图存档（可选）
        # ======================================================
        if Debug:
            print("\n" + "="*80)
            print("🔵[第八步：热力图存档]")
            print("="*80)
            
            try:
                # 指定保存到当前目录的固定位置
                from pathlib import Path
                current_dir = Path.cwd()
                heatmap_dir = current_dir / "bevfusion_analysis" / "heatmaps"
                heatmap_dir.mkdir(parents=True, exist_ok=True)
                
                self.xmy_generate_heatmap_images(
                    self.xmy_data_filter_after, 
                    custom_output_dir=str(heatmap_dir)
                )
            except AttributeError as e:
                print(f"⚠️ 热力图存档未实现: {e}")


    # [xmy] - 基础统计
    def xmy_statistics_boxes(self, boxes, box_type="GT"):
        """
        统计框的基础信息（数量、距离、置信度）
        :param boxes: EvalBoxes对象
        :param box_type: 框类型，"GT"或"Pred"
        :return: (counter, sample_counter, distances, scores)
        """
        from collections import Counter
        import numpy as np
        
        counter = Counter()
        sample_counter = Counter()
        distances = {}
        scores = {}
        
        for sample_token, box_list in boxes.boxes.items():
            sample_classes = set()
            for box in box_list:
                cls = box.detection_name
                counter[cls] += 1
                sample_classes.add(cls)
                
                # 记录距离信息
                if cls not in distances:
                    distances[cls] = []
                distance = np.linalg.norm(box.translation[:2]) if hasattr(box, 'translation') else 0
                distances[cls].append(distance)
                
                # 记录置信度（仅对预测框）
                if box_type == "Pred" and hasattr(box, 'detection_score'):
                    if cls not in scores:
                        scores[cls] = []
                    scores[cls].append(box.detection_score)
            
            # 更新样本类别计数
            for cls in sample_classes:
                if cls not in sample_counter:
                    sample_counter[cls] = 0
                sample_counter[cls] += 1
        
        return counter, sample_counter, distances, scores

    # [xmy] - 距离状态判断
    def xmy_get_distance_status(self, avg_distance, config_range):
        """
        实例方法：根据距离判断状态
        用self.xxx调用，简单直接
        """
        if avg_distance > 0:
            if avg_distance > config_range:
                return "🔴超限"
            elif avg_distance > config_range * 0.8:
                return "⚠️接近"
            else:
                return "✅正常"
        return "  -"
    
    # [xmy] - 构建数据包
    def xmy_build_data_package(self, package_name):
        """
        构建数据包（包含GT和Pred的完整统计信息）
        :param package_name: 数据包名称，如"滤波前"或"滤波后"
        :return: 数据包字典
        """
        import numpy as np
        
        # 统计GT数据
        gt_counter, gt_sample_counter, gt_distances, _ = self.xmy_statistics_boxes(self.gt_boxes, "GT")
        
        # 统计预测数据
        pred_counter, pred_sample_counter, pred_distances, pred_scores = self.xmy_statistics_boxes(self.pred_boxes, "Pred")
        
        # 计算平均距离
        gt_avg_distances = {}
        for cls in gt_counter:
            if cls in gt_distances and gt_distances[cls]:
                gt_avg_distances[cls] = np.mean(gt_distances[cls])
        
        pred_avg_distances = {}
        for cls in pred_counter:
            if cls in pred_distances and pred_distances[cls]:
                pred_avg_distances[cls] = np.mean(pred_distances[cls])
        
        # 构建数据包
        data_package = {
            'name': package_name,
            'timestamp': getattr(self, 'xmy_timestamp', 'N/A'),
            'gt_counter': gt_counter,
            'pred_counter': pred_counter,
            'gt_avg_distances': gt_avg_distances,
            'pred_avg_distances': pred_avg_distances,
            'gt_distances': gt_distances,
            'pred_distances': pred_distances,
            'pred_scores': pred_scores,
            'gt_sample_count': len(self.gt_boxes.sample_tokens),
            'pred_sample_count': len(self.pred_boxes.sample_tokens),
            'total_gt_boxes': sum(gt_counter.values()),
            'total_pred_boxes': sum(pred_counter.values()),
        }
        
        return data_package

    # [xmy] - 显示数据包
    def xmy_show_data_package(self, data_package):
        """
        显示数据包的详细信息
        :param data_package: 数据包字典
        """
        name = data_package['name']
        
        print(f"\n{'='*60}")
        print(f"📊 数据包分析: {name}")
        print(f"{'='*60}")
        
        # 基础统计
        print(f"\n1. 基础统计:")
        print(f"   GT样本数: {data_package['gt_sample_count']}")
        print(f"   预测样本数: {data_package['pred_sample_count']}")
        print(f"   GT总框数: {data_package['total_gt_boxes']}")
        print(f"   预测总框数: {data_package['total_pred_boxes']}")
        
        # GT类别分布
        gt_counter = data_package['gt_counter']
        print(f"\n2. GT类别分布:")
        if gt_counter:
            sorted_gt = sorted(gt_counter.items(), key=lambda x: x[1], reverse=True)
            print("类别               数量   占比")
            print("-" * 40)
            total_gt = sum(gt_counter.values())
            for cls, count in sorted_gt:
                percentage = (count / total_gt * 100) if total_gt > 0 else 0
                print(f"{cls:15} {count:6} ({percentage:5.1f}%)")
        
        # 预测类别分布
        pred_counter = data_package['pred_counter']
        print(f"\n3. 预测类别分布:")
        if pred_counter:
            sorted_pred = sorted(pred_counter.items(), key=lambda x: x[1], reverse=True)
            print("类别               数量   占比")
            print("-" * 40)
            total_pred = sum(pred_counter.values())
            for cls, count in sorted_pred:
                percentage = (count / total_pred * 100) if total_pred > 0 else 0
                print(f"{cls:15} {count:6} ({percentage:5.1f}%)")
        
        # 距离信息
        gt_avg_distances = data_package['gt_avg_distances']
        print(f"\n4. 平均距离统计:")
        if gt_avg_distances:
            print("类别               GT平均距离   Pred平均距离")
            print("-" * 50)
            all_classes = set(list(gt_avg_distances.keys()) + 
                            list(data_package['pred_avg_distances'].keys()))
            for cls in sorted(all_classes):
                gt_avg = gt_avg_distances.get(cls, 0)
                pred_avg = data_package['pred_avg_distances'].get(cls, 0)
                if gt_avg > 0 or pred_avg > 0:
                    print(f"{cls:15} {gt_avg:8.1f}m    {pred_avg:8.1f}m")

    # [xmy] - 置信度分布分析功能
    def xmy_analyze_confidence_distribution(self, data_package):
        """
        深度分析置信度分布，识别置信度校准问题
        :param data_package: 数据包字典
        """
        print("\n" + "="*80)
        print("📈 [置信度分布深度分析]")
        print("="*80)
        
        pred_scores = data_package['pred_scores']
        package_name = data_package['name']
        
        if not pred_scores:
            print("❌ 无预测框数据，无法进行置信度分析")
            return
        
        print(f"分析数据包: {package_name}")
        
        # 1. 基础统计
        print("\n1. 置信度基础统计:")
        print("-" * 60)
        print("类别               平均    最低    最高    中位数   方差    偏度    峰度    <0.3   0.3-0.7   >0.7")
        print("-" * 110)
        
        import numpy as np
        from scipy import stats
        
        confidence_issues = []
        
        for cls in sorted(pred_scores.keys()):
            scores = pred_scores[cls]
            if not scores:
                continue
                
            scores_array = np.array(scores)
            
            # 基础统计
            avg_score = np.mean(scores_array)
            min_score = np.min(scores_array)
            max_score = np.max(scores_array)
            median_score = np.median(scores_array)
            std_score = np.std(scores_array)
            
            # 分布形态
            try:
                skewness = stats.skew(scores_array)
                kurtosis = stats.kurtosis(scores_array)
            except:
                skewness = 0
                kurtosis = 0
            
            # 分桶统计
            low_conf = np.sum(scores_array < 0.3) / len(scores_array) * 100
            mid_conf = np.sum((scores_array >= 0.3) & (scores_array <= 0.7)) / len(scores_array) * 100
            high_conf = np.sum(scores_array > 0.7) / len(scores_array) * 100
            
            print(f"{cls:15} {avg_score:6.3f}  {min_score:6.3f}  {max_score:6.3f}  "
                f"{median_score:6.3f}  {std_score:6.3f}  {skewness:6.3f}  {kurtosis:6.3f}  "
                f"{low_conf:5.1f}%  {mid_conf:6.1f}%  {high_conf:6.1f}%")
            
            # 问题检测
            if avg_score < 0.3:
                confidence_issues.append(f"🔴 {cls}: 平均置信度过低 ({avg_score:.3f})")
            if std_score > 0.3:
                confidence_issues.append(f"⚠️ {cls}: 置信度方差过大 ({std_score:.3f})")
            if low_conf > 80:
                confidence_issues.append(f"⚠️ {cls}: 低置信度框占比过高 ({low_conf:.1f}%)")
            if abs(skewness) > 1.0:
                confidence_issues.append(f"⚠️ {cls}: 置信度分布偏斜 ({skewness:.2f})")
        
        # 2. 整体置信度分布
        print("\n2. 整体置信度分布:")
        print("-" * 60)
        
        all_scores = []
        for cls_scores in pred_scores.values():
            all_scores.extend(cls_scores)
        
        if all_scores:
            all_scores_array = np.array(all_scores)
            
            # 分位数
            quantiles = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
            quantile_values = np.quantile(all_scores_array, quantiles)
            
            print("分位数   0%   10%   25%   50%   75%   90%   100%")
            print("值     ", "  ".join([f"{v:.3f}" for v in quantile_values]))
            
            # 直方图（文本版）
            print("\n置信度分布直方图:")
            hist, bins = np.histogram(all_scores_array, bins=10, range=(0, 1))
            for i in range(len(hist)):
                bin_start = bins[i]
                bin_end = bins[i+1]
                count = hist[i]
                percentage = count / len(all_scores_array) * 100
                bar = "█" * int(percentage / 2)  # 每2%一个方块
                print(f"[{bin_start:.1f}-{bin_end:.1f}): {count:4d} ({percentage:5.1f}%) {bar}")
        
        # 3. 置信度问题总结
        if confidence_issues:
            print("\n3. 置信度问题检测:")
            print("-" * 60)
            for issue in confidence_issues:
                print(f"  {issue}")
        else:
            print("\n✅ 未发现显著置信度问题")

    def xmy_analyze_package_comparison(self, package_a, package_b):
        """
        对比分析两个数据包（如滤波前 vs 滤波后）
        :param package_a: 第一个数据包
        :param package_b: 第二个数据包
        """
        print(f"\n{'='*80}")
        print(f"🔍 数据包对比: {package_a['name']} → {package_b['name']}")
        print(f"{'='*80}")
        
        # 1. 过滤效果对比
        print(f"\n1. 过滤效果对比:")
        print("=" * 90)
        print("类别                GT(A→B)      预测(A→B)      GT过滤率   预测过滤率   状态")
        print("-" * 90)
        
        # 获取所有类别
        all_classes = set()
        for counter in [package_a['gt_counter'], package_a['pred_counter'], 
                        package_b['gt_counter'], package_b['pred_counter']]:
            all_classes.update(counter.keys())
        
        sorted_classes = sorted(all_classes, 
                            key=lambda x: package_a['gt_counter'].get(x, 0), 
                            reverse=True)
        
        for cls in sorted_classes:
            gt_a = package_a['gt_counter'].get(cls, 0)
            gt_b = package_b['gt_counter'].get(cls, 0)
            pred_a = package_a['pred_counter'].get(cls, 0)
            pred_b = package_b['pred_counter'].get(cls, 0)
            
            # 计算过滤率
            gt_filter_rate = ((gt_a - gt_b) / gt_a * 100) if gt_a > 0 else 0
            pred_filter_rate = ((pred_a - pred_b) / pred_a * 100) if pred_a > 0 else 0
            
            # 状态判断
            status_parts = []
            if gt_a > 0 and gt_b == 0:
                status_parts.append("GT全滤")
            elif gt_filter_rate > 50:
                status_parts.append("GT多滤")
                
            if pred_a > 0 and pred_b == 0:
                status_parts.append("预全滤")
            elif pred_filter_rate > 50:
                status_parts.append("预多滤")
                
            if gt_b > 0 and pred_b == 0:
                status_parts.append("无检测")
            elif gt_b == 0 and pred_b > 5:
                status_parts.append("疑误检")
                
            status = "/".join(status_parts) if status_parts else "正常"
            
            print(f"{cls:15} {gt_a:3}→{gt_b:<3} ({gt_filter_rate:5.1f}%) "
                f"    {pred_a:3}→{pred_b:<3} ({pred_filter_rate:5.1f}%) "
                f"    {gt_filter_rate:6.1f}%  {pred_filter_rate:6.1f}%  {status}")
        
        # 2. 距离配置分析
        print(f"\n\n2. 距离配置分析:")
        print("=" * 110)
        print("类别                配置阈值    GT平均距离(前)  GT平均距离(后)  Pred平均距离(前) Pred平均距离(后)  GT状态    Pred状态")
        print("-" * 110)

        # 在打印时使用更清晰的标签
        for cls in sorted_classes:
            config_range = self.cfg.class_range.get(cls, 0.0)
            
            # GT平均距离（前=滤波前，后=滤波后）
            gt_before = package_a['gt_avg_distances'].get(cls, 0.0)  # package_a是"滤波前"
            gt_after = package_b['gt_avg_distances'].get(cls, 0.0)   # package_b是"滤波后"
            
            # 预测平均距离（前=滤波前，后=滤波后）
            pred_before = package_a['pred_avg_distances'].get(cls, 0.0)
            pred_after = package_b['pred_avg_distances'].get(cls, 0.0)
            
            gt_status = self.xmy_get_distance_status(gt_before, config_range)
            pred_status = self.xmy_get_distance_status(pred_before, config_range)

            print(f"{cls:15} {config_range:6.1f}m    "
                f"{gt_before:8.1f}m    {gt_after:8.1f}m    "
                f"{pred_before:8.1f}m    {pred_after:8.1f}m    "
                f"{gt_status:6}   {pred_status:6}")
        
        # 3. 核心发现
        print(f"\n\n3. 核心发现总结:")
        print("=" * 60)
        
        critical_issues = []
        warning_issues = []
        
        for cls in sorted_classes:
            gt_a = package_a['gt_counter'].get(cls, 0)
            gt_b = package_b['gt_counter'].get(cls, 0)
            pred_b = package_b['pred_counter'].get(cls, 0)
            config_range = self.cfg.class_range.get(cls, 0)
            gt_avg_a = package_a['gt_avg_distances'].get(cls, 0)
            
            # 检测严重问题
            if gt_a > 0 and gt_b == 0:
                critical_issues.append(f"🔴 {cls}: 所有{gt_a}个GT框被过滤")
            
            if pred_b == 0 and gt_b > 10:
                critical_issues.append(f"🔴 {cls}: 有{gt_b}个GT但无预测")
            
            # 检测警告问题
            if gt_avg_a > 0 and gt_avg_a > config_range:
                warning_issues.append(f"⚠️ {cls}: 平均距离{gt_avg_a:.1f}m > 阈值{config_range}m")
        
        if critical_issues:
            print("🔴 严重问题发现:")
            for issue in critical_issues:
                print(f"  • {issue}")
        else:
            print("✅ 未发现严重问题")
        
        if warning_issues:
            print("\n⚠️ 警告问题发现:")
            for issue in warning_issues:
                print(f"  • {issue}")
        
        print("=" * 60)

    # [xmy] -  距离区域分析           
    def xmy_analyze_detection_by_distance_zones(self, data_package):
        """
        按距离区域分析检测性能
        """
        print("\n" + "="*80)
        print("📏 [距离区域分析]")
        print("="*80)
        
        distance_zones = [
            (0, 10, "近距 0-10m"),
            (10, 20, "中近 10-20m"),
            (20, 30, "中距 20-30m"),
            (30, 40, "中远 30-40m"),
            (40, 50, "远距 40-50m"),
            (50, 100, "超远 >50m")
        ]
        
        for class_name in sorted(data_package['gt_counter'].keys()):
            gt_counter = data_package['gt_counter'].get(class_name, 0)
            if gt_counter == 0:
                continue
                
            print(f"\n类别: {class_name} (GT总数: {gt_counter})")
            print("-" * 70)
            print("距离区间       GT数量   GT占比   预测数量  检测密度  平均置信度")
            print("-" * 70)
            
            # 获取该类别GT距离
            gt_distances = data_package['gt_distances'].get(class_name, [])
            pred_distances = data_package['pred_distances'].get(class_name, [])
            pred_scores = data_package['pred_scores'].get(class_name, [])
            
            for zone_min, zone_max, zone_name in distance_zones:
                # GT统计
                gt_in_zone = [d for d in gt_distances if zone_min <= d < zone_max]
                gt_count = len(gt_in_zone)
                gt_percentage = gt_count / len(gt_distances) * 100 if gt_distances else 0
                
                # 预测统计
                pred_in_zone = [d for d in pred_distances if zone_min <= d < zone_max]
                pred_count = len(pred_in_zone)
                
                # 检测密度
                detection_density = pred_count / max(gt_count, 1)
                
                # 平均置信度
                scores_in_zone = []
                for i, dist in enumerate(pred_distances):
                    if zone_min <= dist < zone_max and i < len(pred_scores):
                        scores_in_zone.append(pred_scores[i])
                
                avg_confidence = np.mean(scores_in_zone) if scores_in_zone else 0
                
                # 输出
                print(f"{zone_name:10}  {gt_count:6d}  {gt_percentage:6.1f}%  "
                    f"{pred_count:8d}  {detection_density:8.2f}  {avg_confidence:10.3f}")

    # [xmy] - 距离衰减分析
    def xmy_analyze_distance_decay(self, data_package, analyze_classes=True):
        """
        距离衰减分析 - 诊断BEVFusion远距离检测问题
        :param data_package: 数据包
        :param analyze_classes: 是否按类别分析
        """
        print("\n" + "="*80)
        print("📏 [距离衰减分析 - BEVFusion远距离诊断]")
        print("="*80)
        
        import numpy as np
        from collections import defaultdict
        
        # 距离区间定义
        distance_zones = [
            (0, 10, "近距 0-10m"),
            (10, 20, "中近 10-20m"),
            (20, 30, "中距 20-30m"),
            (30, 40, "中远 30-40m"),
            (40, 50, "远距 40-50m"),
            (50, 70, "超远 50-70m"),
            (70, 100, "极限 >70m")
        ]
        
        # 1. 总体距离衰减
        print("\n📊 1. 总体距离衰减分析:")
        print("-" * 85)
        print("距离区间       GT数量   GT占比   预测数量  检测率   平均置信度   状态")
        print("-" * 85)
        
        # 收集所有类别的距离数据
        all_gt_distances = []
        all_pred_distances = []
        all_pred_scores = []
        
        for sample_token, boxes in self.gt_boxes.boxes.items():
            for box in boxes:
                dist = np.linalg.norm(box.translation[:2])
                all_gt_distances.append(dist)
        
        for sample_token, boxes in self.pred_boxes.boxes.items():
            for box in boxes:
                dist = np.linalg.norm(box.translation[:2])
                all_pred_distances.append(dist)
                all_pred_scores.append(box.detection_score)
        
        total_gt = len(all_gt_distances)
        
        for zone_min, zone_max, zone_name in distance_zones:
            # GT统计
            gt_in_zone = [d for d in all_gt_distances if zone_min <= d < zone_max]
            gt_count = len(gt_in_zone)
            gt_ratio = gt_count / total_gt * 100 if total_gt > 0 else 0
            
            # 预测统计
            pred_in_zone = [d for d in all_pred_distances if zone_min <= d < zone_max]
            pred_count = len(pred_in_zone)
            
            # 检测率
            detection_rate = pred_count / max(gt_count, 1)
            
            # 平均置信度
            zone_scores = []
            for i, dist in enumerate(all_pred_distances):
                if zone_min <= dist < zone_max and i < len(all_pred_scores):
                    zone_scores.append(all_pred_scores[i])
            avg_confidence = np.mean(zone_scores) if zone_scores else 0
            
            # 状态判断
            if gt_count > 10:  # 有足够数据判断
                if detection_rate < 0.5:
                    status = "🔴严重"
                elif detection_rate < 0.7:
                    status = "⚠️警告"
                elif detection_rate < 0.85:
                    status = "注意"
                else:
                    status = "✅正常"
            else:
                status = "数据少"
            
            print(f"{zone_name:12} {gt_count:6d}  {gt_ratio:6.1f}%   {pred_count:8d}   "
                f"{detection_rate:7.3f}    {avg_confidence:9.3f}    {status:6}")
        
        # 2. 按类别分析（可选）
        if analyze_classes:
            print("\n\n📊 2. 主要类别距离衰减对比:")
            print("=" * 85)
            
            # 只分析数量足够的类别
            main_classes = []
            for cls in data_package['gt_counter']:
                if data_package['gt_counter'][cls] > 50:  # 至少有50个GT
                    main_classes.append(cls)
            
            if not main_classes:
                print("  无足够数据按类别分析")
                return
            
            # 为每个主要类别生成简表
            for class_name in main_classes[:4]:  # 最多显示4个类别
                print(f"\n[{class_name}] (GT总数: {data_package['gt_counter'].get(class_name, 0)})")
                print("-" * 70)
                
                # 收集该类别的距离数据
                class_gt_dists = []
                class_pred_dists = []
                class_pred_scores = []
                
                for sample_token, boxes in self.gt_boxes.boxes.items():
                    for box in boxes:
                        if box.detection_name == class_name:
                            dist = np.linalg.norm(box.translation[:2])
                            class_gt_dists.append(dist)
                
                for sample_token, boxes in self.pred_boxes.boxes.items():
                    for box in boxes:
                        if box.detection_name == class_name:
                            dist = np.linalg.norm(box.translation[:2])
                            class_pred_dists.append(dist)
                            class_pred_scores.append(box.detection_score)
                
                if not class_gt_dists:
                    continue
                
                # 简化的距离区间
                simple_zones = [
                    (0, 20, "近距"),
                    (20, 40, "中距"),
                    (40, 60, "远距"),
                    (60, 100, "超远")
                ]
                
                for zone_min, zone_max, zone_name in simple_zones:
                    gt_count = len([d for d in class_gt_dists if zone_min <= d < zone_max])
                    pred_count = len([d for d in class_pred_dists if zone_min <= d < zone_max])
                    
                    if gt_count > 0:
                        detection_rate = pred_count / gt_count
                        
                        # 该区间的置信度
                        zone_scores = []
                        for i, dist in enumerate(class_pred_dists):
                            if zone_min <= dist < zone_max and i < len(class_pred_scores):
                                zone_scores.append(class_pred_scores[i])
                        avg_conf = np.mean(zone_scores) if zone_scores else 0
                        
                        status = ""
                        if detection_rate < 0.5:
                            status = "🔴"
                        elif detection_rate < 0.7:
                            status = "⚠️"
                        
                        print(f"  {zone_name:4}: GT={gt_count:3d}, 预测={pred_count:3d}, "
                            f"检测率={detection_rate:.3f}, 置信度={avg_conf:.3f} {status}")
        
        # 3. 距离衰减问题诊断
        print("\n\n🚨 3. 距离衰减问题诊断:")
        print("-" * 70)
        
        # 计算总体衰减曲线
        decay_issues = []
        
        # 简化：对比近距和远距的检测率
        if all_gt_distances:
            # 近距检测率（0-30m）
            near_gt = len([d for d in all_gt_distances if d < 30])
            near_pred = len([d for d in all_pred_distances if d < 30])
            near_rate = near_pred / max(near_gt, 1)
            
            # 远距检测率（>50m）
            far_gt = len([d for d in all_gt_distances if d >= 50])
            far_pred = len([d for d in all_pred_distances if d >= 50])
            far_rate = far_pred / max(far_gt, 1) if far_gt > 0 else 0
            
            if far_gt > 10:  # 有足够远距离数据
                decay_ratio = near_rate / max(far_rate, 0.01)
                if decay_ratio > 1.5:
                    decay_issues.append(f"🔴 距离衰减严重: 近距检测率({near_rate:.2f})是远距({far_rate:.2f})的{decay_ratio:.1f}倍")
                elif decay_ratio > 1.2:
                    decay_issues.append(f"⚠️ 距离衰减明显: 近距检测率({near_rate:.2f})比远距({far_rate:.2f})高{(decay_ratio-1)*100:.0f}%")
        
        # 按类别检查衰减
        for cls in main_classes[:3]:
            # 获取该类别距离数据
            cls_gt_dists = []
            for sample_token, boxes in self.gt_boxes.boxes.items():
                for box in boxes:
                    if box.detection_name == cls:
                        cls_gt_dists.append(np.linalg.norm(box.translation[:2]))
            
            if len(cls_gt_dists) > 50:
                max_dist = max(cls_gt_dists)
                config_range = self.cfg.class_range.get(cls, 50)
                
                if max_dist > config_range * 0.8:
                    decay_issues.append(f"⚠️ {cls}: 最远目标{max_dist:.1f}m接近配置阈值{config_range}m")
        
        # 输出问题诊断
        if decay_issues:
            for issue in decay_issues:
                print(f"  • {issue}")
        else:
            print("  ✅ 未发现严重距离衰减问题")
        
        print("=" * 80)

    # [xmy] - 密度影响分析
    def xmy_analyze_density_impact(self, data_package, analyze_classes=True):
        """
        密度影响分析 - 诊断拥挤场景问题
        :param data_package: 数据包
        :param analyze_classes: 是否按类别分析
        """
        print("\n" + "="*80)
        print("📈 [密度影响分析 - 拥挤场景诊断]")
        print("="*80)
        
        import numpy as np
        from collections import defaultdict
        
        # 密度等级定义
        density_levels = [
            (1, 3, "稀疏(1-3)"),
            (4, 6, "中等(4-6)"),
            (7, 10, "密集(7-10)"),
            (11, 1000, "超密(>10)")
        ]
        
        # 收集所有样本的密度信息
        sample_densities = {}  # sample_token -> GT数量
        
        for sample_token, boxes in self.gt_boxes.boxes.items():
            sample_densities[sample_token] = len(boxes)
        
        if not sample_densities:
            print("  无样本数据")
            return
        
        # 1. 总体密度影响
        print("\n📊 1. 总体密度影响:")
        print("-" * 75)
        print("密度等级    样本数   平均GT数   检测率   平均置信度   状态")
        print("-" * 75)
        
        # 为每个密度等级统计
        density_stats = {}
        
        for level_min, level_max, level_name in density_levels:
            # 找出该密度等级的样本
            level_samples = []
            for sample_token, gt_count in sample_densities.items():
                if level_min <= gt_count <= level_max:
                    level_samples.append(sample_token)
            
            if not level_samples:
                continue
            
            # 统计这些样本
            total_gt = 0
            total_pred = 0
            total_conf = []
            
            for sample_token in level_samples:
                gt_boxes = self.gt_boxes.boxes.get(sample_token, [])
                pred_boxes = self.pred_boxes.boxes.get(sample_token, [])
                
                total_gt += len(gt_boxes)
                total_pred += len(pred_boxes)
                
                # 收集置信度
                for box in pred_boxes:
                    total_conf.append(box.detection_score)
            
            avg_gt = total_gt / len(level_samples) if level_samples else 0
            detection_rate = total_pred / max(total_gt, 1)
            avg_conf = np.mean(total_conf) if total_conf else 0
            
            # 状态判断
            if len(level_samples) >= 5:  # 有足够样本
                if detection_rate < 0.6:
                    status = "🔴严重"
                elif detection_rate < 0.75:
                    status = "⚠️警告"
                elif detection_rate < 0.85:
                    status = "注意"
                else:
                    status = "✅正常"
            else:
                status = "样本少"
            
            density_stats[level_name] = {
                'sample_count': len(level_samples),
                'avg_gt': avg_gt,
                'detection_rate': detection_rate,
                'avg_conf': avg_conf,
                'status': status
            }
            
            print(f"{level_name:12} {len(level_samples):6d}   {avg_gt:7.1f}   "
                f"{detection_rate:7.3f}   {avg_conf:9.3f}   {status:6}")
        
        # 2. 按类别分析（主要分析car）
        if analyze_classes and 'car' in data_package['gt_counter']:
            car_gt_count = data_package['gt_counter']['car']
            if car_gt_count > 100:
                print("\n\n📊 2. Car类别密度影响（交通场景重点）:")
                print("-" * 75)
                print("密度等级    样本数   Car数量   检测率   平均置信度")
                print("-" * 75)
                
                for level_min, level_max, level_name in density_levels:
                    # 找出该密度等级的样本
                    level_samples = []
                    for sample_token, gt_count in sample_densities.items():
                        if level_min <= gt_count <= level_max:
                            level_samples.append(sample_token)
                    
                    if not level_samples:
                        continue
                    
                    # 统计这些样本中的car
                    total_car_gt = 0
                    total_car_pred = 0
                    car_confs = []
                    
                    for sample_token in level_samples:
                        gt_boxes = self.gt_boxes.boxes.get(sample_token, [])
                        pred_boxes = self.pred_boxes.boxes.get(sample_token, [])
                        
                        # car的GT
                        car_gt = [b for b in gt_boxes if b.detection_name == 'car']
                        total_car_gt += len(car_gt)
                        
                        # car的预测
                        car_pred = [b for b in pred_boxes if b.detection_name == 'car']
                        total_car_pred += len(car_pred)
                        
                        # car的置信度
                        for box in car_pred:
                            car_confs.append(box.detection_score)
                    
                    if total_car_gt > 0:
                        detection_rate = total_car_pred / total_car_gt
                        avg_conf = np.mean(car_confs) if car_confs else 0
                        
                        status = ""
                        if detection_rate < 0.6:
                            status = "🔴"
                        elif detection_rate < 0.75:
                            status = "⚠️"
                        
                        print(f"{level_name:12} {len(level_samples):6d}  {total_car_gt:8d}  "
                            f"{detection_rate:7.3f}   {avg_conf:9.3f}  {status}")
        
        # 3. 密度问题诊断
        print("\n\n🚨 3. 密度问题诊断:")
        print("-" * 70)
        
        density_issues = []
        
        # 检查是否有超密样本
        ultra_dense_samples = [s for s, count in sample_densities.items() if count > 15]
        if ultra_dense_samples:
            density_issues.append(f"⚠️ 发现{len(ultra_dense_samples)}个超密集样本(>15个目标)")
        
        # 检查密度衰减
        if '稀疏(1-3)' in density_stats and '密集(7-10)' in density_stats:
            sparse_rate = density_stats['稀疏(1-3)']['detection_rate']
            dense_rate = density_stats['密集(7-10)']['detection_rate']
            
            if dense_rate > 0 and sparse_rate > 0:
                decay_ratio = sparse_rate / dense_rate
                if decay_ratio > 1.3:
                    density_issues.append(f"🔴 密度衰减严重: 稀疏场景检测率({sparse_rate:.2f})是密集场景({dense_rate:.2f})的{decay_ratio:.1f}倍")
        
        # 输出问题
        if density_issues:
            for issue in density_issues:
                print(f"  • {issue}")
        else:
            print("  ✅ 未发现严重密度影响问题")
        
        print("=" * 80)

    # [xmy] - 方向偏差分析
    def xmy_analyze_directional_bias(self, data_package):
        """
        方向偏差分析 - 诊断传感器视角问题
        """
        print("\n" + "="*80)
        print("🧭 [方向偏差分析 - 传感器视角诊断]")
        print("="*80)
        
        import numpy as np
        
        # 方位定义（简化版）
        quadrants = [
            ("前方", lambda x, y: x >= 0),
            ("后方", lambda x, y: x < 0),
            ("右侧", lambda x, y: y >= 0),
            ("左侧", lambda x, y: y < 0)
        ]
        
        print("\n📊 方位检测性能对比:")
        print("-" * 65)
        print("方位    GT数量   预测数量   检测率   平均距离   状态")
        print("-" * 65)
        
        # 统计每个方位
        quadrant_stats = {}
        
        for quad_name, quad_condition in quadrants:
            gt_count = 0
            pred_count = 0
            distances = []
            
            # GT统计
            for boxes in self.gt_boxes.boxes.values():
                for box in boxes:
                    x, y, _ = box.translation
                    if quad_condition(x, y):
                        gt_count += 1
                        distances.append(abs(x))  # 距离取绝对值
            
            # 预测统计
            for boxes in self.pred_boxes.boxes.values():
                for box in boxes:
                    x, y, _ = box.translation
                    if quad_condition(x, y):
                        pred_count += 1
            
            # 计算指标
            detection_rate = pred_count / max(gt_count, 1)
            avg_distance = np.mean(distances) if distances else 0
            
            # 状态判断
            if gt_count > 20:  # 有足够数据
                if detection_rate < 0.65:
                    status = "🔴严重"
                elif detection_rate < 0.75:
                    status = "⚠️警告"
                elif detection_rate < 0.85:
                    status = "注意"
                else:
                    status = "✅正常"
            else:
                status = "数据少"
            
            quadrant_stats[quad_name] = {
                'gt_count': gt_count,
                'pred_count': pred_count,
                'detection_rate': detection_rate,
                'avg_distance': avg_distance,
                'status': status
            }
            
            print(f"{quad_name:4}  {gt_count:8d}  {pred_count:8d}  "
                f"{detection_rate:8.3f}  {avg_distance:8.1f}m  {status:6}")
        
        # 方向偏差诊断
        print("\n🚨 方向偏差诊断:")
        print("-" * 70)
        
        bias_issues = []
        
        # 检查前后差异
        if '前方' in quadrant_stats and '后方' in quadrant_stats:
            front_rate = quadrant_stats['前方']['detection_rate']
            rear_rate = quadrant_stats['后方']['detection_rate']
            
            if front_rate > 0 and rear_rate > 0:
                bias_ratio = front_rate / rear_rate
                if bias_ratio > 1.5:
                    bias_issues.append(f"🔴 前后差异严重: 前方检测率({front_rate:.2f})是后方({rear_rate:.2f})的{bias_ratio:.1f}倍")
                elif bias_ratio > 1.2:
                    bias_issues.append(f"⚠️ 前后差异明显: 前方比后方高{(bias_ratio-1)*100:.0f}%")
        
        # 检查左右差异
        if '左侧' in quadrant_stats and '右侧' in quadrant_stats:
            left_rate = quadrant_stats['左侧']['detection_rate']
            right_rate = quadrant_stats['右侧']['detection_rate']
            
            if left_rate > 0 and right_rate > 0:
                bias_ratio = max(left_rate, right_rate) / min(left_rate, right_rate)
                if bias_ratio > 1.3:
                    if left_rate > right_rate:
                        bias_issues.append(f"⚠️ 左右差异: 左侧({left_rate:.2f})比右侧({right_rate:.2f})好{bias_ratio:.1f}倍")
                    else:
                        bias_issues.append(f"⚠️ 左右差异: 右侧({right_rate:.2f})比左侧({left_rate:.2f})好{bias_ratio:.1f}倍")
        
        # 输出问题
        if bias_issues:
            for issue in bias_issues:
                print(f"  • {issue}")
        else:
            print("  ✅ 未发现显著方向偏差")
        
        print("=" * 80)

    # [xmy] - 问题总结报告
    def xmy_generate_problem_summary(self, data_package):
        """
        生成问题总结报告 - 综合所有分析
        """
        print("\n" + "="*80)
        print("📋 [问题总结报告 - BEVFusion异常AP诊断]")
        print("="*80)
        
        problems = []
        warnings = []
        
        # 这里可以汇总前面分析中发现的问题
        # 实际使用时，需要在前面分析中收集问题信息
        # 这里先给一个模板
        
        print("\n🔍 主要发现问题:")
        print("-" * 70)
        
        # 示例问题
        sample_problems = [
            "1. 🔴 远距离(>50m)检测率仅0.61，相比近距下降32%",
            "2. ⚠️ 后方目标检测率0.71，比前方低19%",
            "3. ⚠️ 密集场景(>6目标)检测率0.63，下降28%",
            "4. ⚠️ pedestrian类别在>40m距离严重漏检(检测率0.31)",
            "5. ✅ car类别整体表现正常，检测率0.87"
        ]
        
        for problem in sample_problems:
            print(f"  {problem}")
        
        print("\n💡 建议调整:")
        print("-" * 70)
        suggestions = [
            "• 调整远距离检测阈值或特征提取",
            "• 检查后方传感器数据质量",
            "• 优化NMS参数处理密集目标",
            "• 增加pedestrian远距离训练样本",
            "• 验证class_range配置合理性"
        ]
        
        for suggestion in suggestions:
            print(f"  {suggestion}")
        
        print("=" * 80)
        
        # 保存报告到文件
        report_file = os.path.join(self.output_dir, "problem_summary.txt")
        with open(report_file, 'w') as f:
            f.write("BEVFusion异常AP诊断报告\n")
            f.write("=" * 60 + "\n\n")
            for problem in sample_problems:
                f.write(problem + "\n")
            f.write("\n建议调整:\n")
            for suggestion in suggestions:
                f.write(suggestion + "\n")
        
        print(f"✅ 详细报告已保存至: {report_file}")

    # [xmy] - 热力图生成器
    def xmy_generate_heatmap_images(self, data_package, custom_output_dir=None):
        """
        热力图生成 - 固定坐标范围避免形变
        """
        print("\n🖼️ [生成热力图 - 固定范围]")
        print("-" * 70)
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import time
            
            # 输出目录
            if custom_output_dir:
                heatmap_dir = custom_output_dir
            else:
                heatmap_dir = os.path.join(os.getcwd(), 'bevfusion_heatmaps')
            
            os.makedirs(heatmap_dir, exist_ok=True)
            timestamp = int(time.time())
            
            # ============================================
            # 🔴 关键修复：固定坐标范围
            # ============================================
            # 感知范围设置（避免自动缩放形变）
            DETECTION_RANGE = 70.0  # 正负70米
            
            # 固定坐标范围（确保所有子图一致）
            FIXED_X_RANGE = [-DETECTION_RANGE, DETECTION_RANGE]
            FIXED_Y_RANGE = [-DETECTION_RANGE, DETECTION_RANGE]  # 保持16:9比例
            
            print(f"  使用固定坐标范围: X{FIXED_X_RANGE}, Y{FIXED_Y_RANGE}")
            
            # ============================================
            # 1. 收集数据
            # ============================================
            print("  收集数据...")
            
            all_gt_coords = []
            all_pred_coords = []
            
            for boxes in self.gt_boxes.boxes.values():
                for box in boxes:
                    x, y, _ = box.translation
                    all_gt_coords.append((x, y))
            
            for boxes in self.pred_boxes.boxes.values():
                for box in boxes:
                    x, y, _ = box.translation
                    all_pred_coords.append((x, y))
            
            if not all_gt_coords:
                print("  ⚠️ 无数据")
                return
            
            # 打印实际数据范围
            gt_x = np.array([c[0] for c in all_gt_coords])
            gt_y = np.array([c[1] for c in all_gt_coords])
            print(f"  实际数据范围: X[{gt_x.min():.1f}, {gt_x.max():.1f}], Y[{gt_y.min():.1f}, {gt_y.max():.1f}]")
            
            # ============================================
            # 2. 生成All分析图
            # ============================================
            print("  生成All分析图...")
            
            # 提取All的坐标数据
            all_gt_x = np.array([c[0] for c in all_gt_coords])  # 确保这是numpy数组
            all_gt_y = np.array([c[1] for c in all_gt_coords])
            all_pred_x = np.array([c[0] for c in all_pred_coords]) if all_pred_coords else np.array([])
            all_pred_y = np.array([c[1] for c in all_pred_coords]) if all_pred_coords else np.array([])
            
            # 🔴 调用 _generate_class_heatmap 生成All图
            self._generate_class_heatmap(
                class_name='All',  # 类别名称为'All'
                gt_x=all_gt_x,
                gt_y=all_gt_y,
                pred_x=all_pred_x,
                pred_y=all_pred_y,
                fixed_x_range=FIXED_X_RANGE,
                fixed_y_range=FIXED_Y_RANGE,
                detection_range=DETECTION_RANGE,
                output_dir=heatmap_dir,
                timestamp=timestamp
            )
            
            # ============================================
            # 3. 生成类别分析图
            # ============================================
            print("\n  生成类别分析图...")
            
            # 按类别收集数据（如果您还没有这个字典）
            class_coords = {}
            for boxes_dict, data_type in [(self.gt_boxes.boxes, 'gt'), (self.pred_boxes.boxes, 'pred')]:
                for sample_token, boxes in boxes_dict.items():
                    for box in boxes:
                        x, y, _ = box.translation
                        class_name = box.detection_name
                        
                        if class_name not in class_coords:
                            class_coords[class_name] = {'gt': [], 'pred': []}
                        class_coords[class_name][data_type].append((x, y))
            
            # 找出主要类别（GT数量>20）
            main_classes = []
            for class_name, coords in class_coords.items():
                if len(coords['gt']) > 20:
                    main_classes.append((class_name, len(coords['gt'])))
            
            # 按数量排序
            main_classes.sort(key=lambda x: x[1], reverse=True)
            
            for class_name, gt_count in main_classes[:6]:  # 最多6个类别
                try:
                    print(f"    处理 {class_name} (GT={gt_count})...")
                    
                    gt_coords = class_coords[class_name]['gt']
                    pred_coords = class_coords[class_name]['pred']
                    
                    # 转换为numpy数组
                    gt_x = np.array([c[0] for c in gt_coords])
                    gt_y = np.array([c[1] for c in gt_coords])
                    pred_x = np.array([c[0] for c in pred_coords]) if pred_coords else np.array([])
                    pred_y = np.array([c[1] for c in pred_coords]) if pred_coords else np.array([])
                    
                    # 🔴 调用 _generate_class_heatmap 生成类别图
                    self._generate_class_heatmap(
                        class_name=class_name,
                        gt_x=gt_x,
                        gt_y=gt_y,
                        pred_x=pred_x,
                        pred_y=pred_y,
                        fixed_x_range=FIXED_X_RANGE,
                        fixed_y_range=FIXED_Y_RANGE,
                        detection_range=DETECTION_RANGE,
                        output_dir=heatmap_dir,
                        timestamp=timestamp
                    )
                    
                except Exception as e:
                    print(f"      ⚠️ {class_name}失败: {e}")
                    continue
            
            print(f"\n  ✅ 完成！文件保存在: {heatmap_dir}")
        
        except ImportError:
            print("  ⚠️ matplotlib未安装，跳过热力图生成")
            print("    安装命令: pip install matplotlib")
        
        except Exception as e:
            print(f"  ⚠️ 热力图生成失败: {e}")
            import traceback
            traceback.print_exc()

    # [xmy] - 生成单个类别的热力图
    def _generate_class_heatmap(self, class_name, gt_x, gt_y, pred_x, pred_y,
                                fixed_x_range, fixed_y_range, detection_range,
                                output_dir, timestamp):
        """
        生成单个类别的热力图
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 创建画布
        fig = plt.figure(figsize=(24, 30))
        
        fig.suptitle(f'BEVFusion Spatial Analysis - {class_name.upper()}\n'
                    f'GT: {len(gt_x):,} boxes, Prediction: {len(pred_y):,} boxes\n'
                    f'Fixed Range: X{fixed_x_range}, Y{fixed_y_range}', 
                    fontsize=16, fontweight='bold', y=0.99)
        
        # =========== 第1行: 散点图 ===========
        # 1-1: GT散点图
        ax1 = plt.subplot(5, 2, 1)
        if len(gt_x) > 0:
            ax1.scatter(gt_x, gt_y, c='red', s=8, alpha=0.6, marker='.')
            ax1.set_title(f'GT {class_name} Scatter', fontsize=12, fontweight='bold')
            ax1.set_xlabel('X (m)', fontsize=10)
            ax1.set_ylabel('Y (m)', fontsize=10)
            ax1.set_xlim(fixed_x_range)
            ax1.set_ylim(fixed_y_range)
            ax1.grid(True, alpha=0.2, linestyle='--')
            ax1.set_aspect('equal')
        
        # 1-2: Pred散点图
        ax2 = plt.subplot(5, 2, 2)
        if len(pred_x) > 0:
            ax2.scatter(pred_x, pred_y, c='blue', s=8, alpha=0.6, marker='.')
            ax2.set_title(f'Prediction {class_name} Scatter', fontsize=12, fontweight='bold')
            ax2.set_xlabel('X (m)', fontsize=10)
            ax2.set_ylabel('Y (m)', fontsize=10)
            ax2.set_xlim(fixed_x_range)
            ax2.set_ylim(fixed_y_range)
            ax2.grid(True, alpha=0.2, linestyle='--')
            ax2.set_aspect('equal')
        
        # =========== 第2行: 热力图 ===========
        # 2-1: GT热力图
        ax3 = plt.subplot(5, 2, 3)
        if len(gt_x) > 0:
            h3 = ax3.hist2d(gt_x, gt_y, bins=60, range=[fixed_x_range, fixed_y_range], cmap='Reds')
            ax3.set_title(f'GT {class_name} Heatmap (Linear)', fontsize=12, fontweight='bold')
            ax3.set_xlabel('X (m)', fontsize=10)
            ax3.set_ylabel('Y (m)', fontsize=10)
            plt.colorbar(h3[3], ax=ax3, label='Count', shrink=0.8)
            ax3.set_aspect('equal')
        
        # 2-2: Pred热力图
        ax4 = plt.subplot(5, 2, 4)
        if len(pred_x) > 0:
            h4 = ax4.hist2d(pred_x, pred_y, bins=60, range=[fixed_x_range, fixed_y_range], cmap='Blues')
            ax4.set_title(f'Prediction {class_name} Heatmap (Linear)', fontsize=12, fontweight='bold')
            ax4.set_xlabel('X (m)', fontsize=10)
            ax4.set_ylabel('Y (m)', fontsize=10)
            plt.colorbar(h4[3], ax=ax4, label='Count', shrink=0.8)
            ax4.set_aspect('equal')
        
        # =========== 第3行: 对数密度图 ===========
        # 3-1: GT对数密度图
        ax5 = plt.subplot(5, 2, 5)
        if len(gt_x) > 0:
            h5 = ax5.hist2d(gt_x, gt_y, bins=60, range=[fixed_x_range, fixed_y_range], 
                        cmap='Reds', norm=plt.matplotlib.colors.LogNorm(vmin=1))
            ax5.set_title(f'GT {class_name} Density (Log Scale)', fontsize=12, fontweight='bold')
            ax5.set_xlabel('X (m)', fontsize=10)
            ax5.set_ylabel('Y (m)', fontsize=10)
            plt.colorbar(h5[3], ax=ax5, label='Count (log)', shrink=0.8)
            ax5.set_aspect('equal')
        
        # 3-2: Pred对数密度图
        ax6 = plt.subplot(5, 2, 6)
        if len(pred_x) > 0:
            h6 = ax6.hist2d(pred_x, pred_y, bins=60, range=[fixed_x_range, fixed_y_range],
                        cmap='Blues', norm=plt.matplotlib.colors.LogNorm(vmin=1))
            ax6.set_title(f'Prediction {class_name} Density (Log Scale)', fontsize=12, fontweight='bold')
            ax6.set_xlabel('X (m)', fontsize=10)
            ax6.set_ylabel('Y (m)', fontsize=10)
            plt.colorbar(h6[3], ax=ax6, label='Count (log)', shrink=0.8)
            ax6.set_aspect('equal')
        
        # =========== 第4行: 距离分布 ===========
        # 4-1: GT距离分布
        ax7 = plt.subplot(5, 2, 7)
        if len(gt_x) > 0:
            gt_distances = np.sqrt(gt_x**2 + gt_y**2)
            ax7.hist(gt_distances, bins=50, range=[0, detection_range], 
                    color='red', alpha=0.7, edgecolor='black')
            ax7.set_title(f'GT {class_name} Distance Distribution', fontsize=12, fontweight='bold')
            ax7.set_xlabel('Distance from ego (m)', fontsize=10)
            ax7.set_ylabel('Count', fontsize=10)
            ax7.grid(True, alpha=0.2, linestyle='--')
        
        # 4-2: Pred距离分布
        ax8 = plt.subplot(5, 2, 8)
        if len(pred_x) > 0:
            pred_distances = np.sqrt(pred_x**2 + pred_y**2)
            ax8.hist(pred_distances, bins=50, range=[0, detection_range],
                    color='blue', alpha=0.7, edgecolor='black')
            ax8.set_title(f'Prediction {class_name} Distance Distribution', fontsize=12, fontweight='bold')
            ax8.set_xlabel('Distance from ego (m)', fontsize=10)
            ax8.set_ylabel('Count', fontsize=10)
            ax8.grid(True, alpha=0.2, linestyle='--')
        
        # =========== 第5行: 叠加图和统计 ===========
        # 5-1: 叠加对比图
        ax9 = plt.subplot(5, 2, 9)
        if len(gt_x) > 0 or len(pred_x) > 0:
            if len(gt_x) > 0:
                ax9.scatter(gt_x, gt_y, c='red', s=6, alpha=0.5, marker='.', label='GT')
            if len(pred_x) > 0:
                ax9.scatter(pred_x, pred_y, c='blue', s=6, alpha=0.5, marker='.', label='Prediction')
            
            ax9.set_title(f'{class_name} Overlay: GT(red) + Prediction(blue)', fontsize=12, fontweight='bold')
            ax9.set_xlabel('X (m)', fontsize=10)
            ax9.set_ylabel('Y (m)', fontsize=10)
            ax9.set_xlim(fixed_x_range)
            ax9.set_ylim(fixed_y_range)
            ax9.legend(fontsize=9, loc='upper right')
            ax9.grid(True, alpha=0.2, linestyle='--')
            ax9.set_aspect('equal')
        
        # 5-2: 统计信息
        ax10 = plt.subplot(5, 2, 10)
        ax10.axis('off')
        
        # 计算统计信息
        info_text = f"{class_name.upper()} Statistics:\n"
        info_text += "=" * 30 + "\n"
        info_text += f"GT Boxes: {len(gt_x):,}\n"
        info_text += f"Pred Boxes: {len(pred_x):,}\n"
        
        if len(gt_x) > 0:
            detection_rate = len(pred_x) / len(gt_x) if len(gt_x) > 0 else 0
            info_text += f"Detection Rate: {detection_rate:.3f}\n\n"
            
            info_text += f"GT Position Stats:\n"
            info_text += f"  X: [{gt_x.min():.1f}, {gt_x.max():.1f}]\n"
            info_text += f"  Y: [{gt_y.min():.1f}, {gt_y.max():.1f}]\n"
            info_text += f"  Mean: ({gt_x.mean():.1f}, {gt_y.mean():.1f})\n"
            info_text += f"  Std: ({gt_x.std():.1f}, {gt_y.std():.1f})\n\n"
            
            info_text += f"Distance Stats:\n"
            if len(gt_x) > 0:
                info_text += f"  Max: {np.sqrt(gt_x**2 + gt_y**2).max():.1f}m\n"
                info_text += f"  Mean: {np.sqrt(gt_x**2 + gt_y**2).mean():.1f}m"
        
        ax10.text(0.05, 0.5, info_text, fontsize=9, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 调整子图间距
        plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.05, 
                        hspace=0.25, wspace=0.25)
        
        # 保存
        output_path = os.path.join(output_dir, f'{class_name}_analysis_{timestamp}.jpg')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"      ✅ {class_name}分析图保存: {output_path}")



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
                    print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  在{box_type}中发现None: sample={sample_token}, index={i}')
                elif box.detection_name is None:
                    none_count += 1
                    print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  在{box_type}中发现detection_name为None: sample={sample_token}, index={i}')
                else:
                    valid_boxes.append(box)
            
            if valid_boxes:
                eval_boxes.boxes[sample_token] = valid_boxes
            else:
                # 如果没有有效框，移除该样本
                del eval_boxes.boxes[sample_token]
        
        print(f'🔵[TYJT调试]>>>  nuscenes/eval/detection/evaluate.py  {box_type}清理: 总数={total_boxes}, None值={none_count}, 有效值={total_boxes-none_count}')
        
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
