"""
step5_1_analyze_training_log_v1.0_0123.py
训练日志分析脚本 - 用于分析mit-BEVFusion训练效果
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 设置中文字体和样式
rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 定义自己的样式（替代seaborn）
def set_custom_style():
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10

class TrainingLogAnalyzer:
    """训练日志分析器"""
    
    def __init__(self, log_path):
        """
        初始化分析器
        
        Args:
            log_path: tfevents文件路径或目录
        """
        self.log_path = Path(log_path)
        self.events_data = {}
        self.metrics_summary = {}
        
    def load_tensorboard_logs(self):
        """加载TensorBoard日志"""
        print(f"正在加载日志文件: {self.log_path}")
        
        if self.log_path.is_dir():
            # 如果是目录，查找所有的tfevents文件
            event_files = list(self.log_path.glob('events.out.tfevents.*'))
            if not event_files:
                raise FileNotFoundError(f"在目录 {self.log_path} 中未找到tfevents文件")
            event_file = event_files[0]
        else:
            event_file = self.log_path
            
        print(f"读取事件文件: {event_file}")
        
        # 加载事件数据
        event_acc = EventAccumulator(str(event_file))
        event_acc.Reload()
        
        # 获取所有标签
        tags = event_acc.Tags()['scalars']
        print(f"找到的指标标签: {tags}")
        
        # 提取数据
        for tag in tags:
            events = event_acc.Scalars(tag)
            self.events_data[tag] = {
                'steps': [e.step for e in events],
                'values': [e.value for e in events],
                'wall_time': [e.wall_time for e in events]
            }
            
        print(f"成功加载 {len(self.events_data)} 个指标")
        return self
    
    def analyze_training_metrics(self):
        """分析训练指标"""
        print("\n" + "="*60)
        print("训练指标分析")
        print("="*60)
        
        # 分离训练和验证指标
        train_metrics = {}
        val_metrics = {}
        
        for metric_name, data in self.events_data.items():
            metric_lower = metric_name.lower()
            if 'train' in metric_lower and 'val' not in metric_lower and 'validation' not in metric_lower:
                train_metrics[metric_name] = data
            elif 'val' in metric_lower or 'validation' in metric_lower:
                val_metrics[metric_name] = data
            elif 'train' in metric_lower:
                train_metrics[metric_name] = data
        
        # 分析训练指标
        print(f"\n训练指标 ({len(train_metrics)}个):")
        for metric_name, data in train_metrics.items():
            values = data['values']
            if values:
                self.metrics_summary[metric_name] = {
                    'min': np.min(values),
                    'max': np.max(values),
                    'mean': np.mean(values),
                    'final': values[-1],
                    'steps': len(values)
                }
                print(f"  {metric_name}:")
                print(f"    最后值: {values[-1]:.6f}, 平均值: {np.mean(values):.6f}, 范围: [{np.min(values):.6f}, {np.max(values):.6f}]")
        
        # 分析验证指标
        print(f"\n验证指标 ({len(val_metrics)}个):")
        for metric_name, data in val_metrics.items():
            values = data['values']
            if values:
                self.metrics_summary[metric_name] = {
                    'min': np.min(values),
                    'max': np.max(values),
                    'mean': np.mean(values),
                    'final': values[-1],
                    'steps': len(values)
                }
                print(f"  {metric_name}:")
                print(f"    最后值: {values[-1]:.6f}, 平均值: {np.mean(values):.6f}, 范围: [{np.min(values):.6f}, {np.max(values):.6f}]")
        
        # 如果没有明确分类，全部显示
        if not train_metrics and not val_metrics:
            print("\n所有指标:")
            for metric_name, data in self.events_data.items():
                values = data['values']
                if values:
                    self.metrics_summary[metric_name] = {
                        'min': np.min(values),
                        'max': np.max(values),
                        'mean': np.mean(values),
                        'final': values[-1],
                        'steps': len(values)
                    }
                    print(f"  {metric_name}:")
                    print(f"    最后值: {values[-1]:.6f}, 平均值: {np.mean(values):.6f}")
        
        return self
    
    def check_training_problems(self):
        """检查常见训练问题"""
        print("\n" + "="*60)
        print("训练问题诊断")
        print("="*60)
        
        problems = []
        
        # 1. 检查过拟合
        train_loss_metrics = [k for k in self.events_data.keys() if 'loss' in k.lower() and 'train' in k.lower()]
        val_loss_metrics = [k for k in self.events_data.keys() if 'loss' in k.lower() and ('val' in k.lower() or 'validation' in k.lower())]
        
        if train_loss_metrics and val_loss_metrics:
            train_loss = self.events_data[train_loss_metrics[0]]['values']
            val_loss = self.events_data[val_loss_metrics[0]]['values']
            
            if len(train_loss) > 10 and len(val_loss) > 10:
                train_final = np.mean(train_loss[-10:])
                val_final = np.mean(val_loss[-10:])
                
                if val_final > train_final * 1.2:  # 验证损失比训练损失高20%
                    problems.append("可能存在过拟合：验证损失明显高于训练损失")
                    print(f"⚠️  过拟合警告: 训练损失={train_final:.4f}, 验证损失={val_final:.4f}, 比值={val_final/train_final:.2f}")
        
        # 2. 检查损失是否收敛
        for metric_name, data in self.events_data.items():
            if 'loss' in metric_name.lower():
                values = data['values']
                if len(values) > 20:
                    first_quarter = np.mean(values[:len(values)//4])
                    last_quarter = np.mean(values[-len(values)//4:])
                    
                    if last_quarter > first_quarter * 0.8:  # 损失下降不明显
                        problems.append(f"{metric_name}: 损失下降不明显，可能学习率设置不当")
                        print(f"⚠️  收敛警告 [{metric_name}]: 初期={first_quarter:.4f}, 后期={last_quarter:.4f}, 下降比例={1-last_quarter/first_quarter:.2%}")
        
        # 3. 检查指标波动
        for metric_name, data in self.events_data.items():
            values = data['values']
            if len(values) > 10:
                # 计算最后20个点的波动
                last_values = values[-20:] if len(values) >= 20 else values
                std_dev = np.std(last_values)
                mean_val = np.mean(last_values)
                
                if mean_val != 0 and std_dev / abs(mean_val) > 0.5:  # 波动过大
                    problems.append(f"{metric_name}: 指标波动较大")
                    print(f"⚠️  波动警告 [{metric_name}]: 标准差/绝对值均值={std_dev/abs(mean_val):.2f}")
        
        # 4. 检查NaN或异常值
        for metric_name, data in self.events_data.items():
            values = np.array(data['values'])
            if np.any(np.isnan(values)):
                problems.append(f"{metric_name}: 包含NaN值")
                print(f"❌  NaN值警告 [{metric_name}]: 包含{np.sum(np.isnan(values))}个NaN值")
            if np.any(np.isinf(values)):
                problems.append(f"{metric_name}: 包含无穷大值")
                print(f"❌  无穷值警告 [{metric_name}]: 包含无穷大值")
        
        # 5. 检查梯度爆炸（如果存在梯度指标）
        grad_metrics = [k for k in self.events_data.keys() if 'grad' in k.lower() or 'gradient' in k.lower()]
        for metric_name in grad_metrics:
            values = self.events_data[metric_name]['values']
            if len(values) > 0:
                max_grad = np.max(np.abs(values))
                if max_grad > 1000:  # 梯度值过大
                    problems.append(f"{metric_name}: 梯度可能爆炸")
                    print(f"⚠️  梯度警告 [{metric_name}]: 最大梯度绝对值={max_grad:.2f}")
        
        if not problems:
            print("✅ 未发现明显训练问题")
        else:
            print(f"\n发现 {len(problems)} 个潜在问题:")
            for i, problem in enumerate(problems, 1):
                print(f"  {i}. {problem}")
        
        return problems
    
    def visualize_metrics(self, output_dir):
        """可视化指标"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\n生成可视化图表到: {output_dir}")
        
        # 设置自定义样式
        set_custom_style()
        
        # 1. 训练损失曲线
        loss_metrics = [k for k in self.events_data.keys() if 'loss' in k.lower()]
        if loss_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('mit-BEVFusion 训练指标分析', fontsize=16, fontweight='bold')
            
            # 损失曲线
            ax = axes[0, 0]
            for metric_name in loss_metrics:
                data = self.events_data[metric_name]
                ax.plot(data['steps'], data['values'], label=metric_name, linewidth=2)
            ax.set_xlabel('训练步数')
            ax.set_ylabel('损失值')
            ax.set_title('训练损失曲线')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # 学习率曲线（如果存在）
            lr_metrics = [k for k in self.events_data.keys() if 'lr' in k.lower() or 'learning_rate' in k.lower()]
            ax = axes[0, 1]
            if lr_metrics:
                for metric_name in lr_metrics:
                    data = self.events_data[metric_name]
                    ax.plot(data['steps'], data['values'], label=metric_name, linewidth=2)
                ax.set_xlabel('训练步数')
                ax.set_ylabel('学习率')
                ax.set_title('学习率变化')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, '未找到学习率指标', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('学习率变化')
            
            # 关键指标对比
            ax = axes[1, 0]
            key_metrics = []
            for metric_name in self.events_data.keys():
                if any(keyword in metric_name.lower() for keyword in ['iou', 'map', 'accuracy', 'precision', 'recall', 'f1', 'ap', 'nds', 'car', 'pedestrian', 'cyclist']):
                    key_metrics.append(metric_name)
            
            if key_metrics:
                for metric_name in key_metrics[:5]:  # 最多显示5个
                    data = self.events_data[metric_name]
                    ax.plot(data['steps'], data['values'], label=metric_name, linewidth=2)
                ax.set_xlabel('训练步数')
                ax.set_ylabel('指标值')
                ax.set_title('关键性能指标')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, '未找到关键性能指标', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('关键性能指标')
            
            # 训练vs验证损失（如果都有）
            ax = axes[1, 1]
            train_loss = [k for k in loss_metrics if 'train' in k.lower()]
            val_loss = [k for k in loss_metrics if 'val' in k.lower() or 'validation' in k.lower()]
            
            if train_loss and val_loss:
                train_data = self.events_data[train_loss[0]]
                val_data = self.events_data[val_loss[0]]
                
                # 对齐步数
                min_len = min(len(train_data['values']), len(val_data['values']))
                ax.plot(train_data['steps'][:min_len], train_data['values'][:min_len], 
                       label='训练损失', linewidth=2)
                ax.plot(val_data['steps'][:min_len], val_data['values'][:min_len], 
                       label='验证损失', linewidth=2)
                ax.set_xlabel('训练步数')
                ax.set_ylabel('损失值')
                ax.set_title('训练 vs 验证损失')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, '缺少训练或验证损失数据', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('训练 vs 验证损失')
            
            plt.tight_layout()
            plot_path = output_dir / 'training_metrics_overview.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"✅ 已保存综合图表: {plot_path}")
            plt.close()
        else:
            print("⚠️  未找到损失指标，跳过损失图表生成")
        
        # 2. 详细的损失分析图（如果有损失指标）
        if loss_metrics:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 损失衰减曲线
            ax = axes[0]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            for i, metric_name in enumerate(loss_metrics):
                if i >= len(colors):
                    break
                data = self.events_data[metric_name]
                ax.plot(data['steps'], data['values'], label=metric_name, 
                       color=colors[i], linewidth=2, alpha=0.8)
            ax.set_xlabel('训练步数')
            ax.set_ylabel('损失值')
            ax.set_title('损失衰减曲线')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # 损失对数曲线（如果所有值都为正）
            ax = axes[1]
            has_positive_values = False
            for i, metric_name in enumerate(loss_metrics):
                if i >= len(colors):
                    break
                data = self.events_data[metric_name]
                values = np.array(data['values'])
                if np.all(values > 0):
                    ax.semilogy(data['steps'], values, label=metric_name, 
                              color=colors[i], linewidth=2, alpha=0.8)
                    has_positive_values = True
            if has_positive_values:
                ax.set_xlabel('训练步数')
                ax.set_ylabel('损失值 (log scale)')
                ax.set_title('损失对数曲线')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, '损失值不全为正，无法显示对数曲线', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('损失对数曲线')
            
            plt.tight_layout()
            plot_path = output_dir / 'loss_analysis.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"✅ 已保存损失分析图: {plot_path}")
            plt.close()
        
        # 3. 生成指标趋势图（针对BEVFusion特定指标）
        bev_metrics = []
        for metric_name in self.events_data.keys():
            metric_lower = metric_name.lower()
            if any(keyword in metric_lower for keyword in ['bev', 'centerpoint', 'lidar', 'camera', 'fusion', '2d', '3d']):
                bev_metrics.append(metric_name)
        
        if bev_metrics:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # BEV相关指标
            ax = axes[0]
            for metric_name in bev_metrics[:6]:  # 最多显示6个
                data = self.events_data[metric_name]
                ax.plot(data['steps'], data['values'], label=metric_name, linewidth=2)
            ax.set_xlabel('训练步数')
            ax.set_ylabel('指标值')
            ax.set_title('BEVFusion特定指标趋势')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # 最终值比较图
            ax = axes[1]
            metric_names = []
            final_values = []
            
            for metric_name in bev_metrics[:10]:  # 最多显示10个
                data = self.events_data[metric_name]
                if len(data['values']) > 0:
                    metric_names.append(metric_name)
                    final_values.append(data['values'][-1])
            
            if metric_names:
                y_pos = np.arange(len(metric_names))
                bars = ax.barh(y_pos, final_values, alpha=0.7)
                
                # 为每个条形添加数值标签
                for i, (bar, val) in enumerate(zip(bars, final_values)):
                    ax.text(val, bar.get_y() + bar.get_height()/2, 
                           f' {val:.4f}', va='center', fontsize=8)
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(metric_names, fontsize=8)
                ax.set_xlabel('最终值')
                ax.set_title('BEV指标最终值对比')
                ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plot_path = output_dir / 'bevfusion_specific_metrics.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"✅ 已保存BEVFusion特定指标图: {plot_path}")
            plt.close()
        
        # 4. 生成CSV报告
        self.generate_report(output_dir)
        
        return output_dir
    
    def generate_report(self, output_dir):
        """生成分析报告"""
        report_path = output_dir / 'training_analysis_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("mit-BEVFusion 训练日志分析报告\n")
            f.write("="*80 + "\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"日志路径: {self.log_path}\n")
            f.write("\n")
            
            f.write("1. 数据概览:\n")
            f.write("-"*40 + "\n")
            f.write(f"总指标数量: {len(self.events_data)}\n")
            for metric_name, data in self.events_data.items():
                f.write(f"  {metric_name}: {len(data['values'])} 个数据点\n")
            
            f.write("\n2. 指标汇总:\n")
            f.write("-"*40 + "\n")
            for metric_name, summary in self.metrics_summary.items():
                f.write(f"{metric_name}:\n")
                f.write(f"  最后值: {summary['final']:.6f}\n")
                f.write(f"  平均值: {summary['mean']:.6f}\n")
                f.write(f"  最小值: {summary['min']:.6f}\n")
                f.write(f"  最大值: {summary['max']:.6f}\n")
                f.write(f"  数据点: {summary['steps']}个\n")
                f.write("\n")
            
            # 检查问题
            problems = []
            for metric_name, data in self.events_data.items():
                values = data['values']
                if 'loss' in metric_name.lower():
                    if len(values) > 20:
                        first_quarter = np.mean(values[:len(values)//4])
                        last_quarter = np.mean(values[-len(values)//4:])
                        if last_quarter > first_quarter * 0.8:
                            problems.append(f"{metric_name}: 损失下降不明显")
                
                if np.any(np.isnan(values)):
                    problems.append(f"{metric_name}: 包含NaN值")
            
            f.write("3. 训练问题诊断:\n")
            f.write("-"*40 + "\n")
            if problems:
                f.write(f"发现 {len(problems)} 个潜在问题:\n")
                for i, problem in enumerate(problems, 1):
                    f.write(f"  {i}. {problem}\n")
            else:
                f.write("未发现明显训练问题\n")
            
            f.write("\n4. 针对BEVFusion的建议:\n")
            f.write("-"*40 + "\n")
            f.write("A. 如果2D检测效果不好:\n")
            f.write("   1. 检查相机标定参数\n")
            f.write("   2. 验证图像预处理流程\n")
            f.write("   3. 调整CenterPoint参数\n")
            f.write("   4. 检查数据增强策略\n")
            f.write("\nB. 如果3D检测效果不好:\n")
            f.write("   1. 检查LiDAR点云预处理\n")
            f.write("   2. 验证BEV特征提取\n")
            f.write("   3. 调整Voxelization参数\n")
            f.write("   4. 检查坐标转换\n")
            f.write("\nC. 如果多模态融合效果不好:\n")
            f.write("   1. 检查时间同步\n")
            f.write("   2. 验证特征对齐\n")
            f.write("   3. 调整融合权重\n")
            f.write("   4. 检查模态间的信息互补性\n")
            f.write("\nD. 通用建议:\n")
            f.write("   1. 检查数据标注质量\n")
            f.write("   2. 验证数据分布\n")
            f.write("   3. 调整学习率调度策略\n")
            f.write("   4. 增加训练迭代次数\n")
            f.write("   5. 尝试不同的优化器\n")
        
        print(f"✅ 已保存分析报告: {report_path}")
        
        # 保存CSV数据
        csv_data = []
        for metric_name, data in self.events_data.items():
            for step, value in zip(data['steps'], data['values']):
                csv_data.append({
                    'metric': metric_name,
                    'step': step,
                    'value': value
                })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = output_dir / 'training_metrics.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ 已保存CSV数据: {csv_path}")
        
        # 保存摘要数据
        summary_data = []
        for metric_name, summary in self.metrics_summary.items():
            summary_data.append({
                'metric': metric_name,
                'final_value': summary['final'],
                'mean_value': summary['mean'],
                'min_value': summary['min'],
                'max_value': summary['max'],
                'data_points': summary['steps']
            })
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            summary_path = output_dir / 'metrics_summary.csv'
            df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
            print(f"✅ 已保存指标摘要: {summary_path}")
    
    def analyze(self, output_dir):
        """执行完整分析流程"""
        print("="*60)
        print("mit-BEVFusion 训练日志分析工具 v1.0")
        print("="*60)
        
        try:
            # 1. 加载日志
            self.load_tensorboard_logs()
            
            # 2. 分析指标
            self.analyze_training_metrics()
            
            # 3. 检查问题
            self.check_training_problems()
            
            # 4. 可视化
            self.visualize_metrics(output_dir)
            
            print("\n" + "="*60)
            print("分析完成!")
            print(f"结果保存在: {output_dir}")
            print("="*60)
            
            return True
            
        except FileNotFoundError as e:
            print(f"\n❌ 文件未找到: {e}")
            print("请检查日志文件路径是否正确")
            return False
        except Exception as e:
            print(f"\n❌ 分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(description='mit-BEVFusion训练日志分析工具')
    parser.add_argument('--log_path', type=str, required=True,
                       help='TensorBoard日志文件路径(tfevents文件)')
    parser.add_argument('--output_dir', type=str, default='output/step5',
                       help='输出目录 (默认: output/step5)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 创建分析器
    analyzer = TrainingLogAnalyzer(args.log_path)
    
    # 执行分析
    success = analyzer.analyze(output_dir)
    
    if success:
        print("\n📊 分析结果:")
        print(f"1. 综合图表: {output_dir}/training_metrics_overview.png")
        print(f"2. 损失分析: {output_dir}/loss_analysis.png")
        print(f"3. 详细报告: {output_dir}/training_analysis_report.txt")
        print(f"4. 原始数据: {output_dir}/training_metrics.csv")
        print(f"5. 指标摘要: {output_dir}/metrics_summary.csv")
        
        print("\n🔧 建议下一步操作:")
        print("1. 查看图表，分析训练趋势和问题")
        print("2. 阅读分析报告中的建议")
        print("3. 检查数据预处理和标注质量")
        print("4. 根据BEVFusion特点调整训练策略")
        print("5. 对比不同配置的训练结果")
    else:
        print("\n❌ 分析失败")

if __name__ == "__main__":
    main()