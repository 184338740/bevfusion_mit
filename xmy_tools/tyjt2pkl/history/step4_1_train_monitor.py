import re
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BEVFusionLogMonitor:
    def __init__(self, log_path, interval=10):
        """
        初始化日志监控器
        :param log_path: 训练日志文件路径
        :param interval: 监控刷新间隔（秒）
        """
        self.log_path = log_path
        self.interval = interval
        self.data = defaultdict(list)  # 存储解析的数据
        self.epoch_pattern = re.compile(r'Epoch \[(\d+)\]\[(\d+)/(\d+)\]')
        self.metric_pattern = re.compile(
            r'lr: ([\d.e-]+), .*?loss/object/loss_heatmap: ([\d.]+), '
            r'loss/object/layer_-1_loss_cls: ([\d.]+), '
            r'loss/object/layer_-1_loss_bbox: ([\d.]+), '
            r'stats/object/matched_ious: ([\d.]+), '
            r'loss: ([\d.]+), grad_norm: ([\d.nan]+)'
        )
        self.total_epochs = None

    def parse_log_line(self, line):
        """解析单条日志内容"""
        epoch_match = self.epoch_pattern.search(line)
        metric_match = self.metric_pattern.search(line)
        
        if epoch_match and metric_match:
            # 提取epoch和批次信息
            epoch = int(epoch_match.group(1))
            batch = int(epoch_match.group(2))
            total_batches = int(epoch_match.group(3))
            
            # 提取指标信息
            lr = float(metric_match.group(1))
            loss_heatmap = float(metric_match.group(2))
            loss_cls = float(metric_match.group(3))
            loss_bbox = float(metric_match.group(4))
            matched_iou = float(metric_match.group(5))
            total_loss = float(metric_match.group(6))
            grad_norm = metric_match.group(7)
            grad_norm = float(grad_norm) if grad_norm != 'nan' else np.nan
            
            return {
                'epoch': epoch,
                'batch': batch,
                'total_batches': total_batches,
                'lr': lr,
                'loss_heatmap': loss_heatmap,
                'loss_cls': loss_cls,
                'loss_bbox': loss_bbox,
                'matched_iou': matched_iou,
                'total_loss': total_loss,
                'grad_norm': grad_norm
            }
        return None

    def update_data(self):
        """更新日志数据（增量解析）"""
        current_lines = 0
        # 读取日志文件（只解析新增内容）
        with open(self.log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            new_lines = lines[current_lines:]
            current_lines = len(lines)
            
            for line in new_lines:
                parsed = self.parse_log_line(line)
                if parsed:
                    # 存储数据
                    for key, value in parsed.items():
                        self.data[key].append(value)
                    # 记录总epoch数（取最大epoch）
                    if self.total_epochs is None or parsed['epoch'] > self.total_epochs:
                        self.total_epochs = parsed['epoch']

    def plot_metrics(self):
        """绘制监控图表"""
        if len(self.data['epoch']) == 0:
            print("暂无训练数据可展示")
            return
        
        # 创建2x2子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'BEVFusion Training Monitor (Epoch: {self.data["epoch"][-1]}/{self.total_epochs})', fontsize=16)
        
        # 准备x轴数据（批次序号）
        x = list(range(1, len(self.data['epoch']) + 1))
        
        # 1. 总损失和分项损失
        ax1.plot(x, self.data['total_loss'], label='Total Loss', linewidth=2, color='red')
        ax1.plot(x, self.data['loss_heatmap'], label='Heatmap Loss', linewidth=1.5, linestyle='--')
        ax1.plot(x, self.data['loss_cls'], label='Cls Loss', linewidth=1.5, linestyle='--')
        ax1.plot(x, self.data['loss_bbox'], label='Bbox Loss', linewidth=1.5, linestyle='--')
        ax1.set_xlabel('Batch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 学习率变化
        ax2.plot(x, self.data['lr'], label='Learning Rate', color='orange', linewidth=2)
        ax2.set_xlabel('Batch')
        ax2.set_ylabel('LR')
        ax2.set_title('Learning Rate Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')  # 对数坐标更易观察
        
        # 3. 匹配IOU
        ax3.plot(x, self.data['matched_iou'], label='Matched IOU', color='green', linewidth=2)
        ax3.set_xlabel('Batch')
        ax3.set_ylabel('IOU')
        ax3.set_title('Matched IOU Curve')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)  # IOU范围0-1
        
        # 4. 梯度范数
        # 过滤nan值
        grad_norm = np.array(self.data['grad_norm'])
        grad_x = [x[i] for i in range(len(x)) if not np.isnan(grad_norm[i])]
        grad_y = [grad_norm[i] for i in range(len(grad_norm)) if not np.isnan(grad_norm[i])]
        
        ax4.plot(grad_x, grad_y, label='Gradient Norm', color='purple', linewidth=2)
        ax4.set_xlabel('Batch')
        ax4.set_ylabel('Grad Norm')
        ax4.set_title('Gradient Norm Curve')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(1)  # 显示1秒后释放
        
    def print_summary(self):
        """打印训练摘要信息"""
        if len(self.data['epoch']) == 0:
            return
        
        latest_idx = -1
        print("\n" + "="*50)
        print(f"训练摘要 (Epoch {self.data['epoch'][latest_idx]}, Batch {self.data['batch'][latest_idx]}/{self.data['total_batches'][latest_idx]})")
        print("="*50)
        print(f"学习率: {self.data['lr'][latest_idx]:.6f}")
        print(f"总损失: {self.data['total_loss'][latest_idx]:.4f}")
        print(f"热力图损失: {self.data['loss_heatmap'][latest_idx]:.4f}")
        print(f"分类损失: {self.data['loss_cls'][latest_idx]:.4f}")
        print(f"回归损失: {self.data['loss_bbox'][latest_idx]:.4f}")
        print(f"匹配IOU: {self.data['matched_iou'][latest_idx]:.4f}")
        print(f"梯度范数: {self.data['grad_norm'][latest_idx]:.4f}" if not np.isnan(self.data['grad_norm'][latest_idx]) else "梯度范数: NaN")
        print("="*50 + "\n")

    def run(self):
        """启动监控器"""
        print(f"启动BEVFusion训练监控器，日志路径：{self.log_path}")
        print(f"监控间隔：{self.interval}秒\n")
        
        try:
            while True:
                self.update_data()
                self.print_summary()
                self.plot_metrics()
                time.sleep(self.interval)
                plt.close('all')  # 关闭旧图，避免内存占用
        except KeyboardInterrupt:
            print("\n监控器已停止")
        except Exception as e:
            print(f"监控器异常：{e}")


if __name__ == "__main__":
    """
    # 替换训练日志路径
    python train_monitor.py --log_path /mnt/bevfusion_mit_xmy/runs/train-tyjt-shape-fixed/20251030_080219.log --interval 15
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='BEVFusion训练日志监控脚本')
    parser.add_argument('--log_path', required=True, help='训练日志文件路径')
    parser.add_argument('--interval', type=int, default=10, help='监控刷新间隔（秒）')
    args = parser.parse_args()
    
    # 启动监控器
    monitor = BEVFusionLogMonitor(args.log_path, args.interval)
    monitor.run()