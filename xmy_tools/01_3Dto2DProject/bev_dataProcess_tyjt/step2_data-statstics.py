import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class DataStatistic:
    def __init__(self, root_dir, matched_data_path):
        self.root_dir = os.path.expanduser(root_dir)  # 解析~为用户目录
        with open(matched_data_path, "r") as f:
            self.matched_data = json.load(f)
        self.label_dir = os.path.join(self.root_dir, "lidar", "label")
        self.stats = defaultdict(dict)

    def compute_stats(self):
        """计算所有统计量，兼容不同标签文件格式"""
        # 1. 基础数量统计
        self.stats["total_matched_frames"] = len(self.matched_data)
        if self.matched_data:
            self.stats["camera_count"] = {
                cam: len([1 for m in self.matched_data if m["cameras"][cam]]) 
                for cam in self.matched_data[0]["cameras"].keys()
            }
        else:
            self.stats["camera_count"] = {}
            print("警告：未找到匹配的数据帧")

        # 2. 标签类别统计
        self.stats["label_classes"] = defaultdict(int)
        self.stats["label_sizes"] = defaultdict(list)
        
        for frame_idx, m in enumerate(self.matched_data):
            label_fname = m["label"]
            label_path = os.path.join(self.label_dir, label_fname)
            
            # 检查文件是否存在
            if not os.path.exists(label_path):
                print(f"警告：第{frame_idx}帧标签文件不存在 - {label_path}")
                continue
            
            # 加载并解析标签（兼容字典/列表两种格式）
            try:
                with open(label_path, "r") as f:
                    label_data = json.load(f)
                
                # 检测标签格式：根节点是字典还是列表
                if isinstance(label_data, dict):
                    # 格式1：{"objects": [...]}
                    objects = label_data.get("objects", [])
                elif isinstance(label_data, list):
                    # 格式2：直接是对象列表 [...]
                    objects = label_data
                else:
                    print(f"警告：第{frame_idx}帧标签格式错误（既非字典也非列表）- {label_path}")
                    continue
                
            except json.JSONDecodeError:
                print(f"警告：第{frame_idx}帧标签文件解析失败 - {label_path}")
                continue
            except Exception as e:
                print(f"警告：第{frame_idx}帧处理出错 - {str(e)}")
                continue
            
            # 统计标签信息
            for obj in objects:
                # 确保obj是字典（避免列表嵌套错误）
                if not isinstance(obj, dict):
                    print(f"警告：第{frame_idx}帧存在非字典格式的目标 - {obj}")
                    continue
                
                cls = obj.get("class", "unknown")  # 默认为unknown
                self.stats["label_classes"][cls] += 1
                
                # 统计尺寸（width/height/length）
                if all(k in obj for k in ["width", "height", "length"]):
                    self.stats["label_sizes"][cls].append([
                        obj["width"], obj["height"], obj["length"]
                    ])
                else:
                    print(f"警告：第{frame_idx}帧目标{cls}缺少尺寸信息")

        # 3. 时间分布统计
        if self.matched_data:
            timestamps = [m["lidar_ts"] for m in self.matched_data if "lidar_ts" in m]
            if timestamps and len(timestamps) > 1:
                self.stats["time_span"] = (min(timestamps), max(timestamps))
                self.stats["frame_interval"] = np.mean(np.diff(sorted(timestamps)))
            else:
                print("警告：时间戳不足，无法计算时间分布")

    def plot_stats(self, save_dir="stats_plots"):
        """可视化统计结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 标签类别分布
        if self.stats["label_classes"]:
            plt.figure(figsize=(10, 6))
            classes, counts = zip(*self.stats["label_classes"].items())
            plt.bar(classes, counts)
            plt.title("3D Label Class Distribution")
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "class_distribution.png"))
            plt.close()
        else:
            print("警告：无标签类别数据，跳过绘图")
        
        # 2. 帧间隔分布
        if "frame_interval" in self.stats:
            timestamps = sorted([m["lidar_ts"] for m in self.matched_data if "lidar_ts" in m])
            intervals = np.diff(timestamps)
            plt.figure(figsize=(10, 6))
            plt.hist(intervals, bins=20)
            plt.title(f"Frame Interval (Avg: {self.stats['frame_interval']:.2f}ms)")
            plt.xlabel("Interval (ms)")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "frame_interval.png"))
            plt.close()

    def save_stats(self, save_path="data_stats.json"):
        """保存统计结果（转换为普通字典）"""
        self.stats["label_classes"] = dict(self.stats["label_classes"])
        self.stats["label_sizes"] = {k: v for k, v in self.stats["label_sizes"].items()}
        with open(save_path, "w") as f:
            json.dump(self.stats, f, indent=4)
        print(f"数据统计已保存至 {save_path}")


# 运行统计
if __name__ == "__main__":
    root_dir = "~/下载/2d3d4d_20250728_weiyuan/datasets/G51102400001M00_20250728191726_2.0HZ"
    matched_data_path = "matched_data.json"  # step1生成的匹配数据
    stat = DataStatistic(root_dir, matched_data_path)
    stat.compute_stats()
    stat.plot_stats()
    stat.save_stats()