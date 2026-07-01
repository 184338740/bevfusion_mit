import os
import json
import re
import numpy as np
from glob import glob

class DataMatcher:
    def __init__(self, root_dir, time_threshold=50):
        self.root_dir = os.path.expanduser(root_dir)  # 解析~为用户目录
        self.time_threshold = time_threshold  # 时间差阈值(ms)
        self.camera_names = [
            "SC_1A_CamR", "SC_1B_CamR", "SC_1C_CamR", "SC_1D_CamR"
        ]  # 4个目标相机
        self.image_ext = "jpg"  # 图像文件后缀
        self.lidar_exts = ["pcd", "npy"]  # 点云文件后缀
        self.label_ext = "json"  # 标签文件后缀
        self.load_timestamps()  # 从文件名提取时间戳

    def _extract_timestamp(self, filename):
        """从文件名提取时间戳（假设为连续数字，如1627459832456）"""
        # 正则匹配文件名中的所有数字，取最长的连续数字作为时间戳
        nums = re.findall(r'\d+', filename)
        if not nums:
            return None
        # 优先选择长度>10的数字（接近毫秒级时间戳）
        for num in sorted(nums, key=lambda x: len(x), reverse=True):
            if len(num) >= 10:
                return int(num)
        return int(nums[-1])  # 否则取最后一组数字

    def load_timestamps(self):
        """直接从文件中提取时间戳（无需单独的时间戳文件）"""
        # 1. 加载Lidar时间戳
        self.lidar_timestamps = {}
        lidar_dir = os.path.join(self.root_dir, "lidar", "pcd")
        for ext in self.lidar_exts:
            for fname in glob(os.path.join(lidar_dir, f"*.{ext}")):
                basename = os.path.basename(fname)
                ts = self._extract_timestamp(basename)
                if ts is not None:
                    self.lidar_timestamps[basename] = ts
        print(f"加载Lidar文件：{len(self.lidar_timestamps)} 个")

        # 2. 加载相机图像时间戳
        self.camera_timestamps = {cam: {} for cam in self.camera_names}
        for cam in self.camera_names:
            img_dir = os.path.join(self.root_dir, cam, "image_dc")
            if not os.path.exists(img_dir):
                print(f"警告：相机{cam}的图像目录不存在：{img_dir}")
                continue
            for fname in glob(os.path.join(img_dir, f"*.{self.image_ext}")):
                basename = os.path.basename(fname)
                ts = self._extract_timestamp(basename)
                if ts is not None:
                    self.camera_timestamps[cam][basename] = ts
            print(f"加载{cam}图像：{len(self.camera_timestamps[cam])} 张")

        # 3. 加载标签时间戳
        self.label_timestamps = {}
        label_dir = os.path.join(self.root_dir, "lidar", "label")
        for fname in glob(os.path.join(label_dir, f"*.{self.label_ext}")):
            basename = os.path.basename(fname)
            ts = self._extract_timestamp(basename)
            if ts is not None:
                self.label_timestamps[basename] = ts
        print(f"加载标签文件：{len(self.label_timestamps)} 个")

    def _find_closest(self, target_ts, ts_dict):
        """寻找与目标时间戳最接近且在阈值内的文件名"""
        if not ts_dict:
            return None
        min_diff = self.time_threshold + 1
        best_fname = None
        for fname, ts in ts_dict.items():
            diff = abs(ts - target_ts)
            if diff < min_diff:
                min_diff = diff
                best_fname = fname
        return best_fname if min_diff <= self.time_threshold else None

    def match_data(self):
        """匹配所有数据，返回匹配表和不匹配数据"""
        matched = []
        unmatched = {
            "lidar": [], 
            "camera": {cam: list(ts_dict.keys()) for cam, ts_dict in self.camera_timestamps.items()},
            "label": list(self.label_timestamps.keys())
        }

        # 以Lidar为基准匹配
        for lidar_fname, lidar_ts in self.lidar_timestamps.items():
            # 匹配标签
            label_fname = self._find_closest(lidar_ts, self.label_timestamps)
            # 匹配每个相机的图像
            camera_fnames = {}
            for cam in self.camera_names:
                cam_ts_dict = self.camera_timestamps[cam]
                cam_fname = self._find_closest(lidar_ts, cam_ts_dict) if cam_ts_dict else None
                camera_fnames[cam] = cam_fname
                # 从相机不匹配列表中移除匹配项
                if cam_fname and cam_fname in unmatched["camera"][cam]:
                    unmatched["camera"][cam].remove(cam_fname)

            # 记录匹配结果（需至少有一个相机和标签匹配）
            valid_camera = any(camera_fnames.values())
            if label_fname and valid_camera:
                matched.append({
                    "lidar": lidar_fname,
                    "label": label_fname,
                    "cameras": camera_fnames,
                    "lidar_ts": lidar_ts
                })
                # 从标签不匹配列表中移除匹配项
                if label_fname in unmatched["label"]:
                    unmatched["label"].remove(label_fname)
            else:
                unmatched["lidar"].append(lidar_fname)

        return matched, unmatched

    def save_mismatch_report(self, unmatched, report_path="mismatch_report.json"):
        """保存不匹配数据报告"""
        with open(report_path, "w") as f:
            json.dump(unmatched, f, indent=4)
        print(f"不匹配数据已保存至 {report_path}")

    def save_matched_data(self, matched, save_path="matched_data.json"):
        """保存匹配成功的数据表"""
        with open(save_path, "w") as f:
            json.dump(matched, f, indent=4)
        print(f"匹配成功的数据已保存至 {save_path}")


# 运行匹配
if __name__ == "__main__":
    # 注意：替换为实际数据根目录（支持~表示用户目录）
    root_dir = "~/下载/2d3d4d_20250728_weiyuan/datasets/G51102400001M00_20250728191726_2.0HZ"
    matcher = DataMatcher(root_dir, time_threshold=50)  # 时间差阈值50ms
    matched_data, unmatched_data = matcher.match_data()
    
    # 保存结果
    matcher.save_matched_data(matched_data)
    matcher.save_mismatch_report(unmatched_data)
    
    # 输出统计信息
    print("\n=== 匹配结果统计 ===")
    print(f"匹配成功的帧数：{len(matched_data)}")
    print(f"不匹配的Lidar文件：{len(unmatched_data['lidar'])} 个")
    print(f"不匹配的标签文件：{len(unmatched_data['label'])} 个")
    for cam, fnames in unmatched_data["camera"].items():
        print(f"不匹配的{cam}图像：{len(fnames)} 张")