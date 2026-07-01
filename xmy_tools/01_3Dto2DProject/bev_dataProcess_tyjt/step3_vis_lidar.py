import os
import re
import numpy as np
import json
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image

# 配置 matplotlib
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

class Calibration:
    def __init__(self, calib_path):
        self.calib_path = os.path.expanduser(calib_path)
        if not os.path.exists(self.calib_path):
            raise FileNotFoundError(f"Calibration file not found: {self.calib_path}")
        with open(self.calib_path, "r") as f:
            self.calib = json.load(f)
        print(f"✅ Loaded calibration: {self.calib_path}")

    def get_extrinsics(self, sensor_name):
        """获取外参矩阵 (sensor → map)"""
        if sensor_name not in self.calib:
            raise KeyError(f"No extrinsics for {sensor_name}")
        params = self.calib[sensor_name]
        # 四元数转旋转矩阵 (w, x, y, z)
        w, x, y, z = params["rw"], params["rx"], params["ry"], params["rz"]
        R = np.array([
            [1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]
        ])
        # 平移向量 (mm → m)
        t = np.array([params["tx"], params["ty"], params["tz"]]) / 1000.0
        # 构造4x4外参矩阵 [R t; 0 1]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_intrinsics(self, sensor_name):
        """获取相机内参 (fx, fy, cx, cy)"""
        if sensor_name not in self.calib:
            raise KeyError(f"No intrinsics for {sensor_name}")
        params = self.calib[sensor_name]
        return {
            "fx": params["fx"],
            "fy": params["fy"],
            "cx": params["cx"],
            "cy": params["cy"],
            "width": params.get("width", 1920),
            "height": params.get("height", 1080)
        }

    def transform_points(self, points, T):
        """将点云通过变换矩阵T转换坐标"""
        points_hom = np.hstack([points, np.ones((len(points), 1))])  # 齐次坐标 (N,4)
        transformed = (T @ points_hom.T).T  # (4,N) → (N,4)
        return transformed[:, :3]  # 去除齐次分量 (N,3)


class ProjectionVisualizer:
    def __init__(self, root_dir, calib_path, dataset_dir, matched_data_path, 
                 save_dir="projection_results", 
                 flip_y=False,  # Y轴翻转开关
                 vertical_offset=0):  # 垂直偏移补偿（像素单位）
        self.root_dir = os.path.expanduser(root_dir)
        self.dataset_dir = os.path.join(self.root_dir, dataset_dir)
        self.save_dir = save_dir
        self.flip_y = flip_y  # 是否翻转相机坐标系Y轴
        self.vertical_offset = vertical_offset  # 投影垂直偏移
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"📌 Projection results saved to: {self.save_dir}")
        print(f"📌 Debug params: flip_y={flip_y}, vertical_offset={vertical_offset}")

        # 加载数据与标定
        self.matched_data = self._load_matched_data(matched_data_path)
        self.calib = Calibration(os.path.join(self.root_dir, calib_path))
        self.group2map = self.calib.get_extrinsics("group2map")  # group→map的变换
        self.cam_mapping = self._auto_match_cameras()  # 相机名称映射
        print(f"✅ Matched cameras: {list(self.cam_mapping.keys())}")

    def _auto_match_cameras(self):
        """自动匹配标定文件相机与图像文件夹"""
        cam_mapping = {}
        pattern = re.compile(r"SC_R1_([A-Za-z])e_CpcN_CAMR")
        for calib_cam in self.calib.calib.keys():
            match = pattern.match(calib_cam)
            if match:
                letter = match.group(1)
                img_cam = f"SC_1{letter}_CamR"
                if os.path.exists(os.path.join(self.dataset_dir, img_cam)):
                    cam_mapping[img_cam] = calib_cam  # {图像文件夹: 标定相机名}
        return cam_mapping

    def _load_matched_data(self, path):
        path = os.path.expanduser(path)
        with open(path, "r") as f:
            return json.load(f)

    def load_lidar(self, lidar_fname):
        """加载点云数据 (N,3)"""
        pcd_path = os.path.join(self.dataset_dir, "lidar", "pcd", lidar_fname)
        if not os.path.exists(pcd_path):
            print(f"❌ Lidar file missing: {pcd_path}")
            return None
        pcd = o3d.io.read_point_cloud(pcd_path)
        return np.asarray(pcd.points)

    def load_image(self, cam_name, img_fname):
        """加载图像数据"""
        img_path = os.path.join(self.dataset_dir, cam_name, "image_dc", img_fname)
        if not os.path.exists(img_path):
            print(f"❌ Image file missing: {img_path}")
            return None
        return np.array(Image.open(img_path))

    def project_lidar_to_image(self, lidar_points, cam_name):
        """
        将点云投影到指定相机的图像上
        步骤: lidar (group坐标系) → map坐标系 → 相机坐标系 → 图像像素
        """
        # 1. 获取相机参数
        calib_cam = self.cam_mapping[cam_name]
        cam2map = self.calib.get_extrinsics(calib_cam)  # 相机→map的变换
        map2cam = np.linalg.inv(cam2map)  # map→相机的变换 (逆矩阵)
        intr = self.calib.get_intrinsics(calib_cam)
        print(f"📌 Camera intrinsics: fx={intr['fx']:.1f}, fy={intr['fy']:.1f}, cx={intr['cx']:.1f}, cy={intr['cy']:.1f}")

        # 2. 坐标转换: group → map
        lidar_in_map = self.calib.transform_points(lidar_points, self.group2map)
        # 3. 坐标转换: map → 相机坐标系
        lidar_in_cam = self.calib.transform_points(lidar_in_map, map2cam)

        # 关键修正1：根据开关选择是否翻转Y轴
        if self.flip_y:
            lidar_in_cam[:, 1] = -lidar_in_cam[:, 1]  # Y轴取反
            print("🔄 Flipped camera Y-axis")

        # 4. 相机坐标系 → 图像像素 (透视投影)
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        z = lidar_in_cam[:, 2]  # 相机坐标系下的z值 (深度)
        
        # 过滤有效点
        valid_mask = (
            (z > 0.5) &  # 只保留50cm以外的点
            (z < 30) &   # 只保留30m以内的点
            (np.abs(lidar_in_cam[:, 0]) < z * 2) &  # 水平方向限制
            (np.abs(lidar_in_cam[:, 1]) < z * 2)    # 垂直方向限制
        )

        # 调试：打印相机坐标系下点的分布
        if np.any(valid_mask):
            print(f"📊 Camera coords stats (valid points):")
            print(f"   X: min={lidar_in_cam[valid_mask,0].min():.2f}, max={lidar_in_cam[valid_mask,0].max():.2f}")
            print(f"   Y: min={lidar_in_cam[valid_mask,1].min():.2f}, max={lidar_in_cam[valid_mask,1].max():.2f}")
            print(f"   Z: min={z[valid_mask].min():.2f}, max={z[valid_mask].max():.2f}")

        # 计算像素坐标（增加垂直偏移补偿）
        u = (fx * lidar_in_cam[valid_mask, 0] / z[valid_mask]) + cx
        v = (fy * lidar_in_cam[valid_mask, 1] / z[valid_mask]) + cy + self.vertical_offset  # 垂直偏移
        pixels = np.column_stack([u, v])

        # 过滤超出图像范围的像素
        img_h, img_w = intr["height"], intr["width"]
        in_bounds = (pixels[:, 0] >= 0) & (pixels[:, 0] < img_w) & \
                    (pixels[:, 1] >= 0) & (pixels[:, 1] < img_h)
        return pixels[in_bounds], lidar_in_cam[valid_mask][in_bounds]

    def visualize_projection(self, frame_idx):
        """可视化指定帧的点云到所有相机的投影"""
        frame_data = self.matched_data[frame_idx]
        print(f"\n===== Processing Frame {frame_idx} =====")

        # 加载点云
        lidar = self.load_lidar(frame_data["lidar"])
        if lidar is None or len(lidar) == 0:
            print("❌ No valid lidar data")
            return

        # 对每个相机进行投影
        for cam_name, img_fname in frame_data["cameras"].items():
            if cam_name not in self.cam_mapping or not img_fname:
                continue  # 跳过未匹配的相机

            # 加载图像
            img = self.load_image(cam_name, img_fname)
            if img is None:
                continue

            # 点云投影到图像
            pixels, points_in_cam = self.project_lidar_to_image(lidar, cam_name)
            print(f"📊 {cam_name}: {len(pixels)} valid projected points")

            # 保存投影结果
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.scatter(pixels[:, 0], pixels[:, 1], s=1, c='r', alpha=0.6)
            plt.title(f"{cam_name} - Lidar Projection (Frame {frame_idx})")
            plt.axis("off")
            save_path = os.path.join(self.save_dir, f"proj_{cam_name}_frame{frame_idx}.png")
            plt.savefig(save_path, bbox_inches="tight", dpi=200)
            plt.close()
            print(f"💾 Saved projection: {save_path}")

    def run(self, frame_idx=0):
        self.visualize_projection(frame_idx)


if __name__ == "__main__":
    # 强制使用X11后端（解决Wayland问题）
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    os.environ["LIBGL_ALWAYS_INDIRECT"] = "0"
    
    # 路径配置
    root_dir = "~/下载/2d3d4d_20250728_weiyuan"
    calib_path = "calib/sensor2map_calib.json"
    dataset_dir = "datasets/G51102400001M00_20250728191726_2.0HZ"
    matched_data_path = "matched_data.json"

    # 关键调试参数（重点调整这两个参数）
    flip_y = False  # 先尝试关闭翻转（之前翻转后点云在下方，可能不需要翻转）
    vertical_offset = -100  # 向上移动100像素（根据结果递增调整，如-150、-200）

    try:
        viz = ProjectionVisualizer(
            root_dir=root_dir,
            calib_path=calib_path,
            dataset_dir=dataset_dir,
            matched_data_path=matched_data_path,
            flip_y=flip_y,
            vertical_offset=vertical_offset
        )
        viz.run(frame_idx=0)
        print("\n🎉 Projection completed successfully")
    except Exception as e:
        print(f"❌ Error: {str(e)}")