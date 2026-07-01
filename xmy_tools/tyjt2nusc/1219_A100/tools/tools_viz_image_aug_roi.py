import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
import matplotlib.lines as mlines

def compute_roi_original(s, p, H_orig, W_orig, H_tar, W_tar):
    """
    计算在原始图像上的ROI矩形（左上角坐标及宽高）
    参数：
        s: 缩放因子
        p: 底部裁剪比例 (0~1)
        H_orig, W_orig: 原始图像尺寸
        H_tar, W_tar: 目标图像尺寸
    返回：
        (x, y, w, h) 原始图像上的矩形（左上角坐标及宽高）
    """
    # 缩放后尺寸
    Hs = H_orig * s
    Ws = W_orig * s

    # 底部裁剪后剩余高度
    H_remain = Hs * (1 - p)

    # 中心裁剪偏移量
    crop_x = (Ws - W_tar) / 2.0
    crop_y = (H_remain - H_tar) / 2.0

    # 逆映射到原始图像
    x_orig = crop_x / s
    y_orig = crop_y / s
    w_orig = W_tar / s
    h_orig = H_tar / s

    return x_orig, y_orig, w_orig, h_orig

def validate_inputs(orig_size, img_path, H_orig_input, W_orig_input):
    """
    校验输入参数一致性
    返回：(valid, image, H_orig, W_orig, message)
    """
    if img_path is not None:
        img = cv2.imread(img_path)
        if img is None:
            return False, None, 0, 0, f"Error: Cannot read image from {img_path}"
        H_real, W_real = img.shape[:2]
        if H_orig_input is not None and W_orig_input is not None:
            if (H_orig_input, W_orig_input) != (H_real, W_real):
                return False, None, H_real, W_real, \
                       f"Error: Size mismatch. Provided size: ({H_orig_input}, {W_orig_input}), image actual size: ({H_real}, {W_real})"
            else:
                return True, img, H_real, W_real, "Success: Image loaded and size matches."
        else:
            return True, img, H_real, W_real, "Success: Image loaded, using actual size."
    else:
        if H_orig_input is None or W_orig_input is None:
            return False, None, 0, 0, "Error: Neither image path nor original size provided."
        else:
            img = np.ones((H_orig_input, W_orig_input, 3), dtype=np.uint8) * 255
            print(f"Warning: No image provided. Using blank canvas of size ({H_orig_input}, {W_orig_input}).")
            return True, img, H_orig_input, W_orig_input, "Warning: Blank canvas created."

def draw_rois(img, H_orig, W_orig, H_tar, W_tar,
              train_resize_lim, train_bot_lim,
              test_resize_lim, test_bot_lim,
              draw_train=True, draw_test=True,
              show_outside_roi=False):
    """
    在图像上绘制ROI矩形
    show_outside_roi: 若为True，扩展坐标轴范围显示完整的矩形（即使超出图像边界）
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), extent=[0, W_orig, H_orig, 0])  # 明确设置坐标范围
    ax.set_title("ROI Visualization on Original Image")
    ax.set_xlabel("Width (pixels)")
    ax.set_ylabel("Height (pixels)")

    legend_elements = []
    all_rects = []  # 存储所有矩形顶点用于范围计算

    # ----- 训练阶段 (冷色系) -----
    if draw_train:
        train_colors = ['cyan', 'blue', 'lime', 'green']
        s_min, s_max = train_resize_lim
        p_min, p_max = train_bot_lim

        combos = [
            (s_min, p_min, 'dashed', train_colors[0], 'Train: s_min, p_min'),
            (s_min, p_max, 'dashed', train_colors[1], 'Train: s_min, p_max'),
            (s_max, p_min, 'dashed', train_colors[2], 'Train: s_max, p_min'),
            (s_max, p_max, 'dashed', train_colors[3], 'Train: s_max, p_max')
        ]

        for s, p, ls, color, label in combos:
            x, y, w, h = compute_roi_original(s, p, H_orig, W_orig, H_tar, W_tar)
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none', linestyle=ls)
            ax.add_patch(rect)
            # 标注坐标（仅当矩形部分在图像内或 show_outside_roi 时显示）
            ax.text(x, y-5, f'({x:.1f}, {y:.1f})', fontsize=8, color=color, ha='left', va='bottom')
            ax.text(x+w, y+h+5, f'({x+w:.1f}, {y+h:.1f})', fontsize=8, color=color, ha='right', va='top')
            legend_elements.append(Patch(edgecolor=color, facecolor='none', linestyle=ls, label=label))
            all_rects.append((x, y, w, h))

    # ----- 测试阶段 (暖色系) -----
    if draw_test:
        test_color = 'red'
        s_test = test_resize_lim[0]
        p_test = test_bot_lim[0]
        x, y, w, h = compute_roi_original(s_test, p_test, H_orig, W_orig, H_tar, W_tar)
        rect = Rectangle((x, y), w, h, linewidth=3, edgecolor=test_color, facecolor='none', linestyle='solid')
        ax.add_patch(rect)
        ax.text(x, y-5, f'({x:.1f}, {y:.1f})', fontsize=8, color=test_color, ha='left', va='bottom')
        ax.text(x+w, y+h+5, f'({x+w:.1f}, {y+h:.1f})', fontsize=8, color=test_color, ha='right', va='top')
        legend_elements.append(Patch(edgecolor=test_color, facecolor='none', linestyle='solid', label='Test (fixed)'))
        all_rects.append((x, y, w, h))

    # ----- 调整坐标轴范围以显示外部矩形 -----
    if show_outside_roi and all_rects:
        # 计算所有矩形的最小外接矩形
        x_min = min([rect[0] for rect in all_rects])
        y_min = min([rect[1] for rect in all_rects])
        x_max = max([rect[0] + rect[2] for rect in all_rects])
        y_max = max([rect[1] + rect[3] for rect in all_rects])
        # 包含图像本身
        x_min = min(x_min, 0)
        y_min = min(y_min, 0)
        x_max = max(x_max, W_orig)
        y_max = max(y_max, H_orig)
        # 设置坐标轴范围，并添加边距
        margin = 0.05 * max(x_max - x_min, y_max - y_min)
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_max + margin, y_min - margin)  # 注意图像坐标y轴向下，所以上下颠倒
    else:
        # 默认仅显示图像区域
        ax.set_xlim(0, W_orig)
        ax.set_ylim(H_orig, 0)  # 保持y轴向下

    if not draw_train and not draw_test:
        ax.text(0.5, 0.5, 'No ROI selected to draw', transform=ax.transAxes,
                ha='center', va='center', fontsize=16, color='gray')
    else:
        ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    output_path = "roi_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()


def main():
    # ===== 用户配置区域 =====
    # 原始图像参数

    # # TYJT-第一版（nusc的image_size + nusc的缩放比）【注意， nusc是900， tyjt是1080】
    # img_path = None   # 设置为None则使用空白画布
    img_path = "/mnt/dataset/tyjt_RawData_demo/2d3d4d_20250728_weiyuan/datasets/G51102400001M00_20250728191726_2.0HZ/SC_1B_CamR/image_dc/1753701446899737835.jpg" 
    H_orig_input = 1080                      # 显式传入原始图像高度（当img_path=None时使用）
    W_orig_input = 1920                       # 显式传入原始图像宽度

    # 目标图像尺寸 (final_dim)
    H_tar = 256
    W_tar = 704

    # augment2d，训练阶段增强参数
    train_resize_lim = [0.38, 0.55]           # 缩放范围 [min, max]
    train_bot_lim = [0.0, 0.01]                # 底部裁剪比例范围 [min, max]

    # augment2d，测试阶段增强参数
    test_resize_lim = [0.48, 0.48]           # 固定缩放
    test_bot_lim = [0.0, 0.0]                  # 固定底部裁剪

    # 控制绘制哪些阶段
    draw_train = True    # 是否绘制训练阶段ROI
    draw_test = True     # 是否绘制测试阶段ROI
    show_outside_roi = True   # 设为True时显示完整矩形，包括图像外部



    # # # TYJT-第一版（nusc的image_size + nusc的缩放比）【注意， nusc是900， tyjt是1080】
    # # img_path = None   # 设置为None则使用空白画布
    # img_path = "/mnt/dataset/tyjt_RawData_demo/2d3d4d_20250728_weiyuan/datasets/G51102400001M00_20250728191726_2.0HZ/SC_1B_CamR/image_dc/1753701446899737835.jpg" 
    # H_orig_input = 1080                      # 显式传入原始图像高度（当img_path=None时使用）
    # W_orig_input = 1920                       # 显式传入原始图像宽度

    # # 目标图像尺寸 (final_dim)
    # H_tar = 256
    # W_tar = 704

    # # augment2d，训练阶段增强参数
    # train_resize_lim = [0.38, 0.55]           # 缩放范围 [min, max]
    # train_bot_lim = [0.0, 0.01]                # 底部裁剪比例范围 [min, max]

    # # augment2d，测试阶段增强参数
    # test_resize_lim = [0.48, 0.48]           # 固定缩放
    # test_bot_lim = [0.0, 0.0]                  # 固定底部裁剪

    # # 控制绘制哪些阶段
    # draw_train = True    # 是否绘制训练阶段ROI
    # draw_test = True     # 是否绘制测试阶段ROI
    # show_outside_roi = True   # 设为True时显示完整矩形，包括图像外部




    # # # TYJT-第二版 图像尺寸配置（TYJT训练第2版nusc尺寸，但修改resize比例：roi_visualization）
    # # img_path = None   # 设置为None则使用空白画布
    # img_path = "/mnt/dataset/tyjt_RawData_demo/2d3d4d_20250728_weiyuan/datasets/G51102400001M00_20250728191726_2.0HZ/SC_1B_CamR/image_dc/1753701446899737835.jpg" 
    # H_orig_input = 1080                      # 显式传入原始图像高度（当img_path=None时使用）
    # W_orig_input = 1920                       # 显式传入原始图像宽度

    # # 目标图像尺寸 (final_dim)
    # H_tar = 256
    # W_tar = 704

    # # augment2d，训练阶段增强参数
    # train_resize_lim = [0.37, 0.43]           # 缩放范围 [min, max]
    # train_bot_lim = [0.0, 0.01]                # 底部裁剪比例范围 [min, max]

    # # augment2d，测试阶段增强参数
    # test_resize_lim = [0.4, 0.4]           # 固定缩放
    # test_bot_lim = [0.0, 0.0]                  # 固定底部裁剪

    # # 控制绘制哪些阶段
    # draw_train = True    # 是否绘制训练阶段ROI
    # draw_test = True     # 是否绘制测试阶段ROI
    # show_outside_roi = True   # 设为True时显示完整矩形，包括图像外部



    # # TYJT-第三版 图像尺寸配置 （TYJT训练第3版，不再使用nusc尺寸，改成tyjt自己的尺寸：roi_visualization.png）
    # img_path = "/mnt/dataset/tyjt_RawData_demo/2d3d4d_20250728_weiyuan/datasets/G51102400001M00_20250728191726_2.0HZ/SC_1B_CamR/image_dc/1753701446899737835.jpg"
    # H_orig_input = 1080                      # 显式传入原始图像高度（当img_path=None时使用）
    # W_orig_input = 1920                       # 显式传入原始图像宽度

    # # 目标图像尺寸 (final_dim)
    # H_tar = 360
    # W_tar = 640

    # # 训练阶段增强参数
    # train_resize_lim = [0.34, 0.42]           # 缩放范围 [min, max]
    # train_bot_lim = [0.0, 0.01]                # 底部裁剪比例范围 [min, max]

    # # 测试阶段增强参数
    # test_resize_lim = [0.333, 0.333]           # 固定缩放
    # test_bot_lim = [0.0, 0.0]                  # 固定底部裁剪

    # # 控制绘制
    # draw_train = True    # 是否绘制训练阶段ROI
    # draw_test = True     # 是否绘制测试阶段ROI
    # show_outside_roi = False   # 设为True时显示完整矩形，包括图像外部


    # ===== 校验输入 =====
    valid, img, H_orig, W_orig, msg = validate_inputs((H_orig_input, W_orig_input), img_path, H_orig_input, W_orig_input)
    print(msg)
    if not valid:
        return

    # ===== 绘制ROI =====
    draw_rois(img, H_orig, W_orig, H_tar, W_tar,
              train_resize_lim, train_bot_lim,
              test_resize_lim, test_bot_lim,
              draw_train, draw_test, show_outside_roi)

if __name__ == "__main__":
    main()