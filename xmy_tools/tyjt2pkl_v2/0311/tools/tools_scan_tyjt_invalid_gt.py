import pickle
import numpy as np

pkl_file = "/data2/xmy/01_project/bevfusion_mit_xmy_1205/datasets/tyjt2pkl/A100_V032h_sub_0428_ShuffleSamples/tyjt_infos_train_split.pkl"

def is_valid_box(box):
    """检查单个3D框是否有效"""
    if len(box) < 7:
        return False
    w, l, h = box[3:6]   # width, length, height (注意顺序可能因数据集而异)
    if w <= 0 or l <= 0 or h <= 0:
        return False
    if not np.isfinite(box[:3]).all():
        return False
    if not np.isfinite(box[6]):   # yaw
        return False
    return True

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--pkl', type=str, required=True, help='Path to the pkl file')
    # args = parser.parse_args()

    # with open(args.pkl, 'rb') as f:

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # 兼容新旧格式
    infos = data.get('infos', data)

    invalid_samples = []
    total_boxes = 0
    invalid_boxes = 0

    for idx, info in enumerate(infos):
        token = info.get('token', f'sample_{idx}')
        gt_boxes = info.get('gt_boxes', [])
        for bidx, box in enumerate(gt_boxes):
            total_boxes += 1
            if not is_valid_box(box):
                invalid_boxes += 1
                invalid_samples.append((token, bidx, box))

    print(f"Total samples: {len(infos)}")
    print(f"Total boxes: {total_boxes}")
    print(f"Invalid boxes: {invalid_boxes}")

    if invalid_samples:
        print("\nInvalid boxes found:")
        for token, bidx, box in invalid_samples:
            print(f"  Sample {token}, box index {bidx}: {box}")
    else:
        print("All boxes are valid.")

if __name__ == '__main__':
    main()