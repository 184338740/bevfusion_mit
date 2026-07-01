import argparse
import os
from os import path as osp

import mmcv
# 仅导入Dataset类，所有公共配置从这里获取
from mmdet3d.datasets import NuScenesDataset, TYJTDataset

# 导入转换器（已依赖Dataset类获取配置）
from tools.data_converter.tyjt_converter import create_nuscenes_infos, export_2d_annotation
from tools.data_converter.create_gt_database import create_groundtruth_database


def parse_args():
    parser = argparse.ArgumentParser(description='Data converter for NuScenes/TYJT (v8.0_1127)')
    parser.add_argument('--dataset', type=str, required=True, choices=['nuscenes', 'tyjt'],
                        help='Dataset type: nuscenes (NuScenes) or tyjt (TYJT)')
    parser.add_argument('--root-path', type=str, default='./data/nuscenes',
                        help='Root path of the raw data (nuscenes/tyjt)')
    parser.add_argument('--out-dir', type=str, default='./data/nuscenes',
                        help='Output directory of the converted pkl files')
    parser.add_argument('--extra-tag', type=str, default='nuscenes',
                        help='Extra tag for the output pkl files (e.g., tyjt -> tyjt_infos_train.pkl)')
    # parser.add_argument('--workers', type=int, default=4, help='Number of workers for data conversion')
    parser.add_argument('--with-radar', action='store_true', help='Whether to use radar data (only for NuScenes)')
    parser.add_argument('--load-augmented', type=str, default=None,
                        help='Load augmented data for training (only for NuScenes)')
    parser.add_argument('--max-sweeps', type=int, default=10,
                        help='Max number of LiDAR sweeps per sample')
    return parser.parse_args()


def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.out_dir)

    # ====================== 仅保留pkl生成必需的配置（全部来自Dataset类）======================
    # 1. 数据集类（用于获取所有公共配置）
    DatasetClass = NuScenesDataset if args.dataset == 'nuscenes' else TYJTDataset
    # 2. 有效类别（生成pkl和GT数据库必需）
    used_classes = DatasetClass.CLASSES
    # 3. 雷达最大sweeps（仅NuScenes用，通过--with-radar控制）
    max_radar_sweeps = 10 if args.dataset == 'nuscenes' and args.with_radar else 0
    # =====================================================================================

    print(f'========================================')
    print(f'Data Conversion Start: {args.dataset}')
    print(f'Root Path: {args.root_path}')
    print(f'Output Dir: {args.out_dir}')
    print(f'Extra Tag: {args.extra_tag}')
    print(f'Used Classes (count: {len(used_classes)}): {used_classes}')
    print(f'Use Radar: {args.with_radar if args.dataset == "nuscenes" else "No"}')
    print(f'Max LiDAR Sweeps: {args.max_sweeps}')
    print(f'========================================')

    # Step1: 生成info文件（pkl）
    if args.dataset == 'nuscenes':
        # NuScenes：官方版本划分
        # 处理trainval集
        create_nuscenes_infos(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version='v1.0-trainval',
            dataset='nuscenes',
            max_sweeps=args.max_sweeps,
            max_radar_sweeps=max_radar_sweeps
        )
        # 处理test集
        if osp.exists(osp.join(args.root_path, 'samples', 'LIDAR_TOP')):
            create_nuscenes_infos(
                root_path=args.root_path,
                info_prefix=args.extra_tag,
                version='v1.0-test',
                dataset='nuscenes',
                max_sweeps=args.max_sweeps,
                max_radar_sweeps=max_radar_sweeps
            )
        # 导出2D标注（可选，NuScenes专属）
        export_2d_annotation(
            args.root_path,
            osp.join(args.root_path, f'{args.extra_tag}_infos_train.pkl'),
            version='v1.0-trainval'
        )
        export_2d_annotation(
            args.root_path,
            osp.join(args.root_path, f'{args.extra_tag}_infos_val.pkl'),
            version='v1.0-trainval'
        )

    else:  # dataset == 'tyjt'
        # Tyjt：6:2:2划分 + 类别merge（依赖TyjtDataset.TYJT_CATEGORY_MAPPING）
        # create_nuscenes_infos(
        #     root_path=args.root_path,
        #     info_prefix=args.extra_tag,
        #     version='v1.0-tyjt',
        #     dataset='tyjt',
        #     max_sweeps=args.max_sweeps,
        #     max_radar_sweeps=0  # Tyjt无雷达，固定关闭
        # )
        # 对于trainval版本
        # create_nuscenes_infos(
        #     root_path=args.root_path,
        #     info_prefix=args.extra_tag,
        #     version='v1.0-tyjt-trainval',
        #     dataset='tyjt',
        #     max_sweeps=args.max_sweeps,
        #     max_radar_sweeps=0,
        #     random_seed=42,
        #     train_val_ratio=0.8  # 可以调整这个比例
        # )

        # # 对于test版本（如果需要）
        # create_nuscenes_infos(
        #     root_path=args.root_path,
        #     info_prefix=args.extra_tag,
        #     version='v1.0-tyjt-test',
        #     dataset='tyjt',
        #     max_sweeps=args.max_sweeps,
        #     max_radar_sweeps=0,
        #     random_seed=42
        # )
        None

    # Step2: 生成GT数据库（仅训练集，依赖Dataset类的used_classes）
    print(f'\n========================================')
    print(f'Creating GT Database for {args.dataset}')
    print(f'GT Database Classes: {used_classes}')
    print(f'========================================')
    # GT数据库核心参数（全部依赖Dataset类或命令行参数）
    dataset_class_name = 'NuScenesDataset' if args.dataset == 'nuscenes' else 'TYJTDataset'
    db_save_path = osp.join(args.out_dir, f'{args.extra_tag}_gt_database')
    db_info_save_path = osp.join(args.out_dir, f'{args.extra_tag}_dbinfos_train.pkl')
    train_info_path = osp.join(args.root_path, f'{args.extra_tag}_infos_train.pkl')

    create_groundtruth_database(
        dataset_class_name=dataset_class_name,
        data_path=args.root_path,
        info_prefix=args.extra_tag,
        info_path=train_info_path,
        used_classes=used_classes,  # 直接来自Dataset类，确保一致
        database_save_path=db_save_path,
        db_info_save_path=db_info_save_path,
        with_mask=False,
        load_augmented=args.load_augmented,
        # workers=args.workers
    )

    # 输出完成信息
    print(f'\n========================================')
    print(f'Data Conversion Completed!')
    print(f'========================================')
    if args.dataset == 'tyjt':
        print(f'Output Files:')
        print(f'  - Train/Val/Test Info: {osp.join(args.root_path, f"{args.extra_tag}_infos_{{train,val,test}}.pkl")}')
        print(f'  - GT Database: {db_save_path}')
        print(f'  - DB Info: {db_info_save_path}')
        print(f'Key Notes:')
        print(f'  1. 6:2:2 scene split (fixed seed: 42)')
        print(f'  2. Category merge uses: TyjtDataset.TYJT_CATEGORY_MAPPING')
        print(f'  3. No radar data (max_radar_sweeps=0)')
    else:
        print(f'Output Files:')
        print(f'  - Train/Val/Test Info: {osp.join(args.root_path, f"{args.extra_tag}_infos_{{train,val,test}}.pkl")}')
        print(f'  - GT Database: {db_save_path}')
        print(f'  - DB Info: {db_info_save_path}')
        print(f'Key Notes:')
        print(f'  1. Official train/val/test split')
        print(f'  2. Use radar: {args.with_radar}')
        print(f'  3. Category mapping uses: NuScenesDataset.NameMapping')


if __name__ == '__main__':
    main()


"""
命令:
# Tyjt 数据转 pkl+GT 数据库（最终命令
python tools/create_data_tyjt_v8.0_1127.py \
  --dataset tyjt \
  --root-path ./data/tyjt \
  --out-dir ./data/tyjt \
  --extra-tag tyjt \
  --max-sweeps 10

# NuScenes 数据转 pkl+GT 数据库（最终命令）
python tools/create_data_tyjt_v8.0_1127.py \
  --dataset nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes \
  --with-radar \
  --max-sweeps 10

"""