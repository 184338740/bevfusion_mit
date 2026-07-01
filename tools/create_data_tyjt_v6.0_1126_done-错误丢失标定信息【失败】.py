# create_data_tyjt_v6.0_1126_done.py
import argparse
import os
import os.path as osp
import mmcv

def ensure_tyjt_database_completeness(root_path, info_prefix):
    """确保TYJT数据库包含所有目标类别"""
    db_path = osp.join(root_path, f'{info_prefix}_dbinfos_train.pkl')
    
    if not osp.exists(db_path):
        print(f"🔵[TYJT修复]>>> 数据库文件不存在: {db_path}")
        return
    
    # TYJT_TARGET_CLASSES = [
    #     'tyjt.car', 'tyjt.truck', 'tyjt.bus', 'tyjt.pedestrian',
    #     'tyjt.motorcycle', 'tyjt.bicycle', 'tyjt.traffic_cone',
    #     'tyjt.barrier', 'tyjt.trolley', 'tyjt.other'
    # ]
    
    # 🔵[xmy修改]>>> 改为BEVFusion标准10类
    BEVFUSION_TARGET_CLASSES = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
        'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]
    
    db_infos = mmcv.load(db_path)
    
    # 检查并修复缺失的类别
    missing_classes = []
    for class_name in BEVFUSION_TARGET_CLASSES:
        if class_name not in db_infos:
            db_infos[class_name] = []
            missing_classes.append(class_name)
    
    if missing_classes:
        print(f"🔵[TYJT修复]>>> 添加 {len(missing_classes)} 个缺失类别: {missing_classes}")
        mmcv.dump(db_infos, db_path)
        print(f"✅ 数据库修复完成，现在包含 {len(db_infos)} 个类别")
        
        # 验证修复结果
        fixed_db_infos = mmcv.load(db_path)
        print(f"🔵[TYJT验证]>>> 修复后数据库类别: {list(fixed_db_infos.keys())}")
    else:
        print(f"✅ 数据库已包含所有 {len(BEVFUSION_TARGET_CLASSES)} 个BEVFusion标准类别")




def tyjt_data_prep(
    root_path,
    info_prefix,
    version,
    dataset_name,
    out_dir,
    max_sweeps=10,
    load_augmented=None,
):
    """TYJT专用数据处理（包含数据库修复）"""
    print(f"🔵[TYJT]>>> TYJT数据处理: {version}")
    
    if load_augmented is None:
        from data_converter import nuscenes_converter
        nuscenes_converter.create_nuscenes_infos(
            root_path, info_prefix, version=version, max_sweeps=max_sweeps
        )
    
    # 只在训练集版本创建groundtruth数据库
    if 'test' not in version:
        from data_converter.create_gt_database import create_groundtruth_database
        info_train_path = osp.join(root_path, f"{info_prefix}_infos_train.pkl")
        if osp.exists(info_train_path):
            print(f"🔵[TYJT]>>> +++++++++++++++++++++++++++++++ 创建groundtruth数据库 +++++++++++++++++++++++++++++++")
            create_groundtruth_database(
                dataset_name,
                root_path,
                info_prefix,
                f"{out_dir}/{info_prefix}_infos_train.pkl",
                load_augmented=load_augmented,
            )
            
            # 🔵[xmy添加]>>> 确保数据库完整性
            print(f"🔵[TYJT修复]>>> 开始数据库完整性检查...")
            ensure_tyjt_database_completeness(root_path, info_prefix)
        else:
            print(f"🔵[TYJT]>>> 训练集info文件不存在，跳过groundtruth数据库创建: {info_train_path}")
    else:
        print(f"🔵[TYJT]>>> 测试集版本，跳过groundtruth数据库创建")

def nuscenes_data_prep(
    root_path,
    info_prefix,
    version,
    dataset_name,
    out_dir,
    max_sweeps=10,
    load_augmented=None,
):
    """官方的NuScenes数据处理（保持不变）"""
    print(f"🔵[NuScenes]>>> 标准NuScenes数据处理: {version}")
    
    if load_augmented is None:
        from data_converter import nuscenes_converter
        nuscenes_converter.create_nuscenes_infos(
            root_path, info_prefix, version=version, max_sweeps=max_sweeps
        )
    
    # 只在训练集版本创建groundtruth数据库
    if 'test' not in version:
        from data_converter.create_gt_database import create_groundtruth_database
        info_train_path = osp.join(root_path, f"{info_prefix}_infos_train.pkl")
        if osp.exists(info_train_path):
            create_groundtruth_database(
                dataset_name,
                root_path,
                info_prefix,
                f"{out_dir}/{info_prefix}_infos_train.pkl",
                load_augmented=load_augmented,
            )
        else:
            print(f"🔵[NuScenes]>>> 训练集info文件不存在，跳过groundtruth数据库创建: {info_train_path}")

def is_tyjt_version(version):
    """判断是否是TYJT版本"""
    return 'tyjt' in version.lower()

def is_test_version(version):
    """判断是否是测试集版本"""
    return 'test' in version.lower()

def main():
    parser = argparse.ArgumentParser(description="Data converter arg parser")
    parser.add_argument("dataset", metavar="dataset", help="name of the dataset (nuscenes or tyjt)")
    parser.add_argument(
        "--root-path",
        type=str,
        required=True,
        help="specify the root path of dataset",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="specify the dataset version",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=10,
        required=False,
        help="specify sweeps of lidar per example",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="output directory for info pkl",
    )
    parser.add_argument("--extra-tag", type=str, default="nuscenes")
    parser.add_argument("--painted", default=False, action="store_true")
    parser.add_argument("--virtual", default=False, action="store_true")
    parser.add_argument(
        "--workers", type=int, default=4, help="number of threads to be used"
    )
    args = parser.parse_args()

    print(f"🔵[Data Converter]>>> 开始处理数据集")
    print(f"  - 数据集: {args.dataset}")
    print(f"  - 根路径: {args.root_path}")
    print(f"  - 版本: {args.version}")
    print(f"  - 输出目录: {args.out_dir}")
    print(f"  - 标签前缀: {args.extra_tag}")

    load_augmented = None
    if args.virtual:
        if args.painted:
            load_augmented = "mvp"
        else:
            load_augmented = "pointpainting"

    # 支持 nuscenes 数据集
    if args.dataset == "nuscenes":
        if args.version == "v1.0-mini":
            # mini版本处理
            print(f"🔵[NuScenes]>>> mini版本: {args.version}")
            nuscenes_data_prep(
                root_path=args.root_path,
                info_prefix=args.extra_tag,
                version=args.version,
                dataset_name="NuScenesDataset",
                out_dir=args.out_dir,
                max_sweeps=args.max_sweeps,
                load_augmented=load_augmented,
            )
        else:
            # 标准NuScenes处理逻辑
            print(f"🔵[NuScenes]>>> 标准NuScenes版本: {args.version}")
            
            if is_test_version(args.version):
                # 标准NuScenes测试版本
                nuscenes_data_prep(
                    root_path=args.root_path,
                    info_prefix=args.extra_tag,
                    version=args.version,
                    dataset_name="NuScenesDataset",
                    out_dir=args.out_dir,
                    max_sweeps=args.max_sweeps,
                    load_augmented=load_augmented,
                )
            else:
                # 标准NuScenes训练验证版本
                train_version = f"{args.version}-trainval"
                test_version = f"{args.version}-test"
                
                nuscenes_data_prep(
                    root_path=args.root_path,
                    info_prefix=args.extra_tag,
                    version=train_version,
                    dataset_name="NuScenesDataset",
                    out_dir=args.out_dir,
                    max_sweeps=args.max_sweeps,
                    load_augmented=load_augmented,
                )
                nuscenes_data_prep(
                    root_path=args.root_path,
                    info_prefix=args.extra_tag,
                    version=test_version,
                    dataset_name="NuScenesDataset",
                    out_dir=args.out_dir,
                    max_sweeps=args.max_sweeps,
                    load_augmented=load_augmented,
                )
    
    # 支持 tyjt 数据集
    elif args.dataset == "tyjt":
        # TYJT专用处理逻辑
        print(f"🔵[TYJT]>>> TYJT数据集: {args.version}")
        
        if is_test_version(args.version):
            # TYJT测试版本 - 直接使用传入的版本号
            print(f"🔵[TYJT]>>> 测试版本，生成测试集pkl")
            tyjt_data_prep(
                root_path=args.root_path,
                info_prefix=args.extra_tag,
                version=args.version,  # 直接使用 v1.0-tyjt-test
                dataset_name="NuScenesDataset",
                out_dir=args.out_dir,
                max_sweeps=args.max_sweeps,
                load_augmented=load_augmented,
            )
        else:
            # TYJT训练验证版本 - 对于tyjt版本，不需要拼接后缀
            if args.version == "v1.0-tyjt":
                # 单版本处理
                print(f"🔵[TYJT]>>> 单版本处理: {args.version}")
                tyjt_data_prep(
                    root_path=args.root_path,
                    info_prefix=args.extra_tag,
                    version=args.version,  # 直接使用 v1.0-tyjt
                    dataset_name="NuScenesDataset",
                    out_dir=args.out_dir,
                    max_sweeps=args.max_sweeps,
                    load_augmented=load_augmented,
                )
            else:
                # 对于已经包含后缀的版本，直接使用
                print(f"🔵[TYJT]>>> 直接使用版本: {args.version}")
                tyjt_data_prep(
                    root_path=args.root_path,
                    info_prefix=args.extra_tag,
                    version=args.version,  # 直接使用 v1.0-tyjt-trainval
                    dataset_name="NuScenesDataset",
                    out_dir=args.out_dir,
                    max_sweeps=args.max_sweeps,
                    load_augmented=load_augmented,
                )
    else:
        print(f"❌ 不支持的dataset: {args.dataset}")

    print(f"🎉 数据集处理完成!")


if __name__ == "__main__":
    main()