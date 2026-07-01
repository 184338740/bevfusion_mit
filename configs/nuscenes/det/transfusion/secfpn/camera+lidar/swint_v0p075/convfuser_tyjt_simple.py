_base_ = './convfuser.py'

# 直接覆盖数据路径
data_root = '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2nusc/1106_deep/output/step1/nuscenes_tyjt/'

data = dict(
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'tyjt_infos_train.pkl',
    ),
    val=dict(
        data_root=data_root,
        ann_file=data_root + 'tyjt_infos_val.pkl',
    ),
    test=dict(
        data_root=data_root,
        ann_file=data_root + 'tyjt_infos_test.pkl',
    )
)

# 类别配置
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
]

model = dict(
    pts_bbox_head=dict(
        num_classes=len(class_names),
    )
)
