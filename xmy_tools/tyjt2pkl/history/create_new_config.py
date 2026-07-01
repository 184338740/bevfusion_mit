# create_new_config.py
import yaml

config = {
    '_base_': ['./_base_/default_runtime.py'],
    
    'dataset_type': 'NuScenesDataset',
    'data_root': '/mnt/dataset/tyjt_RawData/',
    
    'point_cloud_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    
    'input_modality': {
        'use_lidar': True,
        'use_camera': True,
        'use_radar': False
    },
    
    'train_pipeline': [
        {
            'type': 'LoadPointsFromFile',
            'coord_type': 'LIDAR', 
            'load_dim': 4,
            'use_dim': [0, 1, 2, 3],
            'file_client_args': {'backend': 'disk'}
        },
        {
            'type': 'LoadMultiViewImageFromFiles',
            'to_float32': True
        }
    ],
    
    'data': {
        'samples_per_gpu': 2,
        'workers_per_gpu': 4,
        'train': {
            'type': 'NuScenesDataset',
            'data_root': '/mnt/dataset/tyjt_RawData/',
            'ann_file': '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/step3_1_output/tyjt_infos_train.pkl',
            'pipeline': '${train_pipeline}',
            'test_mode': False
        },
        'val': {
            'type': 'NuScenesDataset', 
            'data_root': '/mnt/dataset/tyjt_RawData/',
            'ann_file': '/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/step3_1_output/tyjt_infos_val.pkl',
            'pipeline': '${train_pipeline}',
            'test_mode': True
        }
    }
}

with open('/mnt/bevfusion_mit_xmy/tyjt_simple.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, indent=2)

print("✅ 创建全新配置文件: /mnt/bevfusion_mit_xmy/tyjt_simple.yaml")
print("使用命令: torchpack dist-run -np 1 python tools/train.py tyjt_simple.yaml")