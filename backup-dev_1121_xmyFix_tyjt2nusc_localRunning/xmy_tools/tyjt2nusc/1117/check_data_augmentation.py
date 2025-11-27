import yaml

def check_data_augmentation():
    """检查数据增强配置"""
    config_path = "/mnt/bevfusion_mit_xmy/configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/tyjt_convfuser_tmp.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=== 数据增强配置分析 ===")
    
    # 检查训练pipeline
    if 'train_pipeline' in config:
        aug_steps = []
        for step in config['train_pipeline']:
            step_type = step.get('type', '')
            if any(keyword in step_type.lower() for keyword in ['flip', 'rotate', 'scale', 'aug', 'random']):
                aug_steps.append(step_type)
        
        print(f"数据增强步骤: {aug_steps}")
        print(f"增强步骤数量: {len(aug_steps)}")
    
    # 检查验证pipeline  
    if 'test_pipeline' in config:
        val_aug_steps = []
        for step in config['test_pipeline']:
            step_type = step.get('type', '')
            if any(keyword in step_type.lower() for keyword in ['flip', 'rotate', 'scale', 'aug', 'random']):
                val_aug_steps.append(step_type)
        
        print(f"验证数据增强步骤: {val_aug_steps}")

if __name__ == "__main__":
    check_data_augmentation()
