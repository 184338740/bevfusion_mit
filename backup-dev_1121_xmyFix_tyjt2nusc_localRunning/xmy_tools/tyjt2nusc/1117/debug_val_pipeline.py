import yaml

def debug_val_pipeline():
    """检查验证pipeline中的过滤步骤"""
    config_path = "/mnt/bevfusion_mit_xmy/configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/tyjt_convfuser_tmp.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=== 验证Pipeline分析 ===")
    
    # 检查验证pipeline
    if 'evaluation' in config and 'pipeline' in config['evaluation']:
        pipeline = config['evaluation']['pipeline']
        print(f"验证pipeline步骤数: {len(pipeline)}")
        
        # 查找过滤步骤
        filter_steps = []
        for i, step in enumerate(pipeline):
            step_type = step.get('type', '')
            if any(keyword in step_type.lower() for keyword in ['filter', 'range', 'threshold']):
                filter_steps.append((i, step_type, step))
        
        print(f"过滤步骤: {len(filter_steps)} 个")
        for idx, step_type, step_config in filter_steps:
            print(f"  步骤 {idx}: {step_type}")
            print(f"    配置: {step_config}")
    
    # 检查测试pipeline（验证时可能使用测试pipeline）
    if 'test_pipeline' in config:
        test_pipeline = config['test_pipeline']
        print(f"测试pipeline步骤数: {len(test_pipeline)}")
        
        # 查找过滤步骤
        test_filter_steps = []
        for i, step in enumerate(test_pipeline):
            step_type = step.get('type', '')
            if any(keyword in step_type.lower() for keyword in ['filter', 'range', 'threshold']):
                test_filter_steps.append((i, step_type, step))
        
        print(f"测试过滤步骤: {len(test_filter_steps)} 个")
        for idx, step_type, step_config in test_filter_steps:
            print(f"  步骤 {idx}: {step_type}")
            print(f"    配置: {step_config}")

if __name__ == "__main__":
    debug_val_pipeline()
