import os

def check_official_nuscenes():
    """检查官方nuscenes版本处理"""
    print("=== 官方nuscenes版本分析 ===")
    
    # 官方nuscenes只有这些标准版本
    official_versions = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    print(f"官方版本: {official_versions}")
    
    print("\n官方训练命令分析:")
    print("torchpack dist-run -np 1 python tools/train.py \\")
    print("    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \\")
    print("    --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \\")
    print("    --run-dir runs/train-bevfusion-det-transfusion-from-scratch")
    
    print("\n关键点: 官方命令没有指定版本参数!")
    print("说明版本信息应该在pkl文件或配置中")

if __name__ == "__main__":
    check_official_nuscenes()
