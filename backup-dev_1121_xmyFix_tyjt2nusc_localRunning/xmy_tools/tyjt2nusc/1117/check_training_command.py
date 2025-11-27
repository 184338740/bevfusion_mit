import subprocess
import os

def check_training_command():
    """检查训练命令"""
    print("=== 训练命令分析 ===")
    
    # 常见的训练命令格式
    print("nuscenes原版通常使用:")
    print("  python tools/train.py configs/xxx.py --cfg-options data.val.version=v1.0-val")
    print("")
    print("或者:")
    print("  python tools/train.py configs/xxx.py --cfg-options data.train.version=v1.0-trainval data.val.version=v1.0-val")
    print("")
    print("您当前的命令可能是:")
    print("  python tools/train.py configs/tyjt/det/transfusion/secfpn/camera+lidar/swint_v0p075/tyjt_convfuser_tmp.yaml")
    print("")
    print("缺少版本参数!")

if __name__ == "__main__":
    check_training_command()
