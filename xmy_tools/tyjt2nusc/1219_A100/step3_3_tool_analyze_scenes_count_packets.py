import re
from collections import defaultdict

def parse_scene_file(filename):
    """解析场景文件，返回 {包前缀: {场景数, 样本总数}}"""
    packet_stats = defaultdict(lambda: {'count': 0, 'samples': 0})
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            # 分割场景名和样本数
            parts = line.rsplit(':', 1)
            scene_name = parts[0].strip()
            samples = int(parts[1].strip())
            # 提取包前缀：取第一个下划线前的部分？实际需要根据包列表定义前缀
            # 包前缀定义见表格中的“数据包”列，例如：
            # 2d3d_20250114, 2d3d_20250218, 2d3d_20250221,
            # 2d3d4d_20241122_wuxi, 2d3d4d_20241218, 2d3d4d_20250117,
            # 2d3d4d_20250213, 2d3d4d_20250218, 2d3d4d_20250403,
            # 2d3d4d_20250408, 2d3d4d_20250618_weiyuan, 2d3d4d_20250708_weiyuan,
            # 2d3d4d_20250728_weiyuan
            # 我们需要从场景名中匹配这些前缀。简单方法：用正则提取日期部分，再拼接。
            # 但场景名格式复杂，最好直接用已知前缀列表进行匹配。
            prefixes = [
                "2d3d_20250114",
                "2d3d_20250218",
                "2d3d_20250221",
                "2d3d4d_20241122_wuxi",
                "2d3d4d_20241218",
                "2d3d4d_20250117",
                "2d3d4d_20250213",
                "2d3d4d_20250218",
                "2d3d4d_20250403",
                "2d3d4d_20250408",
                "2d3d4d_20250618_weiyuan",
                "2d3d4d_20250708_weiyuan",
                "2d3d4d_20250728_weiyuan"
            ]
            matched = None
            for p in prefixes:
                if scene_name.startswith(p):
                    matched = p
                    break
            if matched:
                packet_stats[matched]['count'] += 1
                packet_stats[matched]['samples'] += samples
            else:
                print(f"警告：无法匹配前缀的场景：{scene_name}")
    return packet_stats

def main():
    train_stats = parse_scene_file('train_scenes.txt')
    val_stats = parse_scene_file('val_scenes.txt')

    # 所有包列表（按序号）
    packets = [
        (1, "2d3d_20250114"),
        (2, "2d3d_20250218"),
        (3, "2d3d_20250221"),
        (4, "2d3d4d_20241122_wuxi"),
        (5, "2d3d4d_20241218"),
        (6, "2d3d4d_20250117"),
        (7, "2d3d4d_20250213"),
        (8, "2d3d4d_20250218"),
        (9, "2d3d4d_20250403"),
        (10, "2d3d4d_20250408"),
        (11, "2d3d4d_20250618_weiyuan"),
        (12, "2d3d4d_20250708_weiyuan"),
        (13, "2d3d4d_20250728_weiyuan")
    ]

    print("序号\t数据包\t\t\t训练集场景数\t训练集样本数\t验证集场景数\t验证集样本数")
    for idx, p in packets:
        train = train_stats.get(p, {'count':0, 'samples':0})
        val = val_stats.get(p, {'count':0, 'samples':0})
        print(f"{idx}\t{p}\t{train['count']}\t\t{train['samples']}\t\t{val['count']}\t\t{val['samples']}")

if __name__ == "__main__":
    main()