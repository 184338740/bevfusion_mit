import numpy as np

def analyze_pcd(pcd_path):
    """分析PCD文件的实际格式"""
    with open(pcd_path, 'rb') as f:
        content = f.read()
    
    # 解析头部
    content_str = content.decode('ascii', errors='ignore')
    lines = content_str.split('\n')
    
    # 解析头部信息
    fields = []
    size = []
    type_list = []
    count = []
    points_count = 0
    data_start = 0
    
    for i, line in enumerate(lines):
        if line.startswith('FIELDS'):
            fields = line.split()[1:]
        elif line.startswith('SIZE'):
            size = line.split()[1:]
        elif line.startswith('TYPE'):
            type_list = line.split()[1:]
        elif line.startswith('COUNT'):
            count = line.split()[1:]
        elif line.startswith('POINTS'):
            points_count = int(line.split()[-1])
        elif line.startswith('DATA'):
            data_start = i + 1
            break
    
    print("PCD头部信息:")
    print(f"字段: {fields}")
    print(f"大小: {size}")
    print(f"类型: {type_list}")
    print(f"计数: {count}")
    print(f"点数: {points_count}")
    
    # 计算头部大小
    header = '\n'.join(lines[:data_start])
    header_size = len(header.encode('ascii'))
    
    # 分析二进制数据
    binary_data = content[header_size:]
    print(f"二进制数据大小: {len(binary_data)} 字节")
    
    # 根据字段计算期望的数据大小
    total_bytes_per_point = 0
    for s, c in zip(size, count):
        total_bytes_per_point += int(s) * int(c)
    
    print(f"每点字节数: {total_bytes_per_point}")
    print(f"期望数据大小: {points_count * total_bytes_per_point} 字节")
    print(f"实际数据大小: {len(binary_data)} 字节")
    
    # 尝试解析数据
    if total_bytes_per_point > 0:
        expected_points = len(binary_data) // total_bytes_per_point
        print(f"实际点数: {expected_points}")
        
        # 解析为适当的数据类型
        if total_bytes_per_point % 4 == 0:
            points = np.frombuffer(binary_data, dtype=np.float32)
            print(f"解析为float32: {len(points)} 个元素")
            if len(points) % len(fields) == 0:
                actual_dim = len(points) // expected_points
                print(f"实际维度: {actual_dim}")
                return actual_dim
    
    return len(fields)

# 测试
pcd_path = '/mnt/dataset/tyjt_RawData/2d3d4d_20250728_weiyuan/datasets/G51102400001M00_20250728191726_2.0HZ/lidar/pcd/1753701491899716616.pcd'
dim = analyze_pcd(pcd_path)
print(f"最终确定的维度: {dim}")
