import pickle
import numpy as np

# 检查PKL文件中的点云路径
with open('/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/step3_1_output/tyjt_infos_train.pkl', 'rb') as f:
    data = pickle.load(f)

sample = data['infos'][0]
print("点云路径:", sample['lidar_path'])

# 直接读取PCD文件分析实际数据
pcd_path = sample['lidar_path']
with open(pcd_path, 'rb') as f:
    content = f.read()
    
# 查找数据开始位置
content_str = content.decode('ascii', errors='ignore')
lines = content_str.split('\n')

data_start = None
for i, line in enumerate(lines):
    if 'DATA' in line:
        data_start = i + 1
        break

if data_start:
    # 计算二进制数据大小
    header_size = len('\n'.join(lines[:data_start]).encode('ascii'))
    binary_data = content[header_size:]
    
    print(f"文件总大小: {len(content)} 字节")
    print(f"头部大小: {header_size} 字节")  
    print(f"数据大小: {len(binary_data)} 字节")
    
    # 尝试解析为float32
    if len(binary_data) % 4 == 0:
        points = np.frombuffer(binary_data, dtype=np.float32)
        print(f"浮点数总数: {len(points)}")
        print(f"可能的点数: {len(points)} / 6 = {len(points) // 6}")
        print(f"可能的点数: {len(points)} / 4 = {len(points) // 4}")
        print(f"可能的点数: {len(points)} / 3 = {len(points) // 3}")
        
        # 检查759553这个数字
        print(f"759553 * 4 = {759553 * 4}")
        print(f"759553 * 3 = {759553 * 3}")
        print(f"759553 * 6 = {759553 * 6}")
