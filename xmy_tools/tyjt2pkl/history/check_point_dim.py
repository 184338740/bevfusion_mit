import pickle
import numpy as np

# 检查PKL文件中的点云路径
with open('/mnt/bevfusion_mit_xmy/xmy_tools/tyjt2pkl/step3_1_output/tyjt_infos_train.pkl', 'rb') as f:
    data = pickle.load(f)

sample = data['infos'][0]
print("点云路径:", sample['lidar_path'])

# 直接读取PCD文件分析
pcd_path = sample['lidar_path']
with open(pcd_path, 'rb') as f:
    content = f.read().decode('ascii', errors='ignore')
    
# 解析PCD头信息
lines = content.split('\n')
for line in lines:
    if line.startswith('FIELDS'):
        fields = line.split()[1:]
        print("点云字段:", fields)
        print("点云维度:", len(fields))
        break
    elif line.startswith('SIZE'):
        sizes = line.split()[1:]
        print("字段大小:", sizes)
    elif line.startswith('TYPE'):
        types = line.split()[1:]
        print("字段类型:", types)
    elif line.startswith('POINTS'):
        points_count = int(line.split()[-1])
        print("点数:", points_count)
        
# 尝试读取数据部分
for i, line in enumerate(lines):
    if 'DATA' in line:
        # 读取几行数据看看格式
        data_lines = lines[i+1:i+6]
        for j, data_line in enumerate(data_lines):
            if data_line.strip():
                values = data_line.split()
                print(f"数据样例 {j+1}: 值数量={len(values)}, 前几个值={values[:5]}")
        break
