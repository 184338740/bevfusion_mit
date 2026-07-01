# 错误信息：cannot reshape array of size 759553 into shape (6)
# 数据大小：3037603 字节

total_elements = 759553
data_size = 3037603

print(f"总元素数: {total_elements}")
print(f"数据字节数: {data_size}")
print(f"每个元素字节数: {data_size / total_elements}")

# 检查可能的维度
for dim in [3, 4, 5, 6]:
    if total_elements % dim == 0:
        points_count = total_elements // dim
        print(f"维度{dim}: {points_count}个点, 每个点{dim}个元素")
        
# 检查数据类型
import numpy as np
print(f"如果是float32: {data_size / 4} 个float32")
print(f"如果是float64: {data_size / 8} 个float64")
