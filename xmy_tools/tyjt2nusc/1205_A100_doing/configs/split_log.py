import os

def split_log_file(input_file="log.txt", lines_per_file=1000):
    """
    将大日志文件按指定行数分割为多个小文件
    :param input_file: 待分割的源文件路径
    :param lines_per_file: 每个分割文件的最大行数
    """
    # 检查源文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：找不到文件 {input_file}")
        return

    # 获取源文件基本信息（用于生成分割文件名）
    file_dir = os.path.dirname(input_file)
    file_name = os.path.basename(input_file)
    file_base, file_ext = os.path.splitext(file_name)

    # 初始化变量
    file_counter = 1  # 分割文件计数器
    current_lines = []  # 临时存储当前文件的行
    total_lines_processed = 0  # 已处理总行数

    print(f"开始分割文件：{input_file}")
    print(f"分割规则：每个文件最多 {lines_per_file} 行")

    try:
        # 以只读模式打开源文件（避免编码问题，指定utf-8）
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                current_lines.append(line)
                total_lines_processed += 1

                # 当当前行列表达到指定行数时，写入新文件
                if len(current_lines) == lines_per_file:
                    # 生成分割文件名（如：log_part_001.txt）
                    output_file = os.path.join(
                        file_dir,
                        f"{file_base}_part_{file_counter:03d}{file_ext}"
                    )
                    # 写入文件
                    with open(output_file, 'w', encoding='utf-8') as out_f:
                        out_f.writelines(current_lines)
                    
                    print(f"已生成：{output_file}（{lines_per_file} 行）")
                    
                    # 重置临时列表，计数器+1
                    current_lines = []
                    file_counter += 1

            # 处理剩余的不足1000行的内容
            if current_lines:
                output_file = os.path.join(
                    file_dir,
                    f"{file_base}_part_{file_counter:03d}{file_ext}"
                )
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    out_f.writelines(current_lines)
                print(f"已生成：{output_file}（{len(current_lines)} 行）")

        # 输出分割完成信息
        print("\n分割完成！")
        print(f"总计处理行数：{total_lines_processed}")
        print(f"生成文件总数：{file_counter}")

    except Exception as e:
        print(f"分割过程中发生错误：{str(e)}")

if __name__ == "__main__":
    # 调用分割函数（可修改input_file指定自定义路径）
    split_log_file(
        input_file="log.txt",  # 待分割的文件路径
        lines_per_file=1000    # 每个文件的行数
    )