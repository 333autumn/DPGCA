"""
切换表头
"""

import random


def process_data(input_file, output_file):
    # 打开输入文件读取数据
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # 处理数据
    processed_data = []
    # 添加表头
    processed_data.append("user_id:token\tjob_id:token\tdirect:token\tlabel:float")
    for line in lines:
        # 分割每一行
        user_id, item_id, _ = line.strip().split("\t")
        # 随机生成 direct 值（0 或 1）
        direct = random.choice([0, 1])
        # 固定 label 值为 1
        label = 1
        # 重新组合数据
        new_line = f"{user_id}\t{item_id}\t{direct}\t{label}"
        processed_data.append(new_line)

    # 写入输出文件
    with open(output_file, 'w') as outfile:
        for line in processed_data:
            outfile.write(line + "\n")


# 指定输入和输出文件
input_file = "jobrec.inter副本"
output_file = "jobrec.inter"

# 调用函数处理数据
process_data(input_file, output_file)
