#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并SMILES片段文件的脚本

本脚本用于将多个SMILES片段文件(.smi)按预定顺序
合并到一个单一的输出文件中。
"""

import os

def combine_smi_files():
    """
    按指定顺序读取多个.smi文件,并将它们的内容合并写入一个新的.smi文件。
    """
    # 按照您要求的文件处理顺序
    #datasets/source_compounds/Fragment_MW_100_to_150.smi
    #datasets/source_compounds/Fragment_MW_up_to_100.smi
    input_filenames = [
        "Fragment_MW_up_to_100.smi",
        "Fragment_MW_100_to_150.smi",
        "Fragment_MW_150_to_200.smi",
        "Fragment_MW_200_to_250.smi"
    ]

    # 目标输出文件名
    output_filename = "ZINC250k.smi"

    print(f"开始合并文件到 {output_filename}...")

    try:
        # 使用'w'模式打开输出文件，如果文件已存在则会清空内容
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            # 遍历输入文件列表
            for filename in input_filenames:
                # 检查输入文件是否存在
                if not os.path.exists(filename):
                    print(f"--> 警告: 文件 '{filename}' 不存在，已跳过。")
                    continue

                print(f"--> 正在添加文件: {filename}")
                # 打开当前要读取的文件
                with open(filename, 'r', encoding='utf-8') as infile:
                    # 读取文件所有内容
                    content = infile.read()
                    # 将内容写入输出文件
                    outfile.write(content)
                    
                    # 确保文件末尾有换行符，以防下一个文件内容紧接在同一行
                    if content and not content.endswith('\n'):
                        outfile.write('\n')

        print(f"\n合并成功!所有内容已保存到 {output_filename}。")

    except IOError as e:
        print(f"发生文件读写错误: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    combine_smi_files()