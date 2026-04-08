#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预处理脚本：将原始文件转换为 APC 系统可识别的格式（CSV 文件，包含"前提,假设"两列）
用法：python preprocess_for_apc.py <input_file> [output_file]
"""

import os
import sys
import csv
import re
import chardet

# ==================== 配置区域（请根据实际文件格式修改） ====================

# 1. 解析模式选择
#    'split'     : 按分隔符拆分一行，取前两部分作为前提和假设
#    'regex'     : 用正则表达式从一行中提取前提和假设
#    'position'  : 按固定字符位置截取
PARSE_MODE = 'split'   # 可选 'split', 'regex', 'position'

# 2. split 模式下的分隔符（支持正则，如 ',', '\t', ' ', '[,，]' 等）
SPLIT_SEP = ','        # 例如英文逗号，或 r'[ ,，]+' 表示逗号/空格/中文逗号

# 3. regex 模式下的正则表达式（需要包含两个捕获组：前提和假设）
#    例如：r'前提[：:]\s*(.+?)[\s,，]+假设[：:]\s*(.+)'
REGEX_PATTERN = r'前提[：:]\s*(.+?)[\s,，]+假设[：:]\s*(.+)'

# 4. position 模式下的起始和结束位置（索引从0开始）
POS_PREMISE_START, POS_PREMISE_END = 0, 20     # 前提占前20字符
POS_HYP_START, POS_HYP_END = 20, 50            # 假设占第20-50字符

# 5. 是否跳过空行或明显乱码的行（推荐保持 True）
SKIP_INVALID = True

# 6. 输出格式：'csv' 或 'txt'（txt 每行是 "前提,假设"，无表头）
OUTPUT_FORMAT = 'csv'   # 推荐 csv，APC 系统两者都支持

# 7. 输出编码
OUTPUT_ENCODING = 'utf-8'

# ==================== 核心处理函数 ====================

def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        raw = f.read(10000)
        result = chardet.detect(raw)
        return result['encoding'] or 'utf-8'

def is_likely_valid(text):
    """简单判断文本是否包含过多乱码（可自定义）"""
    if not text or len(text.strip()) < 2:
        return False
    ascii_printable = set(chr(i) for i in range(32, 127))
    chinese_range = re.compile(r'[\u4e00-\u9fff]')
    count_valid = 0
    for ch in text.strip():
        if ch in ascii_printable or chinese_range.match(ch):
            count_valid += 1
    ratio = count_valid / len(text.strip()) if text.strip() else 0
    return ratio > 0.7

def parse_line(line):
    """根据配置解析一行，返回 (premise, hypothesis) 或 (None, None)"""
    line = line.strip()
    if not line:
        return None, None

    if PARSE_MODE == 'split':
        parts = re.split(SPLIT_SEP, line, maxsplit=1)
        if len(parts) >= 2:
            premise = parts[0].strip()
            hypothesis = parts[1].strip()
            return premise, hypothesis
        else:
            return None, None

    elif PARSE_MODE == 'regex':
        match = re.search(REGEX_PATTERN, line)
        if match:
            premise = match.group(1).strip()
            hypothesis = match.group(2).strip()
            return premise, hypothesis
        else:
            return None, None

    elif PARSE_MODE == 'position':
        if len(line) >= POS_HYP_END:
            premise = line[POS_PREMISE_START:POS_PREMISE_END].strip()
            hypothesis = line[POS_HYP_START:POS_HYP_END].strip()
            return premise, hypothesis
        else:
            return None, None

    else:
        raise ValueError(f"未知的 PARSE_MODE: {PARSE_MODE}")

def preprocess_file(input_path, output_path):
    """主处理流程"""
    encoding = detect_encoding(input_path)
    print(f"检测到文件编码: {encoding}")

    data_rows = []
    skipped = 0

    with open(input_path, 'r', encoding=encoding, errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            premise, hypothesis = parse_line(line)
            if premise is None or hypothesis is None:
                if SKIP_INVALID:
                    skipped += 1
                continue

            if SKIP_INVALID and not (is_likely_valid(premise) and is_likely_valid(hypothesis)):
                skipped += 1
                continue

            data_rows.append([premise, hypothesis])

    if not data_rows:
        print("错误：没有提取到任何有效数据行。请检查配置的解析规则是否匹配文件格式。")
        sys.exit(1)

    print(f"成功提取 {len(data_rows)} 行，跳过 {skipped} 行无效内容")

    if OUTPUT_FORMAT == 'csv':
        with open(output_path, 'w', encoding=OUTPUT_ENCODING, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['前提', '假设'])
            writer.writerows(data_rows)
        print(f"CSV 文件已保存至: {output_path}")
    else:
        with open(output_path, 'w', encoding=OUTPUT_ENCODING) as f:
            for premise, hypothesis in data_rows:
                f.write(f"{premise},{hypothesis}\n")
        print(f"TXT 文件已保存至: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python preprocess_for_apc.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"错误：输入文件 '{input_file}' 不存在")
        sys.exit(1)

    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        base, ext = os.path.splitext(input_file)
        output_file = base + "_for_apc.csv" if OUTPUT_FORMAT == 'csv' else base + "_for_apc.txt"

    preprocess_file(input_file, output_file)