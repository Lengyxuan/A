import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw = f.read(10000)
        result = chardet.detect(raw)
        return result['encoding'] or 'utf-8'

def show_valid_lines(file_path, max_lines=10):
    encoding = detect_encoding(file_path)
    print(f"检测到编码: {encoding}")
    print("尝试显示有效行（跳过明显乱码）:\n")
    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
        count = 0
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            # 简单判断：如果包含常见中文字符或英文单词，认为是有效
            if any('\u4e00' <= ch <= '\u9fff' for ch in line) or line.isascii():
                print(f"第{line_num}行: {line[:200]}")
                count += 1
                if count >= max_lines:
                    break
        if count == 0:
            print("没有找到任何看起来像正常文本的行。文件可能全是乱码或格式特殊。")

if __name__ == '__main__':
    show_valid_lines('faiss.txt')