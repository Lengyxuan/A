import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
from io import BytesIO
import csv
import chardet
import traceback

app = Flask(__name__)
CORS(app)

# ---------- 离线模型加载 ----------
print("正在加载 NLI 模型（纯离线模式，使用本地缓存）...")
classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli",
    device=-1,
    local_files_only=True
)
LABELS = ["蕴含", "矛盾", "中立"]
print("模型加载完成！")

# ---------- 验证函数 ----------
def verify(premise, hypothesis):
    text = f"{premise} [SEP] {hypothesis}"
    result = classifier(text, candidate_labels=LABELS)
    return result['labels'][0], result['scores'][0]

# ---------- 生成图表（base64） ----------
def generate_chart_base64(results_df):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    label_counts = results_df['预测结果'].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x=label_counts.index, y=label_counts.values, ax=ax1, palette="viridis")
    ax1.set_title('验证结果分布（柱状图）')
    ax1.set_xlabel('逻辑关系')
    ax1.set_ylabel('数量')
    ax2.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('验证结果分布（饼图）')
    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

# ---------- TXT 解析函数（每行格式：前提,假设） ----------
def parse_txt(file_bytes, encoding):
    content = file_bytes.decode(encoding)
    lines = content.strip().splitlines()
    data_rows = []
    for line_num, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        # 支持逗号分隔
        parts = line.split(',', 1)
        if len(parts) == 2:
            premise = parts[0].strip()
            hypothesis = parts[1].strip()
            data_rows.append([premise, hypothesis])
        else:
            # 如果分隔符不是逗号，尝试制表符
            parts = line.split('\t', 1)
            if len(parts) == 2:
                premise = parts[0].strip()
                hypothesis = parts[1].strip()
                data_rows.append([premise, hypothesis])
            else:
                print(f"警告：第 {line_num} 行格式不正确，已跳过: {line[:50]}")
    if not data_rows:
        raise ValueError("TXT 文件没有有效的行（每行需要包含逗号或制表符分隔的“前提,假设”）")
    return pd.DataFrame(data_rows, columns=['前提', '假设'])

# ---------- API 路由（支持 .csv 和 .txt） ----------
@app.route('/validate', methods=['POST'])
def validate():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "未上传文件"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "文件名为空"}), 400

        filename = file.filename
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

        # 读取原始内容并检测编码
        raw_data = file.read()
        detected = chardet.detect(raw_data)
        encoding = detected.get('encoding', 'utf-8')
        print(f"检测到文件编码: {encoding}, 文件类型: {ext}")
        file.seek(0)

        # 根据扩展名选择解析方式
        if ext == 'csv':
            # CSV 解析（增强列名兼容性）
            try:
                df = pd.read_csv(
                    file,
                    encoding=encoding,
                    on_bad_lines='skip',
                    engine='python',
                    quoting=csv.QUOTE_MINIMAL,
                    skipinitialspace=True
                )
                print(f"CSV 解析成功，共 {len(df)} 行，列名: {list(df.columns)}")

                # 检查并标准化列名（支持中英文）
                if '前提' in df.columns and '假设' in df.columns:
                    pass  # 已经是标准列名
                elif 'premise' in df.columns and 'hypothesis' in df.columns:
                    df.rename(columns={'premise': '前提', 'hypothesis': '假设'}, inplace=True)
                    print("已将英文列名映射为中文")
                else:
                    # 尝试模糊匹配（可选，更友好的错误提示）
                    return jsonify({"error": "CSV 文件必须包含 '前提' 和 '假设' 两列（或英文 'premise' 和 'hypothesis'）"}), 400
            except Exception as e:
                # 回退到逐行解析（要求第一行必须为 '前提,假设'）
                file.seek(0)
                data_rows = []
                try:
                    content = file.read().decode(encoding)
                    reader = csv.reader(content.splitlines())
                    header = next(reader)
                    if len(header) != 2 or header[0].strip() != '前提' or header[1].strip() != '假设':
                        return jsonify({"error": "CSV 第一行必须是 '前提,假设' 两列"}), 400
                    for row in reader:
                        if len(row) >= 2:
                            premise = row[0].strip()
                            hypothesis = ','.join(row[1:]).strip()
                            data_rows.append([premise, hypothesis])
                    df = pd.DataFrame(data_rows, columns=['前提', '假设'])
                    print(f"CSV 回退解析成功，共 {len(df)} 行")
                except Exception as e2:
                    return jsonify({"error": f"CSV 解析失败: {str(e)} | 回退失败: {str(e2)}"}), 400
        elif ext == 'txt':
            # TXT 解析（每行 前提,假设）
            try:
                df = parse_txt(raw_data, encoding)
                print(f"TXT 解析成功，共 {len(df)} 行")
            except Exception as e:
                return jsonify({"error": f"TXT 解析失败: {str(e)}。请确保每行格式为：前提,假设（逗号分隔）"}), 400
        else:
            return jsonify({"error": f"不支持的文件类型：{ext}。请上传 .csv 或 .txt 文件"}), 400

        if df.empty:
            return jsonify({"error": "文件没有有效数据行"}), 400

        # 验证逻辑
        results = []
        for idx, row in df.iterrows():
            premise = str(row['前提']) if pd.notna(row['前提']) else ''
            hypothesis = str(row['假设']) if pd.notna(row['假设']) else ''
            label, score = verify(premise, hypothesis)
            results.append({
                "前提": premise,
                "假设": hypothesis,
                "预测结果": label,
                "置信度": round(score, 4)
            })
        result_df = pd.DataFrame(results)
        chart_base64 = generate_chart_base64(result_df)
        stats = result_df['预测结果'].value_counts().to_dict()
        return jsonify({
            "success": True,
            "results": results,
            "stats": stats,
            "chart": chart_base64
        })
    except Exception as e:
        print("=" * 50)
        print("验证过程发生未预期错误:")
        traceback.print_exc()
        print("=" * 50)
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500

# ---------- 前端页面（支持 .csv 和 .txt） ----------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>语义验证系统</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #4CAF50;
            display: inline-block;
            padding-bottom: 5px;
        }
        .upload-area {
            margin: 20px 0;
            padding: 20px;
            border: 2px dashed #ccc;
            border-radius: 8px;
            text-align: center;
            background-color: #fafafa;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 4px;
            cursor: pointer;
        }
        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .loading {
            display: none;
            margin: 20px 0;
            text-align: center;
            color: #666;
        }
        .results {
            margin-top: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        .stats {
            display: flex;
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: #e8f5e9;
            padding: 10px 20px;
            border-radius: 8px;
        }
        .error {
            color: red;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>🔍 语义验证系统</h1>
    <p>上传 CSV 或 TXT 文件（每行格式：前提,假设）</p>
    <div class="upload-area">
        <input type="file" id="fileInput" accept=".csv,.txt">
        <button id="validateBtn">开始验证</button>
    </div>
    <div class="loading" id="loading">⏳ 验证中，请稍候...</div>
    <div id="errorMsg" class="error"></div>
    <div id="resultsArea" style="display: none;">
        <div class="stats" id="stats"></div>
        <div class="chart-container" id="chartContainer"></div>
        <h2>详细结果</h2>
        <div style="overflow-x: auto;">
            <table id="resultTable">
                <thead><tr><th>前提</th><th>假设</th><th>预测结果</th><th>置信度</th></tr></thead>
                <tbody id="tableBody"></tbody>
            </table>
        </div>
    </div>
</div>
<script>
    const fileInput = document.getElementById('fileInput');
    const validateBtn = document.getElementById('validateBtn');
    const loadingDiv = document.getElementById('loading');
    const resultsArea = document.getElementById('resultsArea');
    const errorMsg = document.getElementById('errorMsg');
    const statsDiv = document.getElementById('stats');
    const chartContainer = document.getElementById('chartContainer');
    const tableBody = document.getElementById('tableBody');
    
    validateBtn.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) {
            errorMsg.innerText = '请选择 CSV 或 TXT 文件';
            return;
        }
        errorMsg.innerText = '';
        resultsArea.style.display = 'none';
        loadingDiv.style.display = 'block';
        validateBtn.disabled = true;
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/validate', { method: 'POST', body: formData });
            const data = await response.json();
            if (data.success) {
                let statsHtml = '';
                for (const [label, count] of Object.entries(data.stats)) {
                    statsHtml += `<div class="stat-card"><h3>${label}</h3><p>${count} 条</p></div>`;
                }
                statsDiv.innerHTML = statsHtml;
                chartContainer.innerHTML = `<img src="data:image/png;base64,${data.chart}" alt="图表">`;
                tableBody.innerHTML = '';
                data.results.forEach(row => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `<td>${escapeHtml(row.前提)}</td><td>${escapeHtml(row.假设)}</td><td>${row.预测结果}</td><td>${row.置信度}</td>`;
                    tableBody.appendChild(tr);
                });
                resultsArea.style.display = 'block';
            } else {
                errorMsg.innerText = '验证失败：' + (data.error || '未知错误');
            }
        } catch (err) {
            errorMsg.innerText = '请求出错：' + err.message;
        } finally {
            loadingDiv.style.display = 'none';
            validateBtn.disabled = false;
        }
    });
    function escapeHtml(str) {
        return str.replace(/[&<>]/g, function(m) {
            if (m === '&') return '&amp;';
            if (m === '<') return '&lt;';
            if (m === '>') return '&gt;';
            return m;
        });
    }
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)