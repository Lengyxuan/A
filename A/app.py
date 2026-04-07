import os
import pandas as pd
import tempfile
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)  # 允许跨域，方便本地测试

# ---------- 加载模型（全局加载一次，避免重复加载） ----------
print("正在加载 NLI 模型，首次加载会下载...")
classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
LABELS = ["蕴含", "矛盾", "中立"]
print("模型加载完成！")

# ---------- 验证函数 ----------
def verify(premise, hypothesis):
    """返回 (标签, 置信度)"""
    text = f"{premise} [SEP] {hypothesis}"
    result = classifier(text, candidate_labels=LABELS)
    return result['labels'][0], result['scores'][0]

# ---------- 生成图表的函数（返回 base64 编码的图片） ----------
def generate_chart_base64(results_df):
    """生成结果分布图，并返回 base64 字符串，以便直接在 HTML 中显示"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    label_counts = results_df['预测结果'].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 柱状图
    sns.barplot(x=label_counts.index, y=label_counts.values, ax=ax1, palette="viridis")
    ax1.set_title('验证结果分布（柱状图）')
    ax1.set_xlabel('逻辑关系')
    ax1.set_ylabel('数量')
    
    # 饼图
    ax2.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('验证结果分布（饼图）')
    
    plt.tight_layout()
    # 将图片保存到内存中
    img = BytesIO()
    plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

# ---------- API 路由：处理验证请求 ----------
@app.route('/validate', methods=['POST'])
def validate():
    if 'file' not in request.files:
        return jsonify({"error": "未上传文件"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "文件名为空"}), 400
    
    # 读取 CSV 文件
    try:
        df = pd.read_csv(file)
        # 检查必需的列
        if '前提' not in df.columns or '假设' not in df.columns:
            return jsonify({"error": "CSV 文件必须包含 '前提' 和 '假设' 两列"}), 400
    except Exception as e:
        return jsonify({"error": f"CSV 解析失败: {str(e)}"}), 400
    
    # 逐行验证
    results = []
    for _, row in df.iterrows():
        premise = str(row['前提'])
        hypothesis = str(row['假设'])
        label, score = verify(premise, hypothesis)
        results.append({
            "前提": premise,
            "假设": hypothesis,
            "预测结果": label,
            "置信度": round(score, 4)
        })
    
    result_df = pd.DataFrame(results)
    
    # 生成图表（base64 编码）
    chart_base64 = generate_chart_base64(result_df)
    
    # 统计信息
    stats = result_df['预测结果'].value_counts().to_dict()
    
    # 返回 JSON 数据：包含结果表格和图表
    return jsonify({
        "success": True,
        "results": results,
        "stats": stats,
        "chart": chart_base64
    })

# ---------- 前端页面（HTML） ----------
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
        .upload-area input {
            margin: 10px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .loading {
            display: none;
            margin: 20px 0;
            text-align: center;
            font-style: italic;
            color: #666;
        }
        .results {
            margin-top: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 14px;
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
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .stats {
            display: flex;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .stat-card {
            background: #e8f5e9;
            padding: 10px 20px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-card h3 {
            margin: 0;
            color: #2e7d32;
        }
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        .chart-container img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .error {
            color: red;
            margin: 10px 0;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>🔍 语义验证系统</h1>
    <p>上传包含「前提」和「假设」两列的 CSV 文件，系统将自动判断每对句子的逻辑关系（蕴含/矛盾/中立）。</p>
    
    <div class="upload-area">
        <input type="file" id="csvFile" accept=".csv">
        <button id="validateBtn">开始验证</button>
    </div>
    
    <div class="loading" id="loading">
        ⏳ 正在验证，请稍候...（首次运行需加载模型，可能较慢）
    </div>
    
    <div id="errorMsg" class="error"></div>
    
    <div id="resultsArea" style="display: none;">
        <div class="stats" id="stats"></div>
        <div class="chart-container" id="chartContainer"></div>
        <h2>详细结果</h2>
        <div style="overflow-x: auto;">
            <table id="resultTable">
                <thead>
                    <tr><th>前提</th><th>假设</th><th>预测结果</th><th>置信度</th></tr>
                </thead>
                <tbody id="tableBody"></tbody>
            </table>
        </div>
    </div>
</div>

<script>
    const fileInput = document.getElementById('csvFile');
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
            errorMsg.innerText = '请先选择一个 CSV 文件';
            return;
        }
        
        // 清空之前的错误和结果
        errorMsg.innerText = '';
        resultsArea.style.display = 'none';
        loadingDiv.style.display = 'block';
        validateBtn.disabled = true;
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/validate', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.success) {
                // 显示统计
                let statsHtml = '';
                for (const [label, count] of Object.entries(data.stats)) {
                    statsHtml += `<div class="stat-card"><h3>${label}</h3><p>${count} 条</p></div>`;
                }
                statsDiv.innerHTML = statsHtml;
                
                // 显示图表
                chartContainer.innerHTML = `<img src="data:image/png;base64,${data.chart}" alt="结果分布图">`;
                
                // 填充表格
                tableBody.innerHTML = '';
                data.results.forEach(row => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>${escapeHtml(row.前提)}</td>
                        <td>${escapeHtml(row.假设)}</td>
                        <td>${row.预测结果}</td>
                        <td>${row.置信度}</td>
                    `;
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
    
    // 简单的防XSS
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
    """返回前端页面"""
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)