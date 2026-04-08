# 🔍 APC 语义验证系统 (APC Semantic Validator)

一个基于 **零样本自然语言推理 (Zero-Shot NLI)** 的轻量级 Web 服务，用于批量分析“前提 (Premise)”与“假设 (Hypothesis)”之间的逻辑关系（蕴含、矛盾、中立）。

项目包含两个核心模块：
1. **Web 验证服务**：基于 Flask 的图形化界面，上传 CSV/TXT 文件即可获得分析结果和可视化图表。
2. **预处理脚本**：帮助将杂乱文本文件转换为系统要求的标准化格式。

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Framework](https://img.shields.io/badge/framework-Flask-red)
![Model](https://img.shields.io/badge/model-DistilBERT--MNLI-orange)

## ✨ 功能特点

- **纯离线运行**：通过环境变量强制离线，确保数据安全，不依赖外部网络请求（需提前缓存模型）。
- **智能编码检测**：自动检测上传文件（CSV/TXT）的编码格式（GBK, UTF-8 等），无需手动转换。
- **灵活的文件支持**：
    - 标准 CSV（支持 `前提/假设` 或 `premise/hypothesis` 列名）。
    - 简易 TXT（每行格式：`前提,假设` 或 Tab 分隔）。
- **可视化分析**：自动生成结果分布柱状图与饼图，直观展示数据倾向。
- **附带预处理工具**：提供独立脚本，支持正则、分割、定长等多种模式清洗原始日志或数据文件。

## 🛠️ 技术栈

- **后端**：Python 3.7+, Flask, Flask-CORS
- **模型推理**：Hugging Face Transformers (`typeform/distilbert-base-uncased-mnli`)
- **数据处理**：Pandas, chardet
- **可视化**：Matplotlib, Seaborn

## 📦 安装与配置

### 1. 环境准备
推荐使用 Conda 或 venv 创建独立环境：
```bash
git clone https://github.com/your-username/apc-semantic-validator.git
cd apc-semantic-validator
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows


安装依赖

pip install -r requirements.txt

创建虚拟环境

python -m venv .venv

进入虚拟环境

.vene\Scripts\activate

启动 Web 验证服务

python app.py
