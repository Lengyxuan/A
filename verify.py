import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 1. 加载预训练的自然语言推理模型
# 该模型能判断两个句子间的逻辑关系：蕴含（一致）、矛盾（冲突）、中立（无关）[reference:20]
print("正在加载模型，首次运行会下载（约300MB），请稍候...")
classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

# 定义中文标签，方便理解
labels = ["蕴含", "矛盾", "中立"]

def verify(premise, hypothesis):
    """给定前提（Premise）和假设（Hypothesis），返回验证结果和置信度"""
    # 将前提和假设组合成一个完整的句子，中间用分隔符隔开
    text = f"{premise} [SEP] {hypothesis}"
    # 调用模型进行分类
    result = classifier(text, candidate_labels=labels)
    # 返回得分最高的标签及其置信度分数
    return result['labels'][0], result['scores'][0]

def visualize_results(results_df, output_dir="results"):
    """生成结果可视化图表并保存"""
    # 设置中文字体，防止图表显示乱码
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 统计结果分布
    label_counts = results_df['预测结果'].value_counts()
    
    # 创建柱状图和饼图
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
    # 保存图表
    chart_path = f"{output_dir}/results_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"可视化图表已保存至: {chart_path}")
    return chart_path

def generate_report(results_df, total_time, output_dir="results"):
    """生成详细的验证报告"""
    label_counts = results_df['预测结果'].value_counts()
    report_lines = [
        "=" * 50,
        "语义验证报告",
        "=" * 50,
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"总验证条数: {len(results_df)}",
        f"总耗时: {total_time:.2f} 秒",
        f"平均每条耗时: {total_time/len(results_df):.2f} 秒",
        "-" * 50,
        "验证结果统计:",
    ]
    for label, count in label_counts.items():
        percentage = (count / len(results_df)) * 100
        report_lines.append(f"  {label}: {count} 条 ({percentage:.1f}%)")
    report_lines.extend([
        "-" * 50,
        "详细验证结果:",
        results_df.to_string(index=False),
        "=" * 50
    ])
    report_content = "\n".join(report_lines)
    report_path = f"{output_dir}/validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"详细报告已保存至: {report_path}")
    return report_path

# 2. 数据准备：读取CSV文件
input_file = "data.csv"
try:
    df = pd.read_csv(input_file, encoding="utf-8")
    print(f"成功读取 {len(df)} 条数据")
except FileNotFoundError:
    print(f"错误：找不到文件 '{input_file}'，请确保该文件存在于当前目录下。")
    exit(1)

# 3. 执行验证
results = []
import time
start_time = time.time()
print("开始验证...")
for index, row in df.iterrows():
    premise = row["前提"]
    hypothesis = row["假设"]
    label, score = verify(premise, hypothesis)
    results.append({
        "前提": premise,
        "假设": hypothesis,
        "预测结果": label,
        "置信度": round(score, 4)
    })
    if (index + 1) % 10 == 0:
        print(f"已处理 {index + 1}/{len(df)} 条...")
end_time = time.time()
total_time = end_time - start_time

# 4. 保存结果到CSV文件
result_df = pd.DataFrame(results)
# 创建results文件夹
import os
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/verification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n验证完成！结果已保存到 {output_file}")

# 5. 打印控制台摘要
print("\n验证结果摘要：")
print(result_df)
print(f"\n总耗时: {total_time:.2f} 秒")

# 6. 生成可视化图表和详细报告
try:
    chart_path = visualize_results(result_df, output_dir)
    report_path = generate_report(result_df, total_time, output_dir)
    print("\n所有结果文件已保存到 'results' 文件夹中。")
except Exception as e:
    print(f"生成可视化图表或报告时出错: {e}")