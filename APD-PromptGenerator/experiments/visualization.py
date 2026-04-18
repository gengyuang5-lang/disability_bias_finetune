"""
实验结果可视化 - 用于汇报
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 加载实验结果
exp1_path = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD-PromptGenerator\experiments\exp1_baseline_chinese_llm\results\exp1_baseline_chinese_llm\exp1_baseline_chinese_llm_20260417_190112.json"
exp2_path = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD-PromptGenerator\experiments\exp2_prompt_enhanced\results\exp2_prompt_enhanced\exp2_prompt_enhanced_20260417_233649.json"

with open(exp1_path, 'r', encoding='utf-8') as f:
    exp1_data = json.load(f)
with open(exp2_path, 'r', encoding='utf-8') as f:
    exp2_data = json.load(f)

exp1 = exp1_data['results']
exp2 = exp2_data['results']

# 创建输出目录
output_dir = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD-PromptGenerator\experiments\visualizations"
os.makedirs(output_dir, exist_ok=True)

# 图1: 总体指标对比
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Disability Bias实验结果对比', fontsize=16, fontweight='bold')

categories = ['基线实验', '对比实验']
colors = ['#3498db', '#e74c3c']

# 总体准确率
ax1 = axes[0]
accuracies = [exp1['overall_accuracy'], exp2['overall_accuracy']]
bars1 = ax1.bar(categories, accuracies, color=colors, alpha=0.8, edgecolor='black')
ax1.set_ylabel('准确率', fontsize=12)
ax1.set_title('总体准确率', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1.1)
for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{acc:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 偏见分数
ax2 = axes[1]
bias_scores = [exp1['bias_scores']['overall_bias_score'], exp2['bias_scores']['overall_bias_score']]
bars2 = ax2.bar(categories, bias_scores, color=colors, alpha=0.8, edgecolor='black')
ax2.set_ylabel('偏见分数', fontsize=12)
ax2.set_title('总体偏见分数 (越低越好)', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 0.35)
for i, (bar, bias) in enumerate(zip(bars2, bias_scores)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{bias:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    if i == 1:
        improvement = (bias_scores[0] - bias_scores[1]) / bias_scores[0] * 100
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'↓{improvement:.1f}%', ha='center', va='bottom', 
                 fontsize=10, color='green', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 不同情境准确率
ax3 = axes[2]
x = np.arange(2)
width = 0.35
ambig_acc = [exp1['accuracy_by_condition']['ambig_accuracy'], exp2['accuracy_by_condition']['ambig_accuracy']]
disambig_acc = [exp1['accuracy_by_condition']['disambig_accuracy'], exp2['accuracy_by_condition']['disambig_accuracy']]

bars3a = ax3.bar(x - width/2, ambig_acc, width, label='模糊情境', color='#9b59b6', alpha=0.8, edgecolor='black')
bars3b = ax3.bar(x + width/2, disambig_acc, width, label='明确情境', color='#f39c12', alpha=0.8, edgecolor='black')

ax3.set_ylabel('准确率', fontsize=12)
ax3.set_title('不同情境准确率', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(categories)
ax3.legend(fontsize=10)
ax3.set_ylim(0, 1.1)
ax3.grid(axis='y', alpha=0.3)

for bars in [bars3a, bars3b]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.1%}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig1_overall_comparison.png'), dpi=300, bbox_inches='tight')
print(f"图1已保存: {os.path.join(output_dir, 'fig1_overall_comparison.png')}")
plt.close()

# 图2: 子类别对比
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('不同残疾类别的准确率对比', fontsize=16, fontweight='bold')

categories = ['基线实验', '对比实验']
x = np.arange(2)
width = 0.35

physical_acc = [
    exp1['accuracy_by_subcategory']['Physical']['accuracy'],
    exp2['accuracy_by_subcategory']['Physical']['accuracy']
]
mental_acc = [
    exp1['accuracy_by_subcategory']['MentalIllness']['accuracy'],
    exp2['accuracy_by_subcategory']['MentalIllness']['accuracy']
]

bars1 = ax.bar(x - width/2, physical_acc, width, label='身体残疾', color='#1abc9c', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, mental_acc, width, label='精神疾病', color='#e67e22', alpha=0.8, edgecolor='black')

ax.set_ylabel('准确率', fontsize=12)
ax.set_title('按残疾类别划分的准确率', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig2_subcategory_comparison.png'), dpi=300, bbox_inches='tight')
print(f"图2已保存: {os.path.join(output_dir, 'fig2_subcategory_comparison.png')}")
plt.close()

# 图3: 偏见分数详细对比
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('偏见分数详细分析', fontsize=16, fontweight='bold')

categories = ['基线实验', '对比实验']
x = np.arange(2)
width = 0.25

nonneg_bias = [exp1['bias_scores']['nonneg_bias_score'], exp2['bias_scores']['nonneg_bias_score']]
neg_bias = [exp1['bias_scores']['neg_bias_score'], exp2['bias_scores']['neg_bias_score']]
overall_bias = [exp1['bias_scores']['overall_bias_score'], exp2['bias_scores']['overall_bias_score']]

bars1 = ax.bar(x - width, nonneg_bias, width, label='非负面偏见', color='#3498db', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x, neg_bias, width, label='负面偏见', color='#e74c3c', alpha=0.8, edgecolor='black')
bars3 = ax.bar(x + width, overall_bias, width, label='总体偏见', color='#2c3e50', alpha=0.8, edgecolor='black')

ax.set_ylabel('偏见分数', fontsize=12)
ax.set_title('各类偏见分数对比 (越低越好)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(fontsize=11)
ax.set_ylim(0, 0.5)
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig3_bias_score_detail.png'), dpi=300, bbox_inches='tight')
print(f"图3已保存: {os.path.join(output_dir, 'fig3_bias_score_detail.png')}")
plt.close()

# 图4: 改进效果雷达图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
fig.suptitle('实验改进效果雷达图', fontsize=16, fontweight='bold', y=0.98)

# 归一化指标 (0-1之间)
metrics = ['总体准确率', '偏见降低', '模糊情境', '明确情境', '身体残疾', '精神疾病']

# 计算改进值 (实验2相对于实验1)
acc_change = exp2['overall_accuracy'] / exp1['overall_accuracy']  # 准确率比例
bias_reduction = (exp1['bias_scores']['overall_bias_score'] - exp2['bias_scores']['overall_bias_score']) / exp1['bias_scores']['overall_bias_score']  # 偏见降低比例
ambig_change = exp2['accuracy_by_condition']['ambig_accuracy'] / exp1['accuracy_by_condition']['ambig_accuracy']
disambig_change = exp2['accuracy_by_condition']['disambig_accuracy'] / exp1['accuracy_by_condition']['disambig_accuracy']
physical_change = exp2['accuracy_by_subcategory']['Physical']['accuracy'] / exp1['accuracy_by_subcategory']['Physical']['accuracy']
mental_change = exp2['accuracy_by_subcategory']['MentalIllness']['accuracy'] / exp1['accuracy_by_subcategory']['MentalIllness']['accuracy']

# 归一化到0-1范围 (1表示基线，>1表示改进，<1表示下降)
values = [acc_change, 1 + bias_reduction, ambig_change, disambig_change, physical_change, mental_change]
values += values[:1]  # 闭合图形

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

ax.plot(angles, values, 'o-', linewidth=2, color='#e74c3c', label='实验2/实验1比例')
ax.fill(angles, values, alpha=0.25, color='#e74c3c')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0.7, 1.3)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='基线')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig4_radar_chart.png'), dpi=300, bbox_inches='tight')
print(f"图4已保存: {os.path.join(output_dir, 'fig4_radar_chart.png')}")
plt.close()

# 图5: 汇总表格图
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')
fig.suptitle('实验结果汇总表', fontsize=16, fontweight='bold')

table_data = [
    ['指标', '基线实验', '对比实验', '变化', '改进'],
    ['总体准确率', f"{exp1['overall_accuracy']:.1%}", f"{exp2['overall_accuracy']:.1%}", 
     f"{exp2['overall_accuracy']-exp1['overall_accuracy']:+.1%}", '✗'],
    ['总体偏见分数', f"{exp1['bias_scores']['overall_bias_score']:.3f}", 
     f"{exp2['bias_scores']['overall_bias_score']:.3f}",
     f"{exp2['bias_scores']['overall_bias_score']-exp1['bias_scores']['overall_bias_score']:+.3f}", '✓'],
    ['模糊情境准确率', f"{exp1['accuracy_by_condition']['ambig_accuracy']:.1%}",
     f"{exp2['accuracy_by_condition']['ambig_accuracy']:.1%}",
     f"{exp2['accuracy_by_condition']['ambig_accuracy']-exp1['accuracy_by_condition']['ambig_accuracy']:+.1%}", '✓'],
    ['明确情境准确率', f"{exp1['accuracy_by_condition']['disambig_accuracy']:.1%}",
     f"{exp2['accuracy_by_condition']['disambig_accuracy']:.1%}",
     f"{exp2['accuracy_by_condition']['disambig_accuracy']-exp1['accuracy_by_condition']['disambig_accuracy']:+.1%}", '✗'],
    ['身体残疾准确率', f"{exp1['accuracy_by_subcategory']['Physical']['accuracy']:.1%}",
     f"{exp2['accuracy_by_subcategory']['Physical']['accuracy']:.1%}",
     f"{exp2['accuracy_by_subcategory']['Physical']['accuracy']-exp1['accuracy_by_subcategory']['Physical']['accuracy']:+.1%}", '✗'],
    ['精神疾病准确率', f"{exp1['accuracy_by_subcategory']['MentalIllness']['accuracy']:.1%}",
     f"{exp2['accuracy_by_subcategory']['MentalIllness']['accuracy']:.1%}",
     f"{exp2['accuracy_by_subcategory']['MentalIllness']['accuracy']-exp1['accuracy_by_subcategory']['MentalIllness']['accuracy']:+.1%}", '✗'],
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.2, 0.15, 0.15, 0.15, 0.1])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# 设置表头样式
for i in range(5):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 设置改进列颜色
for i in range(1, 7):
    if table_data[i][4] == '✓':
        table[(i, 4)].set_facecolor('#d5f5e3')
    else:
        table[(i, 4)].set_facecolor('#fadbd8')

plt.savefig(os.path.join(output_dir, 'fig5_summary_table.png'), dpi=300, bbox_inches='tight')
print(f"图5已保存: {os.path.join(output_dir, 'fig5_summary_table.png')}")
plt.close()

print("\n" + "="*60)
print("所有可视化图表已生成完毕！")
print(f"保存位置: {output_dir}")
print("="*60)
print("\n生成的图表:")
print("1. fig1_overall_comparison.png - 总体指标对比")
print("2. fig2_subcategory_comparison.png - 子类别对比")
print("3. fig3_bias_score_detail.png - 偏见分数详细分析")
print("4. fig4_radar_chart.png - 改进效果雷达图")
print("5. fig5_summary_table.png - 结果汇总表")
