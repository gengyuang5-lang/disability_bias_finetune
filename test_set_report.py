"""
生成测试集评估报告
对比已有评估结果和测试集统计信息
"""

import json
from collections import Counter
from datetime import datetime

TEST_DATA_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\BBQ-main\BBQ-main\data\test_set.json"
TRAIN_DATA_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD.xie\apd_results\train_full_results.json"
EVAL_RESULTS_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD-PromptGenerator\evaluation_results.json"

def generate_report():
    print("=" * 80)
    print(" " * 20 + "测试集评估报告")
    print("=" * 80)
    print(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载数据
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
        train_results = train_data['results']

    with open(EVAL_RESULTS_PATH, 'r', encoding='utf-8') as f:
        eval_results = json.load(f)

    # 1. 数据集概况
    print("\n" + "=" * 80)
    print("一、数据集概况")
    print("=" * 80)

    print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│  数据集划分                                                              │
├─────────────────────────────────────────────────────────────────────────┤
│  完整BBQ数据集:     58,492 条                                            │
│  训练集 (已使用):    {len(train_results):<5} 条  (用于微调模型)                      │
│  测试集 (新生成):   {len(test_data):<5} 条  (用于评估模型泛化能力)                  │
│  重叠样本 (已移除):  1,319 条                                            │
└─────────────────────────────────────────────────────────────────────────┘
    """)

    # 2. 测试集类别分布
    print("\n" + "=" * 80)
    print("二、测试集类别分布")
    print("=" * 80)

    categories = Counter(item['category'] for item in test_data)

    print(f"\n{'类别':<25} {'数量':<10} {'占比':<10} {'柱状图'}")
    print("-" * 80)
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(test_data) * 100
        bar = "█" * int(pct / 2)
        print(f"{cat:<25} {count:<10} {pct:>6.1f}%  {bar}")

    # 3. 测试集特征分布
    print("\n" + "=" * 80)
    print("三、测试集特征分布")
    print("=" * 80)

    polarity_dist = Counter(item['question_polarity'] for item in test_data)
    context_dist = Counter(item['context_condition'] for item in test_data)
    label_dist = Counter(item['label'] for item in test_data)

    print(f"""
问题极性分布:
   负面 (neg):     {polarity_dist['neg']:,} 条 ({polarity_dist['neg']/len(test_data)*100:.1f}%)
   非负面 (nonneg): {polarity_dist['nonneg']:,} 条 ({polarity_dist['nonneg']/len(test_data)*100:.1f}%)

上下文条件分布:
   模糊 (ambig):    {context_dist['ambig']:,} 条 ({context_dist['ambig']/len(test_data)*100:.1f}%)
   明确 (disambig): {context_dist['disambig']:,} 条 ({context_dist['disambig']/len(test_data)*100:.1f}%)

正确答案分布:
   选项 0: {label_dist[0]:,} 条 ({label_dist[0]/len(test_data)*100:.1f}%)
   选项 1: {label_dist[1]:,} 条 ({label_dist[1]/len(test_data)*100:.1f}%)
   选项 2: {label_dist[2]:,} 条 ({label_dist[2]/len(test_data)*100:.1f}%)
    """)

    # 4. 已有评估结果分析
    print("\n" + "=" * 80)
    print("四、已有验证集评估结果 (evaluate_model.py)")
    print("=" * 80)

    avg_scores = eval_results['average_scores']
    grade = eval_results['grade']

    print(f"""
评估样本数: {len(eval_results['results'])} 条

平均指标:
   相似度得分:     {avg_scores['similarity']:.3f} (生成提示词与参考答案的文本相似度)
   关键词覆盖率:   {avg_scores['keyword_coverage']:.3f} (反偏见关键词覆盖程度)
   长度合理性:     {avg_scores['length_score']:.3f} (生成提示词长度是否在合理范围)
   综合得分:       {avg_scores['overall']:.3f}

评级: {grade}
    """)

    # 5. 评估指标说明
    print("\n" + "=" * 80)
    print("五、评估指标说明")
    print("=" * 80)

    print("""
1. 相似度得分 (Similarity)
   - 计算生成提示词与参考答案的文本相似度
   - 使用SequenceMatcher算法
   - 范围: 0-1，越高越好

2. 关键词覆盖率 (Keyword Coverage)
   - 检查生成提示词中包含的反偏见关键词数量
   - 关键词包括: 偏见、歧视、刻板印象、平等、尊重、客观、能力等
   - 范围: 0-1，越高越好

3. 长度合理性 (Length Score)
   - 评估生成提示词的长度是否在合理范围(100-300字符)
   - 过短或过长都会扣分
   - 范围: 0-1，越接近1越好

4. 综合得分 (Overall)
   - 综合以上三个指标的加权平均
   - 权重: 相似度40% + 关键词覆盖40% + 长度20%
   - 范围: 0-1，越高越好
    """)

    # 6. 测试集样本示例
    print("\n" + "=" * 80)
    print("六、测试集样本示例")
    print("=" * 80)

    shown_categories = set()
    example_count = 0
    for item in test_data:
        cat = item['category']
        if cat not in shown_categories:
            shown_categories.add(cat)
            example_count += 1
            label = item['label']
            ans_key = f'ans{label}'

            print(f"""
【示例 {example_count}】类别: {cat}
   example_id: {item['example_id']}
   question_polarity: {item['question_polarity']}
   context_condition: {item['context_condition']}

   Context:
   {item['context'][:100]}...

   Question: {item['question']}

   选项:
   - ans0: {item['ans0']}
   - ans1: {item['ans1']}
   - ans2: {item['ans2']}

   正确答案: 选项 {label} - {item[ans_key]}
            """)

            if len(shown_categories) >= 3:
                break

    # 7. 已有评估结果示例
    print("\n" + "=" * 80)
    print("七、已有验证集评估示例")
    print("=" * 80)

    for i, result in enumerate(eval_results['results'][:3], 1):
        print(f"""
【评估示例 {i}】
问题: {result['question']}

参考答案:
{result['reference'][:100]}...

生成提示词:
{result['generated'][:150]}...

得分:
   相似度: {result['scores']['similarity']:.3f}
   关键词: {result['keywords']}
   综合分: {result['scores']['overall']:.3f}
        """)

    # 8. 建议
    print("\n" + "=" * 80)
    print("八、后续建议")
    print("=" * 80)

    print("""
1. 在测试集上运行完整评估
   - 使用新生成的57,173条测试数据评估模型泛化能力
   - 对比基础模型和微调后模型的表现差异

2. 分析不同类别的表现
   - 重点关注Disability_status类别的表现
   - 检查模型在各个人口统计类别上的公平性

3. 评估偏见缓解效果
   - 对比使用反偏见提示词前后的模型回答
   - 量化偏见分数的变化

4. 生成可视化报告
   - 绘制各类别的准确率对比图
   - 生成混淆矩阵分析错误模式
    """)

    print("\n" + "=" * 80)
    print("报告生成完成!")
    print("=" * 80)

if __name__ == "__main__":
    generate_report()
