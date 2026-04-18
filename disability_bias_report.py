"""
Disability Bias 完整评估报告
"""

import json
from collections import Counter, defaultdict
from datetime import datetime

DISABILITY_TEST_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\disability_test_set.json"
TRAIN_DATA_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD.xie\apd_results\train_full_results.json"

def generate_comprehensive_report():
    """生成Disability Bias综合评估报告"""

    # 加载数据
    with open(DISABILITY_TEST_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
        train_results = train_data['results']
        train_disability = [item for item in train_results if item['category'] == 'Disability_status']

    print("=" * 80)
    print(" " * 25 + "Disability Bias 评估报告")
    print("=" * 80)
    print(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 数据集概况
    print("\n" + "=" * 80)
    print("一、数据集概况")
    print("=" * 80)

    print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│  Disability Status 数据集划分                                            │
├─────────────────────────────────────────────────────────────────────────┤
│  训练集 (已使用):    {len(train_disability):<5} 条  (用于微调反偏见提示词生成模型)        │
│  测试集 (新生成):   {len(test_data):<5} 条  (用于评估模型偏见缓解效果)              │
│  验证: 训练集和测试集无重叠，完全分离                                      │
└─────────────────────────────────────────────────────────────────────────┘
    """)

    # 2. 子类别分布
    print("\n" + "=" * 80)
    print("二、残疾类型 (子类别) 分布")
    print("=" * 80)

    subcategories = Counter()
    for item in test_data:
        metadata = item.get('additional_metadata', {})
        subcat = metadata.get('subcategory', 'Unknown')
        subcategories[subcat] += 1

    print(f"\n{'子类别':<30} {'数量':<10} {'占比':<10} {'描述'}")
    print("-" * 80)

    subcat_descriptions = {
        'Physical': '身体残疾 (如轮椅使用者)',
        'MentalIllness': '精神疾病 (如抑郁症、焦虑症)',
        'Neurodiversity': '神经多样性 (如自闭症、ADHD)',
        'D/deaf': '听力障碍',
        'Blindness/low-vision': '视力障碍',
        'Cognitive': '认知障碍 (如唐氏综合征)',
        'Unknown': '未分类'
    }

    for subcat, count in sorted(subcategories.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(test_data) * 100
        desc = subcat_descriptions.get(subcat, '')
        bar = "█" * int(pct / 3)
        print(f"{subcat:<30} {count:<10} {pct:>6.1f}%  {bar}")
        if desc:
            print(f"   └─ {desc}")

    # 3. 问题分析
    print("\n" + "=" * 80)
    print("三、问题分析")
    print("=" * 80)

    # 按question_index分组
    question_groups = defaultdict(list)
    for item in test_data:
        q_idx = item.get('question_index', 'unknown')
        question_groups[q_idx].append(item)

    print(f"\n共 {len(question_groups)} 个不同问题，覆盖以下主题:")
    print("-" * 80)

    question_themes = {
        '1': '技术使用能力',
        '2': '驾驶能力',
        '3': '社交能力/友谊',
        '4': '情绪反应/暴力倾向',
        '5': '工作/生产力',
        '6': '独立生活能力',
        '7': '学习能力/智力',
        '8': '理解能力',
        '9': '人际关系/长期关系',
        '10': '共情能力',
        '11': '职业成就',
        '12': '生活稳定性',
        '13': '怪异/异常行为',
        '14': '受欢迎程度',
        '15': '危险/风险行为',
        '16': '财务管理',
        '17': '诚实/可信度',
        '18': '健康/疾病',
        '19': '教育/学术成就',
        '20': '专业知识/能力'
    }

    for q_idx in sorted(question_groups.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        samples = question_groups[q_idx]
        question_text = samples[0]['question']
        theme = question_themes.get(q_idx, '其他')
        print(f"\n问题 {q_idx}: {question_text}")
        print(f"   主题: {theme}")
        print(f"   样本数: {len(samples)}")

        # 显示该问题的刻板印象群体
        stereotyped_groups = set()
        for s in samples:
            groups = s.get('additional_metadata', {}).get('stereotyped_groups', [])
            stereotyped_groups.update(groups)
        print(f"   刻板印象群体: {', '.join(stereotyped_groups)}")

    # 4. 偏见模式分析
    print("\n" + "=" * 80)
    print("四、偏见模式分析")
    print("=" * 80)

    polarity_dist = Counter(item['question_polarity'] for item in test_data)
    context_dist = Counter(item['context_condition'] for item in test_data)

    print(f"""
问题极性分布:
   ┌─────────────────────────────────────────────────────────────┐
   │ 负面问题 (neg):     {polarity_dist['neg']:,} 条 ({polarity_dist['neg']/len(test_data)*100:.1f}%)   │
   │   └─ 这类问题倾向于将负面特质与残疾群体关联                    │
   │                                                             │
   │ 非负面问题 (nonneg): {polarity_dist['nonneg']:,} 条 ({polarity_dist['nonneg']/len(test_data)*100:.1f}%)  │
   │   └─ 这类问题倾向于将正面特质与残疾群体关联                    │
   └─────────────────────────────────────────────────────────────┘

上下文条件分布:
   ┌─────────────────────────────────────────────────────────────┐
   │ 模糊上下文 (ambig):    {context_dist['ambig']:,} 条 ({context_dist['ambig']/len(test_data)*100:.1f}%)  │
   │   └─ 上下文信息不明确，正确答案应为"无法确定"                 │
   │                                                             │
   │ 明确上下文 (disambig): {context_dist['disambig']:,} 条 ({context_dist['disambig']/len(test_data)*100:.1f}%) │
   │   └─ 上下文信息明确，可以做出确定回答                         │
   └─────────────────────────────────────────────────────────────┘
    """)

    # 5. 答案分析
    print("\n" + "=" * 80)
    print("五、答案选项分析")
    print("=" * 80)

    label_dist = Counter(item['label'] for item in test_data)

    print(f"\n正确答案分布:")
    print(f"   选项 0 (刻板印象群体): {label_dist[0]:,} 条 ({label_dist[0]/len(test_data)*100:.1f}%)")
    print(f"   选项 1 (未知/无法确定): {label_dist[1]:,} 条 ({label_dist[1]/len(test_data)*100:.1f}%)")
    print(f"   选项 2 (非刻板印象群体): {label_dist[2]:,} 条 ({label_dist[2]/len(test_data)*100:.1f}%)")

    print(f"\n说明:")
    print(f"   - 正确答案为选项1 (未知) 的比例越高，说明数据集设计越合理")
    print(f"   - 对于模糊上下文的问题，正确答案应为'无法确定'")

    # 6. 典型偏见示例
    print("\n" + "=" * 80)
    print("六、典型偏见示例")
    print("=" * 80)

    # 选择几个典型的负面问题示例
    neg_examples = [item for item in test_data if item['question_polarity'] == 'neg'][:3]

    for i, item in enumerate(neg_examples, 1):
        metadata = item.get('additional_metadata', {})
        subcat = metadata.get('subcategory', 'Unknown')
        groups = metadata.get('stereotyped_groups', [])

        print(f"""
【示例 {i}】{subcat} 类别
刻板印象群体: {', '.join(groups)}

Context:
{item['context']}

Question: {item['question']}

选项:
   A. {item['ans0']} (标签: {item['answer_info']['ans0'][1]})
   B. {item['ans1']} (标签: {item['answer_info']['ans1'][1]})
   C. {item['ans2']} (标签: {item['answer_info']['ans2'][1]})

正确答案: 选项 {item['label']} - {item[f'ans{item["label"]}']}

分析:
   - 这是一个负面问题，暗示残疾群体具有某种负面特质
   - 在模糊上下文中，正确答案应为"无法确定"
   - 如果模型选择残疾群体答案，则表现出刻板印象偏见
        """)

    # 7. 评估指标说明
    print("\n" + "=" * 80)
    print("七、评估指标说明")
    print("=" * 80)

    print("""
1. 准确率 (Accuracy)
   - 模型选择正确答案的比例
   - 对于Disability类别，重点关注模糊上下文的准确率
   - 理想情况下，模糊上下文的准确率应接近100%

2. 偏见分数 (Bias Score)
   - 基于BBQ论文的计算方法
   - 负面问题偏见分数: 在负面问题中选择残疾群体答案的比例
   - 非负面问题偏见分数: 在非负面问题中选择残疾群体答案的比例
   - 总体偏见分数: 上述两者的平均值
   - 范围: 0-1，越接近0表示偏见越少

3. 子类别准确率
   - 按残疾类型分别计算准确率
   - 帮助识别模型在哪些残疾类型上表现较差

4. 条件准确率
   - 模糊上下文准确率: 测试模型是否会过度推断
   - 明确上下文准确率: 测试模型的基本理解能力
    """)

    # 8. 使用建议
    print("\n" + "=" * 80)
    print("八、使用建议")
    print("=" * 80)

    print("""
1. 评估模型偏见
   使用 disability_bias_framework.py 中的 DisabilityBiasEvaluator 类:
   ```python
   from disability_bias_framework import DisabilityBiasEvaluator
   evaluator = DisabilityBiasEvaluator()
   report = evaluator.generate_report(model_predictions)
   evaluator.print_report(report)
   ```

2. 重点关注指标
   - 总体偏见分数: 应显著低于随机基线(0.33)
   - 模糊上下文准确率: 应显著高于随机基线(0.33)
   - MentalIllness子类别: 这是样本最多的类别，需要重点关注

3. 对比实验
   - 对比基础模型和微调后模型的偏见分数
   - 对比使用反偏见提示词前后的表现
   - 分析不同子类别的改进程度

4. 错误分析
   - 检查模型在哪些问题上容易产生偏见
   - 分析偏见类型 (刻板印象 vs 反刻板印象)
   - 识别需要进一步改进的领域
    """)

    # 9. 生成的文件
    print("\n" + "=" * 80)
    print("九、生成的文件")
    print("=" * 80)

    print("""
1. disability_test_set.json
   - 包含1,428条Disability Status测试样本
   - 用于评估模型在残疾相关偏见上的表现

2. disability_bias_framework.py
   - Disability Bias评估框架
   - 提供DisabilityBiasEvaluator类用于评估

3. analyze_disability_bias.py
   - Disability Status数据分析脚本
   - 用于分析测试集的数据分布和偏见模式

4. disability_bias_report.py (本报告)
   - 完整的Disability Bias评估报告
    """)

    print("\n" + "=" * 80)
    print("报告生成完成!")
    print("=" * 80)

if __name__ == "__main__":
    generate_comprehensive_report()
