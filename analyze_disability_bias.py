"""
分析Disability Status类别的偏见数据
提取相关测试样本并分析偏见模式
"""

import json
from collections import Counter, defaultdict

TEST_DATA_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\BBQ-main\BBQ-main\data\test_set.json"
TRAIN_DATA_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD.xie\apd_results\train_full_results.json"
DISABILITY_TEST_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\disability_test_set.json"

def extract_disability_samples():
    """提取Disability Status类别的样本"""
    print("=" * 70)
    print("Disability Status 类别分析")
    print("=" * 70)

    # 加载测试集
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 筛选Disability Status类别
    disability_samples = [item for item in test_data if item['category'] == 'Disability_status']

    print(f"\nDisability Status 测试样本数: {len(disability_samples)}")

    # 保存Disability测试集
    with open(DISABILITY_TEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(disability_samples, f, ensure_ascii=False, indent=2)
    print(f"已保存到: {DISABILITY_TEST_PATH}")

    return disability_samples

def analyze_subcategories(disability_samples):
    """分析子类别分布"""
    print("\n" + "=" * 70)
    print("子类别分布 (Subcategory)")
    print("=" * 70)

    subcategories = Counter()
    stereotyped_groups = Counter()

    for item in disability_samples:
        metadata = item.get('additional_metadata', {})
        subcat = metadata.get('subcategory', 'Unknown')
        subcategories[subcat] += 1

        groups = metadata.get('stereotyped_groups', [])
        for group in groups:
            stereotyped_groups[group] += 1

    print(f"\n{'子类别':<30} {'数量':<10} {'占比'}")
    print("-" * 70)
    for subcat, count in sorted(subcategories.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(disability_samples) * 100
        print(f"{subcat:<30} {count:<10} {pct:.1f}%")

    print(f"\n{'刻板印象群体':<30} {'出现次数'}")
    print("-" * 70)
    for group, count in sorted(stereotyped_groups.items(), key=lambda x: x[1], reverse=True):
        print(f"{group:<30} {count}")

def analyze_question_patterns(disability_samples):
    """分析问题模式"""
    print("\n" + "=" * 70)
    print("问题模式分析")
    print("=" * 70)

    # 按question_index分组
    question_groups = defaultdict(list)
    for item in disability_samples:
        q_idx = item.get('question_index', 'unknown')
        question_groups[q_idx].append(item)

    print(f"\n问题数量: {len(question_groups)} 个不同问题")

    # 分析每个问题
    print(f"\n{'问题索引':<10} {'样本数':<10} {'问题内容'}")
    print("-" * 70)
    for q_idx in sorted(question_groups.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        samples = question_groups[q_idx]
        question_text = samples[0]['question']
        print(f"{q_idx:<10} {len(samples):<10} {question_text}")

def analyze_answer_patterns(disability_samples):
    """分析答案模式"""
    print("\n" + "=" * 70)
    print("答案模式分析")
    print("=" * 70)

    # 统计正确答案分布
    label_dist = Counter(item['label'] for item in disability_samples)

    print(f"\n正确答案分布:")
    for label, count in sorted(label_dist.items()):
        pct = count / len(disability_samples) * 100
        print(f"   选项 {label}: {count} ({pct:.1f}%)")

    # 分析答案选项内容
    print(f"\n答案选项分析 (示例):")
    print("-" * 70)

    shown_questions = set()
    for item in disability_samples:
        q = item['question']
        if q not in shown_questions and len(shown_questions) < 5:
            shown_questions.add(q)
            print(f"\n问题: {q}")
            print(f"   ans0: {item['ans0']} (标签: {item['answer_info']['ans0'][1]})")
            print(f"   ans1: {item['ans1']} (标签: {item['answer_info']['ans1'][1]})")
            print(f"   ans2: {item['ans2']} (标签: {item['answer_info']['ans2'][1]})")
            print(f"   正确答案: 选项 {item['label']}")

def analyze_bias_types(disability_samples):
    """分析偏见类型"""
    print("\n" + "=" * 70)
    print("偏见类型分析")
    print("=" * 70)

    # 按question_polarity分析
    polarity_dist = Counter(item['question_polarity'] for item in disability_samples)
    context_dist = Counter(item['context_condition'] for item in disability_samples)

    print(f"\n问题极性分布:")
    for polarity, count in polarity_dist.items():
        pct = count / len(disability_samples) * 100
        desc = "负面/刻板印象问题" if polarity == "neg" else "非负面问题"
        print(f"   {polarity}: {count} ({pct:.1f}%) - {desc}")

    print(f"\n上下文条件分布:")
    for condition, count in context_dist.items():
        pct = count / len(disability_samples) * 100
        desc = "信息不明确" if condition == "ambig" else "信息明确"
        print(f"   {condition}: {count} ({pct:.1f}%) - {desc}")

def show_examples(disability_samples):
    """显示典型示例"""
    print("\n" + "=" * 70)
    print("典型示例")
    print("=" * 70)

    # 按子类别选择示例
    subcat_examples = defaultdict(list)
    for item in disability_samples:
        metadata = item.get('additional_metadata', {})
        subcat = metadata.get('subcategory', 'Unknown')
        if len(subcat_examples[subcat]) < 2:
            subcat_examples[subcat].append(item)

    for subcat, examples in list(subcat_examples.items())[:5]:
        print(f"\n【子类别: {subcat}】")
        for i, item in enumerate(examples, 1):
            print(f"\n  示例 {i}:")
            print(f"     Context: {item['context'][:100]}...")
            print(f"     Question: {item['question']}")
            print(f"     Polarity: {item['question_polarity']}")
            print(f"     Condition: {item['context_condition']}")
            label = item['label']
            ans_key = f'ans{label}'
            print(f"     正确答案: {item[ans_key]}")

def compare_with_train():
    """对比训练集和测试集的Disability样本"""
    print("\n" + "=" * 70)
    print("训练集 vs 测试集对比")
    print("=" * 70)

    # 加载训练集
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
        train_results = train_data['results']

    # 加载测试集
    with open(DISABILITY_TEST_PATH, 'r', encoding='utf-8') as f:
        test_samples = json.load(f)

    train_disability = [item for item in train_results if item['category'] == 'Disability_status']

    print(f"\n样本数量对比:")
    print(f"   训练集 Disability 样本: {len(train_disability)}")
    print(f"   测试集 Disability 样本: {len(test_samples)}")

    # 检查是否有重叠
    train_ids = set(item['example_id'] for item in train_disability)
    test_ids = set(item['example_id'] for item in test_samples)
    overlap = train_ids & test_ids

    if overlap:
        print(f"\n[!] 警告: 发现 {len(overlap)} 个重叠样本")
    else:
        print(f"\n[OK] 无重叠，训练集和测试集完全分离")

def main():
    # 1. 提取Disability样本
    disability_samples = extract_disability_samples()

    # 2. 分析子类别
    analyze_subcategories(disability_samples)

    # 3. 分析问题模式
    analyze_question_patterns(disability_samples)

    # 4. 分析答案模式
    analyze_answer_patterns(disability_samples)

    # 5. 分析偏见类型
    analyze_bias_types(disability_samples)

    # 6. 显示示例
    show_examples(disability_samples)

    # 7. 对比训练集
    compare_with_train()

    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)
    print(f"\nDisability测试集已保存: {DISABILITY_TEST_PATH}")
    print(f"样本数: {len(disability_samples)} 条")

if __name__ == "__main__":
    main()
