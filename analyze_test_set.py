"""
分析测试集的统计信息
"""

import json
from collections import Counter

TEST_DATA_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\BBQ-main\BBQ-main\data\test_set.json"
TRAIN_DATA_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD.xie\apd_results\train_full_results.json"

def main():
    print("=" * 70)
    print("测试集统计分析")
    print("=" * 70)

    # 加载测试集
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 加载训练集
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
        train_results = train_data['results']

    print(f"\n数据集规模:")
    print(f"   测试集: {len(test_data)} 条")
    print(f"   训练集: {len(train_results)} 条")

    # 分析测试集的类别分布
    categories = Counter(item['category'] for item in test_data)

    print(f"\n测试集类别分布:")
    print(f"   {'类别':<25} {'数量':<10} {'占比':<10}")
    print("   " + "-" * 45)
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(test_data) * 100
        print(f"   {cat:<25} {count:<10} {pct:.1f}%")

    # 分析question_polarity分布
    polarity_dist = Counter(item['question_polarity'] for item in test_data)
    print(f"\n问题极性分布:")
    for polarity, count in polarity_dist.items():
        pct = count / len(test_data) * 100
        print(f"   {polarity}: {count} ({pct:.1f}%)")

    # 分析context_condition分布
    context_dist = Counter(item['context_condition'] for item in test_data)
    print(f"\n上下文条件分布:")
    for condition, count in context_dist.items():
        pct = count / len(test_data) * 100
        print(f"   {condition}: {count} ({pct:.1f}%)")

    # 分析标签分布
    label_dist = Counter(item['label'] for item in test_data)
    print(f"\n标签分布:")
    for label, count in label_dist.items():
        pct = count / len(test_data) * 100
        print(f"   标签 {label}: {count} ({pct:.1f}%)")

    # 显示一些示例
    print("\n" + "=" * 70)
    print("测试集示例 (每个类别各一条)")
    print("=" * 70)

    shown_categories = set()
    for item in test_data:
        cat = item['category']
        if cat not in shown_categories:
            shown_categories.add(cat)
            print(f"\n类别: {cat}")
            print(f"   example_id: {item['example_id']}")
            print(f"   context: {item['context'][:80]}...")
            print(f"   question: {item['question']}")
            label = item['label']
            ans_key = f'ans{label}'
            print(f"   正确答案 (label): {label} - {item[ans_key]}")
            if len(shown_categories) >= 5:  # 只显示5个类别
                break

    # 验证训练集和测试集没有重叠
    print("\n" + "=" * 70)
    print("验证训练集和测试集没有重叠")
    print("=" * 70)

    train_ids = set(item['example_id'] for item in train_results)
    test_ids = set(item['example_id'] for item in test_data)

    overlap = train_ids & test_ids

    if overlap:
        print(f"\n[!] 警告: 发现 {len(overlap)} 个重叠的example_id!")
    else:
        print(f"\n[OK] 验证通过: 训练集和测试集没有重叠")
        print(f"   训练集example_id数量: {len(train_ids)}")
        print(f"   测试集example_id数量: {len(test_ids)}")

    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()
