"""
从完整BBQ数据集中剥离训练集，生成测试集
用于测试模型在未见过数据上的表现
"""

import json

# 路径配置
FULL_DATASET_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\BBQ-main\BBQ-main\data\Combined_full.jsonl"
TRAIN_DATA_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD.xie\apd_results\train_full_results.json"
TEST_OUTPUT_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\BBQ-main\BBQ-main\data\test_set.jsonl"
TEST_JSON_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\BBQ-main\BBQ-main\data\test_set.json"

def load_full_dataset():
    """加载完整的BBQ数据集"""
    print("[1/4] 加载完整数据集...")
    data = []
    with open(FULL_DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"   完整数据集样本数: {len(data)}")
    return data

def load_train_data():
    """加载训练集数据"""
    print("[2/4] 加载训练集数据...")
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        train_results = json.load(f)

    # 提取训练集中的example_id
    train_example_ids = set()
    for item in train_results['results']:
        train_example_ids.add(item['example_id'])

    print(f"   训练集样本数: {len(train_example_ids)}")
    return train_example_ids

def create_test_set(full_data, train_ids):
    """从完整数据集中剥离训练集，生成测试集"""
    print("[3/4] 生成测试集...")

    test_data = []
    overlap_count = 0

    for item in full_data:
        example_id = item.get('example_id')
        if example_id in train_ids:
            overlap_count += 1
        else:
            test_data.append(item)

    print(f"   重叠样本数(将被移除): {overlap_count}")
    print(f"   测试集样本数: {len(test_data)}")

    return test_data

def save_test_set(test_data):
    """保存测试集"""
    print("[4/4] 保存测试集...")

    # 保存为JSONL格式
    with open(TEST_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"   JSONL格式已保存: {TEST_OUTPUT_PATH}")

    # 同时保存为JSON格式（方便查看）
    with open(TEST_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"   JSON格式已保存: {TEST_JSON_PATH}")

def analyze_categories(data, name="数据集"):
    """分析数据集中的类别分布"""
    categories = {}
    for item in data:
        cat = item.get('category', 'Unknown')
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n{name}类别分布:")
    for cat, count in sorted(categories.items()):
        print(f"   {cat}: {count}")
    return categories

def main():
    print("=" * 70)
    print("从完整BBQ数据集中剥离训练集，生成测试集")
    print("=" * 70)

    # 1. 加载完整数据集
    full_data = load_full_dataset()

    # 2. 加载训练集
    train_ids = load_train_data()

    # 3. 生成测试集
    test_data = create_test_set(full_data, train_ids)

    # 4. 保存测试集
    save_test_set(test_data)

    # 5. 统计分析
    print("\n" + "=" * 70)
    print("统计分析")
    print("=" * 70)

    print(f"\n数据集划分:")
    print(f"   完整数据集: {len(full_data)} 条")
    print(f"   训练集: {len(train_ids)} 条")
    print(f"   测试集: {len(test_data)} 条")
    print(f"   验证: {len(full_data)} - {len(train_ids)} = {len(full_data) - len(train_ids)} (测试集)")

    # 分析类别分布
    full_categories = analyze_categories(full_data, "完整数据集")
    test_categories = analyze_categories(test_data, "测试集")

    # 显示一些测试集示例
    print("\n" + "=" * 70)
    print("测试集示例 (前3条)")
    print("=" * 70)
    for i, item in enumerate(test_data[:3], 1):
        print(f"\n示例 {i}:")
        print(f"   example_id: {item.get('example_id')}")
        print(f"   category: {item.get('category')}")
        print(f"   context: {item.get('context', '')[:100]}...")
        print(f"   question: {item.get('question')}")

    print("\n" + "=" * 70)
    print("测试集生成完成!")
    print("=" * 70)
    print(f"\n测试集文件:")
    print(f"   - {TEST_OUTPUT_PATH}")
    print(f"   - {TEST_JSON_PATH}")
    print(f"\n你可以使用测试集来评估模型的泛化能力!")

if __name__ == "__main__":
    main()
