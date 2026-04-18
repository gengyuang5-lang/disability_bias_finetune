import json

# 加载测试数据
test_data_path = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\disability_test_set.json"
with open(test_data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 60)
print("测试数据集信息")
print("=" * 60)
print(f"总样本数: {len(data)}")

if len(data) > 0:
    print(f"\n第一条样本信息:")
    print(f"  - 问题: {data[0]['question']}")
    print(f"  - 类别: {data[0]['category']}")
    print(f"  - 子类别: {data[0].get('additional_metadata', {}).get('subcategory', 'N/A')}")
    print(f"  - 上下文条件: {data[0]['context_condition']}")
    print(f"  - 正确答案: {data[0]['label']}")

print("\n" + "=" * 60)
print("数据集已准备好，可以开始实验！")
print("=" * 60)
