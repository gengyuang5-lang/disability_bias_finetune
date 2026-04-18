"""
检查训练集和验证集是否有重叠
"""

import json

# 路径配置
TRAIN_DATA_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\biasalert\BiasAlert-repo\code\LLaMA-Factory\data\apd_sft_train.json"
VAL_DATA_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\biasalert\BiasAlert-repo\code\LLaMA-Factory\data\apd_sft_val.json"

def load_data(path):
    """加载数据"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_question(sample):
    """提取问题内容"""
    messages = sample['messages']
    user_content = messages[1]['content']
    # 提取 Question 部分
    import re
    match = re.search(r'Question: (.+?)\n\n', user_content)
    if match:
        return match.group(1).strip()
    return user_content[:100]

def check_overlap():
    print("=" * 70)
    print("检查训练集和验证集重叠情况")
    print("=" * 70)
    
    # 加载数据
    print("\n[1/3] 加载数据...")
    train_data = load_data(TRAIN_DATA_PATH)
    val_data = load_data(VAL_DATA_PATH)
    
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(val_data)}")
    
    # 提取问题
    print("\n[2/3] 提取问题...")
    train_questions = set(extract_question(sample) for sample in train_data)
    val_questions = set(extract_question(sample) for sample in val_data)
    
    print(f"训练集唯一问题数: {len(train_questions)}")
    print(f"验证集唯一问题数: {len(val_questions)}")
    
    # 检查重叠
    print("\n[3/3] 检查重叠...")
    overlap = train_questions & val_questions
    
    if overlap:
        print(f"\n[!] 发现 {len(overlap)} 个重叠问题！")
        print("\n重叠的问题：")
        for i, q in enumerate(list(overlap)[:5], 1):
            print(f"  {i}. {q}")
        if len(overlap) > 5:
            print(f"  ... 还有 {len(overlap) - 5} 个")
    else:
        print("\n[OK] 没有发现重叠！训练集和验证集是完全分开的。")
    
    # 显示一些示例
    print("\n" + "=" * 70)
    print("数据示例")
    print("=" * 70)
    
    print("\n训练集前3个样本的问题：")
    for i, sample in enumerate(train_data[:3], 1):
        q = extract_question(sample)
        print(f"  {i}. {q}")
    
    print("\n验证集前3个样本的问题：")
    for i, sample in enumerate(val_data[:3], 1):
        q = extract_question(sample)
        print(f"  {i}. {q}")
    
    # 参考答案分析
    print("\n" + "=" * 70)
    print("参考答案分析")
    print("=" * 70)
    
    train_answers = set(sample['messages'][2]['content'] for sample in train_data)
    val_answers = set(sample['messages'][2]['content'] for sample in val_data)
    
    print(f"\n训练集唯一参考答案数: {len(train_answers)}")
    print(f"验证集唯一参考答案数: {len(val_answers)}")
    
    print("\n训练集参考答案示例：")
    for i, ans in enumerate(list(train_answers)[:3], 1):
        print(f"  {i}. {ans[:80]}...")
    
    print("\n验证集参考答案示例：")
    for i, ans in enumerate(list(val_answers)[:3], 1):
        print(f"  {i}. {ans[:80]}...")
    
    print("\n" + "=" * 70)
    print("检查完成!")
    print("=" * 70)

if __name__ == "__main__":
    check_overlap()
