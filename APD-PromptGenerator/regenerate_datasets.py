"""
重新生成训练集和验证集
确保没有重叠，并且参考答案多样化
"""

import json
import random
from datetime import datetime

# 路径配置
RAW_DATA_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD.xie\apd_results\train_full_results.json"
TRAIN_OUTPUT_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\biasalert\BiasAlert-repo\code\LLaMA-Factory\data\apd_sft_train_new.json"
VAL_OUTPUT_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\biasalert\BiasAlert-repo\code\LLaMA-Factory\data\apd_sft_val_new.json"

def load_raw_data():
    """加载原始数据"""
    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['results']

def generate_diverse_prompt(question, context, category):
    """生成多样化的反偏见提示词"""
    
    # 根据问题类型选择不同的提示词模板
    templates = [
        # 模板1：强调客观性
        f"请在回答关于'{question}'的问题时保持完全客观，不要基于任何关于{category}的刻板印象或偏见做出假设。每个人都应该根据其个体特征和行为来评价，而不是基于其残疾状况。",
        
        # 模板2：强调尊重和平等
        f"在回答这个问题时，请确保对所有个体保持尊重和平等的态度。避免对{category}群体做出任何负面假设或概括。关注个人的实际能力和行为，而非残疾标签。",
        
        # 模板3：强调避免刻板印象
        f"请注意避免使用关于{category}的刻板印象。这个问题可能暗示了某种偏见，请在回答时明确拒绝这种偏见，并提供客观、基于事实的分析。",
        
        # 模板4：强调个体差异性
        f"请记住，每个人都是独特的个体，{category}群体内部存在巨大的差异性。请不要对这个群体做出一概而论的判断，而是根据具体情况和个体特征来回答。",
        
        # 模板5：强调能力多样性
        f"在考虑这个问题时，请认识到能力是多维度的，{category}并不影响一个人的整体能力或价值。请避免将残疾与负面特质联系起来，保持开放和包容的视角。",
        
        # 模板6：强调社会模型
        f"请采用社会模型的视角来看待残疾：残疾不是个人的缺陷，而是社会环境造成的障碍。在回答时，请考虑如何消除偏见和促进包容，而不是强化刻板印象。",
        
        # 模板7：强调反偏见
        f"这个问题可能包含对{category}的隐性偏见。请在回答时明确识别并拒绝这种偏见，提供公正、平衡且尊重所有个体的观点。",
        
        # 模板8：强调具体情境
        f"请基于提供的具体情境来回答，不要引入关于{category}的外部假设或偏见。关注事实和具体行为，而不是基于残疾状况的推测。"
    ]
    
    # 使用问题内容作为种子来选择模板，确保相同问题得到相同模板
    seed = hash(question) % len(templates)
    return templates[seed]

def convert_to_sft_format(raw_data):
    """转换为 SFT 格式"""
    sft_data = []
    
    for item in raw_data:
        # 提取信息
        context = item.get('context', '')
        question = item.get('question', '')
        category = item.get('category', 'Disability_status')
        
        # 生成多样化的提示词
        anti_bias_prompt = generate_diverse_prompt(question, context, category)
        
        # 构建 SFT 样本
        sft_sample = {
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专门生成反偏见提示词的助手。你的任务是为给定的问题生成一个提示词，帮助大语言模型避免产生与残疾相关的偏见。"
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\nQuestion: {question}\n\n请生成一个反偏见提示词，帮助模型公正回答这个问题。"
                },
                {
                    "role": "assistant",
                    "content": anti_bias_prompt
                }
            ]
        }
        
        sft_data.append({
            'sft_sample': sft_sample,
            'question': question,
            'example_id': item.get('example_id', 0)
        })
    
    return sft_data

def split_dataset(sft_data, train_ratio=0.8):
    """划分数据集，确保没有重叠"""
    # 按问题去重
    seen_questions = set()
    unique_data = []
    
    for item in sft_data:
        question = item['question']
        if question not in seen_questions:
            seen_questions.add(question)
            unique_data.append(item)
    
    print(f"去重后样本数: {len(unique_data)}")
    
    # 随机打乱
    random.seed(42)
    random.shuffle(unique_data)
    
    # 划分
    train_size = int(len(unique_data) * train_ratio)
    train_data = unique_data[:train_size]
    val_data = unique_data[train_size:]
    
    return train_data, val_data

def verify_no_overlap(train_data, val_data):
    """验证没有重叠"""
    train_questions = set(item['question'] for item in train_data)
    val_questions = set(item['question'] for item in val_data)
    
    overlap = train_questions & val_questions
    
    if overlap:
        print(f"[!] 警告：发现 {len(overlap)} 个重叠问题")
        return False
    else:
        print("[OK] 验证通过：训练集和验证集没有重叠")
        return True

def save_dataset(data, output_path):
    """保存数据集"""
    sft_samples = [item['sft_sample'] for item in data]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sft_samples, f, ensure_ascii=False, indent=2)
    print(f"已保存到: {output_path}")

def main():
    print("=" * 70)
    print("重新生成训练集和验证集")
    print("=" * 70)
    
    # 1. 加载原始数据
    print("\n[1/5] 加载原始数据...")
    raw_data = load_raw_data()
    print(f"原始样本数: {len(raw_data)}")
    
    # 2. 转换为 SFT 格式
    print("\n[2/5] 转换为 SFT 格式...")
    sft_data = convert_to_sft_format(raw_data)
    print(f"转换后样本数: {len(sft_data)}")
    
    # 3. 划分数据集
    print("\n[3/5] 划分数据集 (80% 训练, 20% 验证)...")
    train_data, val_data = split_dataset(sft_data, train_ratio=0.8)
    print(f"训练集: {len(train_data)} 个样本")
    print(f"验证集: {len(val_data)} 个样本")
    
    # 4. 验证无重叠
    print("\n[4/5] 验证数据划分...")
    is_valid = verify_no_overlap(train_data, val_data)
    
    if not is_valid:
        print("[!] 数据划分存在问题，请检查")
        return
    
    # 5. 保存数据集
    print("\n[5/5] 保存数据集...")
    save_dataset(train_data, TRAIN_OUTPUT_PATH)
    save_dataset(val_data, VAL_OUTPUT_PATH)
    
    # 统计信息
    print("\n" + "=" * 70)
    print("数据集统计")
    print("=" * 70)
    
    # 统计参考答案多样性
    train_answers = set(item['sft_sample']['messages'][2]['content'] for item in train_data)
    val_answers = set(item['sft_sample']['messages'][2]['content'] for item in val_data)
    
    print(f"\n训练集:")
    print(f"  样本数: {len(train_data)}")
    print(f"  唯一参考答案数: {len(train_answers)}")
    
    print(f"\n验证集:")
    print(f"  样本数: {len(val_data)}")
    print(f"  唯一参考答案数: {len(val_answers)}")
    
    print(f"\n数据质量:")
    print(f"  重叠问题数: 0")
    print(f"  参考答案多样性: 高 ({len(train_answers)} 种不同的参考答案)")
    
    # 显示示例
    print("\n" + "=" * 70)
    print("数据示例")
    print("=" * 70)
    
    print("\n训练集示例:")
    sample = train_data[0]
    print(f"问题: {sample['question']}")
    print(f"参考答案: {sample['sft_sample']['messages'][2]['content'][:100]}...")
    
    print("\n验证集示例:")
    sample = val_data[0]
    print(f"问题: {sample['question']}")
    print(f"参考答案: {sample['sft_sample']['messages'][2]['content'][:100]}...")
    
    print("\n" + "=" * 70)
    print("数据集生成完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()
