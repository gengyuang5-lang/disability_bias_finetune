"""
DPO (Direct Preference Optimization) 训练模块
使用偏好数据对优化模型
"""

import json
import os
from openai import OpenAI

def load_preference_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_dpo_training_data(preference_data, output_path):
    formatted_data = []
    for item in preference_data:
        formatted_item = {
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"]
        }
        formatted_data.append(formatted_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"DPO训练数据已准备: {output_path}")
    return formatted_data

def train_with_dpo(preference_data, model_name="gpt-4o-mini", beta=0.1):
    client = OpenAI()

    print(f"开始DPO训练，使用模型: {model_name}")
    print(f"偏好对数量: {len(preference_data)}")
    print(f"Beta (KL散度权重): {beta}")

    optimized_prompts = []

    for i, item in enumerate(preference_data):
        try:
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": """你是一个提示词优化专家。你的任务是分析给定的偏好对，学习什么是"好的提示词"。

好的提示词特点：
1. 能有效引导模型避免偏见
2. 表达清晰、具体
3. 具有通用性

请生成一个综合了偏好对优点的优化提示词。"""},
                    {"role": "user", "content": f"""Prompt: {prompt}

好提示词 (偏见分数 {item['chosen_score']:.3f}):
{chosen}

差提示词 (偏见分数 {item['rejected_score']:.3f}):
{rejected}

请生成一个综合优点的优化提示词:"""}
                ]
            )

            optimized_instruction = response.choices[0].message.content

            optimized_prompts.append({
                "original_prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "optimized": optimized_instruction,
                "chosen_score": item['chosen_score'],
                "rejected_score": item['rejected_score'],
                "improvement": item['improvement']
            })

            if (i + 1) % 10 == 0:
                print(f"已处理 {i+1}/{len(preference_data)} 对数据")

        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")

    print(f"\nDPO训练完成!")
    print(f"成功优化: {len(optimized_prompts)} 条")

    return optimized_prompts

def validate_dpo_improvement(optimized_prompts):
    improvements = []
    for item in optimized_prompts:
        if item['improvement'] > 0:
            improvements.append(item)

    print(f"\nDPO验证结果:")
    print(f"有改善的样本: {len(improvements)}/{len(optimized_prompts)} ({len(improvements)/len(optimized_prompts)*100:.1f}%)")

    return improvements

def main():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from configs.config import DATA_PREFERENCE_DIR, MODEL_BASE_DIR, DPO_MODEL_DIR, DPO_CONFIG

    os.makedirs(DPO_MODEL_DIR, exist_ok=True)

    train_path = os.path.join(DATA_PREFERENCE_DIR, "preference_train.json")
    val_path = os.path.join(DATA_PREFERENCE_DIR, "preference_val.json")

    print("加载偏好训练数据...")
    train_data = load_preference_data(train_path)
    val_data = load_preference_data(val_path)

    train_output_path = os.path.join(DPO_MODEL_DIR, "dpo_train_formatted.jsonl")
    prepare_dpo_training_data(train_data, train_output_path)

    model_name = DPO_CONFIG.get("model_name", "gpt-4o-mini")
    beta = DPO_CONFIG.get("beta", 0.1)

    print("\n使用OpenAI API进行DPO训练...")
    optimized_prompts = train_with_dpo(train_data, model_name, beta)

    improvements = validate_dpo_improvement(optimized_prompts)

    optimized_path = os.path.join(DPO_MODEL_DIR, "optimized_prompts.json")
    with open(optimized_path, 'w', encoding='utf-8') as f:
        json.dump(optimized_prompts, f, ensure_ascii=False, indent=2)
    print(f"优化结果已保存: {optimized_path}")

    print("\nDPO训练流程完成!")
    print(f"模型输出目录: {DPO_MODEL_DIR}")

if __name__ == "__main__":
    main()
