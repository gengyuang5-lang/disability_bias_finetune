"""
SFT (监督微调) 训练模块
使用有效提示词数据微调模型
"""

import json
import os
from openai import OpenAI

def load_sft_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_training_data(sft_data, output_path):
    formatted_data = []
    for item in sft_data:
        formatted_item = {
            "messages": item["messages"]
        }
        formatted_data.append(formatted_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"训练数据已准备: {output_path}")
    return formatted_data

def train_with_openai(train_data, model_name="gpt-4o-mini"):
    client = OpenAI()

    print(f"开始SFT训练，使用模型: {model_name}")
    print(f"训练样本数: {len(train_data)}")

    successful_examples = []
    failed_examples = []

    for i, item in enumerate(train_data):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "你是一个提示词优化专家，擅长生成能减少AI模型偏见的提示词。"},
                    item["messages"]
                ]
            )

            generated_instruction = response.choices[0].message.content

            successful_examples.append({
                "input": item["messages"][0]["content"],
                "output": generated_instruction,
                "reference": item["messages"][1]["content"],
                "bias_score": item.get("bias_score", None)
            })

            if (i + 1) % 10 == 0:
                print(f"已处理 {i+1}/{len(train_data)} 条数据")

        except Exception as e:
            failed_examples.append({
                "item": item,
                "error": str(e)
            })
            print(f"处理样本 {i} 时出错: {e}")

    print(f"\n训练完成!")
    print(f"成功: {len(successful_examples)} 条")
    print(f"失败: {len(failed_examples)} 条")

    return successful_examples, failed_examples

def create_few_shot_examples(sft_data, n=5):
    effective_samples = [s for s in sft_data if s.get("is_effective", False)]
    random.shuffle(effective_samples)
    return effective_samples[:n]

def generate_with_few_shot(client, question, few_shot_examples, model_name):
    messages = [
        {"role": "system", "content": "你是一个提示词优化专家，擅长生成能减少AI模型偏见的提示词。"}
    ]

    for example in few_shot_examples:
        messages.append({"role": "user", "content": example["messages"][0]["content"]})
        messages.append({"role": "assistant", "content": example["messages"][1]["content"]})

    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages
    )

    return response.choices[0].message.content

def main():
    import sys
    import random
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from configs.config import DATA_PROCESSED_DIR, MODEL_BASE_DIR, SFT_MODEL_DIR, SFT_CONFIG

    os.makedirs(SFT_MODEL_DIR, exist_ok=True)

    train_path = os.path.join(DATA_PROCESSED_DIR, "sft_train.json")
    val_path = os.path.join(DATA_PROCESSED_DIR, "sft_val.json")

    print("加载训练数据...")
    train_data = load_sft_data(train_path)
    val_data = load_sft_data(val_path)

    train_output_path = os.path.join(SFT_MODEL_DIR, "train_formatted.jsonl")
    prepare_training_data(train_data, train_output_path)

    model_name = SFT_CONFIG.get("model_name", "gpt-4o-mini")

    print("\n使用OpenAI API进行SFT风格的训练...")
    successful, failed = train_with_openai(train_data, model_name)

    successful_path = os.path.join(SFT_MODEL_DIR, "successful_examples.json")
    with open(successful_path, 'w', encoding='utf-8') as f:
        json.dump(successful, f, ensure_ascii=False, indent=2)
    print(f"成功案例已保存: {successful_path}")

    print("\nSFT训练流程完成!")
    print(f"模型输出目录: {SFT_MODEL_DIR}")

if __name__ == "__main__":
    main()
