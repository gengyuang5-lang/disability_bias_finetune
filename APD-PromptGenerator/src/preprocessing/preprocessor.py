"""
数据预处理模块
将收集的数据转换为训练格式
"""

import json
import os
import random
from collections import defaultdict

def load_iteration_data(data_dir):
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and filename.startswith('iteration_'):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.append(data)
    return sorted(all_data, key=lambda x: x['iteration'])

def convert_to_sft_format(iteration_data, effective_only=False):
    sft_data = []
    for item in iteration_data:
        if effective_only and not item['is_effective']:
            continue

        question = item['samples'][0]['context'] if item['samples'] else ""
        q_text = item['samples'][0]['question'] if item['samples'] else ""

        sft_data.append({
            "messages": [
                {
                    "role": "user",
                    "content": f"问题：{q_text}\n语境：{question}"
                },
                {
                    "role": "assistant",
                    "content": item['instruction']
                }
            ],
            "bias_score": item['bias_score'],
            "is_effective": item['is_effective'],
            "iteration": item['iteration']
        })
    return sft_data

def convert_to_preference_format(iteration_data):
    preference_data = []
    grouped_by_iteration = defaultdict(list)

    for item in iteration_data:
        grouped_by_iteration[item['iteration']].append(item)

    iterations = sorted(grouped_by_iteration.keys())

    for i in range(len(iterations) - 1):
        curr_iter = iterations[i]
        next_iter = iterations[i + 1]

        curr_items = grouped_by_iteration[curr_iter]
        next_items = grouped_by_iteration[next_iter]

        if not curr_items or not next_items:
            continue

        if curr_items[0]['bias_score'] > next_items[0]['bias_score']:
            chosen = next_items[0]
            rejected = curr_items[0]
        else:
            chosen = curr_items[0]
            rejected = next_items[0]

        question = chosen['samples'][0]['context'] if chosen['samples'] else ""
        q_text = chosen['samples'][0]['question'] if chosen['samples'] else ""

        preference_data.append({
            "prompt": f"问题：{q_text}\n语境：{question}",
            "chosen": chosen['instruction'],
            "rejected": rejected['instruction'],
            "chosen_score": chosen['bias_score'],
            "rejected_score": rejected['bias_score'],
            "improvement": rejected['bias_score'] - chosen['bias_score']
        })

    return preference_data

def create_negative_samples(iteration_data, min_score_threshold=0.3):
    negative_data = []
    for item in iteration_data:
        if item['bias_score'] >= min_score_threshold:
            question = item['samples'][0]['context'] if item['samples'] else ""
            q_text = item['samples'][0]['question'] if item['samples'] else ""

            negative_data.append({
                "prompt": f"问题：{q_text}\n语境：{question}",
                "instruction": item['instruction'],
                "bias_score": item['bias_score'],
                "is_effective": False
            })
    return negative_data

def split_dataset(data, train_ratio=0.8):
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]

def save_processed_data(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"数据已保存到: {output_path}")

def main():
    from configs.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_PREFERENCE_DIR

    print("加载迭代数据...")
    iteration_data = load_iteration_data(DATA_RAW_DIR)
    print(f"共加载 {len(iteration_data)} 个迭代数据")

    print("\n转换为SFT格式...")
    sft_data = convert_to_sft_format(iteration_data)
    print(f"SFT数据量: {len(sft_data)} 条")

    print("\n转换为偏好数据格式...")
    preference_data = convert_to_preference_format(iteration_data)
    print(f"偏好数据对: {len(preference_data)} 对")

    print("\n创建负样本...")
    negative_data = create_negative_samples(iteration_data)
    print(f"高偏见样本: {len(negative_data)} 条")

    train_sft, val_sft = split_dataset(sft_data)
    print(f"\nSFT训练集: {len(train_sft)} 条, 验证集: {len(val_sft)} 条")

    train_pref, val_pref = split_dataset(preference_data)
    print(f"偏好训练集: {len(train_pref)} 条, 验证集: {len(val_pref)} 条")

    save_processed_data(train_sft, os.path.join(DATA_PROCESSED_DIR, "sft_train.json"))
    save_processed_data(val_sft, os.path.join(DATA_PROCESSED_DIR, "sft_val.json"))
    save_processed_data(train_pref, os.path.join(DATA_PREFERENCE_DIR, "preference_train.json"))
    save_processed_data(val_pref, os.path.join(DATA_PREFERENCE_DIR, "preference_val.json"))

    print("\n数据预处理完成!")

if __name__ == "__main__":
    main()
