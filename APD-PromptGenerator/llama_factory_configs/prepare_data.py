"""
数据转换脚本
将 APD-PromptGenerator 数据转换为 LLaMA-Factory 格式
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.config import DATA_PROCESSED_DIR, DATA_PREFERENCE_DIR

def convert_to_sharegpt_format(sft_data):
    """转换为 ShareGPT 格式"""
    sharegpt_data = []
    for item in sft_data:
        messages = []
        for msg in item["messages"]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        sharegpt_data.append({"messages": messages})
    return sharegpt_data

def convert_to_dpo_format(preference_data):
    """转换为 DPO 格式"""
    dpo_data = []
    for item in preference_data:
        dpo_data.append({
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"]
        })
    return dpo_data

def main():
    # LLaMA-Factory 数据目录
    llama_factory_data_dir = "E:/desktop/CDS546 Project Topics (AY2025-26)(2)/biasalert/BiasAlert-repo/code/LLaMA-Factory/data"
    
    print("=" * 60)
    print("准备 LLaMA-Factory 训练数据")
    print("=" * 60)
    
    # 1. 转换 SFT 数据
    print("\n1. 转换 SFT 数据...")
    sft_train_path = os.path.join(DATA_PROCESSED_DIR, "sft_train.json")
    sft_val_path = os.path.join(DATA_PROCESSED_DIR, "sft_val.json")
    
    if os.path.exists(sft_train_path):
        with open(sft_train_path, 'r', encoding='utf-8') as f:
            sft_train = json.load(f)
        sft_train_sharegpt = convert_to_sharegpt_format(sft_train)
        
        output_path = os.path.join(llama_factory_data_dir, "apd_sft_data.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sft_train_sharegpt, f, ensure_ascii=False, indent=2)
        print(f"   SFT 训练数据已保存: {output_path}")
        print(f"   样本数: {len(sft_train_sharegpt)}")
    else:
        print(f"   警告: 未找到 {sft_train_path}")
        print("   请先运行数据收集和预处理脚本")
    
    # 2. 转换 DPO 数据
    print("\n2. 转换 DPO 数据...")
    dpo_train_path = os.path.join(DATA_PREFERENCE_DIR, "preference_train.json")
    
    if os.path.exists(dpo_train_path):
        with open(dpo_train_path, 'r', encoding='utf-8') as f:
            dpo_train = json.load(f)
        dpo_data = convert_to_dpo_format(dpo_train)
        
        output_path = os.path.join(llama_factory_data_dir, "apd_dpo_data.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dpo_data, f, ensure_ascii=False, indent=2)
        print(f"   DPO 训练数据已保存: {output_path}")
        print(f"   样本数: {len(dpo_data)}")
    else:
        print(f"   警告: 未找到 {dpo_train_path}")
        print("   请先运行数据收集和预处理脚本")
    
    # 3. 复制 dataset_info.json
    print("\n3. 复制 dataset_info.json...")
    dataset_info_src = os.path.join(os.path.dirname(__file__), "dataset_info.json")
    dataset_info_dst = os.path.join(llama_factory_data_dir, "dataset_info.json")
    
    if os.path.exists(dataset_info_src):
        import shutil
        shutil.copy2(dataset_info_src, dataset_info_dst)
        print(f"   dataset_info.json 已复制到: {dataset_info_dst}")
    
    print("\n" + "=" * 60)
    print("数据准备完成!")
    print("=" * 60)
    print("\n下一步:")
    print("1. 运行 SFT 训练: bash llama_factory_configs/sft_qlora.sh")
    print("2. 运行 DPO 训练: bash llama_factory_configs/dpo_qlora.sh")

if __name__ == "__main__":
    main()
