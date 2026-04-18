"""
APD 数据收集完成后自动开始训练
"""

import json
import os
import shutil
import subprocess
import time

# 路径配置
APD_OUTPUT_DIR = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD.xie\apd_results"
LLAMA_FACTORY_DIR = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\biasalert\BiasAlert-repo\code\LLaMA-Factory"
PROJECT_DIR = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD-PromptGenerator"

def wait_for_apd_completion():
    """等待 APD 完成"""
    print("=" * 60)
    print("等待 APD 数据收集完成...")
    print("=" * 60)
    
    sft_file = os.path.join(APD_OUTPUT_DIR, "sft_data.json")
    dpo_file = os.path.join(APD_OUTPUT_DIR, "dpo_data.json")
    
    while not (os.path.exists(sft_file) and os.path.exists(dpo_file)):
        print("等待 APD 完成... (检查 sft_data.json 和 dpo_data.json)")
        time.sleep(30)  # 每 30 秒检查一次
    
    print("\n✓ APD 数据收集完成!")
    return sft_file, dpo_file

def split_train_val(data, train_ratio=0.8):
    """划分训练集和验证集"""
    import random
    random.seed(42)
    
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    train_size = int(len(data_copy) * train_ratio)
    train_data = data_copy[:train_size]
    val_data = data_copy[train_size:]
    
    return train_data, val_data

def prepare_data_for_llama_factory():
    """准备 LLaMA-Factory 格式的数据"""
    print("\n" + "=" * 60)
    print("准备 LLaMA-Factory 训练数据")
    print("=" * 60)
    
    # 1. 读取 SFT 数据
    with open(os.path.join(APD_OUTPUT_DIR, "sft_data.json"), 'r', encoding='utf-8') as f:
        sft_data = json.load(f)
    
    # 2. 划分训练集和验证集 (80/20)
    sft_train, sft_val = split_train_val(sft_data, train_ratio=0.8)
    
    print(f"SFT 数据: 训练集 {len(sft_train)} 条, 验证集 {len(sft_val)} 条")
    
    # 3. 转换为 ShareGPT 格式
    def convert_to_sharegpt(data):
        return [{"messages": item["messages"]} for item in data]
    
    sft_train_sharegpt = convert_to_sharegpt(sft_train)
    sft_val_sharegpt = convert_to_sharegpt(sft_val)
    
    # 4. 保存到 LLaMA-Factory 数据目录
    llama_factory_data_dir = os.path.join(LLAMA_FACTORY_DIR, "data")
    
    with open(os.path.join(llama_factory_data_dir, "apd_sft_train.json"), 'w', encoding='utf-8') as f:
        json.dump(sft_train_sharegpt, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(llama_factory_data_dir, "apd_sft_val.json"), 'w', encoding='utf-8') as f:
        json.dump(sft_val_sharegpt, f, ensure_ascii=False, indent=2)
    
    # 5. 处理 DPO 数据
    with open(os.path.join(APD_OUTPUT_DIR, "dpo_data.json"), 'r', encoding='utf-8') as f:
        dpo_data = json.load(f)
    
    dpo_train, dpo_val = split_train_val(dpo_data, train_ratio=0.8)
    
    print(f"DPO 数据: 训练集 {len(dpo_train)} 条, 验证集 {len(dpo_val)} 条")
    
    with open(os.path.join(llama_factory_data_dir, "apd_dpo_train.json"), 'w', encoding='utf-8') as f:
        json.dump(dpo_train, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(llama_factory_data_dir, "apd_dpo_val.json"), 'w', encoding='utf-8') as f:
        json.dump(dpo_val, f, ensure_ascii=False, indent=2)
    
    # 6. 更新 dataset_info.json
    dataset_info = {
        "apd_sft_train": {
            "file_name": "apd_sft_train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        },
        "apd_sft_val": {
            "file_name": "apd_sft_val.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        },
        "apd_dpo_train": {
            "file_name": "apd_dpo_train.json",
            "formatting": "pairwise",
            "ranking": True,
            "columns": {
                "prompt": "prompt",
                "chosen": "chosen",
                "rejected": "rejected"
            }
        },
        "apd_dpo_val": {
            "file_name": "apd_dpo_val.json",
            "formatting": "pairwise",
            "ranking": True,
            "columns": {
                "prompt": "prompt",
                "chosen": "chosen",
                "rejected": "rejected"
            }
        }
    }
    
    with open(os.path.join(llama_factory_data_dir, "dataset_info.json"), 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    print("\n✓ 数据准备完成!")
    return len(sft_train), len(sft_val), len(dpo_train), len(dpo_val)

def update_training_scripts():
    """更新训练脚本使用正确的数据集"""
    print("\n" + "=" * 60)
    print("更新训练脚本")
    print("=" * 60)
    
    # 更新 SFT 脚本
    sft_script_path = os.path.join(PROJECT_DIR, "llama_factory_configs", "sft_qlora.sh")
    with open(sft_script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换数据集名称
    content = content.replace('--dataset apd_prompt_sft', '--dataset apd_sft_train')
    
    with open(sft_script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 更新 DPO 脚本
    dpo_script_path = os.path.join(PROJECT_DIR, "llama_factory_configs", "dpo_qlora.sh")
    with open(dpo_script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = content.replace('--dataset apd_prompt_dpo', '--dataset apd_dpo_train')
    
    with open(dpo_script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ 训练脚本已更新")

def run_training():
    """运行训练"""
    print("\n" + "=" * 60)
    print("开始 LLaMA-Factory 训练")
    print("=" * 60)
    
    # 1. SFT 训练
    print("\n[1/2] SFT QLoRA 训练...")
    sft_script = os.path.join(PROJECT_DIR, "llama_factory_configs", "sft_qlora.sh")
    
    result = subprocess.run(
        ["bash", sft_script],
        cwd=LLAMA_FACTORY_DIR,
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print("SFT 训练失败!")
        return False
    
    print("✓ SFT 训练完成!")
    
    # 2. DPO 训练
    print("\n[2/2] DPO QLoRA 训练...")
    dpo_script = os.path.join(PROJECT_DIR, "llama_factory_configs", "dpo_qlora.sh")
    
    result = subprocess.run(
        ["bash", dpo_script],
        cwd=LLAMA_FACTORY_DIR,
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print("DPO 训练失败!")
        return False
    
    print("✓ DPO 训练完成!")
    return True

def main():
    print("=" * 60)
    print("APD + LLaMA-Factory 自动训练流程")
    print("=" * 60)
    
    # 1. 等待 APD 完成
    wait_for_apd_completion()
    
    # 2. 准备数据
    sft_train_count, sft_val_count, dpo_train_count, dpo_val_count = prepare_data_for_llama_factory()
    
    # 3. 更新脚本
    update_training_scripts()
    
    # 4. 开始训练
    success = run_training()
    
    if success:
        print("\n" + "=" * 60)
        print("训练全部完成!")
        print("=" * 60)
        print(f"SFT: {sft_train_count} 训练 / {sft_val_count} 验证")
        print(f"DPO: {dpo_train_count} 训练 / {dpo_val_count} 验证")
        print(f"模型保存位置: {PROJECT_DIR}/models/")
    else:
        print("\n训练过程中出现错误!")

if __name__ == "__main__":
    main()
