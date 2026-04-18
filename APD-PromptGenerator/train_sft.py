"""
SFT 训练脚本 - 使用原生 transformers
适配 RTX 3060 Laptop (6GB)
"""

import json
import os
import torch
from transformers import (
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# 路径配置
MODEL_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD-PromptGenerator\models\Qwen2.5-1.5B-Instruct-base\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
OUTPUT_DIR = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD-PromptGenerator\models\sft_qlora"
DATA_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\biasalert\BiasAlert-repo\code\LLaMA-Factory\data\apd_sft_train.json"

def load_data():
    """加载训练数据"""
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为文本格式
    texts = []
    for item in data:
        messages = item['messages']
        text = ""
        for msg in messages:
            if msg['role'] == 'system':
                text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg['role'] == 'user':
                text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg['role'] == 'assistant':
                text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        texts.append(text)
    
    return Dataset.from_dict({"text": texts})

def main():
    print("=" * 60)
    print("SFT QLoRA 训练")
    print("=" * 60)
    
    # 1. 加载模型和 tokenizer
    print("\n[1/4] 加载模型...")
    tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = Qwen2ForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 2. 配置 LoRA
    print("[2/4] 配置 LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 3. 加载数据
    print("[3/4] 加载数据...")
    dataset = load_data()
    print(f"训练样本数: {len(dataset)}")
    
    # 4. 配置训练参数
    print("[4/4] 开始训练...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        fp16=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        optim="adamw_torch",
        warmup_ratio=0.1,
    )
    
    # 5. 数据预处理
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            padding="max_length",
            return_tensors=None
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # 6. 创建 Trainer
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # 7. 开始训练
    trainer.train()
    
    # 8. 保存模型
    print("\n保存模型...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("SFT 训练完成!")
    print(f"模型保存位置: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
