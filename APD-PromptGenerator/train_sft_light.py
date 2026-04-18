"""
SFT QLoRA 训练脚本 - 轻量级版本
适配 RTX 3060 Laptop (6GB) - 使用更激进的内存优化
"""

import json
import os
import torch
from transformers import (
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
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
    print("SFT QLoRA 训练 (轻量级)")
    print("=" * 60)
    
    # 清理 GPU 缓存
    torch.cuda.empty_cache()
    
    # 1. 配置 4-bit 量化
    print("\n[1/5] 配置 4-bit 量化...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # 2. 加载模型和 tokenizer
    print("[2/5] 加载模型...")
    tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("正在加载模型到 GPU...")
    model = Qwen2ForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # 3. 准备模型用于训练
    print("[3/5] 准备模型用于训练...")
    model = prepare_model_for_kbit_training(model)
    
    # 4. 配置 LoRA - 使用更小的 rank
    print("[4/5] 配置 LoRA...")
    lora_config = LoraConfig(
        r=4,  # 减小 rank
        lora_alpha=8,  # 相应减小 alpha
        target_modules=["q_proj", "v_proj"],  # 只训练注意力层
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 5. 加载数据
    print("[5/5] 加载数据...")
    dataset = load_data()
    print(f"训练样本数: {len(dataset)}")
    
    # 6. 配置训练参数 - 使用更保守的设置
    print("\n开始训练...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # 减小梯度累积
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=50,
        fp16=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        optim="adamw_torch",
        warmup_steps=5,
        max_grad_norm=0.3,
        dataloader_pin_memory=False,  # 禁用 pin_memory
    )
    
    # 7. 数据预处理 - 使用更短的长度
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,  # 减小最大长度
            padding="max_length",
            return_tensors=None
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # 8. 创建 Trainer
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # 9. 开始训练
    print("\n训练开始，这可能需要一些时间...")
    trainer.train()
    
    # 10. 保存模型
    print("\n保存模型...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("SFT 训练完成!")
    print(f"模型保存位置: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
