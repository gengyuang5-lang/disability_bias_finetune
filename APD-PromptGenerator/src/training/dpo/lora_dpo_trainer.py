"""
真正的DPO (Direct Preference Optimization) 微调训练脚本
使用HuggingFace TRL库进行本地微调
"""

import os
import sys
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from trl import DPOTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from configs.lora_config import (
    BASE_MODEL_CONFIG, LORA_CONFIG, DPO_TRAINING_CONFIG,
    DATA_PREFERENCE_DIR, DPO_MODEL_DIR, SFT_MODEL_DIR, MAX_seq_LENGTH
)

def load_preference_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_dataset(preference_data):
    formatted_data = []
    for item in preference_data:
        formatted_data.append({
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"]
        })
    return Dataset.from_list(formatted_data)

def tokenize_function(examples, tokenizer, max_length):
    prompts = examples["prompt"]
    choosen = examples["chosen"]
    rejected = examples["rejected"]

    chosen_texts = []
    rejected_texts = []

    for prompt, chosen, reject in zip(prompts, choosen, rejected):
        chosen_text = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n{chosen}<|end|>"
        rejected_text = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n{reject}<|end|>"
        chosen_texts.append(chosen_text)
        rejected_texts.append(rejected_text)

    chosen_tokenized = tokenizer(
        chosen_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None
    )

    rejected_tokenized = tokenizer(
        rejected_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None
    )

    return {
        "prompt": prompts,
        "chosen": chosen_tokenized["input_ids"],
        "rejected": rejected_tokenized["input_ids"],
        "chosen_attention_mask": chosen_tokenized["attention_mask"],
        "rejected_attention_mask": rejected_tokenized["attention_mask"]
    }

def prepare_dataset(data_path, tokenizer):
    print(f"加载偏好数据: {data_path}")
    raw_data = load_preference_data(data_path)

    dataset = Dataset.from_list(raw_data)

    def process_fn(examples):
        return tokenize_function(examples, tokenizer, MAX_seq_LENGTH)

    tokenized_dataset = dataset.map(
        process_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing preference dataset"
    )

    return tokenized_dataset

def create_peft_model(model_name, ref_model_name=None):
    print(f"加载基础模型: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )

    print("加载参考模型 (用于DPO)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    ref_model.eval()

    return model, ref_model

def setup_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|end|>"

    tokenizer.padding_side = "left"

    return tokenizer

def main():
    print("=" * 60)
    print("开始真正的DPO微调训练")
    print("=" * 60)

    os.makedirs(DPO_MODEL_DIR, exist_ok=True)

    train_path = os.path.join(DATA_PREFERENCE_DIR, "preference_train.json")
    val_path = os.path.join(DATA_PREFERENCE_DIR, "preference_val.json")

    sft_model_path = os.path.join(SFT_MODEL_DIR, "final_model")
    if os.path.exists(sft_model_path):
        print(f"检测到已完成的SFT模型: {sft_model_path}")
        model_name = sft_model_path
    else:
        model_name = BASE_MODEL_CONFIG["model_name"]
        print(f"未找到SFT模型，使用基础模型: {model_name}")

    print("\n1. 设置Tokenizer...")
    tokenizer = setup_tokenizer(model_name)

    print("\n2. 准备训练数据集...")
    train_dataset = prepare_dataset(train_path, tokenizer)

    print("\n3. 准备验证数据集...")
    val_dataset = prepare_dataset(val_path, tokenizer)

    print(f"   训练集大小: {len(train_dataset)}")
    print(f"   验证集大小: {len(val_dataset)}")

    print("\n4. 创建模型和参考模型...")
    model, ref_model = create_peft_model(model_name)

    print("\n5. 配置DPO训练参数...")
    training_args = TrainingArguments(
        output_dir=DPO_MODEL_DIR,
        num_train_epochs=DPO_TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=DPO_TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=DPO_TRAINING_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=DPO_TRAINING_CONFIG["gradient_accumulation_steps"],
        eval_strategy=DPO_TRAINING_CONFIG["eval_strategy"],
        eval_steps=DPO_TRAINING_CONFIG["eval_steps"],
        save_strategy=DPO_TRAINING_CONFIG["save_strategy"],
        save_steps=DPO_TRAINING_CONFIG["save_steps"],
        save_total_limit=DPO_TRAINING_CONFIG["save_total_limit"],
        learning_rate=DPO_TRAINING_CONFIG["learning_rate"],
        warmup_ratio=DPO_TRAINING_CONFIG["warmup_ratio"],
        lr_scheduler_type=DPO_TRAINING_CONFIG["lr_scheduler_type"],
        logging_steps=DPO_TRAINING_CONFIG["logging_steps"],
        bf16=DPO_TRAINING_CONFIG["bf16"],
        seed=DPO_TRAINING_CONFIG["seed"],
        report_to="none"
    )

    print("\n6. 初始化DPO训练器...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=DPO_TRAINING_CONFIG["beta"],
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    print("\n7. 开始DPO训练...")
    dpo_trainer.train()

    print("\n8. 保存最终模型...")
    final_model_path = os.path.join(DPO_MODEL_DIR, "final_model")
    dpo_trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print("\n" + "=" * 60)
    print("DPO微调训练完成!")
    print(f"模型保存位置: {final_model_path}")
    print("=" * 60)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("警告: 未检测到CUDA，将使用CPU训练（非常慢）")
    main()
