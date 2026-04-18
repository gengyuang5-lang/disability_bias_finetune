"""
真正的SFT微调训练脚本
使用HuggingFace Transformers + PEFT (LoRA)进行本地微调
"""

import os
import sys
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from configs.lora_config import (
    BASE_MODEL_CONFIG, LORA_CONFIG, TRAINING_CONFIG,
    DATA_PROCESSED_DIR, SFT_MODEL_DIR, MAX_seq_LENGTH
)

def load_sft_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_conversation(messages, tokenizer, max_length):
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            formatted += f"<|user|>\n{content}<|end|>\n"
        elif role == "assistant":
            formatted += f"<|assistant|>\n{content}<|end|>\n"
    formatted += "<|assistant|>"
    return formatted

def tokenize_function(examples, tokenizer, max_length):
    texts = []
    for messages in examples["messages"]:
        text = format_conversation(messages, tokenizer, max_length)
        texts.append(text)

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    for i, labels in enumerate(tokenized["labels"]):
        user_turn = tokenizer.encode("<|user|>", add_special_tokens=False)
        assistant_turn = tokenizer.encode("<|assistant|>", add_special_tokens=False)

        for j, token_id in enumerate(labels):
            if token_id == tokenizer.pad_token_id:
                labels[j] = -100
            elif token_id in user_turn:
                labels[j] = -100

    return tokenized

def prepare_dataset(data_path, tokenizer):
    print(f"加载数据: {data_path}")
    raw_data = load_sft_data(data_path)

    dataset = Dataset.from_list(raw_data)

    def process_fn(examples):
        return tokenize_function(examples, tokenizer, MAX_seq_LENGTH)

    tokenized_dataset = dataset.map(
        process_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )

    return tokenized_dataset

def create_peft_model(model_name, lora_config):
    print(f"加载基础模型: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )

    print("应用LoRA配置...")
    lora_config_obj = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config["lora_r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["target_modules"],
        bias=lora_config["bias"]
    )

    model = get_peft_model(model, lora_config_obj)
    model.print_trainable_parameters()

    return model

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

def compute_metrics(eval_pred):
    loss = eval_pred.predictions
    return {"eval_loss": float(loss.mean())}

def main():
    print("=" * 60)
    print("开始真正的SFT微调训练")
    print("=" * 60)

    os.makedirs(SFT_MODEL_DIR, exist_ok=True)

    train_path = os.path.join(DATA_PROCESSED_DIR, "sft_train.json")
    val_path = os.path.join(DATA_PROCESSED_DIR, "sft_val.json")

    model_name = BASE_MODEL_CONFIG["model_name"]

    print("\n1. 设置Tokenizer...")
    tokenizer = setup_tokenizer(model_name)

    print("\n2. 准备训练数据集...")
    train_dataset = prepare_dataset(train_path, tokenizer)

    print("\n3. 准备验证数据集...")
    val_dataset = prepare_dataset(val_path, tokenizer)

    print(f"   训练集大小: {len(train_dataset)}")
    print(f"   验证集大小: {len(val_dataset)}")

    print("\n4. 创建PEFT模型 (LoRA)...")
    model = create_peft_model(model_name, LORA_CONFIG)

    print("\n5. 配置训练参数...")
    training_args = TrainingArguments(
        output_dir=SFT_MODEL_DIR,
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        eval_strategy=TRAINING_CONFIG["eval_strategy"],
        eval_steps=TRAINING_CONFIG["eval_steps"],
        save_strategy=TRAINING_CONFIG["save_strategy"],
        save_steps=TRAINING_CONFIG["save_steps"],
        save_total_limit=TRAINING_CONFIG["save_total_limit"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
        bf16=TRAINING_CONFIG["bf16"],
        seed=TRAINING_CONFIG["seed"],
        remove_unused_columns=TRAINING_CONFIG["remove_unused_columns"],
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    print("\n6. 开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    print("\n7. 保存最终模型...")
    final_model_path = os.path.join(SFT_MODEL_DIR, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print("\n" + "=" * 60)
    print("SFT微调训练完成!")
    print(f"模型保存位置: {final_model_path}")
    print("=" * 60)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("警告: 未检测到CUDA，将使用CPU训练（非常慢）")
    main()
