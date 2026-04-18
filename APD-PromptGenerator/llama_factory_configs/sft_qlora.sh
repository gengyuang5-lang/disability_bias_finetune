#!/bin/bash
# SFT QLoRA 训练脚本 - 适配 RTX 3060 Laptop (6GB)
# 使用 Qwen-2.5-1.5B-Instruct 模型

cd "E:/desktop/CDS546 Project Topics (AY2025-26)(2)/biasalert/BiasAlert-repo/code/LLaMA-Factory"

python src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path "E:/desktop/CDS546 Project Topics (AY2025-26)(2)/tune/APD-PromptGenerator/models/Qwen2.5-1.5B-Instruct-base" \
    --dataset apd_sft_train \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj \
    --lora_rank 8 \
    --lora_alpha 16 \
    --output_dir "E:/desktop/CDS546 Project Topics (AY2025-26)(2)/tune/APD-PromptGenerator/models/sft_qlora" \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --plot_loss \
    --bf16 \
    --max_length 1024 \
    --max_new_tokens 256 \
    --cutoff_len 1024 \
    --preprocessing_num_workers 4 \
    --flash_attn auto \
    --use_unsloth False

echo "SFT 训练完成!"
