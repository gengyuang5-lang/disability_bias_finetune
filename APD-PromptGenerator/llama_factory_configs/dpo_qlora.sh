#!/bin/bash
# DPO QLoRA 训练脚本 - 适配 RTX 3060 Laptop (6GB)
# 基于 SFT 后的模型进行 DPO 训练

cd "E:/desktop/CDS546 Project Topics (AY2025-26)(2)/biasalert/BiasAlert-repo/code/LLaMA-Factory"

python src/train.py \
    --stage dpo \
    --do_train \
    --model_name_or_path "E:/desktop/CDS546 Project Topics (AY2025-26)(2)/tune/APD-PromptGenerator/models/sft_qlora" \
    --dataset apd_dpo_train \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj \
    --lora_rank 8 \
    --lora_alpha 16 \
    --output_dir "E:/desktop/CDS546 Project Topics (AY2025-26)(2)/tune/APD-PromptGenerator/models/dpo_qlora" \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 1e-5 \
    --num_train_epochs 2 \
    --plot_loss \
    --bf16 \
    --max_length 1024 \
    --max_new_tokens 256 \
    --cutoff_len 1024 \
    --preprocessing_num_workers 4 \
    --flash_attn auto \
    --use_unsloth False \
    --beta 0.1

echo "DPO 训练完成!"
