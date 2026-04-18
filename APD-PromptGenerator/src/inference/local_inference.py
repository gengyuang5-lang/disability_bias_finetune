"""
本地模型推理脚本
使用微调后的LoRA模型生成anti-bias提示词
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from configs.lora_config import (
    BASE_MODEL_CONFIG, DPO_MODEL_DIR, SFT_MODEL_DIR, MODEL_BASE_DIR
)

def load_model_and_tokenizer(model_path, use_lora=True):
    print(f"加载模型: {model_path}")

    base_model_name = BASE_MODEL_CONFIG["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )

    if use_lora:
        model = PeftModel.from_pretrained(
            base_model,
            model_path,
            torch_dtype=torch.bfloat16
        )
        model.eval()
    else:
        model = base_model

    return model, tokenizer

def generate_prompt(model, tokenizer, question, context="", max_new_tokens=256, temperature=0.7):
    prompt_text = f"问题：{question}"
    if context:
        prompt_text = f"语境：{context}\n{prompt_text}"

    messages = [
        {"role": "user", "content": prompt_text}
    ]

    prompt = f"<|user|>\n{prompt_text}<|end|>\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "<|assistant|>" in full_output:
        response = full_output.split("<|assistant|>")[-1].strip()
    else:
        response = full_output[len(prompt):].strip()

    return response

def create_inference_pipeline(model_path, use_lora=True):
    model, tokenizer = load_model_and_tokenizer(model_path, use_lora)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

    return pipe

def batch_generate(pipe, questions, contexts=None):
    contexts = contexts or [""] * len(questions)
    results = []

    for i, (question, context) in enumerate(zip(questions, contexts)):
        print(f"处理 {i+1}/{len(questions)}...")

        prompt_text = f"语境：{context}\n问题：{question}" if context else f"问题：{question}"
        prompt = f"<|user|>\n{prompt_text}<|end|>\n<|assistant|>\n"

        output = pipe(prompt)[0]["generated_text"]

        if "<|assistant|>" in output:
            response = output.split("<|assistant|>")[-1].strip()
        else:
            response = output[len(prompt):].strip()

        results.append({
            "question": question,
            "context": context,
            "generated_prompt": response
        })

    return results

def main():
    print("=" * 60)
    print("APD-PromptGenerator 本地推理")
    print("=" * 60)

    dpo_model_path = os.path.join(DPO_MODEL_DIR, "final_model")
    sft_model_path = os.path.join(SFT_MODEL_DIR, "final_model")

    if os.path.exists(dpo_model_path):
        model_path = dpo_model_path
        model_type = "DPO"
    elif os.path.exists(sft_model_path):
        model_path = sft_model_path
        model_type = "SFT"
    else:
        print("错误: 未找到训练好的模型")
        print(f"请先运行 SFT 和 DPO 训练脚本")
        return

    print(f"使用 {model_type} 模型: {model_path}")

    print("\n加载模型中...")
    model, tokenizer = load_model_and_tokenizer(model_path, use_lora=True)

    test_questions = [
        "有自闭症的人适合做老师吗？",
        "抑郁症患者能正常工作吗？",
        "坐轮椅的人可以独立生活吗？",
    ]

    test_contexts = [
        "一位有自闭症谱系障碍的老师正在教室给学生上课",
        "某公司正在招聘新员工",
        "一位坐着轮椅的人正在寻找公寓",
    ]

    print("\n生成测试:")
    for i, (q, ctx) in enumerate(zip(test_questions, test_contexts)):
        print(f"\n--- 样本 {i+1} ---")
        print(f"问题: {q}")
        print(f"语境: {ctx}")

        prompt = generate_prompt(model, tokenizer, q, ctx)
        print(f"生成提示词: {prompt}")

    print("\n" + "=" * 60)
    print("推理完成!")
    print("=" * 60)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("警告: 未检测到CUDA，将使用CPU推理（很慢）")
    main()
