"""
使用真实验证数据测试训练好的反偏见提示词生成模型
"""

import json
import torch
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer
from peft import PeftModel

# 路径配置
BASE_MODEL_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD-PromptGenerator\models\Qwen2.5-1.5B-Instruct-base\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LORA_MODEL_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD-PromptGenerator\models\sft_qlora"
VAL_DATA_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\biasalert\BiasAlert-repo\code\LLaMA-Factory\data\apd_sft_val.json"

def load_val_data():
    """加载验证数据"""
    with open(VAL_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def test_model():
    print("=" * 60)
    print("使用验证数据测试反偏见提示词生成模型")
    print("=" * 60)
    
    # 检查 GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"\n使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("\n使用 CPU")
    
    # 加载验证数据
    print("\n[1/4] 加载验证数据...")
    val_data = load_val_data()
    print(f"验证样本数: {len(val_data)}")
    
    # 加载基础模型和 tokenizer
    print("[2/4] 加载基础模型...")
    tokenizer = Qwen2Tokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = Qwen2ForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    if not torch.cuda.is_available():
        model = model.to(device)
    
    # 加载 LoRA 权重
    print("[3/4] 加载 LoRA 权重...")
    model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)
    model.eval()
    
    # 测试前5个样本
    print("[4/4] 开始测试...")
    test_samples = val_data[:5]
    
    print("\n" + "=" * 60)
    print("生成反偏见提示词对比")
    print("=" * 60)
    
    for i, sample in enumerate(test_samples, 1):
        messages = sample['messages']
        user_message = messages[1]['content']  # 用户问题
        reference_answer = messages[2]['content']  # 参考答案
        
        # 提取原始问题
        question_start = user_message.find("Question:")
        question_end = user_message.find("\n\n请生成")
        if question_start != -1 and question_end != -1:
            original_question = user_message[question_start:question_end].strip()
        else:
            original_question = user_message[:100] + "..."
        
        print(f"\n{'='*60}")
        print(f"测试样本 {i}/{len(test_samples)}")
        print(f"{'='*60}")
        print(f"原始问题:\n{original_question}")
        print("-" * 60)
        print(f"参考答案:\n{reference_answer}")
        print("-" * 60)
        
        # 构建提示
        prompt = f"""<|im_start|>system
你是一个专门生成反偏见提示词的助手。你的任务是为给定的问题生成一个提示词，帮助大语言模型避免产生与残疾相关的偏见。<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
        
        # 生成
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 提取生成的提示词
        assistant_start = generated_text.find("<|im_start|>assistant")
        if assistant_start != -1:
            generated_prompt = generated_text[assistant_start + len("<|im_start|>assistant"):].strip()
            generated_prompt = generated_prompt.replace("<|im_end|>", "").strip()
            print(f"模型生成:\n{generated_prompt}")
        else:
            print(f"模型生成:\n{generated_text}")
        
        # 简单评估
        print("-" * 60)
        if "偏见" in generated_prompt or "歧视" in generated_prompt or "平等" in generated_prompt or "尊重" in generated_prompt:
            print("✅ 生成质量：包含反偏见关键词")
        else:
            print("⚠️ 生成质量：可能缺少反偏见关键词")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    test_model()
