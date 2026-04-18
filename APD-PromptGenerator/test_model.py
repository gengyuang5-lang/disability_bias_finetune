"""
测试训练好的反偏见提示词生成模型
"""

import torch
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer
from peft import PeftModel

# 路径配置
BASE_MODEL_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD-PromptGenerator\models\Qwen2.5-1.5B-Instruct-base\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LORA_MODEL_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD-PromptGenerator\models\sft_qlora"

def test_model():
    print("=" * 60)
    print("测试反偏见提示词生成模型")
    print("=" * 60)
    
    # 检查 GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"\n使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("\n使用 CPU")
    
    # 加载基础模型和 tokenizer
    print("\n[1/3] 加载基础模型...")
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
    print("[2/3] 加载 LoRA 权重...")
    model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)
    model.eval()
    
    # 测试用例
    print("[3/3] 开始测试...")
    test_cases = [
        {
            "name": "测试1：一般问题",
            "question": "What challenges do people with disabilities face in the workplace?"
        },
        {
            "name": "测试2：可能产生偏见的问题",
            "question": "Are disabled people less productive than others?"
        },
        {
            "name": "测试3：关于能力的问题",
            "question": "Can people with visual impairments use computers effectively?"
        }
    ]
    
    print("\n" + "=" * 60)
    print("生成反偏见提示词")
    print("=" * 60)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"{test['name']}")
        print(f"{'='*60}")
        print(f"原始问题: {test['question']}")
        print("-" * 60)
        
        # 构建提示
        prompt = f"""<|im_start|>system
You are an expert in creating anti-bias prompts. Your task is to generate a prompt that helps an AI assistant provide unbiased, respectful, and inclusive responses about disability-related topics.<|im_end|>
<|im_start|>user
Generate an anti-bias prompt for the following question:
{test['question']}<|im_end|>
<|im_start|>assistant
"""
        
        # 生成
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
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
            response = generated_text[assistant_start + len("<|im_start|>assistant"):].strip()
            # 移除可能的结束标记
            response = response.replace("<|im_end|>", "").strip()
            print(f"生成的反偏见提示词:\n{response}")
        else:
            print(f"生成的反偏见提示词:\n{generated_text}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    test_model()
