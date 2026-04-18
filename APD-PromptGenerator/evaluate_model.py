"""
评估训练好的反偏见提示词生成模型的性能
使用验证数据进行定量评估
"""

import json
import torch
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer
from peft import PeftModel
from difflib import SequenceMatcher
import re

# 路径配置
BASE_MODEL_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD-PromptGenerator\models\Qwen2.5-1.5B-Instruct-base\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LORA_MODEL_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD-PromptGenerator\models\sft_qlora"
VAL_DATA_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\biasalert\BiasAlert-repo\code\LLaMA-Factory\data\apd_sft_val.json"

def load_val_data():
    """加载验证数据"""
    with open(VAL_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def similarity_score(a, b):
    """计算两个字符串的相似度"""
    return SequenceMatcher(None, a, b).ratio()

def contains_anti_bias_keywords(text):
    """检查是否包含反偏见关键词"""
    keywords = [
        '偏见', '歧视', '刻板印象', 'stereotype', 'bias', 'discrimination',
        '平等', '尊重', '包容', 'inclusive', 'respect', 'equality',
        '客观', '中立', '公正', 'objective', 'neutral', 'fair',
        '假设', 'assume', 'assumption',
        '能力', 'ability', 'capability',
        '独立', 'independent',
        '多样', 'diversity'
    ]
    text_lower = text.lower()
    found_keywords = [kw for kw in keywords if kw.lower() in text_lower]
    return len(found_keywords), found_keywords

def evaluate_prompt_quality(generated, reference):
    """评估生成提示词的质量"""
    scores = {}
    
    # 1. 相似度分数 (0-1)
    scores['similarity'] = similarity_score(generated, reference)
    
    # 2. 关键词覆盖率
    gen_kw_count, gen_keywords = contains_anti_bias_keywords(generated)
    ref_kw_count, ref_keywords = contains_anti_bias_keywords(reference)
    scores['keyword_coverage'] = gen_kw_count / max(ref_kw_count, 3)  # 至少期望3个关键词
    
    # 3. 长度合理性 (100-300字符为合理范围)
    gen_len = len(generated)
    if 100 <= gen_len <= 300:
        scores['length_score'] = 1.0
    elif gen_len < 100:
        scores['length_score'] = gen_len / 100
    else:
        scores['length_score'] = max(0, 1 - (gen_len - 300) / 300)
    
    # 4. 综合分数
    scores['overall'] = (
        scores['similarity'] * 0.4 +
        min(scores['keyword_coverage'], 1.0) * 0.4 +
        scores['length_score'] * 0.2
    )
    
    return scores, gen_keywords

def evaluate_model():
    print("=" * 70)
    print("反偏见提示词生成模型性能评估")
    print("=" * 70)
    
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
    
    # 评估
    print("[4/4] 开始评估...")
    print("\n" + "=" * 70)
    
    results = []
    total_scores = {
        'similarity': 0,
        'keyword_coverage': 0,
        'length_score': 0,
        'overall': 0
    }
    
    # 只测试前10个样本（避免内存问题）
    test_samples = val_data[:10]
    
    for i, sample in enumerate(test_samples, 1):
        messages = sample['messages']
        user_message = messages[1]['content']
        reference_answer = messages[2]['content']
        
        # 提取原始问题
        question_match = re.search(r'Question: (.+?)\n\n', user_message)
        if question_match:
            question = question_match.group(1)
        else:
            question = f"Sample {i}"
        
        print(f"\n样本 {i}/{len(test_samples)}: {question[:50]}...")
        
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
        else:
            generated_prompt = generated_text
        
        # 评估
        scores, keywords = evaluate_prompt_quality(generated_prompt, reference_answer)
        
        # 累加分数
        for key in total_scores:
            total_scores[key] += scores[key]
        
        results.append({
            'question': question,
            'reference': reference_answer,
            'generated': generated_prompt,
            'scores': scores,
            'keywords': keywords
        })
        
        print(f"  相似度: {scores['similarity']:.3f}")
        print(f"  关键词: {len(keywords)}个 {keywords[:3]}")
        print(f"  综合分: {scores['overall']:.3f}")
    
    # 计算平均分
    print("\n" + "=" * 70)
    print("评估结果汇总")
    print("=" * 70)
    
    n = len(test_samples)
    avg_scores = {key: total_scores[key] / n for key in total_scores}
    
    print(f"\n测试样本数: {n}")
    print(f"\n平均指标:")
    print(f"  相似度分数:     {avg_scores['similarity']:.3f} / 1.000")
    print(f"  关键词覆盖率:   {avg_scores['keyword_coverage']:.3f} / 1.000")
    print(f"  长度合理性:     {avg_scores['length_score']:.3f} / 1.000")
    print(f"  综合评分:       {avg_scores['overall']:.3f} / 1.000")
    
    # 评级
    overall = avg_scores['overall']
    if overall >= 0.8:
        grade = "优秀 (Excellent)"
    elif overall >= 0.6:
        grade = "良好 (Good)"
    elif overall >= 0.4:
        grade = "及格 (Pass)"
    else:
        grade = "需改进 (Needs Improvement)"
    
    print(f"\n模型评级: {grade}")
    
    # 详细结果
    print("\n" + "=" * 70)
    print("详细生成结果")
    print("=" * 70)
    
    for i, result in enumerate(results, 1):
        print(f"\n样本 {i}:")
        print(f"问题: {result['question'][:60]}...")
        print(f"生成: {result['generated'][:100]}...")
        print(f"分数: {result['scores']['overall']:.3f}")
    
    # 保存结果
    output_file = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD-PromptGenerator\evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'average_scores': avg_scores,
            'grade': grade,
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {output_file}")
    
    print("\n" + "=" * 70)
    print("评估完成!")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_model()
