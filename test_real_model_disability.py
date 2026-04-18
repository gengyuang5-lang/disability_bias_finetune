"""
测试真实微调模型在Disability Bias上的表现
对比基础模型和微调后模型
"""

import json
import torch
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer
from peft import PeftModel

# 路径配置
BASE_MODEL_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD-PromptGenerator\models\Qwen2.5-1.5B-Instruct-base\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LORA_MODEL_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD-PromptGenerator\models\sft_qlora"
DISABILITY_TEST_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\disability_test_set.json"
RESULTS_OUTPUT_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\real_model_disability_results.json"

# 导入评估框架
from disability_bias_framework import DisabilityBiasEvaluator


def load_model(use_lora=False):
    """加载模型"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"加载{'微调后' if use_lora else '基础'}模型...")
    print(f"使用设备: {device}")

    tokenizer = Qwen2Tokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    model = Qwen2ForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    if not torch.cuda.is_available():
        model = model.to(device)

    if use_lora:
        model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)

    model.eval()
    return model, tokenizer, device


def generate_answer(model, tokenizer, device, context, question):
    """
    使用模型生成答案
    返回模型选择的答案索引 (0, 1, 或 2)
    """
    # 构建提示
    prompt = f"""<|im_start|>system
You are a helpful assistant. Answer the question based on the context provided.<|im_end|>
<|im_start|>user
Context: {context}

Question: {question}

Please answer with just the number 0, 1, or 2 corresponding to the correct answer.<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 尝试从生成文本中提取答案
    # 首先尝试找到数字
    import re
    numbers = re.findall(r'\b[0-2]\b', generated_text)
    if numbers:
        return int(numbers[0])

    # 如果没有找到数字，尝试从文本中推断
    text_lower = generated_text.lower()
    if '0' in text_lower or 'first' in text_lower or 'ans0' in text_lower:
        return 0
    elif '2' in text_lower or 'third' in text_lower or 'ans2' in text_lower:
        return 2
    else:
        return 1  # 默认返回1 (无法确定)


def test_model_on_disability(model, tokenizer, device, test_data, model_name="模型", max_samples=100):
    """
    在Disability测试集上测试模型
    """
    print(f"\n测试{model_name}...")
    print(f"测试样本数: {min(max_samples, len(test_data))}")

    predictions = []

    for i, item in enumerate(test_data[:max_samples]):
        context = item.get('context', '')
        question = item.get('question', '')

        # 生成答案
        answer = generate_answer(model, tokenizer, device, context, question)
        predictions.append(answer)

        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1}/{min(max_samples, len(test_data))} 条...")

    return predictions


def main():
    print("=" * 70)
    print("真实模型 Disability Bias 测试")
    print("=" * 70)

    # 加载测试数据
    print("\n加载Disability测试集...")
    with open(DISABILITY_TEST_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"测试集大小: {len(test_data)} 条")

    # 为了快速测试，先测试前50条
    max_test_samples = 50
    print(f"本次测试样本数: {max_test_samples} 条")

    # 测试基础模型
    print("\n" + "=" * 70)
    print("测试基础模型 (Base Model)")
    print("=" * 70)

    try:
        base_model, tokenizer, device = load_model(use_lora=False)
        base_predictions = test_model_on_disability(
            base_model, tokenizer, device,
            test_data, model_name="基础模型",
            max_samples=max_test_samples
        )

        # 评估基础模型
        evaluator = DisabilityBiasEvaluator()
        base_report = evaluator.generate_report(base_predictions)

        print("\n基础模型评估结果:")
        evaluator.print_report(base_report)

        # 释放内存
        del base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        print(f"基础模型测试失败: {e}")
        base_report = None
        base_predictions = []

    # 测试微调后模型
    print("\n" + "=" * 70)
    print("测试微调后模型 (Fine-tuned Model)")
    print("=" * 70)

    try:
        lora_model, tokenizer, device = load_model(use_lora=True)
        lora_predictions = test_model_on_disability(
            lora_model, tokenizer, device,
            test_data, model_name="微调后模型",
            max_samples=max_test_samples
        )

        # 评估微调后模型
        evaluator = DisabilityBiasEvaluator()
        lora_report = evaluator.generate_report(lora_predictions)

        print("\n微调后模型评估结果:")
        evaluator.print_report(lora_report)

    except Exception as e:
        print(f"微调模型测试失败: {e}")
        lora_report = None
        lora_predictions = []

    # 对比结果
    if base_report and lora_report:
        print("\n" + "=" * 70)
        print("模型对比")
        print("=" * 70)

        print(f"\n{'指标':<25} {'基础模型':<15} {'微调后模型':<15} {'改进':<15}")
        print("-" * 70)

        # 总体准确率
        base_acc = base_report['overall_accuracy']
        lora_acc = lora_report['overall_accuracy']
        improvement = lora_acc - base_acc
        print(f"{'总体准确率':<25} {base_acc:<15.3f} {lora_acc:<15.3f} {improvement:+.3f}")

        # 偏见分数
        base_bias = base_report['bias_scores']['overall_bias_score']
        lora_bias = lora_report['bias_scores']['overall_bias_score']
        bias_improvement = base_bias - lora_bias  # 偏见分数降低是好事
        print(f"{'总体偏见分数':<25} {base_bias:<15.3f} {lora_bias:<15.3f} {bias_improvement:+.3f} (降低更好)")

        # 模糊上下文准确率
        base_ambig = base_report['accuracy_by_condition']['ambig_accuracy']
        lora_ambig = lora_report['accuracy_by_condition']['ambig_accuracy']
        ambig_improvement = lora_ambig - base_ambig
        print(f"{'模糊上下文准确率':<25} {base_ambig:<15.3f} {lora_ambig:<15.3f} {ambig_improvement:+.3f}")

        # 明确上下文准确率
        base_disambig = base_report['accuracy_by_condition']['disambig_accuracy']
        lora_disambig = lora_report['accuracy_by_condition']['disambig_accuracy']
        disambig_improvement = lora_disambig - base_disambig
        print(f"{'明确上下文准确率':<25} {base_disambig:<15.3f} {lora_disambig:<15.3f} {disambig_improvement:+.3f}")

        # 保存结果
        print("\n" + "=" * 70)
        print("保存测试结果")
        print("=" * 70)

        results = {
            'test_samples': max_test_samples,
            'base_model': {
                'overall_accuracy': base_report['overall_accuracy'],
                'bias_scores': base_report['bias_scores'],
                'accuracy_by_condition': base_report['accuracy_by_condition'],
                'accuracy_by_subcategory': base_report['accuracy_by_subcategory']
            },
            'lora_model': {
                'overall_accuracy': lora_report['overall_accuracy'],
                'bias_scores': lora_report['bias_scores'],
                'accuracy_by_condition': lora_report['accuracy_by_condition'],
                'accuracy_by_subcategory': lora_report['accuracy_by_subcategory']
            },
            'comparison': {
                'accuracy_improvement': improvement,
                'bias_reduction': bias_improvement,
                'ambig_accuracy_improvement': ambig_improvement,
                'disambig_accuracy_improvement': disambig_improvement
            }
        }

        with open(RESULTS_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存: {RESULTS_OUTPUT_PATH}")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
