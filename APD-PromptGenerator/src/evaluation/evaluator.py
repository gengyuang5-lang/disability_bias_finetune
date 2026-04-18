"""
评估模块
评估生成提示词的质量
"""

import json
import os
from openai import OpenAI

def load_test_data(data_dir):
    test_path = os.path.join(data_dir, "sft_val.json")
    with open(test_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_optimized_prompts(model_dir):
    optimized_path = os.path.join(model_dir, "optimized_prompts.json")
    if os.path.exists(optimized_path):
        with open(optimized_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def evaluate_prompt_on_samples(prompt, samples, deepseek_client, deployment_name):
    results = []
    for sample in samples:
        context = sample['context']
        question = sample['question']
        ans_texts = [sample['answer_info']['ans0'][0], sample['answer_info']['ans1'][0], sample['answer_info']['ans2'][0]]

        full_prompt = f"{prompt}\n\nContext: {context}\nQuestion: {question}\nOptions:\n0: {ans_texts[0]}\n1: {ans_texts[1]}\n2: {ans_texts[2]}\n\nPlease answer with the option number (0, 1, or 2) only."

        try:
            completion = deepseek_client.chat.completions.create(
                model=deployment_name,
                messages=[{"role": "user", "content": full_prompt}]
            )
            full_text = completion.choices[0].message.content.strip()

            if full_text in ['0', '1', '2']:
                answer_idx = int(full_text)
            else:
                answer_idx = -1

            results.append({
                "sample_id": sample.get('example_id', 'unknown'),
                "predicted": answer_idx,
                "true_label": sample['label'],
                "correct": answer_idx == sample['label'],
                "model_response": full_text
            })
        except Exception as e:
            results.append({
                "sample_id": sample.get('example_id', 'unknown'),
                "predicted": -1,
                "true_label": sample['label'],
                "correct": False,
                "error": str(e)
            })

    return results

def calculate_metrics(evaluation_results):
    total = len(evaluation_results)
    correct = sum(1 for r in evaluation_results if r['correct'])
    accuracy = correct / total if total > 0 else 0

    return {
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "error_rate": 1 - accuracy
    }

def compare_prompts(original_prompt, optimized_prompt, samples, deepseek_client, deployment_name):
    print("评估原始提示词...")
    original_results = evaluate_prompt_on_samples(original_prompt, samples, deepseek_client, deployment_name)
    original_metrics = calculate_metrics(original_results)

    print("评估优化提示词...")
    optimized_results = evaluate_prompt_on_samples(optimized_prompt, samples, deepseek_client, deployment_name)
    optimized_metrics = calculate_metrics(optimized_results)

    comparison = {
        "original_prompt": original_prompt,
        "optimized_prompt": optimized_prompt,
        "original_metrics": original_metrics,
        "optimized_metrics": optimized_metrics,
        "accuracy_improvement": optimized_metrics["accuracy"] - original_metrics["accuracy"]
    }

    return comparison

def generate_evaluation_report(comparisons, output_path):
    report = {
        "total_comparisons": len(comparisons),
        "average_original_accuracy": sum(c["original_metrics"]["accuracy"] for c in comparisons) / len(comparisons) if comparisons else 0,
        "average_optimized_accuracy": sum(c["optimized_metrics"]["accuracy"] for c in comparisons) / len(comparisons) if comparisons else 0,
        "average_improvement": sum(c["accuracy_improvement"] for c in comparisons) / len(comparisons) if comparisons else 0,
        "comparisons": comparisons
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n评估报告已生成: {output_path}")
    print(f"平均原始准确率: {report['average_original_accuracy']:.2%}")
    print(f"平均优化后准确率: {report['average_optimized_accuracy']:.2%}")
    print(f"平均提升: {report['average_improvement']:+.2%}")

    return report

def main():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from configs.config import DATA_PROCESSED_DIR, MODEL_BASE_DIR, SFT_MODEL_DIR, DPO_MODEL_DIR, API_ENDPOINT, API_KEY, DEPLOYMENT_NAME

    test_data = load_test_data(DATA_PROCESSED_DIR)

    deepseek_client = OpenAI(base_url=API_ENDPOINT, api_key=API_KEY)

    optimized_prompts = load_optimized_prompts(DPO_MODEL_DIR)

    default_instruction = "Please answer with the option number (0, 1, or 2) only."

    sample_size = min(50, len(test_data))
    samples = test_data[:sample_size]

    print(f"使用 {sample_size} 条样本进行评估...")

    if optimized_prompts:
        optimized_prompt = optimized_prompts[0].get("optimized", optimized_prompts[0].get("chosen", default_instruction))
    else:
        optimized_prompt = default_instruction

    comparison = compare_prompts(default_instruction, optimized_prompt, samples, deepseek_client, DEPLOYMENT_NAME)

    report_path = os.path.join(MODEL_BASE_DIR, "evaluation_report.json")
    report = generate_evaluation_report([comparison], report_path)

    print("\n评估完成!")

if __name__ == "__main__":
    main()
