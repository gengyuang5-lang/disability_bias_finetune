"""
实验1: 基线实验 - 中文大模型 + 传统BBQ

测试中文大模型(DeepSeek/Kimi)在传统BBQ数据集上的Disability Bias表现
"""

import json
import sys
import os
import requests
import time

# 添加共享工具目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import save_experiment_result, load_experiment_config, log_experiment

# 添加tune目录到路径以导入评估框架
tune_dir = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune"
sys.path.append(tune_dir)
from disability_bias_framework import DisabilityBiasEvaluator


def call_azure_api(api_key, api_endpoint, deployment_name, prompt, temperature=0.1, max_tokens=10, max_retries=3):
    """调用Azure OpenAI API (带重试机制)"""
    url = f"{api_endpoint}chat/completions"
    headers = {
        "api-key": api_key,  # Azure使用api-key而不是Authorization Bearer
        "Content-Type": "application/json"
    }
    data = {
        "model": deployment_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"API调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 * (attempt + 1)
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print("达到最大重试次数，放弃此请求")
                return None
    return None


def extract_answer(text):
    """从模型输出中提取答案 (0, 1, 2)"""
    import re

    # 查找数字0, 1, 2
    numbers = re.findall(r'\b[0-2]\b', text)
    if numbers:
        return int(numbers[0])

    # 如果没有找到数字，尝试从文本中推断
    text_lower = text.lower()
    if '0' in text_lower or 'first' in text_lower or 'ans0' in text_lower or '第一个' in text:
        return 0
    elif '2' in text_lower or 'third' in text_lower or 'ans2' in text_lower or '第三个' in text:
        return 2
    else:
        return 1  # 默认返回1 (无法确定)


def create_prompt(context, question, ans0, ans1, ans2):
    """创建标准BBQ提示词"""
    prompt = f"""根据以下情境回答问题：

情境：{context}

问题：{question}

选项：
0. {ans0}
1. {ans1}
2. {ans2}

请只回答数字0、1或2。"""
    return prompt


def run_experiment(config):
    """运行基线实验"""
    experiment_name = config['experiment_name']
    log_experiment(experiment_name, "开始基线实验")

    # 加载测试数据
    test_data_path = config['data_paths']['input']
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    max_samples = config['parameters']['test_samples']
    test_data = test_data[:max_samples]
    log_experiment(experiment_name, f"加载测试数据: {len(test_data)}条")

    # 获取API配置
    api_provider = config['parameters']['api_provider']
    api_config = config['api_config'][api_provider]
    api_key = api_config['api_key']
    api_endpoint = api_config.get('api_endpoint', '')
    deployment_name = api_config.get('deployment_name', '')

    if api_key == f"YOUR_{api_provider.upper()}_API_KEY" or not api_key:
        log_experiment(experiment_name, "警告: 请先在config.json中设置API Key")
        return {"error": "API Key未设置"}

    # 测试模型
    predictions = []
    correct_count = 0

    for i, item in enumerate(test_data):
        context = item.get('context', '')
        question = item.get('question', '')
        ans0 = item.get('ans0', '')
        ans1 = item.get('ans1', '')
        ans2 = item.get('ans2', '')
        correct_label = item.get('label', 1)

        # 创建提示词
        prompt = create_prompt(context, question, ans0, ans1, ans2)

        # 调用API
        if api_provider == 'azure':
            response = call_azure_api(
                api_key, api_endpoint, deployment_name, prompt,
                temperature=config['parameters']['temperature'],
                max_tokens=config['parameters']['max_tokens']
            )
        else:
            response = None

        if response:
            answer = extract_answer(response)
            predictions.append(answer)

            if answer == correct_label:
                correct_count += 1
        else:
            # API调用失败，使用默认答案
            predictions.append(1)

        if (i + 1) % 10 == 0:
            accuracy_so_far = correct_count / (i + 1)
            log_experiment(experiment_name, f"已处理 {i+1}/{len(test_data)} 条，当前准确率: {accuracy_so_far:.3f}")

        # 避免API限流
        time.sleep(0.5)

    # 评估结果
    evaluator = DisabilityBiasEvaluator()
    report = evaluator.generate_report(predictions)

    final_accuracy = correct_count / len(predictions)
    log_experiment(experiment_name, f"实验完成，最终准确率: {final_accuracy:.3f}")

    # 整理结果
    results = {
        "api_provider": api_provider,
        "test_samples": len(test_data),
        "overall_accuracy": report['overall_accuracy'],
        "bias_scores": report['bias_scores'],
        "accuracy_by_condition": report['accuracy_by_condition'],
        "accuracy_by_subcategory": report['accuracy_by_subcategory'],
        "predictions": predictions
    }

    return results


def main():
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    config = load_experiment_config(config_path)

    print(f"=" * 70)
    print(f"实验: {config['experiment_name']}")
    print(f"描述: {config['description']}")
    print(f"=" * 70)

    # 运行实验
    results = run_experiment(config)

    # 保存结果
    if 'error' not in results:
        output_dir = config.get('data_paths', {}).get('output', 'results')
        save_experiment_result(
            experiment_name=config['experiment_name'],
            results=results,
            output_dir=output_dir
        )

    print(f"\n实验完成!")


if __name__ == "__main__":
    main()
