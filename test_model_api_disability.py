"""
使用API测试模型在Disability Bias上的表现
不需要本地PyTorch环境
"""

import json
import time
from datetime import datetime

# 路径配置
DISABILITY_TEST_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\disability_test_set.json"
RESULTS_OUTPUT_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\api_model_disability_results.json"

# 导入评估框架
from disability_bias_framework import DisabilityBiasEvaluator


def load_test_data(max_samples=50):
    """加载测试数据"""
    with open(DISABILITY_TEST_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:max_samples]


def simulate_model_predictions(test_data, model_type="random"):
    """
    模拟模型预测
    在实际使用时，这里应该替换为真实的API调用
    """
    import random
    random.seed(42)

    predictions = []

    for item in test_data:
        if model_type == "random":
            # 随机预测
            pred = random.randint(0, 2)
        elif model_type == "biased":
            # 有偏见预测 - 总是选择disabled标签
            for i in range(3):
                ans_key = f'ans{i}'
                if ans_key in item.get('answer_info', {}):
                    if item['answer_info'][ans_key][1] == 'disabled':
                        pred = i
                        break
            else:
                pred = 1
        elif model_type == "perfect":
            # 完美预测 - 总是选择正确答案
            pred = item['label']
        else:
            pred = random.randint(0, 2)

        predictions.append(pred)

    return predictions


def test_model(model_name="模拟模型", model_type="random", max_samples=50):
    """测试模型"""
    print(f"\n测试 {model_name}...")
    print(f"测试样本数: {max_samples}")

    # 加载测试数据
    test_data = load_test_data(max_samples)

    # 获取模型预测
    print("获取模型预测...")
    predictions = simulate_model_predictions(test_data, model_type)

    # 评估
    evaluator = DisabilityBiasEvaluator()
    report = evaluator.generate_report(predictions)

    print(f"\n{model_name} 评估结果:")
    evaluator.print_report(report)

    return report, predictions


def compare_models(reports):
    """对比多个模型的结果"""
    print("\n" + "=" * 70)
    print("模型对比")
    print("=" * 70)

    print(f"\n{'指标':<25} ", end="")
    for name in reports.keys():
        print(f"{name:<15} ", end="")
    print()
    print("-" * 70)

    # 总体准确率
    print(f"{'总体准确率':<25} ", end="")
    for report in reports.values():
        print(f"{report['overall_accuracy']:<15.3f} ", end="")
    print()

    # 偏见分数
    print(f"{'总体偏见分数':<25} ", end="")
    for report in reports.values():
        bias = report['bias_scores']['overall_bias_score']
        print(f"{bias:<15.3f} ", end="")
    print()

    # 模糊上下文准确率
    print(f"{'模糊上下文准确率':<25} ", end="")
    for report in reports.values():
        ambig = report['accuracy_by_condition']['ambig_accuracy']
        print(f"{ambig:<15.3f} ", end="")
    print()

    # 明确上下文准确率
    print(f"{'明确上下文准确率':<25} ", end="")
    for report in reports.values():
        disambig = report['accuracy_by_condition']['disambig_accuracy']
        print(f"{disambig:<15.3f} ", end="")
    print()


def save_results(reports, max_samples):
    """保存测试结果"""
    print("\n" + "=" * 70)
    print("保存测试结果")
    print("=" * 70)

    results = {
        'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_samples': max_samples,
        'models': {}
    }

    for name, report in reports.items():
        results['models'][name] = {
            'overall_accuracy': report['overall_accuracy'],
            'bias_scores': report['bias_scores'],
            'accuracy_by_condition': report['accuracy_by_condition'],
            'accuracy_by_subcategory': report['accuracy_by_subcategory']
        }

    with open(RESULTS_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {RESULTS_OUTPUT_PATH}")


def main():
    print("=" * 70)
    print("模型 Disability Bias 测试 (API版本)")
    print("=" * 70)
    print("\n注意: 当前使用模拟预测，请替换为实际API调用")

    max_samples = 50  # 测试样本数

    # 测试三种模型
    reports = {}

    # 1. 随机基线
    reports['随机基线'], _ = test_model(
        model_name="随机基线",
        model_type="random",
        max_samples=max_samples
    )

    # 2. 有偏见模型
    reports['有偏见模型'], _ = test_model(
        model_name="有偏见模型",
        model_type="biased",
        max_samples=max_samples
    )

    # 3. 理想模型
    reports['理想模型'], _ = test_model(
        model_name="理想模型",
        model_type="perfect",
        max_samples=max_samples
    )

    # 对比结果
    compare_models(reports)

    # 保存结果
    save_results(reports, max_samples)

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
    print("\n要使用实际模型测试，请:")
    print("1. 修改 simulate_model_predictions 函数，添加API调用")
    print("2. 或者在有PyTorch/transformers的环境中运行 test_real_model_disability.py")


if __name__ == "__main__":
    main()
