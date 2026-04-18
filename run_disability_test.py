"""
运行Disability Bias测试
评估模型在Disability测试集上的表现
"""

import json
import sys

# 导入评估框架
from disability_bias_framework import DisabilityBiasEvaluator

# 路径配置
DISABILITY_TEST_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\disability_test_set.json"
RESULTS_OUTPUT_PATH = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\disability_test_results.json"

def simulate_baseline_predictions(test_data):
    """
    模拟基线预测 (随机选择)
    用于对比评估
    """
    import random
    random.seed(42)
    return [random.randint(0, 2) for _ in range(len(test_data))]

def simulate_biased_predictions(test_data):
    """
    模拟有偏见的预测
    总是选择残疾群体答案 (测试偏见检测能力)
    """
    predictions = []
    for item in test_data:
        # 找到disabled标签的选项
        for i in range(3):
            ans_key = f'ans{i}'
            if ans_key in item.get('answer_info', {}):
                if item['answer_info'][ans_key][1] == 'disabled':
                    predictions.append(i)
                    break
        else:
            predictions.append(1)  # 默认选择中间选项
    return predictions

def simulate_anti_bias_predictions(test_data):
    """
    模拟反偏见预测
    优先选择正确答案
    """
    predictions = []
    for item in test_data:
        correct_label = item['label']
        # 80%概率选择正确答案，20%随机
        import random
        random.seed(42)
        if random.random() < 0.8:
            predictions.append(correct_label)
        else:
            predictions.append(random.randint(0, 2))
    return predictions

def run_baseline_test():
    """运行基线测试 (随机预测)"""
    print("=" * 70)
    print("Disability Bias 基线测试 (随机预测)")
    print("=" * 70)

    evaluator = DisabilityBiasEvaluator()
    test_data = evaluator.data

    print(f"\n测试样本数: {len(test_data)}")
    print("\n生成随机预测...")

    predictions = simulate_baseline_predictions(test_data)
    report = evaluator.generate_report(predictions)

    print("\n" + "=" * 70)
    print("基线测试结果 (随机预测)")
    print("=" * 70)
    evaluator.print_report(report)

    return report

def run_biased_test():
    """运行偏见测试 (总是选择残疾群体答案)"""
    print("\n" + "=" * 70)
    print("Disability Bias 偏见测试 (总是选择残疾群体答案)")
    print("=" * 70)

    evaluator = DisabilityBiasEvaluator()
    test_data = evaluator.data

    print(f"\n测试样本数: {len(test_data)}")
    print("\n生成偏见预测 (总是选择残疾群体)...")

    predictions = simulate_biased_predictions(test_data)
    report = evaluator.generate_report(predictions)

    print("\n" + "=" * 70)
    print("偏见测试结果 (总是选择残疾群体答案)")
    print("=" * 70)
    evaluator.print_report(report)

    return report

def run_anti_bias_test():
    """运行反偏见测试 (80%正确率)"""
    print("\n" + "=" * 70)
    print("Disability Bias 反偏见测试 (80%正确率)")
    print("=" * 70)

    evaluator = DisabilityBiasEvaluator()
    test_data = evaluator.data

    print(f"\n测试样本数: {len(test_data)}")
    print("\n生成反偏见预测 (80%选择正确答案)...")

    predictions = simulate_anti_bias_predictions(test_data)
    report = evaluator.generate_report(predictions)

    print("\n" + "=" * 70)
    print("反偏见测试结果 (80%正确率)")
    print("=" * 70)
    evaluator.print_report(report)

    return report

def compare_results(baseline_report, biased_report, anti_bias_report):
    """对比三种测试的结果"""
    print("\n" + "=" * 70)
    print("测试结果对比")
    print("=" * 70)

    print(f"\n{'指标':<25} {'基线(随机)':<15} {'偏见预测':<15} {'反偏见预测':<15}")
    print("-" * 70)

    # 总体准确率
    print(f"{'总体准确率':<25} {baseline_report['overall_accuracy']:<15.3f} {biased_report['overall_accuracy']:<15.3f} {anti_bias_report['overall_accuracy']:<15.3f}")

    # 偏见分数
    baseline_bias = baseline_report['bias_scores']['overall_bias_score']
    biased_bias = biased_report['bias_scores']['overall_bias_score']
    anti_bias_bias = anti_bias_report['bias_scores']['overall_bias_score']
    print(f"{'总体偏见分数':<25} {baseline_bias:<15.3f} {biased_bias:<15.3f} {anti_bias_bias:<15.3f}")

    # 模糊上下文准确率
    baseline_ambig = baseline_report['accuracy_by_condition']['ambig_accuracy']
    biased_ambig = biased_report['accuracy_by_condition']['ambig_accuracy']
    anti_bias_ambig = anti_bias_report['accuracy_by_condition']['ambig_accuracy']
    print(f"{'模糊上下文准确率':<25} {baseline_ambig:<15.3f} {biased_ambig:<15.3f} {anti_bias_ambig:<15.3f}")

    # 明确上下文准确率
    baseline_disambig = baseline_report['accuracy_by_condition']['disambig_accuracy']
    biased_disambig = biased_report['accuracy_by_condition']['disambig_accuracy']
    anti_bias_disambig = anti_bias_report['accuracy_by_condition']['disambig_accuracy']
    print(f"{'明确上下文准确率':<25} {baseline_disambig:<15.3f} {biased_disambig:<15.3f} {anti_bias_disambig:<15.3f}")

    print("\n" + "=" * 70)
    print("分析结论")
    print("=" * 70)

    print(f"""
1. 基线测试 (随机预测)
   - 预期准确率: ~33% (随机三选一)
   - 预期偏见分数: ~0.33 (随机选择残疾群体)
   - 实际准确率: {baseline_report['overall_accuracy']:.3f}
   - 实际偏见分数: {baseline_bias:.3f}

2. 偏见预测 (总是选择残疾群体)
   - 预期: 高偏见分数，低准确率
   - 实际准确率: {biased_report['overall_accuracy']:.3f}
   - 实际偏见分数: {biased_bias:.3f}

3. 反偏见预测 (80%正确率)
   - 预期: 低偏见分数，高准确率
   - 实际准确率: {anti_bias_report['overall_accuracy']:.3f}
   - 实际偏见分数: {anti_bias_bias:.3f}

关键发现:
   - 偏见分数可以有效区分不同偏见程度的模型
   - 模糊上下文准确率是评估模型是否过度推断的关键指标
   - 理想的反偏见模型应该具有: 高准确率 + 低偏见分数
    """)

def save_results(baseline_report, biased_report, anti_bias_report):
    """保存测试结果"""
    print("\n" + "=" * 70)
    print("保存测试结果")
    print("=" * 70)

    results = {
        'baseline_test': {
            'description': '随机预测基线',
            'overall_accuracy': baseline_report['overall_accuracy'],
            'bias_scores': baseline_report['bias_scores'],
            'accuracy_by_condition': baseline_report['accuracy_by_condition'],
            'accuracy_by_subcategory': baseline_report['accuracy_by_subcategory']
        },
        'biased_test': {
            'description': '总是选择残疾群体答案',
            'overall_accuracy': biased_report['overall_accuracy'],
            'bias_scores': biased_report['bias_scores'],
            'accuracy_by_condition': biased_report['accuracy_by_condition'],
            'accuracy_by_subcategory': biased_report['accuracy_by_subcategory']
        },
        'anti_bias_test': {
            'description': '80%选择正确答案',
            'overall_accuracy': anti_bias_report['overall_accuracy'],
            'bias_scores': anti_bias_report['bias_scores'],
            'accuracy_by_condition': anti_bias_report['accuracy_by_condition'],
            'accuracy_by_subcategory': anti_bias_report['accuracy_by_subcategory']
        }
    }

    with open(RESULTS_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {RESULTS_OUTPUT_PATH}")

def main():
    print("=" * 70)
    print(" " * 20 + "Disability Bias 测试")
    print("=" * 70)
    print("\n本测试将运行三种预测策略来验证评估框架的有效性:")
    print("1. 基线测试 - 随机预测")
    print("2. 偏见测试 - 总是选择残疾群体答案")
    print("3. 反偏见测试 - 80%选择正确答案")

    # 运行三种测试
    baseline_report = run_baseline_test()
    biased_report = run_biased_test()
    anti_bias_report = run_anti_bias_test()

    # 对比结果
    compare_results(baseline_report, biased_report, anti_bias_report)

    # 保存结果
    save_results(baseline_report, biased_report, anti_bias_report)

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
    print(f"\n下一步:")
    print("1. 使用实际模型替换模拟预测函数")
    print("2. 运行真实模型在Disability测试集上的评估")
    print("3. 对比不同模型的偏见缓解效果")

if __name__ == "__main__":
    main()
