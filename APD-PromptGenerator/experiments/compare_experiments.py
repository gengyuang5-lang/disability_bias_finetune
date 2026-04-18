"""
实验对比分析脚本

对比两个实验的结果，生成对比报告

使用方法:
python compare_experiments.py <exp1_name> <exp2_name>

示例:
python compare_experiments.py exp1_baseline_chinese_llm exp2_prompt_enhanced
"""

import json
import sys
import os
import glob
from datetime import datetime


def find_latest_result(experiment_name):
    """找到实验的最新结果文件"""
    result_dir = os.path.join(os.path.dirname(__file__), experiment_name, 'results', experiment_name)

    if not os.path.exists(result_dir):
        return None

    # 查找所有结果文件
    pattern = os.path.join(result_dir, f"{experiment_name}_*.json")
    files = glob.glob(pattern)

    if not files:
        return None

    # 返回最新的文件
    latest_file = max(files, key=os.path.getctime)
    return latest_file


def load_experiment_result(filepath):
    """加载实验结果"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def compare_experiments(exp1_name, exp2_name):
    """对比两个实验"""
    print("=" * 70)
    print(f"实验对比分析: {exp1_name} vs {exp2_name}")
    print("=" * 70)

    # 加载实验结果
    exp1_file = find_latest_result(exp1_name)
    exp2_file = find_latest_result(exp2_name)

    if not exp1_file:
        print(f"错误: 找不到实验 {exp1_name} 的结果文件")
        return

    if not exp2_file:
        print(f"错误: 找不到实验 {exp2_name} 的结果文件")
        return

    print(f"\n加载实验结果:")
    print(f"  实验1: {exp1_file}")
    print(f"  实验2: {exp2_file}")

    exp1_data = load_experiment_result(exp1_file)
    exp2_data = load_experiment_result(exp2_file)

    exp1_results = exp1_data.get('results', {})
    exp2_results = exp2_data.get('results', {})

    # 对比基本信息
    print(f"\n{'='*70}")
    print("基本信息对比")
    print(f"{'='*70}")
    print(f"{'指标':<25} {'实验1 (基线)':<20} {'实验2 (对比)':<20}")
    print("-" * 70)
    print(f"{'实验名称':<25} {exp1_name:<20} {exp2_name:<20}")
    print(f"{'测试样本数':<25} {exp1_results.get('test_samples', 'N/A'):<20} {exp2_results.get('test_samples', 'N/A'):<20}")
    print(f"{'API提供商':<25} {exp1_results.get('api_provider', 'N/A'):<20} {exp2_results.get('api_provider', 'N/A'):<20}")

    # 对比核心指标
    print(f"\n{'='*70}")
    print("核心指标对比")
    print(f"{'='*70}")
    print(f"{'指标':<25} {'实验1':<15} {'实验2':<15} {'改进':<15}")
    print("-" * 70)

    # 总体准确率
    acc1 = exp1_results.get('overall_accuracy', 0)
    acc2 = exp2_results.get('overall_accuracy', 0)
    acc_diff = acc2 - acc1
    acc_improved = "↑" if acc_diff > 0 else "↓" if acc_diff < 0 else "→"
    print(f"{'总体准确率':<25} {acc1:<15.3f} {acc2:<15.3f} {acc_diff:+.3f} {acc_improved}")

    # 偏见分数
    bias1 = exp1_results.get('bias_scores', {}).get('overall_bias_score', 0)
    bias2 = exp2_results.get('bias_scores', {}).get('overall_bias_score', 0)
    bias_diff = bias1 - bias2  # 偏见分数降低是好事
    bias_improved = "↑" if bias_diff > 0 else "↓" if bias_diff < 0 else "→"
    print(f"{'总体偏见分数':<25} {bias1:<15.3f} {bias2:<15.3f} {bias_diff:+.3f} {bias_improved} (降低更好)")

    # 模糊上下文准确率
    ambig1 = exp1_results.get('accuracy_by_condition', {}).get('ambig_accuracy', 0)
    ambig2 = exp2_results.get('accuracy_by_condition', {}).get('ambig_accuracy', 0)
    ambig_diff = ambig2 - ambig1
    ambig_improved = "↑" if ambig_diff > 0 else "↓" if ambig_diff < 0 else "→"
    print(f"{'模糊上下文准确率':<25} {ambig1:<15.3f} {ambig2:<15.3f} {ambig_diff:+.3f} {ambig_improved}")

    # 明确上下文准确率
    disambig1 = exp1_results.get('accuracy_by_condition', {}).get('disambig_accuracy', 0)
    disambig2 = exp2_results.get('accuracy_by_condition', {}).get('disambig_accuracy', 0)
    disambig_diff = disambig2 - disambig1
    disambig_improved = "↑" if disambig_diff > 0 else "↓" if disambig_diff < 0 else "→"
    print(f"{'明确上下文准确率':<25} {disambig1:<15.3f} {disambig2:<15.3f} {disambig_diff:+.3f} {disambig_improved}")

    # 子类别对比
    print(f"\n{'='*70}")
    print("子类别准确率对比")
    print(f"{'='*70}")
    print(f"{'子类别':<25} {'实验1':<15} {'实验2':<15} {'改进':<15}")
    print("-" * 70)

    subcat1 = exp1_results.get('accuracy_by_subcategory', {})
    subcat2 = exp2_results.get('accuracy_by_subcategory', {})

    all_subcats = set(subcat1.keys()) | set(subcat2.keys())
    for subcat in sorted(all_subcats):
        val1 = subcat1.get(subcat, {}).get('accuracy', 0) if isinstance(subcat1.get(subcat), dict) else subcat1.get(subcat, 0)
        val2 = subcat2.get(subcat, {}).get('accuracy', 0) if isinstance(subcat2.get(subcat), dict) else subcat2.get(subcat, 0)
        diff = val2 - val1
        improved = "↑" if diff > 0 else "↓" if diff < 0 else "→"
        print(f"{subcat:<25} {val1:<15.3f} {val2:<15.3f} {diff:+.3f} {improved}")

    # 总结
    print(f"\n{'='*70}")
    print("总结")
    print(f"{'='*70}")

    improvements = []
    if acc_diff > 0:
        improvements.append(f"准确率提升 {acc_diff:.3f}")
    if bias_diff > 0:
        improvements.append(f"偏见分数降低 {bias_diff:.3f}")
    if ambig_diff > 0:
        improvements.append(f"模糊上下文准确率提升 {ambig_diff:.3f}")

    if improvements:
        print("✓ 改进效果:")
        for imp in improvements:
            print(f"  - {imp}")
    else:
        print("✗ 未观察到明显改进")

    # 保存对比报告
    report = {
        "comparison_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_1": {
            "name": exp1_name,
            "file": exp1_file,
            "results": {
                "overall_accuracy": acc1,
                "bias_score": bias1,
                "ambig_accuracy": ambig1,
                "disambig_accuracy": disambig1
            }
        },
        "experiment_2": {
            "name": exp2_name,
            "file": exp2_file,
            "results": {
                "overall_accuracy": acc2,
                "bias_score": bias2,
                "ambig_accuracy": ambig2,
                "disambig_accuracy": disambig2
            }
        },
        "improvements": {
            "accuracy": acc_diff,
            "bias_reduction": bias_diff,
            "ambig_accuracy": ambig_diff,
            "disambig_accuracy": disambig_diff
        },
        "summary": improvements
    }

    report_path = os.path.join(os.path.dirname(__file__), 'comparison_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n对比报告已保存: {report_path}")


def main():
    if len(sys.argv) != 3:
        print("使用方法: python compare_experiments.py <exp1_name> <exp2_name>")
        print("示例: python compare_experiments.py exp1_baseline_chinese_llm exp2_prompt_enhanced")
        sys.exit(1)

    exp1_name = sys.argv[1]
    exp2_name = sys.argv[2]

    compare_experiments(exp1_name, exp2_name)


if __name__ == "__main__":
    main()
