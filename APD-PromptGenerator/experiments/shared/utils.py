"""
实验共享工具函数
"""

import json
import os
from datetime import datetime


def save_experiment_result(experiment_name, results, output_dir="results"):
    """
    保存实验结果

    Args:
        experiment_name: 实验名称
        results: 实验结果字典
        output_dir: 输出目录
    """
    # 创建结果目录
    result_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(result_dir, exist_ok=True)

    # 生成文件名（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(result_dir, filename)

    # 添加元数据
    results_with_meta = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "results": results
    }

    # 保存结果
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_with_meta, f, ensure_ascii=False, indent=2)

    print(f"实验结果已保存: {filepath}")
    return filepath


def load_experiment_config(config_path):
    """
    加载实验配置

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def log_experiment(experiment_name, message, log_file="experiment.log"):
    """
    记录实验日志

    Args:
        experiment_name: 实验名称
        message: 日志消息
        log_file: 日志文件路径
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{experiment_name}] {message}\n"

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)

    print(log_entry.strip())


def compare_results(result1, result2, metrics=None):
    """
    对比两个实验结果

    Args:
        result1: 第一个实验结果
        result2: 第二个实验结果
        metrics: 要对比的指标列表

    Returns:
        对比结果字典
    """
    if metrics is None:
        metrics = result1.keys()

    comparison = {}
    for metric in metrics:
        if metric in result1 and metric in result2:
            val1 = result1[metric]
            val2 = result2[metric]
            diff = val2 - val1
            comparison[metric] = {
                "experiment_1": val1,
                "experiment_2": val2,
                "difference": diff,
                "improvement": diff > 0
            }

    return comparison
