"""
实验模板脚本

使用说明:
1. 复制此文件夹作为新实验的基础
2. 修改 config.json 中的配置
3. 在 run_experiment() 函数中实现实验逻辑
4. 运行: python run.py
"""

import json
import sys
import os

# 添加共享工具目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import save_experiment_result, load_experiment_config, log_experiment


def run_experiment(config):
    """
    实现实验逻辑

    Args:
        config: 实验配置字典

    Returns:
        实验结果字典
    """
    experiment_name = config['experiment_name']
    log_experiment(experiment_name, "开始实验")

    # TODO: 在这里实现你的实验逻辑
    # 示例:
    # results = {
    #     "metric1": value1,
    #     "metric2": value2
    # }

    results = {
        "status": "completed",
        "note": "请修改此脚本以实现实际实验逻辑"
    }

    log_experiment(experiment_name, "实验完成")
    return results


def main():
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    config = load_experiment_config(config_path)

    print(f"=" * 60)
    print(f"运行实验: {config['experiment_name']}")
    print(f"描述: {config['description']}")
    print(f"=" * 60)

    # 运行实验
    results = run_experiment(config)

    # 保存结果
    output_dir = config.get('data_paths', {}).get('output', 'results')
    save_experiment_result(
        experiment_name=config['experiment_name'],
        results=results,
        output_dir=output_dir
    )

    print(f"\n实验完成!")


if __name__ == "__main__":
    main()
