"""
APD-PromptGenerator 主入口脚本
一键运行完整训练流程
"""

import os
import sys

def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def run_step(name, func):
    print_header(name)
    try:
        func()
        print(f"✓ {name} 完成")
    except Exception as e:
        print(f"✗ {name} 失败: {e}")
        return False
    return True

def main():
    print_header("APD-PromptGenerator 训练流程")

    steps = [
        ("Step 1: 数据收集", lambda: exec(open("src/data_collection/data_collector.py", encoding='utf-8').read())),
        ("Step 2: 数据预处理", lambda: exec(open("src/preprocessing/preprocessor.py", encoding='utf-8').read())),
        ("Step 3: SFT训练", lambda: exec(open("src/training/sft/sft_trainer.py", encoding='utf-8').read())),
        ("Step 4: DPO训练", lambda: exec(open("src/training/dpo/dpo_trainer.py", encoding='utf-8').read())),
        ("Step 5: 模型评估", lambda: exec(open("src/evaluation/evaluator.py", encoding='utf-8').read())),
    ]

    for name, func in steps:
        if not run_step(name, func):
            print("\n训练流程中断")
            sys.exit(1)

    print_header("训练流程完成!")
    print("\n下一步:")
    print("1. 查看 models/ 目录下的训练结果")
    print("2. 使用 src/inference/prompt_generator.py 生成提示词")
    print("3. 查看 evaluation_report.json 了解模型表现")

if __name__ == "__main__":
    main()
