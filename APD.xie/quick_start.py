"""
快速启动脚本
先运行阶段1，查看数据规模和统计信息
"""

import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from create_full_dataset import FullDatasetGenerator

def main():
    print("=" * 60)
    print("APD 完整数据集生成 - 快速启动")
    print("=" * 60)
    print("\n本脚本将运行阶段1：加载和划分数据")
    print("预计时间: 5分钟")
    print("=" * 60)
    
    generator = FullDatasetGenerator()
    
    # 1. 加载全部数据
    generator.load_all_data()
    
    # 2. 划分数据集
    generator.split_data()
    
    # 3. 保存原始数据集
    generator.save_raw_datasets()
    
    # 4. 保存统计信息
    generator.save_stats()
    
    print("\n" + "=" * 60)
    print("阶段1完成！")
    print("=" * 60)
    print("\n数据已准备就绪，可以开始生成提示词了。")
    print("\n下一步选项:")
    print("  1. 运行完整流程: python run_full_pipeline.py all")
    print("  2. 从阶段2开始: python run_full_pipeline.py from_stage 2")
    print("  3. 单独运行阶段2: 修改本脚本只运行阶段2")
    print("\n提示: 建议使用 run_full_pipeline.py 进行分阶段运行")
    print("      这样可以避免长时间运行中断导致的数据丢失。")

if __name__ == "__main__":
    main()
