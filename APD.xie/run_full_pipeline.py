"""
APD 完整数据流水线
分阶段运行，支持断点续传
"""

import json
import os
import sys
from datetime import datetime

# 导入之前的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from create_full_dataset import FullDatasetGenerator
from augment_dataset import DataAugmenter

OUTPUT_DIR = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD.xie\apd_results_full"
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'pipeline_checkpoint.json')

class PipelineRunner:
    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.checkpoint = self.load_checkpoint()
        
    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def load_checkpoint(self) -> dict:
        """加载检查点"""
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'stage': 'init',
            'completed': []
        }
    
    def save_checkpoint(self, stage: str):
        """保存检查点"""
        self.checkpoint['stage'] = stage
        self.checkpoint['completed'].append({
            'stage': stage,
            'timestamp': datetime.now().isoformat()
        })
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.checkpoint, f, ensure_ascii=False, indent=2)
        self.log(f"检查点已保存: {stage}")
    
    def stage1_load_and_split(self):
        """阶段1: 加载和划分数据"""
        self.log("\n" + "=" * 60)
        self.log("阶段1: 加载和划分数据")
        self.log("=" * 60)
        
        generator = FullDatasetGenerator()
        
        # 加载全部数据
        generator.load_all_data()
        
        # 划分数据集
        generator.split_data()
        
        # 保存原始数据集
        generator.save_raw_datasets()
        
        # 保存统计信息
        generator.save_stats()
        
        self.save_checkpoint('stage1_complete')
        self.log("\n阶段1完成！")
        self.log(f"原始数据集已保存到: {OUTPUT_DIR}")
        
        return generator
    
    def stage2_generate_train_prompts(self):
        """阶段2: 生成训练集提示词"""
        self.log("\n" + "=" * 60)
        self.log("阶段2: 生成训练集提示词")
        self.log("=" * 60)
        self.log("预计时间: 2-3小时")
        self.log("=" * 60)
        
        generator = FullDatasetGenerator()
        
        # 加载已划分的数据
        train_file = os.path.join(OUTPUT_DIR, 'train_full.json')
        with open(train_file, 'r', encoding='utf-8') as f:
            generator.train_data = json.load(f)
        
        self.log(f"加载训练集: {len(generator.train_data)} 条")
        
        # 生成训练集提示词
        train_sft = generator.process_split(generator.train_data, 'train')
        
        # 保存
        train_file = os.path.join(OUTPUT_DIR, 'sft_train_full.json')
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_sft, f, ensure_ascii=False, indent=2)
        
        self.save_checkpoint('stage2_complete')
        self.log("\n阶段2完成！")
        
        return generator
    
    def stage3_generate_val_prompts(self):
        """阶段3: 生成验证集提示词"""
        self.log("\n" + "=" * 60)
        self.log("阶段3: 生成验证集提示词")
        self.log("=" * 60)
        self.log("预计时间: 30-45分钟")
        self.log("=" * 60)
        
        generator = FullDatasetGenerator()
        
        # 加载已划分的数据
        val_file = os.path.join(OUTPUT_DIR, 'val_full.json')
        with open(val_file, 'r', encoding='utf-8') as f:
            generator.val_data = json.load(f)
        
        self.log(f"加载验证集: {len(generator.val_data)} 条")
        
        # 生成验证集提示词
        val_sft = generator.process_split(generator.val_data, 'val')
        
        # 保存
        val_file = os.path.join(OUTPUT_DIR, 'sft_val_full.json')
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_sft, f, ensure_ascii=False, indent=2)
        
        self.save_checkpoint('stage3_complete')
        self.log("\n阶段3完成！")
        
        return generator
    
    def stage4_generate_test_prompts(self):
        """阶段4: 生成测试集提示词"""
        self.log("\n" + "=" * 60)
        self.log("阶段4: 生成测试集提示词")
        self.log("=" * 60)
        self.log("预计时间: 30-45分钟")
        self.log("=" * 60)
        
        generator = FullDatasetGenerator()
        
        # 加载已划分的数据
        test_file = os.path.join(OUTPUT_DIR, 'test_full.json')
        with open(test_file, 'r', encoding='utf-8') as f:
            generator.test_data = json.load(f)
        
        self.log(f"加载测试集: {len(generator.test_data)} 条")
        
        # 生成测试集提示词
        test_sft = generator.process_split(generator.test_data, 'test')
        
        # 保存
        test_file = os.path.join(OUTPUT_DIR, 'sft_test_full.json')
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_sft, f, ensure_ascii=False, indent=2)
        
        # 合并所有SFT数据
        train_file = os.path.join(OUTPUT_DIR, 'sft_train_full.json')
        val_file = os.path.join(OUTPUT_DIR, 'sft_val_full.json')
        
        with open(train_file, 'r', encoding='utf-8') as f:
            train_sft = json.load(f)
        with open(val_file, 'r', encoding='utf-8') as f:
            val_sft = json.load(f)
        
        all_sft = train_sft + val_sft + test_sft
        all_sft_file = os.path.join(OUTPUT_DIR, 'sft_data_full.json')
        with open(all_sft_file, 'w', encoding='utf-8') as f:
            json.dump(all_sft, f, ensure_ascii=False, indent=2)
        
        self.save_checkpoint('stage4_complete')
        self.log("\n阶段4完成！")
        
        return generator
    
    def stage5_augment_data(self):
        """阶段5: 数据增强"""
        self.log("\n" + "=" * 60)
        self.log("阶段5: 数据增强")
        self.log("=" * 60)
        
        augmenter = DataAugmenter()
        augmenter.run()
        
        self.save_checkpoint('stage5_complete')
        self.log("\n阶段5完成！")
    
    def run_all(self):
        """运行完整流程"""
        self.log("\n" + "=" * 60)
        self.log("APD 完整数据流水线")
        self.log("=" * 60)
        self.log(f"当前检查点: {self.checkpoint['stage']}")
        
        # 阶段1: 加载和划分
        if self.checkpoint['stage'] in ['init']:
            self.stage1_load_and_split()
        else:
            self.log("\n阶段1已完成，跳过")
        
        # 阶段2: 生成训练集提示词
        if self.checkpoint['stage'] in ['init', 'stage1_complete']:
            self.stage2_generate_train_prompts()
        else:
            self.log("\n阶段2已完成，跳过")
        
        # 阶段3: 生成验证集提示词
        if self.checkpoint['stage'] in ['init', 'stage1_complete', 'stage2_complete']:
            self.stage3_generate_val_prompts()
        else:
            self.log("\n阶段3已完成，跳过")
        
        # 阶段4: 生成测试集提示词
        if self.checkpoint['stage'] in ['init', 'stage1_complete', 'stage2_complete', 'stage3_complete']:
            self.stage4_generate_test_prompts()
        else:
            self.log("\n阶段4已完成，跳过")
        
        # 阶段5: 数据增强
        if self.checkpoint['stage'] in ['init', 'stage1_complete', 'stage2_complete', 'stage3_complete', 'stage4_complete']:
            self.stage5_augment_data()
        else:
            self.log("\n阶段5已完成，跳过")
        
        self.log("\n" + "=" * 60)
        self.log("所有阶段完成！")
        self.log("=" * 60)
    
    def run_from_stage(self, stage: int):
        """从指定阶段开始运行"""
        stage_map = {
            1: 'init',
            2: 'stage1_complete',
            3: 'stage2_complete',
            4: 'stage3_complete',
            5: 'stage4_complete'
        }
        
        if stage in stage_map:
            self.checkpoint['stage'] = stage_map[stage]
            self.run_all()
        else:
            self.log(f"无效的阶段: {stage}")


def print_usage():
    print("""
用法:
  python run_full_pipeline.py [命令] [参数]

命令:
  all              - 运行完整流程（自动检测已完成的阶段）
  from_stage <n>   - 从指定阶段开始运行 (1-5)
  reset            - 重置检查点，重新运行

阶段说明:
  阶段1: 加载和划分数据（约5分钟）
  阶段2: 生成训练集提示词（约2-3小时）
  阶段3: 生成验证集提示词（约30-45分钟）
  阶段4: 生成测试集提示词（约30-45分钟）
  阶段5: 数据增强（约10分钟）

示例:
  python run_full_pipeline.py all
  python run_full_pipeline.py from_stage 3
  python run_full_pipeline.py reset
""")


def main():
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1]
    runner = PipelineRunner()
    
    if command == 'all':
        runner.run_all()
    elif command == 'from_stage':
        if len(sys.argv) < 3:
            print("请指定阶段号")
            return
        stage = int(sys.argv[2])
        runner.run_from_stage(stage)
    elif command == 'reset':
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("检查点已重置")
        runner.run_all()
    else:
        print_usage()


if __name__ == "__main__":
    main()
