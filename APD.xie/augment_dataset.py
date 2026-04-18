"""
数据集增强脚本
对生成的SFT训练数据进行增强，增加数据多样性
"""

import json
import random
import os
from typing import List, Dict
from datetime import datetime

INPUT_DIR = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD.xie\apd_results_full"
OUTPUT_DIR = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD.xie\apd_results_full_augmented"

class DataAugmenter:
    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    def log(self, message: str):
        """打印日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def load_sft_data(self, filename: str) -> List[Dict]:
        """加载SFT数据"""
        filepath = os.path.join(INPUT_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def paraphrase_prompt(self, prompt: str) -> str:
        """改写提示词，生成不同表述"""
        # 改写模板
        templates = [
            "请在回答时{action}，不要{target}。",
            "重要提醒：{reminder}，避免{bias}。",
            "注意：{note}，请保持{principle}。",
            "提示：{hint}，{suggestion}。",
            "请记住：{principle}，不要{avoid}。"
        ]
        
        # 根据原提示词内容选择合适的改写方式
        if "客观" in prompt or "中立" in prompt:
            return f"请在回答时保持客观公正的态度，不要基于任何刻板印象做出判断。"
        elif "平等" in prompt or "对待" in prompt:
            return f"重要提醒：每个人都应被平等对待，避免因残疾状况而产生偏见。"
        elif "刻板印象" in prompt or "假设" in prompt:
            return f"注意：请避免基于刻板印象的假设，根据具体情境理性分析。"
        else:
            return f"提示：请在回答时保持公正，不要对残疾群体做出片面判断。"
    
    def add_context_variation(self, record: Dict) -> Dict:
        """添加上下文变体"""
        # 复制原记录
        new_record = record.copy()
        
        # 改写提示词
        original_prompt = record['anti_bias_prompt']
        new_prompt = self.paraphrase_prompt(original_prompt)
        
        # 更新消息
        new_record['messages'] = [
            record['messages'][0],  # system
            record['messages'][1],  # user
            {
                "role": "assistant",
                "content": new_prompt
            }
        ]
        new_record['anti_bias_prompt'] = new_prompt
        new_record['augmented'] = True
        new_record['original_prompt'] = original_prompt
        
        return new_record
    
    def augment_dataset(self, data: List[Dict], augmentation_factor: int = 2) -> List[Dict]:
        """增强数据集"""
        self.log(f"原始数据: {len(data)} 条")
        self.log(f"增强倍数: {augmentation_factor}x")
        
        augmented_data = data.copy()  # 保留原始数据
        
        for i in range(1, augmentation_factor):
            self.log(f"生成第 {i+1} 轮增强数据...")
            for record in data:
                # 为每条记录生成变体
                variant = self.add_context_variation(record)
                augmented_data.append(variant)
        
        self.log(f"增强后数据: {len(augmented_data)} 条")
        return augmented_data
    
    def save_augmented_data(self, data: List[Dict], filename: str):
        """保存增强后的数据"""
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.log(f"已保存: {filepath}")
    
    def run(self):
        """运行增强流程"""
        self.log("=" * 60)
        self.log("数据集增强")
        self.log("=" * 60)
        
        # 1. 加载训练集
        self.log("\n加载训练集...")
        train_data = self.load_sft_data('sft_train_full.json')
        
        # 2. 增强训练集
        self.log("\n增强训练集...")
        train_augmented = self.augment_dataset(train_data, augmentation_factor=2)
        
        # 3. 保存增强后的训练集
        self.save_augmented_data(train_augmented, 'sft_train_augmented.json')
        
        # 4. 复制验证集和测试集（不需要增强）
        self.log("\n复制验证集和测试集...")
        val_data = self.load_sft_data('sft_val_full.json')
        test_data = self.load_sft_data('sft_test_full.json')
        
        self.save_augmented_data(val_data, 'sft_val.json')
        self.save_augmented_data(test_data, 'sft_test.json')
        
        # 5. 合并所有数据
        all_data = train_augmented + val_data + test_data
        self.save_augmented_data(all_data, 'sft_data_all.json')
        
        # 6. 保存统计信息
        stats = {
            'train_original': len(train_data),
            'train_augmented': len(train_augmented),
            'val': len(val_data),
            'test': len(test_data),
            'total': len(all_data),
            'augmentation_factor': 2
        }
        
        stats_file = os.path.join(OUTPUT_DIR, 'augmentation_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        self.log("\n" + "=" * 60)
        self.log("数据增强完成！")
        self.log("=" * 60)
        self.log(f"输出目录: {OUTPUT_DIR}")
        self.log(f"训练集: {stats['train_original']} → {stats['train_augmented']} 条")
        self.log(f"验证集: {stats['val']} 条")
        self.log(f"测试集: {stats['test']} 条")
        self.log(f"总计: {stats['total']} 条")


def main():
    augmenter = DataAugmenter()
    augmenter.run()


if __name__ == "__main__":
    main()
