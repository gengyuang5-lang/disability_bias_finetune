# 实验1: 基线实验 - 中文大模型 + 传统BBQ

## 实验目的
测量中文大模型(DeepSeek/Kimi)在传统BBQ数据集上的原始Disability Bias水平，作为后续对比实验的基准。

## 实验设计

### 实验组
- **基线组**: 中文大模型 + BBQ原始问题

### 评估指标
- 总体准确率
- 总体偏见分数 (越低越好)
- 模糊上下文准确率
- 明确上下文准确率
- 各子类别准确率

## 实验步骤
1. 准备BBQ Disability测试集 (1428条)
2. 调用中文大模型API (DeepSeek/Kimi)
3. 记录模型对每个问题的回答
4. 使用DisabilityBiasEvaluator评估偏见水平
5. 保存实验结果

## 预期结果
- 预期准确率: 30-50% (随机基线为33%)
- 预期偏见分数: 0.3-0.7 (需要实际测量)
- 模糊上下文准确率应低于明确上下文

## 使用方法

### 1. 配置API Key
编辑 `config.json`，设置你的API Key：
```json
"api_config": {
    "deepseek": {
        "api_key": "your_actual_api_key_here"
    }
}
```

### 2. 运行实验
```bash
cd experiments/exp1_baseline_chinese_llm
python run.py
```

### 3. 查看结果
结果将保存在 `results/exp1_baseline_chinese_llm/` 目录下

## 注意事项
- 确保API Key有效且有足够额度
- 测试100条样本大约需要5-10分钟 (含API调用间隔)
- 如需测试更多样本，修改config.json中的test_samples参数
