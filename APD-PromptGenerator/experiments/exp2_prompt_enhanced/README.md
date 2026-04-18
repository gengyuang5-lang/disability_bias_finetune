# 实验2: 对比实验 - 中文大模型 + 微调模型提示词 + BBQ

## 实验目的
测试中文大模型在加入微调模型生成的反偏见提示词后，Disability Bias的改善效果。

## 实验设计

### 实验组
- **对比组**: 中文大模型 + BBQ问题 + 微调模型生成的反偏见提示词

### 评估指标
- 总体准确率
- 总体偏见分数 (越低越好)
- 模糊上下文准确率
- 明确上下文准确率
- 各子类别准确率
- 与基线实验的对比改进

## 实验步骤
1. 加载微调模型 (sft_qlora)
2. 为每个BBQ问题生成反偏见提示词
3. 将提示词与BBQ问题组合
4. 调用中文大模型API (DeepSeek/Kimi)
5. 记录模型回答
6. 使用DisabilityBiasEvaluator评估偏见水平
7. 与基线实验结果对比

## 预期结果
- 准确率应高于基线实验
- 偏见分数应低于基线实验
- 模糊上下文准确率提升更明显

## 使用方法

### 1. 配置API Key和模型路径
编辑 `config.json`：
```json
{
  "api_config": {
    "deepseek": {
      "api_key": "your_actual_api_key_here"
    }
  },
  "parameters": {
    "prompt_generator_path": "path/to/your/sft_qlora"
  }
}
```

### 2. 运行实验
```bash
cd experiments/exp2_prompt_enhanced
python run.py
```

### 3. 查看结果
- 评估结果：`results/exp2_prompt_enhanced/`
- 生成的提示词：`results/exp2_prompt_enhanced_prompts.json`

## 与实验1的对比

运行完实验1和实验2后，可以使用对比分析脚本：
```bash
python ../compare_experiments.py exp1_baseline_chinese_llm exp2_prompt_enhanced
```

## 注意事项
- 本地模型加载需要足够的内存 (~6GB)
- 提示词生成会缓存，避免重复生成
- 测试100条样本大约需要10-15分钟
