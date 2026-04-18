# APD-PromptGenerator + LLaMA-Factory 使用指南

## 硬件配置

- **GPU**: RTX 3060 Laptop (6GB 显存)
- **内存**: 16GB
- **模型**: Qwen-2.5-1.5B-Instruct
- **训练方式**: QLoRA (4-bit 量化)

## 快速开始

### 1. 环境检查

确保 LLaMA-Factory 已安装:
```bash
cd "E:/desktop/CDS546 Project Topics (AY2025-26)(2)/biasalert/BiasAlert-repo/code/LLaMA-Factory"
pip install -r requirements.txt
```

### 2. 一键训练

```powershell
# 打开 PowerShell，进入项目目录
cd "E:/desktop/CDS546 Project Topics (AY2025-26)(2)/tune/APD-PromptGenerator"

# 运行训练脚本
.\run_training.ps1
```

选择选项:
- `1`: 完整流程 (数据收集 → 预处理 → SFT → DPO → 合并)
- `2`: 仅训练 (跳过数据收集)
- `3`: 仅 SFT
- `4`: 仅 DPO
- `5`: 仅合并模型

### 3. 分步执行

如果不想用一键脚本，可以分步执行:

#### Step 1: 数据收集
```bash
python src/data_collection/data_collector.py
```

#### Step 2: 数据预处理
```bash
python src/preprocessing/preprocessor.py
```

#### Step 3: 准备 LLaMA-Factory 数据
```bash
python llama_factory_configs/prepare_data.py
```

#### Step 4: SFT 训练
```bash
bash llama_factory_configs/sft_qlora.sh
```

#### Step 5: DPO 训练
```bash
bash llama_factory_configs/dpo_qlora.sh
```

#### Step 6: 合并模型
```bash
cd "E:/desktop/CDS546 Project Topics (AY2025-26)(2)/biasalert/BiasAlert-repo/code/LLaMA-Factory"
python src/export_model.py \
    --model_name_or_path "E:/desktop/CDS546 Project Topics (AY2025-26)(2)/tune/APD-PromptGenerator/models/dpo_qlora" \
    --template qwen \
    --finetuning_type lora \
    --export_dir "E:/desktop/CDS546 Project Topics (AY2025-26)(2)/tune/APD-PromptGenerator/models/final_merged"
```

### 4. 推理测试

训练完成后，使用微调后的模型生成提示词:

```bash
# 使用 LLaMA-Factory 的 CLI 工具
cd "E:/desktop/CDS546 Project Topics (AY2025-26)(2)/biasalert/BiasAlert-repo/code/LLaMA-Factory"
python src/cli_demo.py \
    --model_name_or_path "E:/desktop/CDS546 Project Topics (AY2025-26)(2)/tune/APD-PromptGenerator/models/final_merged" \
    --template qwen
```

## 训练参数说明

### SFT 训练参数
- **模型**: Qwen/Qwen2.5-1.5B-Instruct
- **LoRA rank**: 8
- **LoRA alpha**: 16
- **Batch size**: 1
- **Gradient accumulation**: 8
- **Learning rate**: 5e-5
- **Epochs**: 3

### DPO 训练参数
- **Learning rate**: 1e-5
- **Beta**: 0.1
- **Epochs**: 2

## 时间估算

| 阶段 | 时间 |
|------|------|
| 数据收集 | 10-20 分钟 |
| 数据预处理 | 5 分钟 |
| 下载模型 | 5-10 分钟 (首次) |
| SFT 训练 | 30-50 分钟 |
| DPO 训练 | 20-30 分钟 |
| 合并模型 | 5 分钟 |
| **总计** | **约 1.5-2 小时** |

## 模型输出

训练完成后，模型保存在:
- SFT 模型: `models/sft_qlora/`
- DPO 模型: `models/dpo_qlora/`
- 合并后模型: `models/final_merged/`

## 常见问题

### Q: 显存不足
**A**: 已经使用 4-bit 量化，如果还是不足，可以:
- 减小 `max_length` 到 512
- 减小 `lora_rank` 到 4

### Q: 模型下载慢
**A**: 设置 HuggingFace 镜像:
```bash
set HF_ENDPOINT=https://hf-mirror.com
```

### Q: 训练中断
**A**: LLaMA-Factory 支持断点续训，重新运行脚本即可

## 文件结构

```
APD-PromptGenerator/
├── llama_factory_configs/
│   ├── sft_qlora.sh          # SFT 训练脚本
│   ├── dpo_qlora.sh          # DPO 训练脚本
│   ├── dataset_info.json     # 数据集配置
│   └── prepare_data.py       # 数据转换脚本
├── run_training.ps1          # 一键训练脚本
└── README_LLaMA_Factory.md   # 本文件
```
