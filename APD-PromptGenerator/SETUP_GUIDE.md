# APD-PromptGenerator 环境配置指南

## 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **GPU** | NVIDIA GPU with 8GB VRAM | NVIDIA GPU with 24GB+ VRAM |
| **内存** | 16GB RAM | 32GB+ RAM |
| **存储** | 50GB 可用空间 | 100GB+ SSD |

## 推荐的GPU配置

- **LLaMA-3.1-8B (推荐)**: 需要 16GB VRAM (如 RTX 4090, A100)
- **Qwen-7B**: 需要 14GB VRAM (如 RTX 3090, A100)
- **LLaMA-3.1-8B with LoRA**: 仅需 8-12GB VRAM

## 安装依赖

```bash
# 创建conda环境
conda create -n apd-promptgen python=3.10
conda activate apd-promptgen

# 安装PyTorch (根据你的CUDA版本选择)
# CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 安装核心依赖
pip install transformers>=4.35.0
pip install peft>=0.6.0
pip install datasets>=2.14.0
pip install trl>=0.7.0
pip install accelerate>=0.24.0
pip install sentencepiece>=0.1.99
pip install protobuf>=3.20.0

# 安装其他工具
pip install openai>=1.0.0
pip install tqdm>=4.65.0
pip install pandas>=2.0.0
pip install scikit-learn>=1.3.0
```

## HuggingFace 模型下载

### 方法1: 直接使用HuggingFace CLI

```bash
# 安装git lfs
git lfs install

# 下载模型
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./models/llama-3.1-8b
```

### 方法2: 使用Python代码

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

## 环境变量设置

```bash
# 设置HuggingFace token
export HF_TOKEN="your_huggingface_token_here"

# 设置PyTorch CUDA路径 (根据安装位置)
export CUDA_HOME=/path/to/cuda
```

## 验证安装

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## 快速开始

```bash
# 1. 克隆项目
cd APD-PromptGenerator

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载基础模型
# (需要HuggingFace账号和Llama模型权限)

# 4. 运行数据收集
python src/data_collection/data_collector.py

# 5. 运行数据预处理
python src/preprocessing/preprocessor.py

# 6. 运行SFT训练
python src/training/sft/lora_sft_trainer.py

# 7. 运行DPO训练
python src/training/dpo/lora_dpo_trainer.py

# 8. 本地推理测试
python src/inference/local_inference.py
```

## 常见问题

### Q: CUDA out of memory

**A:** 减小batch size，或使用8-bit量化:
```python
# 在加载模型时使用量化
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)
```

### Q: HuggingFace token验证失败

**A:** 确保已登录HuggingFace并接受模型协议:
```bash
huggingface-cli login
```

### Q: 模型下载太慢

**A:** 使用镜像站点:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
