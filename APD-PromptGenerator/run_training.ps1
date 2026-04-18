# APD-PromptGenerator 一键训练脚本 (PowerShell)
# 适配 Windows + RTX 3060 Laptop

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  APD-PromptGenerator 训练流程" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$ErrorActionPreference = "Stop"

# 设置路径
$ProjectDir = "E:/desktop/CDS546 Project Topics (AY2025-26)(2)/tune/APD-PromptGenerator"
$LLaMAFactoryDir = "E:/desktop/CDS546 Project Topics (AY2025-26)(2)/biasalert/BiasAlert-repo/code/LLaMA-Factory"

function Step-1-DataCollection {
    Write-Host "`n[Step 1/5] 数据收集..." -ForegroundColor Yellow
    Set-Location $ProjectDir
    python src/data_collection/data_collector.py
    if ($LASTEXITCODE -ne 0) { throw "数据收集失败" }
    Write-Host "✓ 数据收集完成" -ForegroundColor Green
}

function Step-2-Preprocessing {
    Write-Host "`n[Step 2/5] 数据预处理..." -ForegroundColor Yellow
    Set-Location $ProjectDir
    python src/preprocessing/preprocessor.py
    if ($LASTEXITCODE -ne 0) { throw "数据预处理失败" }
    Write-Host "✓ 数据预处理完成" -ForegroundColor Green
}

function Step-3-PrepareLLaMAFactory {
    Write-Host "`n[Step 3/5] 准备 LLaMA-Factory 数据..." -ForegroundColor Yellow
    Set-Location $ProjectDir
    python llama_factory_configs/prepare_data.py
    if ($LASTEXITCODE -ne 0) { throw "数据准备失败" }
    Write-Host "✓ LLaMA-Factory 数据准备完成" -ForegroundColor Green
}

function Step-4-SFT {
    Write-Host "`n[Step 4/5] SFT QLoRA 训练..." -ForegroundColor Yellow
    Write-Host "这将使用 Qwen-2.5-1.5B-Instruct 模型进行 SFT 训练" -ForegroundColor Gray
    Write-Host "预计时间: 30-50 分钟" -ForegroundColor Gray
    
    Set-Location $ProjectDir
    bash llama_factory_configs/sft_qlora.sh
    if ($LASTEXITCODE -ne 0) { throw "SFT 训练失败" }
    Write-Host "✓ SFT 训练完成" -ForegroundColor Green
}

function Step-5-DPO {
    Write-Host "`n[Step 5/5] DPO QLoRA 训练..." -ForegroundColor Yellow
    Write-Host "这将基于 SFT 模型进行 DPO 训练" -ForegroundColor Gray
    Write-Host "预计时间: 20-30 分钟" -ForegroundColor Gray
    
    Set-Location $ProjectDir
    bash llama_factory_configs/dpo_qlora.sh
    if ($LASTEXITCODE -ne 0) { throw "DPO 训练失败" }
    Write-Host "✓ DPO 训练完成" -ForegroundColor Green
}

function Step-6-Merge {
    Write-Host "`n[Step 6/6] 合并 LoRA 权重..." -ForegroundColor Yellow
    
    Set-Location $LLaMAFactoryDir
    python src/export_model.py `
        --model_name_or_path "${ProjectDir}/models/dpo_qlora" `
        --template qwen `
        --finetuning_type lora `
        --export_dir "${ProjectDir}/models/final_merged" `
        --export_size 2 `
        --export_device cpu `
        --export_legacy_format False
    
    if ($LASTEXITCODE -ne 0) { throw "模型合并失败" }
    Write-Host "✓ 模型合并完成" -ForegroundColor Green
}

# 主流程
Write-Host "`n请选择要执行的步骤:" -ForegroundColor Cyan
Write-Host "1. 完整流程 (数据收集 → 预处理 → SFT → DPO → 合并)"
Write-Host "2. 仅训练 (跳过数据收集和预处理)"
Write-Host "3. 仅 SFT 训练"
Write-Host "4. 仅 DPO 训练"
Write-Host "5. 仅合并模型"

$choice = Read-Host "`n请输入选项 (1-5)"

try {
    switch ($choice) {
        "1" {
            Step-1-DataCollection
            Step-2-Preprocessing
            Step-3-PrepareLLaMAFactory
            Step-4-SFT
            Step-5-DPO
            Step-6-Merge
        }
        "2" {
            Step-3-PrepareLLaMAFactory
            Step-4-SFT
            Step-5-DPO
            Step-6-Merge
        }
        "3" {
            Step-3-PrepareLLaMAFactory
            Step-4-SFT
        }
        "4" {
            Step-5-DPO
            Step-6-Merge
        }
        "5" {
            Step-6-Merge
        }
        default {
            Write-Host "无效选项" -ForegroundColor Red
            exit 1
        }
    }
    
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "  训练流程全部完成!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "`n模型保存位置: ${ProjectDir}/models/final_merged" -ForegroundColor Cyan
}
catch {
    Write-Host "`n========================================" -ForegroundColor Red
    Write-Host "  训练流程出错!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "错误信息: $_" -ForegroundColor Red
    exit 1
}
