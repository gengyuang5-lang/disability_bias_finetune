"""
测试API连接
"""

import json
import requests

# 加载配置
config_path = r"E:\desktop\CDS546 Project Topics (AY2025-26)(2)\tune\APD-PromptGenerator\experiments\exp1_baseline_chinese_llm\config.json"
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

api_config = config['api_config']['azure']
api_key = api_config['api_key']
api_endpoint = api_config['api_endpoint']
deployment_name = api_config['deployment_name']

print("=" * 60)
print("测试Azure API连接")
print("=" * 60)
print(f"API Endpoint: {api_endpoint}")
print(f"Deployment: {deployment_name}")

# 测试提示词
test_prompt = "请回答：1+1等于几？只回答数字。"

url = f"{api_endpoint}chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": deployment_name,
    "messages": [{"role": "user", "content": test_prompt}],
    "temperature": 0.1,
    "max_tokens": 10
}

print(f"\n发送测试请求...")
print(f"提示词: {test_prompt}")

try:
    response = requests.post(url, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    result = response.json()
    answer = result['choices'][0]['message']['content']
    print(f"\n✓ API连接成功!")
    print(f"模型回答: {answer}")
except Exception as e:
    print(f"\n✗ API连接失败: {e}")
    if hasattr(e, 'response') and e.response:
        print(f"错误详情: {e.response.text}")

print("\n" + "=" * 60)
