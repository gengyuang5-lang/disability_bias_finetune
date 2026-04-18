"""
推理模块
使用训练好的模型生成anti-bias提示词
"""

import json
import os
from openai import OpenAI

def load_trained_prompts(model_dir):
    optimized_path = os.path.join(model_dir, "optimized_prompts.json")
    if os.path.exists(optimized_path):
        with open(optimized_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def generate_prompt_from_template(question, context="", template=None):
    if template is None:
        template = "在回答涉及残障人士的问题时，请仅基于题目提供的信息作答，避免任何刻板印象或偏见。"

    return template

def generate_prompt_with_llm(question, context="", client=None, model_name="gpt-4o-mini"):
    if client is None:
        client = OpenAI()

    full_question = f"语境：{context}\n问题：{question}" if context else question

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": """你是一个提示词优化专家，专门生成能减少AI模型偏见的提示词。
对于用户输入的每一个问题，你需要生成一个能引导AI公平、无偏见回答的提示词。

提示词应该：
1. 明确要求仅基于题目信息作答
2. 提醒避免刻板印象
3. 保持中立和公平
4. 简短但有效"""},
                {"role": "user", "content": f"请为以下问题生成一个anti-bias提示词：\n\n{full_question}"}
            ]
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"生成失败: {str(e)}"

def batch_generate_prompts(questions, context_list=None, client=None, model_name="gpt-4o-mini"):
    results = []
    context_list = context_list or [""] * len(questions)

    for i, (question, context) in enumerate(zip(questions, context_list)):
        print(f"处理 {i+1}/{len(questions)}...")
        prompt = generate_prompt_with_llm(question, context, client, model_name)
        results.append({
            "question": question,
            "context": context,
            "generated_prompt": prompt
        })

    return results

def generate_with_few_shot(question, context, few_shot_examples, client=None, model_name="gpt-4o-mini"):
    if client is None:
        client = OpenAI()

    messages = [
        {"role": "system", "content": """你是一个提示词优化专家，擅长生成能减少AI模型偏见的提示词。
你会通过few-shot学习来生成更好的提示词。"""}
    ]

    for example in few_shot_examples:
        messages.append({"role": "user", "content": example["question"])
        messages.append({"role": "assistant", "content": example["prompt"])

    full_question = f"语境：{context}\n问题：{question}" if context else question
    messages.append({"role": "user", "content": full_question})

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"生成失败: {str(e)}"

def main():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from configs.config import MODEL_BASE_DIR, DPO_MODEL_DIR

    sample_questions = [
        "有自闭症的人适合做老师吗？",
        "抑郁症患者能正常工作吗？",
        "坐轮椅的人可以独立生活吗？"
    ]

    print("加载训练好的提示词...")
    trained_prompts = load_trained_prompts(DPO_MODEL_DIR)

    if trained_prompts:
        print(f"已加载 {len(trained_prompts)} 个优化过的提示词")
        template = trained_prompts[0].get("optimized", trained_prompts[0].get("chosen"))
    else:
        print("未找到训练好的提示词，使用默认模板")
        template = None

    print("\n使用模板方式生成提示词:")
    for q in sample_questions:
        prompt = generate_prompt_from_template(q, template=template)
        print(f"问题: {q}")
        print(f"提示词: {prompt}")
        print()

if __name__ == "__main__":
    main()
