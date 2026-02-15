# 无需配置镜像（加载本地模型）
import os
import re
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import pandas as pd

# ===================== 1. 加载本地模型（路径替换为你的） =====================
model_path = r"D:\code\gpt2-chinese-cluecorpussmall"  # 替换成你的模型文件夹路径

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")

# 手动添加特殊令牌
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
tokenizer.pad_token = tokenizer.eos_token

# 加载模型并调整嵌入层
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype="auto"
)
model.resize_token_embeddings(len(tokenizer))

# ===================== 2. 优化：文本清洗+生成后截断（核心） =====================
def clean_generation_output(text):
    """增强版清洗：不仅去乱码，还截断重复内容"""
    # 基础清洗
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])', '', text)
    text = re.sub(r'([？！。，])+', r'\1', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9？！。，：；""''（）【】]', '', text)
    
    # 新增：截断重复的核心短语（比如“大模型是什麼”重复）
    # 匹配连续重复的3字以上短语，只保留1次
    text = re.sub(r'(.{3,}?)\1+', r'\1', text)
    
    # 新增：按句号/问号截断，只保留第一个完整回答
    if '。' in text:
        text = text.split('。')[0] + '。'
    elif '？' in text:
        text = text.split('？')[0] + '？'
    
    return text.strip()

# ===================== 3. 优化：训练数据格式（让模型更易学习） =====================
# 调整数据格式：增加分隔符，明确问题/答案边界
dataset_raw = [
    {"question": "什么是大模型？", "answer": "大模型是基于海量数据训练的深度学习模型，能处理多种自然语言任务。"},
    {"question": "微调需要多少数据？", "answer": "轻量级微调只需几十条数据，就能让模型适配特定任务。"},
    {"question": "GPT2模型有什么特点？", "answer": "GPT2是小型生成式模型，体积小、运行快，适合入门学习。"}
]

# 优化格式：【问题】xxx\n【答案】xxx（让模型明确学习“问题→答案”的映射）
df = pd.DataFrame(dataset_raw)
df["text"] = df.apply(
    lambda x: f"【问题】{x['question']}\n【答案】{x['answer']}{tokenizer.eos_token}",  # 加结束令牌
    axis=1
)
dataset = Dataset.from_pandas(df)

# 分词处理（缩短长度，适配小模型）
def preprocess_function(examples):
    inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=64,  # 从128缩短到64，减少小模型学习负担
        padding="max_length"
    )
    # 优化：对padding部分的label做掩码（-100，模型不学习填充部分）
    inputs["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in input_ids]
        for input_ids in inputs["input_ids"]
    ]
    return inputs

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

# ===================== 4. 优化：LoRA+训练参数（适配小模型） =====================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # 从4提升到8，增强微调效果
    lora_alpha=32,
    lora_dropout=0.05,  # 降低dropout，小数据集少正则化
    target_modules=["c_attn"],
    bias="none",
    inference_mode=False
)

model_peft = get_peft_model(model, lora_config)
print("=== 可训练参数 ===")
model_peft.print_trainable_parameters()

# 核心：优化训练参数（小模型专用）
training_args = TrainingArguments(
    output_dir="./gpt2-lora-finetune",
    per_device_train_batch_size=1,  # 降到1，提升稳定性
    gradient_accumulation_steps=8,  # 累积梯度，变相增大批次
    learning_rate=5e-5,  # 从1e-4降到5e-5，小模型学习率要小
    num_train_epochs=20,  # 增加到20轮，让模型学透小数据集
    logging_steps=1,
    save_strategy="no",
    fp16=False,
    report_to="none",
    optim="adamw_torch",  # 指定优化器，提升训练效果
    max_grad_norm=1.0,  # 梯度裁剪，防止梯度爆炸
)

# 数据整理器（优化：禁用mlm，适配因果语言模型）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)

# 启动训练
trainer = Trainer(
    model=model_peft,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)
trainer.train()
print("\n=== 微调完成 ===")

# ===================== 5. 优化：生成参数（彻底抑制重复） =====================
def generate_answer(prompt):
    """封装生成函数，用最优参数生成"""
    # 构造输入格式（和训练数据一致）
    input_text = f"【问题】{prompt}\n【答案】"
    inputs = tokenizer(input_text, return_tensors="pt").to(model_peft.device)
    
    # 极致抑制重复的生成参数
    outputs = model_peft.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.01,  # 极低随机性，强制模型输出学过的内容
        top_p=0.1,         # 极小采样范围
        repetition_penalty=2.0,  # 高重复惩罚，彻底杜绝重复
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,   # 关闭采样，用贪心搜索（小模型更稳定）
        num_beams=5,       # 束搜索提升质量
        early_stopping=True,  # 生成到结束令牌就停止
    )
    
    # 解码+清洗
    raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 只提取【答案】后的内容
    if "【答案】" in raw_response:
        raw_response = raw_response.split("【答案】")[-1]
    cleaned_response = clean_generation_output(raw_response)
    return cleaned_response

# 测试微调效果
test_prompt = "什么是大模型？"
final_answer = generate_answer(test_prompt)
print(f"\n=== 最终微调效果 ===")
print(f"问题：{test_prompt}")
print(f"答案：{final_answer}")

# 测试第二个问题
test_prompt2 = "微调需要多少数据？"
final_answer2 = generate_answer(test_prompt2)
print(f"\n问题：{test_prompt2}")
print(f"答案：{final_answer2}")