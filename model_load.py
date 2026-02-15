# 第一步：配置国内镜像，解决模型下载问题
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 第二步：导入核心库
from transformers import AutoTokenizer, AutoModel
import torch

# 第三步：加载bge-small-zh模型（指定使用慢分词器，避免转换报错）
model_name = "BAAI/bge-small-zh"
try:
    # 强制使用慢分词器，避免sentencepiece转换问题
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False  # 关键：禁用fast tokenizer，彻底解决转换报错
    )
    model = AutoModel.from_pretrained(model_name)
    print("✅ 模型和分词器加载成功（网络下载）")
except Exception as e:
    # 兜底：本地加载（如果网络仍有问题）
    print(f"⚠️  网络加载失败，尝试本地加载：{e}")
    print("👉 手动下载模型到 D:\bge-small-zh，地址：https://hf-mirror.com/BAAI/bge-small-zh")
    model_path = r"D:\bge-small-zh"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModel.from_pretrained(model_path)
    print("✅ 模型和分词器加载成功（本地加载）")

# 第四步：定义向量生成函数（bge-small-zh核心）
def get_text_embedding(texts):
    """生成文本的归一化向量"""
    if isinstance(texts, str):
        texts = [texts]
    
    # 编码文本（bge-small-zh标准配置）
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # 生成向量（禁用梯度，提升速度）
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # 取[CLS]位置向量
    
    # 向量归一化（必做，保证相似度计算准确）
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings

# 第五步：构建知识库+语义检索
knowledge_base = [
    "什么是大模型？大模型是基于海量数据训练的深度学习模型，能处理多种自然语言任务。",
    "微调需要多少数据？轻量级微调只需几十条数据，就能让模型适配特定任务。",
    "GPT2模型有什么特点？GPT2是小型生成式模型，体积小、运行快，适合入门学习。"
]

# 预生成知识库向量
kb_embeddings = get_text_embedding(knowledge_base)


def rag_qa_with_score(question, threshold=0.7):
    q_embedding = get_text_embedding(question)
    similarity_scores = torch.matmul(q_embedding, kb_embeddings.T)
    
    # 把分数打印出来！（最关键）
    print("相似度分数（和3条知识库）：", similarity_scores)
    
    best_score = similarity_scores.max().item()
    best_idx = torch.argmax(similarity_scores).item()

    # 低于阈值 → 拒绝回答
    if best_score < threshold:
        return f"[不知道] 最高分只有 {best_score:.2f}"
    
    best_text = knowledge_base[best_idx]
    answer = best_text.split("？")[1].strip()
    return answer

# 第六步：测试效果
print("\n=== BAAI/bge-small-zh 问答效果 ===")
questions = [
    "大模型是啥？",
    "微调需要几条数据？",
    "GPT2有啥特点？",
    "今天晚上吃什么？"
]

for q in questions:
    print(f"问题：{q}")
    print(f"答案：{rag_qa_with_score(q)}\n")

