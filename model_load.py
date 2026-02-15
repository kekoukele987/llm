# 第一步：配置国内镜像 + 导入核心库
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoTokenizer, AutoModel
import torch

# 第二步：加载bge-small-zh模型（固定写法）
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-zh", use_fast=False)
model = AutoModel.from_pretrained("BAAI/bge-small-zh")

# 第三步：基础向量生成函数（核心不变，加注释）
def get_text_embedding(texts):
    """
    生成文本的归一化语义向量（bge-small-zh核心）
    :param texts: 单个文本/文本列表
    :return: 归一化后的向量（shape: [文本数, 768]）
    """
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
    
    # 向量归一化（必须！否则matmul不是余弦相似度）
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings

# 第四步：进阶1 - 带相似度分数的问答（修正笔误）
def rag_qa_with_score(question, threshold=0.5):
    """
    带相似度分数的问答函数（工业界标准写法）
    :param question: 用户问题
    :param threshold: 相似度阈值（低于此值拒绝回答）
    :return: 答案/不知道
    """
    # 生成问题向量
    q_embedding = get_text_embedding(question)
    # 计算和所有知识库的相似度（核心：修正similarities→similarity_scores）
    similarity_scores = torch.matmul(q_embedding, kb_embeddings.T)
    
    # 打印分数（关键：让你知道为什么选这条答案）
    print(f"\n【{question}】的相似度分数：")
    for i, score in enumerate(similarity_scores[0]):
        print(f"  - 和知识库{i+1}的相似度：{score.item():.4f}")
    
    # 找最高分和对应索引
    best_score = similarity_scores.max().item()
    best_idx = torch.argmax(similarity_scores).item()

    # 低于阈值 → 拒绝回答（避免胡说）
    if best_score < threshold:
        return f"🤷‍♂️ 我不知道（最高相似度仅 {best_score:.4f}，低于阈值{threshold}）"
    
    # 提取答案
    best_text = knowledge_base[best_idx]
    answer = best_text.split("？")[1].strip()
    return f"✅ 答案：{answer}（相似度：{best_score:.4f}）"

# 第五步：进阶2 - 批量向量化（处理大量数据）
def batch_embed(texts, batch_size=2):
    """
    批量生成向量（真实场景必用，避免显存溢出）
    :param texts: 文本列表（可上千/上万条）
    :param batch_size: 每批处理的文本数（根据显存调整）
    :return: 所有文本的向量（shape: [总条数, 768]）
    """
    all_embeddings = []
    # 分批处理
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]  # 取当前批次
        batch_emb = get_text_embedding(batch_texts)  # 生成当前批次向量
        all_embeddings.append(batch_emb)  # 存入列表
    
    # 拼接所有批次的向量
    return torch.cat(all_embeddings, dim=0)

# 第六步：进阶3 - 简易向量库（模拟FAISS/Chroma）
class SimpleVectorDB:
    """手写最简向量库（理解工业界向量库的核心逻辑）"""
    def __init__(self):
        self.texts = []  # 存储原始文本
        self.embeddings = None  # 存储文本向量
    
    def add_texts(self, texts):
        """添加文本到向量库（并生成向量）"""
        self.texts = texts
        self.embeddings = batch_embed(texts)  # 批量生成向量
    
    def search(self, query, top_k=1):
        """
        语义检索（找最相似的top_k条）
        :param query: 用户问题
        :param top_k: 返回最相似的k条
        :return: 最相似的文本列表
        """
        q_emb = get_text_embedding(query)
        # 计算相似度
        scores = torch.matmul(q_emb, self.embeddings.T)
        # 取top_k个最高分的索引
        top_k_indices = scores.topk(top_k).indices[0].tolist()
        # 返回对应的文本
        return [self.texts[idx] for idx in top_k_indices]

# ===================== 测试所有进阶功能 =====================
# 1. 构建知识库
knowledge_base = [
    "什么是大模型？大模型是基于海量数据训练的深度学习模型，能处理多种自然语言任务。",
    "微调需要多少数据？轻量级微调只需几十条数据，就能让模型适配特定任务。",
    "GPT2模型有什么特点？GPT2是小型生成式模型，体积小、运行快，适合入门学习。"
]

# 2. 预生成知识库向量（用批量向量化）
kb_embeddings = batch_embed(knowledge_base)
print("✅ 知识库向量形状：", kb_embeddings.shape)  # 输出 torch.Size([3, 768])

# 3. 测试进阶1：带分数的问答
print("=== 进阶1：带相似度分数的问答 ===")
print(rag_qa_with_score("大模型不是啥？", threshold=0.5))
print(rag_qa_with_score("GPT2有啥优势？", threshold=0.5))
print(rag_qa_with_score("Python怎么学？", threshold=0.5))  # 低于阈值，返回不知道

# 4. 测试进阶3：简易向量库
print("\n=== 进阶3：简易向量库检索 ===")
db = SimpleVectorDB()
db.add_texts(knowledge_base)
top1_text = db.search("微调需要多少数据？")[0]
print("检索到的最相似文本：", top1_text)