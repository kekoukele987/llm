import json
import torch
from transformers import AutoTokenizer, AutoModel
import os

# ========== 核心：只加载本地模型（无任何网络请求） ==========
# 替换成你手动下载的模型文件夹路径！！！
LOCAL_MODEL_PATH = r"D:\code\llm\bge-small-zh"

# 加载本地分词器（关闭网络，强制本地加载）
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_PATH,
    use_fast=False,
    local_files_only=True  # 关键：只加载本地文件，不发网络请求
)

# 加载本地模型（关闭网络，强制本地加载）
model = AutoModel.from_pretrained(
    LOCAL_MODEL_PATH,
    local_files_only=True,  # 关键：只加载本地文件
    torch_dtype=torch.float32  # 适配CPU，避免显存问题
)

# ========== 基础向量生成函数（不变） ==========
def get_text_embedding(texts):
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings

def batch_embed(texts, batch_size=2):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_emb = get_text_embedding(batch_texts)
        all_embeddings.append(batch_emb)
    return torch.cat(all_embeddings, dim=0)

# ========== JSON版向量库（不变） ==========
class SimpleVectorDB_JSON:
    def __init__(self):
        self.texts = []
        self.embeddings = None

    def add_texts(self, texts):
        self.texts = texts
        self.embeddings = batch_embed(texts)

    def search(self, query, top_k=1):
        if self.embeddings is None or len(self.texts) == 0:
            return ["❌ 向量库为空，请先添加文本"]
        
        q_emb = get_text_embedding(query)
        scores = torch.matmul(q_emb, self.embeddings.T)
        top_k_indices = scores.topk(top_k).indices[0].tolist()
        return [self.texts[idx] for idx in top_k_indices]

    def save_to_json(self, filepath="vector_db.json"):
        if self.embeddings is None or len(self.texts) == 0:
            print("❌ 向量库为空，无需保存")
            return
        
        embeddings_list = self.embeddings.tolist()
        json_data = {
            "metadata": {
                "version": "1.0",
                "text_count": len(self.texts),
                "embedding_dim": len(embeddings_list[0])
            },
            "texts": self.texts,
            "embeddings": embeddings_list
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        print(f"✅ 向量库已保存为JSON文件：{filepath}")

    def load_from_json(self, filepath="vector_db.json"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            
            self.texts = json_data["texts"]
            self.embeddings = torch.tensor(json_data["embeddings"])
            
            print(f"✅ 从JSON加载成功：")
            print(f"  - 文本数量：{len(self.texts)}")
            print(f"  - 向量维度：{self.embeddings.shape}")
        except FileNotFoundError:
            print(f"❌ 未找到JSON文件：{filepath}")
        except KeyError as e:
            print(f"❌ JSON文件格式错误，缺少字段：{e}")

# ========== 测试（不变） ==========
if __name__ == "__main__":
    knowledge_base = [
        "什么是大模型？大模型是基于海量数据训练的深度学习模型，能处理多种自然语言任务。",
        "微调需要多少数据？轻量级微调只需几十条数据，就能让模型适配特定任务。",
        "GPT2模型有什么特点？GPT2是小型生成式模型，体积小、运行快，适合入门学习。"
    ]

    db = SimpleVectorDB_JSON()
    db.add_texts(knowledge_base)
    print("=== 原始检索测试 ===")
    print(db.search("GPT2有啥特点？")[0])

    db.save_to_json("my_vector_db.json")

    print("\n=== 从JSON加载后检索测试 ===")
    new_db = SimpleVectorDB_JSON()
    new_db.load_from_json("my_vector_db.json")
    print(new_db.search("大模型是啥？")[0])