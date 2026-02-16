

# 1. 加载文档
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()

from langchain_community.document_loaders import TextLoader

file_path = "90-文档-Data/黑悟空/黑悟空wiki.txt"
loader = TextLoader(file_path, encoding="utf-8")  # LangChain 自带的加载器
docs = loader.load()  # 输出：LangChain 格式的 Document（有 page_content 属性）
# print(docs[0].page_content[:500])  # 打印前500个字符看看内容

# 2. 文档分块
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# 3. 设置嵌入模型
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name=r"D:\code\llm\bge-small-zh",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 4. 创建向量存储
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)

# 5. 构建用户查询
question = "黑悟空有哪些游戏场景？"

# 6. 在向量存储中搜索相关文档，并准备上下文内容
retrieved_docs = vector_store.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 7. 构建提示模板
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
                基于以下上下文，回答问题。如果上下文中没有相关信息，
                请说"我无法从提供的上下文中找到相关信息"。
                上下文: {context}
                问题: {question}
                回答:"""
                                          )

# 8. 使用大语言模型生成答案
from langchain_ollama import ChatOllama # pip install langchain-ollama
llm = ChatOllama (model="qwen:1.8b")
answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer.content)
