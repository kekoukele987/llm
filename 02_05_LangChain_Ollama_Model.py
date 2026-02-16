# 1. 补全前置代码
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 加载文档+分块
file_path = "90-文档-Data/黑悟空/黑悟空wiki.txt"
loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# 嵌入模型+向量库
embeddings = HuggingFaceEmbeddings(
    model_name=r"D:\code\llm\bge-small-zh",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)
retriever = vector_store.as_retriever(search_kwargs={"k":3})  # 生成检索器

# 提示词模板
prompt = ChatPromptTemplate.from_template("""
基于以下上下文，回答问题。如果上下文中没有相关信息，
请说"我无法从提供的上下文中找到相关信息"。
上下文: {context}
问题: {question}
回答:""")

# 大模型
llm = ChatOllama(model="qwen:1.8b")

# 2. 构建 LCEL 链
chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 3. 调用链（只需一行）
result = chain.invoke("黑神话：悟空是腾讯开发的吗？")
print(result)