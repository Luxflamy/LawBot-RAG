import streamlit as st
import os
import json
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import Docx2txtLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# Configuration and loading
load_dotenv(find_dotenv())  # Load environment variables from .env file

# Document-specific configurations
with open("data/fakao_gpt4.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

from langchain.docstore.document import Document
texts = [
    Document(page_content=item["output"], metadata={"question": item["input"], "type": item["type"]})
    for item in json_data
]

# Text splitting and embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
texts = text_splitter.split_documents(texts)
texts = texts[:500]  # 限制最多处理前 500 条文档块
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002", api_key=os.environ["OPENAI_API_KEY"]
)

# Vector store and retriever
vector_store = vector_store = FAISS.from_documents(texts, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# Question-answering system
prompt_template = """
你是一位专业的中文法律咨询助手。请根据以下法律知识内容，使用中文回答用户提出的问题。

法律参考资料：
{context}

用户问题：
{question}

请用简洁、正式、专业的中文回答。如果无法确定答案，请明确说明。
"""

QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), 
    chain_type="stuff", 
    retriever=retriever, 
    chain_type_kwargs={"prompt": QA_PROMPT}
)
llm = OpenAI(temperature=0.5, max_tokens=1000)

def rag_answer(query):
    try:
        response = qa.run(query)
        return f"【RAG模式回答】：\n{response}"
    except Exception as e:
        return f"RAG模式出错：{e}"

def prompt_only_answer(query):
    prompt = f"""你是一位专业的中文法律咨询助手，请根据以下问题提供权威解答：

问题：{query}

请以“根据我国现行法律规定…”开头回答。"""
    try:
        response = llm(prompt)
        return f"【Prompt模式回答】：\n{response}"
    except Exception as e:
        return f"Prompt模式出错：{e}"

def main():
    st.set_page_config(page_title="法律服务助手")

    st.title("🧑‍⚖️ AI 法律咨询助手")
    st.markdown("请选择模式，并输入您要咨询的法律问题：")

    # 模式选择
    mode = st.radio("问答模式", ["RAG 模式（知识库检索）", "Prompt-only 模式"])

    query = st.text_input("请输入问题：", placeholder="例如：公司辞退员工是否需要赔偿？")

    if query:
        if mode == "RAG 模式（知识库检索）":
            response = rag_answer(query)
        else:
            response = prompt_only_answer(query)

        st.markdown("### 💡 回答：")
        st.write(response)

if __name__ == "__main__":
    main()
