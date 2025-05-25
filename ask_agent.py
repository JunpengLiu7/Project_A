import os
import subprocess
import gradio as gr
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from requests.exceptions import ConnectionError

# 检查并尝试启动 Ollama
def ensure_ollama_running():
    import requests
    try:
        _ = requests.get("http://localhost:11434")
    except ConnectionError:
        print("🔄 Ollama 未运行，正在尝试启动服务...")
        if os.name == "nt":
            subprocess.Popen(["start", "cmd", "/k", "ollama serve"], shell=True)
        else:
            subprocess.Popen(["ollama", "serve"])
        print("⏳ 等待 Ollama 启动...")
        import time
        time.sleep(10)

# 初始化 Ollama + Langchain
def initialize_qa():
    ensure_ollama_running()
    print("🔍 正在加载 Mistral 模型...")
    llm = Ollama(model="mistral", temperature=0.3)

    print("🔍 正在加载向量数据库...")
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="md_vector_db", embedding_function=embedding)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
        return_source_documents=True
    )
    return qa

qa = initialize_qa()

def ask_question(query):
    result = qa(query)
    answer = result["result"]
    sources = result["source_documents"]

    source_info = "\n\n📚 来源段落：\n"
    for i, doc in enumerate(sources):
        snippet = doc.page_content.strip()[:150].replace("\n", " ")
        source_info += f"{i+1}. {snippet}...\n"

    return answer + source_info

# 启动 Gradio UI
iface = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(lines=3, placeholder="请输入你的问题，例如：这篇论文的方法有什么亮点？"),
    outputs="text",
    title="📄 Markdown 论文语义问答",
    description="基于本地向量数据库与 Mistral 模型，支持结构化论文问答"
)

if __name__ == "__main__":
    iface.launch()
