import os
import subprocess
import gradio as gr
import time
from requests.exceptions import ConnectionError

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.vectorstores.faiss import FAISS

# === 图像索引加载 ===

def load_faiss_index(index_dir="faiss_image_index"):
    index_faiss = os.path.join(index_dir, "index.faiss")
    index_pkl = os.path.join(index_dir, "index.pkl")

    if not os.path.exists(index_faiss) or not os.path.exists(index_pkl):
        print("⚠️ 图像索引不存在，跳过图像检索。")
        return None, []
    index = FAISS.load_local(index_dir, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    with open(os.path.join(index_dir, "paths.txt"), "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f.readlines()]
    return index, paths

def search_image_vectors(query, top_k=4):
    try:
        index, paths = load_faiss_index()
        if index is None:
            return []
        docs_and_scores = index.similarity_search_with_score(query, k=top_k)
        results = [doc.metadata["source"] for doc, _ in docs_and_scores]
        return results
    except Exception as e:
        print("❌ 图像检索失败：", e)
        return []

# === Ollama 启动检测 ===

def ensure_ollama_running():
    import requests
    try:
        requests.get("http://localhost:11434")
    except ConnectionError:
        print("🔄 Ollama 未运行，正在尝试启动服务...")
        if os.name == "nt":
            subprocess.Popen(["start", "cmd", "/k", "ollama serve"], shell=True)
        else:
            subprocess.Popen(["ollama", "serve"])
        time.sleep(10)

# === 文本问答链初始化 ===

def init_text_qa():
    print("🔍 正在加载文本向量数据库...")
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="md_vector_db", embedding_function=embedding)

    llm = Ollama(model="mistral", temperature=0.3)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
        return_source_documents=True
    )
    return qa

# === 初始化系统 ===

ensure_ollama_running()
qa = init_text_qa()

# === 多模态问答主函数 ===

def multimodal_qa(query):
    result = qa(query)
    answer = result["result"]
    source_docs = result["source_documents"]

    text_info = "\n\n📚 来源段落：\n"
    for i, doc in enumerate(source_docs):
        snippet = doc.page_content.strip().replace("\n", " ")[:150]
        text_info += f"{i+1}. {snippet}...\n"

    # 图像搜索
    image_paths = search_image_vectors(query)
    return answer + text_info, image_paths

# === Gradio UI ===

iface = gr.Interface(
    fn=multimodal_qa,
    inputs=gr.Textbox(lines=3, placeholder="请输入问题，例如：这篇论文的配图展示了什么？"),
    outputs=[
        gr.Textbox(label="回答"),
        gr.Gallery(label="相关图片", columns=2, height="auto")
    ],
    title="🧠 多模态语义问答系统",
    description="结合结构化论文（Markdown + 图像）进行语义搜索与回答"
)

if __name__ == "__main__":
    iface.launch()
