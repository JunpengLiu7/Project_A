import os
import glob
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.document_loaders import TextLoader

def load_and_split_markdown_files(markdown_dir: str):
    all_docs = []
    md_files = glob.glob(os.path.join(markdown_dir, "**/*.md"), recursive=True)
    print(f"📂 找到 {len(md_files)} 个 Markdown 文件")

    for md_file in md_files:
        print(f"📄 正在处理: {md_file}")
        loader = TextLoader(md_file, encoding="utf-8")
        documents = loader.load()

        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
            ("#", "Heading1"),
            ("##", "Heading2"),
            ("###", "Heading3")
        ])
        split_docs = splitter.split_text(documents[0].page_content)

        for doc in split_docs:
            doc.metadata["source"] = md_file

        all_docs.extend(split_docs)

    print(f"🔪 总切分段落数: {len(all_docs)}")
    return all_docs

def build_md_vector_db():
    markdown_dir = "Markdown"
    vector_dir = "md_vector_db"
    os.makedirs(vector_dir, exist_ok=True)

    docs = load_and_split_markdown_files(markdown_dir)
    if not docs:
        print("⚠️ 未找到可嵌入的 Markdown 内容，退出。")
        return

    print("🧠 正在嵌入并写入向量数据库...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=vector_dir
    )
    vectordb.persist()
    print(f"✅ 嵌入完成，数据库保存在: {vector_dir}")

if __name__ == "__main__":
    build_md_vector_db()
