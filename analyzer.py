import os
import pandas as pd
from typing import List
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from transformers import pipeline


class PaperAnalyzer:
    def __init__(self):
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        # 初始化 zero-shot 分类器
        try:
            self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        except Exception as e:
            print(f"❌ Zero-shot 分类器初始化失败: {str(e)}")
            self.classifier = None

    def analyze_papers(self, papers: pd.DataFrame):
        print("\n📝 分析报告：")
        print(f" 【LLM领域最新{len(papers)}篇论文分析】\n")
        for i, row in papers.iterrows():
            print(f"📄 论文 {i+1}: {row['title']}")
            print(f"📅 发表日期: {row['published']}")
            print(f"✍️ 作者: {row['authors']}")
            print(f"🔍 摘要摘要: {row['summary'][:300]}...\n")
        print("💡 分析建议:")
        print("- 这些论文展示了该领域的最新研究方向")
        print("- 建议重点关注第一篇论文的方法创新点")
        print("- 对比各论文的实验结果部分可能有新发现\n")

    def classify_paper_theme(self, summary: str, candidate_labels: List[str]) -> str:
        if self.classifier is None:
            print("⚠️ 未加载分类器，返回 Unknown")
            return "Unknown"
        try:
            result = self.classifier(summary, candidate_labels)
            return result["labels"][0]  # 返回置信度最高的标签
        except Exception as e:
            print(f"❌ 分类失败: {str(e)}")
            return "Unknown"

    def create_vector_db(self, papers: pd.DataFrame):
        print("🛠️ 正在创建向量数据库...")
        print("📥 开始构建向量数据库...")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        docs = []
        for i, row in papers.iterrows():
            content = f"Title: {row['title']}\nAbstract: {row['summary']}"
            print(f"📄 文档 {i+1}：{row['title']}（{len(content)} 字）")
            docs.append(content)

        if not docs:
            print("⚠️ 没有可处理的文档，终止创建数据库。")
            return None

        splits = text_splitter.create_documents(docs)
        print(f"🔪 文本切片后共有：{len(splits)} 段")
        print(f"🔍 使用的嵌入模型：{self.embedding_model_name}")

        try:
            vectordb = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
            print("✅ 向量数据库创建成功并写入文档！")
            return vectordb
        except Exception as e:
            print("❌ 向量数据库创建失败：", str(e))
            raise
