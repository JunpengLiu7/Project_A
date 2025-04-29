from transformers import pipeline
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

class PaperAnalyzer:
    def __init__(self):
        # 使用 HuggingFace 的 BART 模型进行摘要生成
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # 使用 HuggingFace 的预训练模型进行文本分类
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"  # 轻量级句子嵌入模型
        )

    def analyze_trends(self, papers_df: pd.DataFrame, domain: str) -> str:
        """分析论文趋势"""
        # 按发表日期排序并选择最新3篇
        top_papers = papers_df.sort_values("published", ascending=False).head(3)

        # 生成简易分析报告
        report = f"【{domain}领域最新3篇论文分析】\n\n"
        
        for idx, row in top_papers.iterrows():
            report += f"📄 论文 {idx+1}: {row['title']}\n"
            report += f"📅 发表日期: {row['published']}\n"
            report += f"✍️ 作者: {', '.join(row['authors'][:3])}{'等' if len(row['authors']) > 3 else ''}\n"
            report += f"🔍 摘要摘要: {row['summary'][:200]}...\n\n"
        
        report += "💡 分析建议:\n"
        report += "- 这些论文展示了该领域的最新研究方向\n"
        report += "- 建议重点关注第一篇论文的方法创新点\n"
        report += "- 对比各论文的实验结果部分可能有新发现"
        
        return report

    def classify_paper_theme(self, paper_text: str, candidate_labels: list) -> str:
        """根据论文内容进行主题分类"""
        classification = self.classifier(paper_text, candidate_labels)
        # 返回预测的主题标签
        return classification["labels"][0]

    def create_vector_db(self, papers_df: pd.DataFrame):
        print("📥 开始构建向量数据库...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        docs = []
        for i, row in papers_df.iterrows():
            content = row.get("structured_text") or row["summary"]
            full_doc = f"Title: {row['title']}\nText: {content}"
            docs.append(full_doc)
            print(f"📄 文档 {i+1}：{row['title']}（{len(full_doc)} 字）")

        if not docs:
            print("⚠️ 无文档生成嵌入，终止。")
            return

        splits = text_splitter.create_documents(docs)

        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        print("✅ 向量数据库创建成功！")
        return vectordb


    def summarize(self, text: str) -> str:
        """摘要生成"""
        summary = self.summarizer(text, max_length=200, min_length=50, do_sample=False)
        return summary[0]['summary_text']
