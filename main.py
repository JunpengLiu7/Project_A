# from paper_hunter import PaperHunter
# from analyzer import PaperAnalyzer
# import argparse

# def main():
#     # 命令行参数解析
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--query", type=str, required=True, help="搜索关键词")
#     parser.add_argument("--domain", type=str, required=True, help="分析领域")
#     parser.add_argument("--max-results", type=int, default=5, help="最大结果数")
#     args = parser.parse_args()
    
#     # 执行流程
#     hunter = PaperHunter()
#     analyzer = PaperAnalyzer()  # 确保使用更新后的类
    
#     print(f"🔍 正在抓取 {args.query} 相关论文...") 
#     papers = hunter.fetch_arxiv(args.query, args.max_results)
    
#     print("📊 正在分析技术趋势...") 
#     report = analyzer.analyze_trends(papers, args.domain)  # 现在这个方法已存在
    
#     print("\n📝 分析报告：")
#     print(report)

#     # 进行主题分类
#     candidate_labels = ["AI", "Machine Learning", "Quantum Computing", "Computer Vision", "Natural Language Processing"]
#     print("\n🔍 正在进行论文主题分类...") 
#     for idx, row in papers.iterrows():
#         theme = analyzer.classify_paper_theme(row['summary'], candidate_labels)
#         print(f"📄 论文 {idx+1} 主题: {theme}")
    
#     print("\n🛠️ 正在创建向量数据库...")
#     analyzer.create_vector_db(papers)
#     print("✅ 分析完成！")

# if __name__ == "__main__":
#     main()


from paper_hunter import PaperHunter
from analyzer import PaperAnalyzer
from mineru_loader import MineruPaperLoader
from config import Config
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="搜索关键词")
    parser.add_argument("--domain", type=str, required=True, help="分析领域")
    parser.add_argument("--max-results", type=int, default=5, help="最大抓取数量")
    args = parser.parse_args()

    hunter = PaperHunter()
    analyzer = PaperAnalyzer()
    loader = MineruPaperLoader()

    print(f"🔍 抓取 {args.query} 相关论文...")
    papers_df = hunter.fetch_arxiv(args.query, args.max_results)

    print("📥 正在下载 PDF 并进行结构化识别...")
    texts = []
    for idx, row in papers_df.iterrows():
        title = row["title"]
        url = row["pdf_url"]
        filename = f"{title.replace(' ', '_')}.pdf"
        path = hunter.download_pdf(url, save_as=filename)
        
        try:
            structured_text = loader.load(path)
            papers_df.at[idx, "structured_text"] = structured_text
            print(f"✅ {title} 结构化完成")
            texts.append(structured_text)
        except Exception as e:
            print(f"❌ Mineru 识别失败：{title} - {e}")
            papers_df.at[idx, "structured_text"] = ""

    print("📊 正在分析技术趋势...")
    report = analyzer.analyze_trends(papers_df, args.domain)
    print("\n📝 趋势报告：\n" + report)

    print("\n🔍 主题分类中...")
    candidate_labels = ["AI", "Machine Learning", "Quantum Computing", "Computer Vision", "Natural Language Processing"]
    for idx, row in papers_df.iterrows():
        text = row["structured_text"] or row["summary"]
        theme = analyzer.classify_paper_theme(text, candidate_labels)
        print(f"📄 论文 {idx+1} 《{row['title']}》主题: {theme}")

    print("\n📦 正在构建向量数据库...")
    analyzer.create_vector_db(papers_df)

    print("\n✅ 全流程完成！")

if __name__ == "__main__":
    main()
