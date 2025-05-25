import argparse
import os
import requests
from paper_hunter import PaperHunter
from analyzer import PaperAnalyzer
from dotenv import load_dotenv
import time
import re
from mineru import *

# 加载 .env 文件中的环境变量
load_dotenv()

def main():
    # 从 .env 文件加载 API 密钥
    mineru_api_key = os.getenv("MINERU_API_KEY")

    if not mineru_api_key:
        print("❌ 错误：API 密钥未设置！请确保在 .env 文件中设置 MINERU_API_KEY")
        return

    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="搜索关键词")
    parser.add_argument("--domain", type=str, required=True, help="分析领域")
    parser.add_argument("--max-results", type=int, default=5, help="最大结果数")
    args = parser.parse_args()

    # 执行流程
    hunter = PaperHunter()
    analyzer = PaperAnalyzer()

    print(f"🔍 正在抓取 {args.query} 相关论文...")
    papers = hunter.fetch_arxiv(args.query, args.max_results)

    print("📊 正在分析技术趋势...")
    report = analyzer.analyze_trends(papers, args.domain)

    print("\n📝 分析报告：")
    print(report)

    # 进行主题分类
    candidate_labels = ["AI", "Machine Learning", "Quantum Computing", "Computer Vision", "Natural Language Processing"]
    print("\n🔍 正在进行论文主题分类...")
    for idx, row in papers.iterrows():
        theme = analyzer.classify_paper_theme(row['summary'], candidate_labels)
        print(f"📄 论文 {idx + 1} 主题: {theme}")

    # 创建向量数据库
    print("\n🛠️ 正在创建向量数据库...")
    analyzer.create_vector_db(papers)
    print("✅ 向量数据库创建完成！")

    # 准备报告文件
    print("\n📥 正在准备 PDF 文件以生成报告...")
    pdf_url = papers.iloc[0]["pdf_url"]
    local_pdf_path = "temp_paper.pdf"

    if not download_pdf(pdf_url, local_pdf_path):
        print("❌ 无法下载 PDF，跳过报告生成。")
        return

    # 使用 Mineru 生成报告
    print("\n🔍 正在生成报告...")
    report_generator = ReportGenerator(mineru_api_key)
    task_id, upload_url = report_generator.generate_report(report, local_pdf_path)

    if task_id:
        print(f"📦 报告生成成功，任务 ID: {task_id}")
        
        # 查询提取结果以获取下载链接
        extract_results = None
        for _ in range(10):  # 尝试查询10次
            extract_results = report_generator.query_extract_results(task_id)
            if extract_results and extract_results.get("extract_result"):
                extract_status = extract_results["extract_result"][0].get("state")
                if extract_status == "success":
                    break
                elif extract_status == "waiting-file":
                    print("等待任务完成...")
                    time.sleep(10)  # 等待10秒后重试
                else:
                    print(f"任务状态: {extract_status}")
                    break
            else:
                print("查询提取结果为空，重试中...")
                time.sleep(10)  # 等待10秒后重试

        if extract_results:
            extract_result = extract_results.get("extract_result", [{}])[0]
            download_url = extract_result.get("full_zip_url")
            print("提取结果: ", download_url)
            if download_url:
                report_generator.download_results(download_url, "report.zip")
            else:
                print("❌ 无法获取下载链接。")
        else:
            print("❌ 无法获取提取结果。")
    else:
        print("❌ 报告生成失败。")

    # 可选：删除临时 PDF 文件
    if os.path.exists(local_pdf_path):
        os.remove(local_pdf_path)
    unzip_file("report.zip")


if __name__ == "__main__":
    main()