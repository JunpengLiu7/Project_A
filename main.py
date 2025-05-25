import argparse
import os
import requests
import time
import re
import glob
import shutil
from dotenv import load_dotenv, find_dotenv
from paper_hunter import PaperHunter
from analyzer import PaperAnalyzer
from mineru import ReportGenerator, download_pdf, unzip_file

# === 环境变量加载 ===
env_path = find_dotenv()
if not env_path:
    print("⚠️ 未找到 .env 文件，请确保存在于项目根目录")
else:
    print(f"🔍 正在加载 .env 文件路径: {env_path}")
    load_dotenv(dotenv_path=env_path, override=True)

mineru_api_key = os.getenv("MINERU_API_KEY")
if mineru_api_key:
    print("✅ 成功加载 MINERU_API_KEY（前8位）:", mineru_api_key[:8], "...")
else:
    print("❌ 错误：未能读取 MINERU_API_KEY，请检查 .env 文件")
    exit(1)

# === 清理函数（多级控制） ===
def clean_workspace_after_process():
    print("\n🧹 清理本轮运行产生的 PDF 和 ZIP...")
    pdf_files = glob.glob("papers/*.pdf")
    for f in pdf_files:
        os.remove(f)
    zip_files = glob.glob("Markdown/*.zip")
    for f in zip_files:
        os.remove(f)
    print(f"✅ 删除 {len(pdf_files)} 个 PDF，{len(zip_files)} 个 ZIP")

def clean_all_workspace():
    print("\n🧹 执行全量清理：PDF + ZIP + Markdown 结构化目录...")
    for f in glob.glob("papers/*.pdf"):
        os.remove(f)
    for f in glob.glob("Markdown/*.zip"):
        os.remove(f)
    for name in os.listdir("Markdown"):
        path = os.path.join("Markdown", name)
        if os.path.isdir(path):
            shutil.rmtree(path)
    print("✅ 所有历史文件清理完成")

# === 主流程 ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="搜索关键词")
    parser.add_argument("--domain", type=str, required=True, help="分析领域")
    parser.add_argument("--max-results", type=int, default=5, help="最大结果数")
    parser.add_argument("--clean", action="store_true", help="清理当前运行产生的 PDF/ZIP")
    parser.add_argument("--clean-all", action="store_true", help="清理所有 PDF、ZIP 和结构化目录")
    args = parser.parse_args()

    hunter = PaperHunter()
    analyzer = PaperAnalyzer()

    print(f"\n🔍 正在抓取 {args.query} 相关论文...")
    papers = hunter.fetch_arxiv(args.query, args.max_results)

    print("📊 正在分析技术趋势...")
    report = analyzer.analyze_papers(papers)
    print(report)

    # 下载 PDF
    os.makedirs("papers", exist_ok=True)
    print(f"\n📥 正在下载 {len(papers)} 篇抓取的论文 PDF...")
    for i, row in papers.iterrows():
        pdf_url = row["pdf_url"]
        title_clean = re.sub(r'[\\/*?:"<>|]', "_", row["title"])
        save_path = f"papers/{title_clean}.pdf"
        print(f"📄 下载论文 {i + 1}: {title_clean}")
        if download_pdf(pdf_url, save_path):
            print(f"✅ 成功: {save_path}")
        else:
            print(f"❌ 失败: {pdf_url}")

    # 分类 + 嵌入
    candidate_labels = ["AI", "Machine Learning", "Quantum Computing", "Computer Vision", "Natural Language Processing"]
    print("\n🔍 正在进行主题分类...")
    for idx, row in papers.iterrows():
        theme = analyzer.classify_paper_theme(row['summary'], candidate_labels)
        print(f"📄 论文 {idx + 1} 主题: {theme}")

    print("\n🛠️ 正在创建向量数据库...")
    analyzer.create_vector_db(papers)
    print("✅ 向量数据库创建完成")

    # Mineru 报告处理
    print("\n📥 生成报告...")
    pdf_files = [os.path.join("papers", f) for f in os.listdir("papers") if f.endswith(".pdf")]
    print("📚 找到 PDF 文件：", pdf_files)

    report_generator = ReportGenerator(mineru_api_key)

    for pdf_path in pdf_files:
        print(f"\n🔍 正在处理: {pdf_path}")
        result = report_generator.generate_report(report, pdf_path)
        if not result:
            print("❌ 报告生成失败，跳过")
            continue

        task_id, _ = result
        for _ in range(10):
            extract_results = report_generator.query_extract_results(task_id)
            if extract_results and extract_results.get("extract_result"):
                state = extract_results["extract_result"][0].get("state")
                if state == "done":
                    break
                print("⏳ 等待任务完成...")
                time.sleep(60)
            else:
                print("⚠️ 无结果，重试中...")
                time.sleep(60)

        if extract_results:
            extract_result = extract_results["extract_result"][0]
            download_url = extract_result.get("full_zip_url")
            if download_url:
                file_name = os.path.basename(pdf_path).replace(".pdf", "")
                report_zip = os.path.join("Markdown", f"{file_name}.zip")
                os.makedirs(os.path.dirname(report_zip), exist_ok=True)
                report_generator.download_results(download_url, report_zip)
                unzip_file(report_zip)
            else:
                print("❌ 未获取下载链接")
        else:
            print("❌ 提取失败")

    # === 末尾清理控制 ===
    if args.clean_all:
        clean_all_workspace()
    elif args.clean:
        clean_workspace_after_process()
    else:
        print("\n🚫 未启用 --clean 或 --clean-all，保留所有中间文件")

if __name__ == "__main__":
    main()
