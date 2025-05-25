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
from image_indexer import build_image_vector_index

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

# === 清理函数 ===
def clean_workspace_after_process():
    print("\n🧹 清理本轮运行产生的 PDF 和 ZIP...")
    pdf_files = glob.glob("papers/*.pdf")
    zip_files = glob.glob("Markdown/*.zip")
    for f in pdf_files + zip_files:
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
    analyzer.analyze_papers(papers)

    os.makedirs("papers", exist_ok=True)
    new_pdf_paths = []

    print(f"\n📥 下载论文 PDF（跳过已有）...")
    for i, row in papers.iterrows():
        title_clean = re.sub(r'[\\/*?:"<>|]', "_", row["title"])
        save_path = f"papers/{title_clean}.pdf"
        if os.path.exists(save_path):
            print(f"⏭️ 跳过已存在: {title_clean}")
        else:
            print(f"📄 下载论文 {i + 1}: {title_clean}")
            if download_pdf(row["pdf_url"], save_path):
                print(f"✅ 成功: {save_path}")
                new_pdf_paths.append(save_path)
            else:
                print(f"❌ 失败: {row['pdf_url']}")

    print("\n🛠️ 正在创建文本向量数据库...")
    analyzer.create_vector_db(papers)
    print("✅ 文本向量数据库创建完成")

    report_generator = ReportGenerator(mineru_api_key)

    for pdf_path in new_pdf_paths:
        name = os.path.basename(pdf_path).replace(".pdf", "")
        target_dir = os.path.join("Markdown", name)
        if os.path.exists(target_dir):
            print(f"⏭️ 跳过已结构化报告: {target_dir}")
            continue

        print(f"\n📥 正在处理: {pdf_path}")
        result = report_generator.generate_report("auto", pdf_path)
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
                zip_path = os.path.join("Markdown", f"{name}.zip")
                report_generator.download_results(download_url, zip_path)
                unzip_file(zip_path)
            else:
                print("❌ 未获取下载链接")
        else:
            print("❌ 提取失败")

    print("\n🖼️ 正在构建图像向量数据库...")
    try:
        build_image_vector_index(root_dir="Markdown", index_dir="faiss_image_index")
    except Exception as e:
        print(f"⚠️ 图像索引构建失败：{e}")
    else:
        print("✅ 图像向量数据库更新完成")

    if args.clean_all:
        clean_all_workspace()
    elif args.clean:
        clean_workspace_after_process()
    else:
        print("\n🚫 未启用 --clean 或 --clean-all，保留所有中间文件")

if __name__ == "__main__":
    main()
