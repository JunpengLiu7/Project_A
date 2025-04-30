import argparse
import os
import requests
from paper_hunter import PaperHunter
from analyzer import PaperAnalyzer
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# apikey = f"Bearer + ' ' + {sload_dotenv().api_key}"

class ReportGenerator:
    def __init__(self, mineru_api_key: str):
        self.mineru_api_url = "https://mineru.net/api/v4/file-urls/batch"
        self.api_key = mineru_api_key

    def generate_report(self, markdown_text: str, file_url: str) -> str:
        """使用 Mineru API 生成报告"""
        headers = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {self.api_key}"
        }
        print(headers)
        data = {
            "enable_formula": True,
            "language": "en",
            "layout_model": "doclayout_yolo",
            "enable_table": True,
            "files": [
                {
                    "name": "demo.pdf",
                    "is_ocr": True,
                    "data_id": "abcd"  # 您可以为每个任务指定一个唯一的数据 ID
                }
            ]
        }
        file_path = ["demo.pdf"]
        try:
            # 向 Mineru API 发送请求生成报告
            response = requests.post(self.mineru_api_url, headers=headers, json=data)
            response.raise_for_status()  # 如果请求失败，会抛出异常

            # 获取返回的 JSON 数据
            result = response.json()

            if response.status_code == 200:
                if result["code"] == 0:
                    batch_id = result["data"]["batch_id"]
                    print(f"任务提交成功，batch_id: {batch_id}")
                    return batch_id
                else:
                    print(f"提交任务失败，原因: {result['msg']}")
                    return ""
            else:
                print(f"响应失败，状态码: {response.status_code}, 返回内容: {result}")
                return ""
        except requests.RequestException as e:
            print(f"请求失败: {e}")
            return ""

    def download_report(self, task_id: str, download_path: str) -> None:
        """使用 Mineru API 下载生成的报告"""
        download_url = f"{self.mineru_api_url}/task/{task_id}/download"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            # 请求下载生成的报告
            response = requests.get(download_url, headers=headers)
            response.raise_for_status()  # 如果请求失败，会抛出异常

            # 保存报告文件
            with open(download_path, "wb") as file:
                file.write(response.content)
            print(f"报告已下载并保存到 {download_path}")

        except requests.RequestException as e:
            print(f"下载报告失败: {e}")

def main():
    # 从 .env 文件加载 API 密钥
    mineru_api_key = os.getenv("MINERU_API_KEY")

    if not mineru_api_key:
        print("错误：API 密钥未设置！请确保在 .env 文件中设置 MINERU_API_KEY")
        return

    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="搜索关键词")
    parser.add_argument("--domain", type=str, required=True, help="分析领域")
    parser.add_argument("--max-results", type=int, default=5, help="最大结果数")
    args = parser.parse_args()

    # 执行流程
    hunter = PaperHunter()  # 实例化 PaperHunter
    analyzer = PaperAnalyzer()  # 实例化 PaperAnalyzer

    print(f"🔍 正在抓取 {args.query} 相关论文...") 
    papers = hunter.fetch_arxiv(args.query, args.max_results)  # 获取论文

    print("📊 正在分析技术趋势...") 
    report = analyzer.analyze_trends(papers, args.domain)  # 分析趋势

    print("\n📝 分析报告：")
    print(report)

    # 进行主题分类
    candidate_labels = ["AI", "Machine Learning", "Quantum Computing", "Computer Vision", "Natural Language Processing"]
    print("\n🔍 正在进行论文主题分类...") 
    for idx, row in papers.iterrows():
        theme = analyzer.classify_paper_theme(row['summary'], candidate_labels)  # 分类
        print(f"📄 论文 {idx+1} 主题: {theme}")

    # 创建向量数据库
    print("\n🛠️ 正在创建向量数据库...")
    analyzer.create_vector_db(papers)
    print("✅ 向量数据库创建完成！")

    # 获取论文的 PDF 文件 URL
    pdf_url = papers.iloc[0]["pdf_url"]  # 以第一篇论文为例

    # 将分析结果生成报告并下载
    print("\n🔍 正在生成报告...")
    report_generator = ReportGenerator(mineru_api_key)  # 使用 Mineru API 生成报告

    task_id = report_generator.generate_report(report, pdf_url)  # 生成报告
    if task_id:
        print(f"报告生成成功，任务 ID: {task_id}")
    
        # 下载报告
        report_generator.download_report(task_id, "generated_report.pdf")  # 下载生成的报告
        print("报告已下载并保存为 generated_report.pdf")
    else:
        print("报告生成失败。")

if __name__ == "__main__":
    main()
