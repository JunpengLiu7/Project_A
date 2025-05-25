import os
import re
import pandas as pd
from paper_hunter import PaperHunter

def sanitize_filename(title: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", title)

def fetch_and_save_papers(query: str, max_results: int = 5, save_dir: str = "papers") -> pd.DataFrame:
    hunter = PaperHunter()
    print(f"🔍 正在抓取 {query} 相关论文（最多 {max_results} 篇）...")

    papers = hunter.fetch_arxiv(query, max_results=max_results)

    if papers.empty:
        print("⚠️ 未获取到论文")
        return papers

    os.makedirs(save_dir, exist_ok=True)

    for idx, row in papers.iterrows():
        clean_title = sanitize_filename(row["title"])
        filename = f"{clean_title}.pdf"
        print(f"📄 [{idx + 1}] 下载论文: {filename}")
        pdf_path = os.path.join(save_dir, filename)
        hunter.download_pdf(row["pdf_url"], save_as=filename)

    return papers
