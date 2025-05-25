import requests
import arxiv
from config import Config
from typing import List, Dict
import pandas as pd
import os
import time
from requests.exceptions import RequestException

class PaperHunter:
    def __init__(self):
        self.config = Config()

    def fetch_arxiv(self, query: str, max_results: int = 5) -> pd.DataFrame:
        """从arXiv获取论文"""
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )

            papers = []
            for result in client.results(search):
                papers.append({
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "published": result.published.strftime("%Y-%m-%d"),
                    "pdf_url": result.pdf_url,
                    "summary": result.summary,
                    "source": "arxiv"
                })

            return pd.DataFrame(papers)
        except RequestException as e:
            print(f"请求 arXiv 失败: {e}")
            return pd.DataFrame()

    def fetch_semantic_scholar(self, query: str, fields: List[str] = None) -> pd.DataFrame:
        """从Semantic Scholar获取论文"""
        if fields is None:
            fields = ["title", "authors", "year", "citationCount", "url"]

        try:
            params = {
                "query": query,
                "fields": ",".join(fields)
            }
            response = requests.get(self.config.SEMANTIC_SCHOLAR_API, params=params)
            response.raise_for_status()  # Raise an exception for 4xx/5xx responses

            return pd.DataFrame(response.json().get("data", []))
        except RequestException as e:
            print(f"请求 Semantic Scholar 失败: {e}")
            return pd.DataFrame()

    def download_pdf(self, url: str, save_as: str = None) -> str:
        """下载论文PDF"""
        try:
            os.makedirs(self.config.PDF_SAVE_PATH, exist_ok=True)

            if not save_as:
                save_as = url.split("/")[-1] + ".pdf"
            filepath = os.path.join(self.config.PDF_SAVE_PATH, save_as)

            # 避免覆盖文件，处理文件名冲突
            if os.path.exists(filepath):
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filepath = os.path.join(self.config.PDF_SAVE_PATH, f"{timestamp}_{save_as}")

            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad responses

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"PDF 文件已下载到 {filepath}")
            return filepath

        except RequestException as e:
            print(f"下载 PDF 文件失败: {e}")
            return ""
