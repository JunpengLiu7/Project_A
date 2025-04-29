import requests
import arxiv
from config import Config
from typing import List, Dict
import pandas as pd

class PaperHunter:
    def __init__(self):
        self.config = Config()

    def fetch_arxiv(self, query: str, max_results: int = 5) -> pd.DataFrame:
        """从arXiv获取论文"""
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

    def fetch_semantic_scholar(self, query: str, fields: List[str] = None) -> pd.DataFrame:
        """从Semantic Scholar获取论文"""
        if fields is None:
            fields = ["title", "authors", "year", "citationCount", "url"]
            
        params = {
            "query": query,
            "fields": ",".join(fields)
        }
        
        response = requests.get(
            self.config.SEMANTIC_SCHOLAR_API,
            params=params
        )
        
        return pd.DataFrame(response.json().get("data", []))
    
    def download_pdf(self, url: str, save_as: str = None) -> str:
        """下载论文PDF"""
        os.makedirs(self.config.PDF_SAVE_PATH, exist_ok=True)
        
        if not save_as:
            save_as = url.split("/")[-1] + ".pdf"
            
        filepath = os.path.join(self.config.PDF_SAVE_PATH, save_as)
        response = requests.get(url)
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
            
        return filepath