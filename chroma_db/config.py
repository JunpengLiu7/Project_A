import os
from dotenv import load_dotenv

load_dotenv()  # 从 .env 文件加载环境变量

class Config:
    # OpenAI API Key（建议放在 .env 文件中而不是写死）
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # API Endpoint 设置
    SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
    ARXIV_API = "http://export.arxiv.org/api/query"
    ACL_ANTHOLOGY = "https://aclanthology.org"

    # 路径配置
    PDF_SAVE_PATH = "./papers"
    CHROMA_DB_PATH = "./chroma_db"
    OUTPUT_PATH = "./output"  # Mineru 输出图像和结果目录

    # 其他参数
    DEFAULT_QUERY = "multimodal large models"
    MAX_RESULTS = 5
