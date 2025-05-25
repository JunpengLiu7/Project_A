import argparse
import os
import requests
from paper_hunter import PaperHunter
from analyzer import PaperAnalyzer
from dotenv import load_dotenv
import time
import re

class ReportGenerator:
    def __init__(self, mineru_api_key: str):
        self.mineru_api_url = "https://mineru.net/api/v4/file-urls/batch"
        self.api_key = mineru_api_key
    def generate_report(self, markdown_text: str, file_path: str) -> tuple:
        """使用 Mineru API 上传文件并生成报告"""
        headers = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {self.api_key}"
        }

        # 构造请求体
        data = {
            "enable_formula": True,
            "language": "en",
            "layout_model": "doclayout_yolo",
            "enable_table": True,
            "files": [
                {
                    "name": os.path.basename(file_path),
                    "is_ocr": True,
                    "data_id": "abcd"
                }
            ]
        }

        try:
            # 第一步：申请上传链接
            response = requests.post(self.mineru_api_url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                print('✅ response success. result:', result)

                if result["code"] == 0:
                    batch_id = result["data"]["batch_id"]
                    urls = result["data"]["file_urls"]
                    print(f'📦 batch_id: {batch_id}, upload_urls: {urls}')

                    # 第二步：上传 PDF 文件
                    with open(file_path, 'rb') as f:
                        upload_response = requests.put(urls[0], data=f)
                        if upload_response.status_code == 200:
                            print(f"✅ PDF 上传成功: {urls[0]}")
                            return batch_id, urls[0]
                        else:
                            print(f"❌ PDF 上传失败: {upload_response.status_code}")
                            return "", ""
                else:
                    print(f"❌ 获取上传链接失败，原因: {result['msg']}")
                return "", ""
        except Exception as e:
            print(f"❌ 请求异常: {e}")
            return "", ""

    def query_extract_results(self, batch_id):
        url = f'https://mineru.net/api/v4/extract-results/batch/{batch_id}'
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.get(url, headers=headers)
        
        if response.status_code == 200 and response.json()["code"] == 0:
            print('✅ 提取结果查询成功:', response.json())
            data = response.json()["data"]
            
            # 提取 full_zip_url 中的实际链接
            if data and data.get("extract_result"):
                extract_result = data["extract_result"][0]
                if extract_result.get("full_zip_url"):
                    # 使用正则表达式提取 URL
                    url_pattern = re.compile(r'<url.*?>(.*?)</url>')
                    match = url_pattern.search(extract_result["full_zip_url"])
                    if match:
                        extract_result["full_zip_url"] = match.group(1).strip()
            return data
        else:
            print(f"❌ 查询提取结果失败: {response.text}")
            return None

    def download_results(self, download_url: str, save_path: str):
        """下载解析结果"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(download_url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"📥 下载成功: {save_path}")
        except requests.RequestException as e:
            print(f"❌ 下载失败: {e}")

def download_pdf(pdf_url: str, save_path: str) -> bool:
    """下载 PDF 到本地路径"""
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ PDF 已保存到本地: {save_path}")
        return True
    except Exception as e:
        print(f"❌ PDF 下载失败: {e}")
        return False
    
import os
import zipfile

def unzip_file(zip_path):
    """解压 ZIP 文件到指定目录"""
    extract_to="."
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ ZIP 文件已解压到: {extract_to}")
    except zipfile.BadZipFile:
        print("❌ 错误: ZIP 文件损坏或不是有效的 ZIP 文件格式")
    except Exception as e:
        print(f"❌ 解压失败: {e}")
