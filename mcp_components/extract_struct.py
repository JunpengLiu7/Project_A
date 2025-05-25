import os
import re
import time
import requests
from pathlib import Path
from zipfile import ZipFile, BadZipFile
from typing import Optional


class MineruClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_upload_url = "https://mineru.net/api/v4/file-urls/batch"
        self.base_result_url = "https://mineru.net/api/v4/extract-results/batch"

    def request_upload_url(self, filename: str) -> Optional[tuple]:
        headers = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "enable_formula": True,
            "language": "en",
            "layout_model": "doclayout_yolo",
            "enable_table": True,
            "files": [
                {
                    "name": filename,
                    "is_ocr": True,
                    "data_id": "abcd"
                }
            ]
        }

        try:
            resp = requests.post(self.base_upload_url, headers=headers, json=data)
            if resp.status_code == 200 and resp.json().get("code") == 0:
                batch_id = resp.json()["data"]["batch_id"]
                file_url = resp.json()["data"]["file_urls"][0]
                return batch_id, file_url
            else:
                print("❌ 获取上传链接失败:", resp.text)
        except Exception as e:
            print("❌ 请求异常:", e)
        return None

    def upload_pdf(self, file_path: str, upload_url: str) -> bool:
        try:
            with open(file_path, 'rb') as f:
                resp = requests.put(upload_url, data=f)
            return resp.status_code == 200
        except Exception as e:
            print("❌ 上传 PDF 失败:", e)
            return False

    def poll_result_url(self, batch_id: str, max_retries: int = 10, wait_seconds: int = 30) -> Optional[str]:
        url = f"{self.base_result_url}/{batch_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        for _ in range(max_retries):
            try:
                resp = requests.get(url, headers=headers)
                if resp.status_code == 200 and resp.json()["code"] == 0:
                    result = resp.json()["data"].get("extract_result", [{}])[0]
                    full_url = result.get("full_zip_url")
                    if full_url:
                        match = re.search(r'<url.*?>(.*?)</url>', full_url)
                        return match.group(1) if match else full_url
            except Exception:
                pass
            print("⏳ 等待结构化完成...")
            time.sleep(wait_seconds)
        return None

    def download_zip(self, download_url: str, save_path: str) -> bool:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.get(download_url, headers=headers, stream=True)
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as e:
            print("❌ 下载 ZIP 失败:", e)
            return False

    def unzip_and_clean(self, zip_path: str, extract_root: str = "Markdown") -> Optional[str]:
        try:
            with ZipFile(zip_path, "r") as zip_ref:
                base = Path(zip_path).stem
                out_dir = Path(extract_root) / base
                idx = 1
                while out_dir.exists():
                    out_dir = Path(extract_root) / f"{base}_{idx}"
                    idx += 1
                zip_ref.extractall(out_dir)
            os.remove(zip_path)
            print(f"✅ 解压完成: {out_dir}")
            return str(out_dir)
        except BadZipFile:
            print("❌ ZIP 文件损坏")
        except Exception as e:
            print("❌ 解压失败:", e)
        return None
