import os
import time
from glob import glob

def process_pdf_files(mineru_api_key, folder_path="papers"):
    report_generator = ReportGenerator(mineru_api_key)
    pdf_files = glob(os.path.join(folder_path, "*.pdf"))

    if not pdf_files:
        print("❌ 未找到任何 PDF 文件。")
        return

    for pdf_file in pdf_files:
        base_name = os.path.splitext(os.path.basename(pdf_file))[0]
        print(f"\n🔍 正在处理 {pdf_file} ...")

        # 可自定义报告内容，这里用空字符串作为占位
        dummy_report = "分析报告占位内容"  

        task_id, upload_url = report_generator.generate_report(dummy_report, pdf_file)

        if task_id:
            print(f"📦 报告任务提交成功，任务 ID: {task_id}")

            # 查询任务结果
            extract_results = None
            for _ in range(10):
                extract_results = report_generator.query_extract_results(task_id)
                if extract_results and extract_results.get("extract_result"):
                    extract_status = extract_results["extract_result"][0].get("state")
                    if extract_status == "success":
                        break
                    elif extract_status == "waiting-file":
                        print("等待任务完成...")
                        time.sleep(10)
                    else:
                        print(f"任务状态: {extract_status}")
                        break
                else:
                    print("查询提取结果为空，重试中...")
                    time.sleep(10)

            if extract_results:
                extract_result = extract_results.get("extract_result", [{}])[0]
                download_url = extract_result.get("full_zip_url")
                if download_url:
                    zip_filename = f"{base_name}.zip"
                    report_generator.download_results(download_url, zip_filename)
                    print(f"✅ 下载完成: {zip_filename}")

                    # 解压并可选删除 ZIP 文件
                    unzip_file(zip_filename)
                    os.remove(zip_filename)
                else:
                    print("❌ 无法获取下载链接。")
            else:
                print("❌ 无法获取提取结果。")
        else:
            print("❌ 报告任务创建失败。")
