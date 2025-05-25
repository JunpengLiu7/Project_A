import os
import time
import subprocess
from datetime import datetime, timedelta

# 用于记录上次运行时间的文件
LAST_RUN_FILE = "last_run.txt"

# 设置间隔周期：7 天
INTERVAL_DAYS = 7

def should_run():
    if not os.path.exists(LAST_RUN_FILE):
        return True

    with open(LAST_RUN_FILE, "r") as f:
        last_run_str = f.read().strip()
        try:
            last_run = datetime.strptime(last_run_str, "%Y-%m-%d")
            return datetime.now() - last_run >= timedelta(days=INTERVAL_DAYS)
        except ValueError:
            return True  # 时间格式有误也重新运行

def update_last_run_time():
    with open(LAST_RUN_FILE, "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d"))

def run_main_script():
    print("🚀 正在运行 main.py ...")
    result = subprocess.run(["python", "main.py", "--query", "LLM", "--domain", "NLP", "--max-results", "5"])
    if result.returncode == 0:
        print("✅ main.py 执行成功")
        update_last_run_time()
    else:
        print("❌ main.py 执行失败")

if __name__ == "__main__":
    if should_run():
        run_main_script()
    else:
        print("⏳ 未到下一次运行时间，跳过本轮。")
