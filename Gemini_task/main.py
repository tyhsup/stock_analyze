import os
import sys
import logging

# 取得 Gemini_task 根目錄，並將 mydjango/demo 加入 sys.path，保證 stock_Django 包可被正常載入
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "demo"))
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)

# 1. 預先載入位於 mydjango/demo/stock_Django/.env 的環境變數
# 這必須在任何其他 module 載入之前執行，以確保 MySQL 設定與 Gemini API Key 提早注入環境
from dotenv import load_dotenv
ENV_PATH = os.path.join(BASE_DIR, "..", "demo", "stock_Django", ".env")

if os.path.exists(ENV_PATH):
    print(f"[Gemini Task] 正在載入環境變數檔案: {os.path.abspath(ENV_PATH)}")
    load_dotenv(os.path.abspath(ENV_PATH))
else:
    print(f"[Gemini Task] 警告: 未能找到環境變數檔案 {os.path.abspath(ENV_PATH)}，請確認專案路徑設定。")

# 2. 設定日誌系統
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(name)s) %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("scheduler.main")

# 3. 延遲載入 FastAPI 應用程式，此時環境變數已完全準備就緒
from app.web_server import app

if __name__ == "__main__":
    logger.info("正在啟動 Gemini Task 股市排程服務...")
    # 執行 Uvicorn 伺服器
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
