import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# 取得目前檔案所在目錄，以建立位於 Gemini_task 根目錄的 scheduler.db
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'scheduler.db')}"

# 建立 SQLite 連線引擎
# 使用 check_same_thread=False 讓多個線程（FastAPI 與 執行緒 Worker）共用同一個連線
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """
    FastAPI 依賴注入 (Dependency Injection) 使用的資料庫 Session 產生器
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
