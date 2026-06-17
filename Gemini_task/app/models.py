from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Text
from .database import Base

class Job(Base):
    """
    任務排程表：記錄系統中的手動任務、自動週期任務以及 LLM 觸發之任務
    """
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    status = Column(String(20), default="pending", nullable=False)  # pending, running, completed, failed, cancelled
    trigger_type = Column(String(20), nullable=False)               # auto, manual, llm
    trigger_time = Column(DateTime, nullable=False, default=datetime.now)
    completion_time = Column(DateTime, nullable=True)
    duration = Column(Float, nullable=True)                          # 單位為秒
    task_type = Column(String(50), nullable=False)                  # 例如: tw_stock_cost, us_stock_cost, tw_investor...
    interval_days = Column(Integer, nullable=True, default=None)    # N 天更新一次 (None 代表非週期性，單次任務)
    remarks = Column(Text, nullable=True)                           # 錯誤訊息、單一股票代號等備註資訊
    last_heartbeat = Column(DateTime, nullable=True)                # 最後心跳時間
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class LlmUsage(Base):
    """
    LLM 呼叫次數記錄表：用於控制每日呼叫次數上限（50 次）
    """
    __tablename__ = "llm_usage"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, unique=True, nullable=False)                 # YYYY-MM-DD
    count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
