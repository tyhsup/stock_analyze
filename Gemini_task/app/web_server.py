import os
import logging
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from sqlalchemy.orm import Session

from .database import get_db, SessionLocal
from .models import Job
from .scheduler import start_scheduler, stop_scheduler
from .llm_parser import parse_natural_language_task, get_current_usage_count, DAILY_LIMIT

logger = logging.getLogger("scheduler.web")

app = FastAPI(title="Gemini Task Job Scheduler", version="1.0.0")

# 允許跨網域存取 (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 請求模型
class JobCreateSchema(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    task_type: str = Field(...)
    trigger_type: str = Field("manual")  # manual, auto, llm
    trigger_time: Optional[str] = Field(None, description="格式: YYYY-MM-DD HH:MM:SS")
    interval_days: Optional[int] = Field(None, description="大於等於 1 則代表週期任務")
    remarks: Optional[str] = Field(None)

class LlmParseRequestSchema(BaseModel):
    prompt: str = Field(..., min_length=1)

# Pydantic 回傳模型
class JobResponseSchema(BaseModel):
    id: int
    name: str
    status: str
    trigger_type: str
    trigger_time: datetime
    completion_time: Optional[datetime]
    duration: Optional[float]
    task_type: str
    interval_days: Optional[int]
    remarks: Optional[str]
    last_heartbeat: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

@app.on_event("startup")
def startup_event():
    """當 Web 伺服器啟動時，自動啟動背景排程器"""
    logger.info("正在啟動背景排程器守護執行緒...")
    start_scheduler(num_workers=2)

@app.on_event("shutdown")
def shutdown_event():
    """當 Web 伺服器關閉時，優雅關閉背景排程器"""
    logger.info("正在停止背景排程器...")
    stop_scheduler()

# ----------------- REST API Endpoints -----------------

@app.get("/api/jobs", response_model=List[JobResponseSchema])
def list_jobs(
    status: Optional[str] = Query(None, description="過濾狀態: pending, running, completed, failed"),
    db: Session = Depends(get_db)
):
    """
    列出所有任務，可根據 status 進行篩選。以 trigger_time 降序排列。
    """
    query = db.query(Job)
    if status:
        query = query.filter(Job.status == status)
    return query.order_by(Job.trigger_time.desc()).all()

@app.post("/api/jobs", response_model=JobResponseSchema)
def create_job(payload: JobCreateSchema, db: Session = Depends(get_db)):
    """
    建立新的任務（手動、自動定期或 LLM 解析確認後任務）
    """
    # 格式化 trigger_time
    parsed_trigger_time = datetime.now()
    if payload.trigger_time:
        try:
            parsed_trigger_time = datetime.strptime(payload.trigger_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise HTTPException(status_code=400, detail="trigger_time 格式不符。必須為 YYYY-MM-DD HH:MM:SS")
            
    # 驗證週期任務
    if payload.trigger_type == "auto":
        if not payload.interval_days or payload.interval_days < 1:
            raise HTTPException(status_code=400, detail="自動週期任務的 interval_days 必須大於等於 1")
            
    new_job = Job(
        name=payload.name,
        status="pending",
        trigger_type=payload.trigger_type,
        trigger_time=parsed_trigger_time,
        task_type=payload.task_type,
        interval_days=payload.interval_days,
        remarks=payload.remarks
    )
    
    try:
        db.add(new_job)
        db.commit()
        db.refresh(new_job)
        logger.info(f"成功手動/自動新增任務: {new_job.name} (ID: {new_job.id})")
        return new_job
    except Exception as e:
        db.rollback()
        logger.error(f"新增任務失敗: {e}")
        raise HTTPException(status_code=500, detail=f"寫入資料庫失敗: {str(e)}")

@app.post("/api/jobs/execute/{job_id}", response_model=JobResponseSchema)
def trigger_job_immediately(job_id: int, db: Session = Depends(get_db)):
    """
    立即觸發某個任務。
    將該任務的 trigger_time 修改為現在，並將狀態重設為 pending。
    Watcher 在下一輪掃描中會將其挑出並送入執行佇列。
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="找不到該任務")
        
    # 如果任務正在運行中，不可重複執行
    if job.status == "running":
        raise HTTPException(status_code=400, detail="該任務目前正在執行中，無法立即觸發")

    job.status = "pending"
    job.trigger_time = datetime.now()
    # 清空之前的完成時間和耗時
    job.completion_time = None
    job.duration = None
    
    try:
        db.commit()
        db.refresh(job)
        logger.info(f"立即手動觸發任務 (ID: {job.id}, Name: {job.name})")
        return job
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"更新資料庫失敗: {str(e)}")

@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: int, db: Session = Depends(get_db)):
    """
    刪除或取消某個任務。
    如果是 pending 狀態，直接物理刪除。
    如果是其他狀態，拒絕或將狀態改為 cancelled。
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="找不到該任務")
        
    if job.status == "running":
        raise HTTPException(status_code=400, detail="任務執行中，無法刪除或取消")
        
    try:
        if job.status == "pending":
            # 直接刪除
            db.delete(job)
            db.commit()
            return {"detail": f"成功刪除任務 ID {job_id}"}
        else:
            # 將狀態改為 cancelled
            job.status = "cancelled"
            db.commit()
            return {"detail": f"成功將任務 ID {job_id} 狀態更改為已取消"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"資料庫操作失敗: {str(e)}")

@app.post("/api/jobs/heartbeat/{job_id}")
def update_job_heartbeat(job_id: int, db: Session = Depends(get_db)):
    """
    更新任務的心跳時間。由執行中的任務執行緒定期呼叫。
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="找不到該任務")
    if job.status != "running":
        raise HTTPException(status_code=400, detail="任務並非執行中狀態，拒絕更新心跳")
        
    job.last_heartbeat = datetime.now()
    try:
        db.commit()
        return {"status": "ok", "last_heartbeat": job.last_heartbeat}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"更新心跳失敗: {str(e)}")

@app.post("/api/llm/parse")
def llm_parse_prompt(payload: LlmParseRequestSchema):
    """
    呼叫 LLM 解析自然語言，回傳結構化的排程任務資訊。
    這不直接寫入資料庫，而是讓前端進行二次確認。
    """
    try:
        result = parse_natural_language_task(payload.prompt)
        return result
    except Exception as e:
        logger.error(f"LLM 解析失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/llm/usage")
def get_llm_usage():
    """
    取得今日 LLM 呼叫配額使用狀態
    """
    count = get_current_usage_count()
    return {"count": count, "limit": DAILY_LIMIT}

# ----------------- 靜態網頁託管 -----------------

# 首先檢查 E:\Infinity\mydjango\Gemini_task\app\static 目錄是否存在，若不存在則先建立它
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)

# 掛載靜態檔案目錄 (用於 CSS, JS 等資源)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    """
    回傳前端 Dashboard 主頁面 (index.html)
    """
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"detail": "靜態 Dashboard 檔案尚未建立。請建立 index.html 後重試。"}
