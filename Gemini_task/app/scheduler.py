import time
import queue
import logging
import threading
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from .database import SessionLocal, engine
from .models import Base, Job
from .executors import EXECUTORS

# 初始化資料表 (SQLite)
Base.metadata.create_all(bind=engine)

logger = logging.getLogger("scheduler.core")

# 互斥任務類型分組定義（同組任務同一時間僅允許一個處於 running 狀態）
MUTEX_GROUPS = [
    {
        "tw_stock_cost", 
        "tw_stock_price_only", 
        "us_stock_cost", 
        "us_stock_price_only"
    }
]

def _recover_stale_running_jobs():
    """
    啟動時恢復機制：將所有 status='running' 的任務重設為 failed
    """
    db = get_db_session()
    try:
        stale_jobs = db.query(Job).filter(Job.status == "running").all()
        for job in stale_jobs:
            job.status = "failed"
            job.completion_time = datetime.now()
            job.remarks = f"系統重啟，自動將未完成的任務標記為 failed | {job.remarks or ''}"[:500]
            logger.info(f"已恢復卡死任務 ID {job.id} ({job.name}) 為 failed")
        db.commit()
    except Exception as e:
        logger.error(f"恢復卡死任務失敗: {e}")
        db.rollback()
    finally:
        db.close()

def _heartbeat_worker(job_id: int, stop_heartbeat_event: threading.Event):
    """
    心跳背景執行緒：每 60 秒更新一次資料庫中的 last_heartbeat
    """
    logger.info(f"Job-{job_id} 心跳執行緒已啟動")
    while not stop_heartbeat_event.wait(timeout=60):
        db = get_db_session()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job and job.status == "running":
                job.last_heartbeat = datetime.now()
                db.commit()
                logger.debug(f"Job-{job_id} 心跳已更新")
            else:
                break
        except Exception as e:
            logger.warning(f"更新 Job-{job_id} 心跳失敗: {e}")
            db.rollback()
        finally:
            db.close()
    logger.info(f"Job-{job_id} 心跳執行緒已退出")

# 全域執行佇列
job_queue = queue.Queue()

# 執行緒控制 Event
stop_event = threading.Event()

# 執行緒參考
watcher_thread = None
worker_threads = []

def get_db_session() -> Session:
    """提供獨立的 Session 供背景執行緒使用"""
    return SessionLocal()

def worker_loop(worker_id: int):
    """
    Worker 執行緒主迴圈：從 Queue 中取出 Job ID 並執行
    """
    logger.info(f"Worker-{worker_id} 啟動完成")
    while not stop_event.is_set():
        try:
            # 採用非阻塞或短超時，讓執行緒能定期檢查 stop_event
            job_id = job_queue.get(timeout=2)
        except queue.Empty:
            continue
            
        logger.info(f"Worker-{worker_id} 開始處理 Job ID: {job_id}")
        db = get_db_session()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if not job:
                logger.warning(f"Job ID {job_id} 不存在於資料庫中")
                job_queue.task_done()
                db.close()
                continue
                
            if job.status != "running":
                logger.warning(f"Job ID {job_id} 狀態非 running (目前為 {job.status})，跳過執行")
                job_queue.task_done()
                db.close()
                continue
            
            start_time = datetime.now()
            
            # 尋找對應的執行器並執行
            executor_func = EXECUTORS.get(job.task_type)
            if not executor_func:
                raise ValueError(f"找不到任務類型 {job.task_type} 對應的執行器")
                
            # 初始化心跳時間
            job.last_heartbeat = datetime.now()
            db.commit()

            # 啟動心跳執行緒
            stop_heartbeat_event = threading.Event()
            hb_thread = threading.Thread(
                target=_heartbeat_worker, 
                args=(job.id, stop_heartbeat_event),
                name=f"Job-{job.id}-Heartbeat",
                daemon=True
            )
            hb_thread.start()

            try:
                # 呼叫對應的腳本
                logger.info(f"正在執行任務: {job.name} (類型: {job.task_type}, 備註: {job.remarks})")
                executor_func(remarks=job.remarks)
            finally:
                # 確保心跳執行緒停止
                stop_heartbeat_event.set()
                hb_thread.join(timeout=2)
            
            # 執行成功：更新狀態與耗時
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # 重新取得 job (防止 Session 斷開)
            job = db.query(Job).filter(Job.id == job_id).first()
            job.status = "completed"
            job.completion_time = end_time
            job.duration = duration
            db.commit()
            logger.info(f"任務執行成功: {job.name}，耗時: {duration:.2f} 秒")
            
            # 處理週期任務的自動 Reschedule
            reschedule_job(db, job)
            
        except Exception as e:
            logger.exception(f"Job ID {job_id} 執行失敗")
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() if 'start_time' in locals() else 0.0
            
            try:
                db.rollback()
                job = db.query(Job).filter(Job.id == job_id).first()
                if job:
                    job.status = "failed"
                    job.completion_time = end_time
                    job.duration = duration
                    error_msg = f"Error: {str(e)}"
                    job.remarks = f"{error_msg} | {job.remarks or ''}"[:500]  # 限制長度以防溢位
                    db.commit()
                    
                    # 即使失敗，週期任務也需 Reschedule，否則排程會中斷
                    reschedule_job(db, job)
            except Exception as db_err:
                logger.error(f"更新 Job ID {job_id} 失敗狀態時發生 DB 錯誤: {db_err}")
                
        finally:
            job_queue.task_done()
            db.close()

    logger.info(f"Worker-{worker_id} 已安全退出")

def reschedule_job(db: Session, job: Job):
    """
    若是週期任務，自動新增下一次的 pending 排程
    """
    if job.interval_days and job.interval_days >= 1:
        # 下次觸發時間為「目前 trigger_time 加上 interval_days 天」
        next_trigger = job.trigger_time + timedelta(days=job.interval_days)
        
        # 檢查是否已存在相同的 pending 任務，避免重複排程
        exists = db.query(Job).filter(
            Job.task_type == job.task_type,
            Job.status == "pending",
            Job.interval_days == job.interval_days,
            Job.trigger_time == next_trigger
        ).first()
        
        if not exists:
            # 清理 remarks 中的 Error 標記，只保留原本的參數（例如股票代號）
            clean_remarks = job.remarks or ""
            if "Error:" in clean_remarks and " | " in clean_remarks:
                clean_remarks = clean_remarks.split(" | ", 1)[-1]
                
            next_job = Job(
                name=job.name,
                status="pending",
                trigger_type="auto",
                trigger_time=next_trigger,
                task_type=job.task_type,
                interval_days=job.interval_days,
                remarks=clean_remarks
            )
            db.add(next_job)
            db.commit()
            logger.info(f"已自動新增週期任務: {job.name} (下次觸發時間: {next_trigger})")

def watcher_loop():
    """
    Watcher 執行緒主迴圈：定期掃描資料庫中到達觸發時間的 pending 任務，並送入佇列
    """
    logger.info("Watcher 執行緒啟動完成")
    while not stop_event.is_set():
        db = get_db_session()
        try:
            now = datetime.now()
            
            # 1. 偵測狀態為 "running" 且 last_heartbeat 超時的任務 (例如 5 分鐘無心跳)
            stale_threshold = now - timedelta(minutes=5)
            stale_jobs = db.query(Job).filter(
                Job.status == "running",
                Job.last_heartbeat < stale_threshold
            ).all()
            
            for stale_job in stale_jobs:
                stale_job.status = "failed"
                stale_job.completion_time = now
                stale_job.remarks = f"Heartbeat timeout: 任務已卡死且超過 5 分鐘無心跳 | {stale_job.remarks or ''}"[:500]
                logger.error(f"偵測到任務卡死 (ID: {stale_job.id}, Name: {stale_job.name})，已強制標記為 failed")
            
            if stale_jobs:
                db.commit()

            # 2. 撈出所有 trigger_time <= now 且狀態為 pending 的任務
            due_jobs = db.query(Job).filter(
                Job.status == "pending",
                Job.trigger_time <= now
            ).order_by(Job.trigger_time.asc()).all()
            
            # 3. 獲取目前所有正在執行的任務類型 (status='running')
            running_jobs = db.query(Job).filter(Job.status == "running").all()
            running_task_types = {j.task_type for j in running_jobs}
            
            for job in due_jobs:
                # 判定當前任務類型是否與任何正在執行的任務衝突
                conflict = False
                for group in MUTEX_GROUPS:
                    if job.task_type in group:
                        # 檢查該組中是否有其他同組的任務類型正在運行
                        intersect = group.intersection(running_task_types)
                        if intersect:
                            conflict = True
                            logger.info(f"任務 ID {job.id} ({job.name}, 類型: {job.task_type}) 由於同性質任務 {list(intersect)} 正在執行，暫緩本次調度")
                            break
                
                if conflict:
                    # 有衝突，跳過這個任務，保留其為 pending，等下一個調度週期再行判定
                    continue
                    
                # 無衝突，將該任務標記為 running 並加入執行佇列
                job.status = "running"
                job.last_heartbeat = now
                db.commit()
                
                # 為了避免在同一次 watcher_loop 掃描中，後續 due_jobs 與目前剛啟動的任務衝突，
                # 必須將目前啟動的 job.task_type 立即加到 running_task_types 集合中
                running_task_types.add(job.task_type)
                
                job_queue.put(job.id)
                logger.info(f"Watcher 已將 Job ID {job.id} ({job.name}) 送入執行佇列")
                
        except Exception as e:
            logger.error(f"Watcher 掃描任務失敗: {e}")
            try:
                db.rollback()
            except:
                pass
        finally:
            db.close()
            
        # 每 10 秒掃描一次資料庫
        time.sleep(10)
        
    logger.info("Watcher 執行緒已安全退出")

def start_scheduler(num_workers: int = 2):
    """
    啟動排程器守護執行緒 (Watcher 與 Workers)
    """
    global watcher_thread, worker_threads, stop_event
    
    if watcher_thread is not None and watcher_thread.is_alive():
        logger.warning("排程器已在運行中，忽略啟動要求")
        return
        
    stop_event.clear()
    
    # 啟動時先將卡住的 running 任務進行恢復 (改為 failed)
    _recover_stale_running_jobs()
    
    # 啟動 Watcher
    watcher_thread = threading.Thread(target=watcher_loop, name="JobWatcher", daemon=True)
    watcher_thread.start()
    
    # 啟動 Workers
    worker_threads = []
    for i in range(num_workers):
        t = threading.Thread(target=worker_loop, args=(i+1,), name=f"JobWorker-{i+1}", daemon=True)
        t.start()
        worker_threads.append(t)
        
    logger.info(f"排程器啟動成功，包含 1 個 Watcher 和 {num_workers} 個 Workers")

def stop_scheduler():
    """
    停止排程器，通知所有執行緒優雅退出
    """
    global watcher_thread, worker_threads
    
    if watcher_thread is None:
        logger.warning("排程器並未運行")
        return
        
    logger.info("正在停止排程器...")
    stop_event.set()
    
    # 等待 Watcher 退出
    if watcher_thread:
        watcher_thread.join(timeout=5)
        
    # 等待 Workers 退出
    for t in worker_threads:
        t.join(timeout=5)
        
    watcher_thread = None
    worker_threads = []
    logger.info("排程器已完全停止")
