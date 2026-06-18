import os
import json
import logging
from datetime import datetime
from django.http import JsonResponse, FileResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

# 載入 Gemini_task 的 SQLAlchemy 核心模組
from app.database import SessionLocal
from app.models import Job
from app.llm_parser import parse_natural_language_task, get_current_usage_count, DAILY_LIMIT

logger = logging.getLogger("scheduler.views")

def scheduler_home(request):
    """回傳排程器首頁 (index.html)"""
    index_path = os.path.join(settings.BASE_DIR.parent, "Gemini_task", "app", "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(open(index_path, 'rb'))
    return JsonResponse({"detail": "靜態 Dashboard 檔案尚未建立。請確認 Gemini_task/app/static/index.html 存在。"}, status=404)

def list_jobs(request):
    """列出所有任務，以 trigger_time 降序排列。支援 status 狀態過濾。"""
    status = request.GET.get("status", None)
    db = SessionLocal()
    try:
        query = db.query(Job)
        if status:
            query = query.filter(Job.status == status)
        jobs = query.order_by(Job.trigger_time.desc()).all()
        
        data = []
        for job in jobs:
            data.append({
                "id": job.id,
                "name": job.name,
                "status": job.status,
                "trigger_type": job.trigger_type,
                "trigger_time": job.trigger_time.isoformat() if job.trigger_time else None,
                "completion_time": job.completion_time.isoformat() if job.completion_time else None,
                "duration": job.duration,
                "task_type": job.task_type,
                "interval_days": job.interval_days,
                "remarks": job.remarks,
                "last_heartbeat": job.last_heartbeat.isoformat() if job.last_heartbeat else None,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "updated_at": job.updated_at.isoformat() if job.updated_at else None,
            })
        return JsonResponse(data, safe=False)
    except Exception as e:
        logger.error(f"獲取任務清單失敗: {e}")
        return JsonResponse({"detail": f"伺服器錯誤: {str(e)}"}, status=500)
    finally:
        db.close()

@csrf_exempt
def create_job(request):
    """建立手動、自動週期或由 AI 解析出來的新任務"""
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
    
    try:
        payload = json.loads(request.body)
        name = payload.get("name")
        task_type = payload.get("task_type")
        trigger_type = payload.get("trigger_type", "manual")
        trigger_time_str = payload.get("trigger_time")
        interval_days = payload.get("interval_days")
        remarks = payload.get("remarks")
    except Exception as e:
        return JsonResponse({"detail": f"無效的 JSON 請求: {str(e)}"}, status=400)
    
    if not name or not task_type:
        return JsonResponse({"detail": "任務名稱與類型為必填項目"}, status=400)
        
    parsed_trigger_time = datetime.now()
    if trigger_time_str:
        try:
            parsed_trigger_time = datetime.strptime(trigger_time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return JsonResponse({"detail": "trigger_time 格式不符。必須為 YYYY-MM-DD HH:MM:SS"}, status=400)
            
    if trigger_type == "auto":
        if not interval_days or int(interval_days) < 1:
            return JsonResponse({"detail": "自動週期任務的 interval_days 必須大於等於 1"}, status=400)
            
    new_job = Job(
        name=name,
        status="pending",
        trigger_type=trigger_type,
        trigger_time=parsed_trigger_time,
        task_type=task_type,
        interval_days=int(interval_days) if interval_days else None,
        remarks=remarks
    )
    
    db = SessionLocal()
    try:
        db.add(new_job)
        db.commit()
        db.refresh(new_job)
        logger.info(f"成功在 Django 整合環境下建立任務: {new_job.name} (ID: {new_job.id})")
        return JsonResponse({
            "id": new_job.id,
            "name": new_job.name,
            "status": new_job.status,
            "trigger_type": new_job.trigger_type,
            "trigger_time": new_job.trigger_time.isoformat() if new_job.trigger_time else None,
            "task_type": new_job.task_type,
            "interval_days": new_job.interval_days,
            "remarks": new_job.remarks,
        })
    except Exception as e:
        db.rollback()
        logger.error(f"建立任務失敗: {e}")
        return JsonResponse({"detail": f"寫入資料庫失敗: {str(e)}"}, status=500)
    finally:
        db.close()

@csrf_exempt
def jobs_list_or_create(request):
    """取得或建立任務"""
    if request.method == "POST":
        return create_job(request)
    return list_jobs(request)

@csrf_exempt
def trigger_job_immediately(request, job_id):
    """立即重新觸發某個任務"""
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
        
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return JsonResponse({"detail": "找不到該任務"}, status=404)
            
        if job.status == "running":
            return JsonResponse({"detail": "該任務目前正在執行中，無法立即觸發"}, status=400)
            
        job.status = "pending"
        job.trigger_time = datetime.now()
        job.completion_time = None
        job.duration = None
        db.commit()
        logger.info(f"立即手動觸發任務 (ID: {job.id}, Name: {job.name})")
        return JsonResponse({
            "id": job.id,
            "status": job.status,
            "trigger_time": job.trigger_time.isoformat(),
        })
    except Exception as e:
        db.rollback()
        logger.error(f"觸發任務失敗: {e}")
        return JsonResponse({"detail": f"更新資料庫失敗: {str(e)}"}, status=500)
    finally:
        db.close()

@csrf_exempt
def delete_job(request, job_id):
    """刪除或取消某個任務"""
    if request.method != "DELETE":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
        
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return JsonResponse({"detail": "找不到該任務"}, status=404)
            
        if job.status == "running":
            return JsonResponse({"detail": "任務執行中，無法刪除或取消"}, status=400)
            
        if job.status == "pending":
            db.delete(job)
            db.commit()
            return JsonResponse({"detail": f"成功刪除任務 ID {job_id}"})
        else:
            job.status = "cancelled"
            db.commit()
            return JsonResponse({"detail": f"成功將任務 ID {job_id} 狀態更改為已取消"})
    except Exception as e:
        db.rollback()
        logger.error(f"刪除任務失敗: {e}")
        return JsonResponse({"detail": f"資料庫操作失敗: {str(e)}"}, status=500)
    finally:
        db.close()

@csrf_exempt
def update_job_heartbeat(request, job_id):
    """由 Worker 執行緒呼叫，更新任務心跳"""
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
        
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return JsonResponse({"detail": "找不到該任務"}, status=404)
        if job.status != "running":
            return JsonResponse({"detail": "任務並非執行中狀態，拒絕更新心跳"}, status=400)
            
        job.last_heartbeat = datetime.now()
        db.commit()
        return JsonResponse({"status": "ok", "last_heartbeat": job.last_heartbeat.isoformat()})
    except Exception as e:
        db.rollback()
        return JsonResponse({"detail": f"更新心跳失敗: {str(e)}"}, status=500)
    finally:
        db.close()

@csrf_exempt
def llm_parse_prompt(request):
    """由 AI 自然語言解析排程需求"""
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)
        
    try:
        payload = json.loads(request.body)
        prompt = payload.get("prompt")
    except Exception as e:
        return JsonResponse({"detail": f"無效的 JSON 請求: {str(e)}"}, status=400)
        
    if not prompt:
        return JsonResponse({"detail": "prompt 為必填欄位"}, status=400)
        
    try:
        result = parse_natural_language_task(prompt)
        return JsonResponse(result)
    except Exception as e:
        logger.error(f"LLM 解析失敗: {e}")
        return JsonResponse({"detail": str(e)}, status=500)

def get_llm_usage(request):
    """取得今日 LLM 呼叫配額資訊"""
    count = get_current_usage_count()
    return JsonResponse({"count": count, "limit": DAILY_LIMIT})
