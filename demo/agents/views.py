import json
import logging
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from .orchestrator import FinancialOrchestrator

logger = logging.getLogger(__name__)

# 初始化全局 Orchestrator 實例
orchestrator = FinancialOrchestrator()

@csrf_exempt
@require_POST
def chat_api(request):
    """
    金融 AI 助理對話 API 接口。
    接受 JSON POST 請求，格式：{"message": "使用者輸入", "history": []}。
    """
    try:
        data = json.loads(request.body)
        user_message = data.get("message", "").strip()
        history = data.get("history", [])
        
        if not user_message:
            return JsonResponse({"status": "error", "message": "訊息內容不可為空"}, status=400)
            
        logger.info(f"[ChatAPI] 收到使用者訊息: {user_message}")
        
        # 執行 Orchestrator 分發調度與推理
        result = orchestrator.chat(user_message, history)
        
        return JsonResponse({
            "status": "success",
            "agent": result.get("agent"),
            "reason": result.get("reason"),
            "response": result.get("response"),
            "route_log": result.get("route_log", [])
        })
        
    except json.JSONDecodeError:
        return JsonResponse({"status": "error", "message": "無效的 JSON 格式"}, status=400)
    except Exception as e:
        logger.error(f"[ChatAPI] 處理對話異常: {e}", exc_info=True)
        return JsonResponse({"status": "error", "message": f"伺服器內部錯誤: {str(e)}"}, status=500)

def chat_panel_template(request):
    """
    提供對話面板 HTML 模板（供 AJAX 或 iframe 呼叫，亦可直接 include）。
    """
    return render(request, "agents/chat_panel.html")


def model_audit_view(request):
    """
    渲染 Excel 財務模型稽核與 Debug 獨立網頁。
    """
    context = {"test_data_json": None}
    if request.GET.get('test') == '1':
        import os
        from io import BytesIO
        from .services.model_auditor import ExcelModelAuditor
        
        test_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch", "test_2330_model.xlsx")
        if os.path.exists(test_file_path):
            try:
                with open(test_file_path, "rb") as f:
                    file_data = BytesIO(f.read())
                    audit_res = ExcelModelAuditor.audit_workbook(file_data)
                    context["test_data_json"] = json.dumps(audit_res)
            except Exception as ex:
                logger.warning(f"[AuditView] 載入 test=1 檔案失敗: {ex}")
                
    return render(request, "agents/model_audit.html", context)


@csrf_exempt
@require_POST
def model_audit_api(request):
    """
    接收上傳的 Excel 財務模型，執行公式稽核、硬編碼偵測與勾稽一致性校驗。
    """
    uploaded_file = request.FILES.get('file')
    if not uploaded_file:
        return JsonResponse({"status": "error", "message": "未檢測到上傳的檔案"}, status=400)
        
    filename = uploaded_file.name
    if not filename.endswith('.xlsx'):
        return JsonResponse({"status": "error", "message": "僅支援上傳 .xlsx 格式的 Excel 活頁簿"}, status=400)
        
    try:
        from .services.model_auditor import ExcelModelAuditor
        # 讀取檔案流進行稽核
        audit_results = ExcelModelAuditor.audit_workbook(uploaded_file)
        return JsonResponse(audit_results)
    except Exception as e:
        logger.error(f"[AuditAPI] Excel 模型稽核失敗: {e}", exc_info=True)
        return JsonResponse({"status": "error", "message": f"稽核過程中發生錯誤: {str(e)}"}, status=500)
