import os
import json
import logging
from datetime import datetime, date
import httpx
from pydantic import BaseModel, Field
from typing import Optional

from google import genai
from google.genai import types

from .database import SessionLocal
from .models import LlmUsage

logger = logging.getLogger("scheduler.llm")

# 限制每日最大 LLM 呼叫次數
DAILY_LIMIT = 50

class TaskParseResult(BaseModel):
    task_type: str = Field(description="必須是以下其中之一：'tw_stock_cost' (台股股價更新)、'us_stock_cost' (美股股價更新)、'tw_stock_price_only' (更新全台灣股價)、'us_stock_price_only' (更新全美國股價)、'twse_investor' (台股上市三大法人)、'tpex_investor' (台股上櫃三大法人)、'us_investor' (美股三大法人持股)、'tw_listed_list_update' (台灣上市公司清單更新)、'tw_otc_list_update' (台灣上櫃公司清單更新)、'us_stock_list_update' (美國上市公司清單更新)")
    name: str = Field(description="此任務的中文名稱，描述要執行的操作，例如：'手動更新台股 2330 股價'")
    interval_days: Optional[int] = Field(None, description="週期天數（若是單次執行任務，請填 null 或 0；如果是定期任務如每 3 天更新一次，填 3）")
    remarks: Optional[str] = Field(None, description="解析出的股票代碼清單，多個以半形逗號分隔，例如 '2330' 或 'AAPL,NVDA'。若無特定股票代碼則填 null")
    trigger_time: Optional[str] = Field(None, description="預計觸發時間，格式必須為 YYYY-MM-DD HH:MM:SS。若是立刻執行或未指定時間，則填 null")

def check_and_increment_quota() -> bool:
    """
    檢查今日 LLM 呼叫次數是否超過上限，若未超過則先不增加，等成功呼叫後再累加。
    此方法只回傳是否可以進行呼叫。
    """
    today_date = date.today()
    db = SessionLocal()
    try:
        usage = db.query(LlmUsage).filter(LlmUsage.date == today_date).first()
        if usage and usage.count >= DAILY_LIMIT:
            logger.warning(f"今日 LLM 呼叫已達上限 ({usage.count}/{DAILY_LIMIT} 次)")
            return False
        return True
    except Exception as e:
        logger.error(f"檢查配額失敗: {e}")
        return True  # 發生資料庫錯誤時放行，以防阻礙正常使用
    finally:
        db.close()

def record_usage_success():
    """
    呼叫 LLM 成功後，累加今日的使用次數
    """
    today_date = date.today()
    db = SessionLocal()
    try:
        usage = db.query(LlmUsage).filter(LlmUsage.date == today_date).first()
        if not usage:
            usage = LlmUsage(date=today_date, count=1)
            db.add(usage)
        else:
            usage.count += 1
        db.commit()
        logger.info(f"LLM 呼叫成功，今日累計次數: {usage.count}/{DAILY_LIMIT}")
    except Exception as e:
        logger.error(f"記錄使用次數失敗: {e}")
        db.rollback()
    finally:
        db.close()

def get_current_usage_count() -> int:
    """
    取得今日已使用的 LLM 呼叫次數
    """
    today_date = date.today()
    db = SessionLocal()
    try:
        usage = db.query(LlmUsage).filter(LlmUsage.date == today_date).first()
        return usage.count if usage else 0
    except Exception:
        return 0
    finally:
        db.close()

def parse_task_with_gemini(prompt: str, api_key: str) -> dict:
    """
    使用雲端 Gemini 3.5 Flash 模型進行結構化解析
    """
    logger.info("發送請求至雲端 Gemini 3.5 Flash...")
    client = genai.Client(api_key=api_key)
    
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    system_instruction = f"""
    你是一個工作排程分析助手。
    請分析使用者的自然語言輸入，並將其解析為對應的排程任務資訊。
    目前系統的基準時間為: {current_time_str}。
    如果使用者提到的時間是相對的（例如「明天早上 9 點」、「下週一」），請以此基準時間計算出絕對時間並格式化為 YYYY-MM-DD HH:MM:SS。
    若未指定時間或指明「立刻」、「現在」，trigger_time 欄位請務必填 null。
    注意：
    - 台股代號一般為 4 位數字（如 2330）。
    - 美股代號為英文字母（如 AAPL, TSLA, NVDA）。
    - 任務類型 (task_type) 必須嚴格限定為:
      1. 'tw_stock_cost' (更新台股股價)
      2. 'us_stock_cost' (更新美股股價)
      3. 'tw_stock_price_only' (更新全台灣股價)
      4. 'us_stock_price_only' (更新全美國股價)
      5. 'twse_investor' (更新台股上市三大法人)
      6. 'tpex_investor' (更新台股上櫃三大法人)
      7. 'us_investor' (更新美股三大法人持股)
      8. 'tw_listed_list_update' (台灣上市公司清單更新)
      9. 'tw_otc_list_update' (台灣上櫃公司清單更新)
      10. 'us_stock_list_update' (美國上市公司清單更新)
    """
    
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=TaskParseResult,
            temperature=0.1,
        ),
    )
    
    # 解析 JSON 回傳值
    result = json.loads(response.text)
    logger.info(f"Gemini 解析成功: {result}")
    return result

def parse_task_with_ollama(prompt: str) -> dict:
    """
    使用本地 Ollama (gemma4:e4b) 模型進行解析 (作為備援)
    """
    logger.info("雲端 Gemini 呼叫失敗，啟用本地 Ollama 備援模式 (gemma4:e4b)...")
    
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    system_prompt = f"""
    你是一個工作排程分析助手。請分析使用者的自然語言輸入，並將其解析為對應的排程任務 JSON 資訊。
    目前系統的基準時間為: {current_time_str}。
    請依此基準時間計算相對時間並格式化為 YYYY-MM-DD HH:MM:SS，若未提及時間或說立刻，trigger_time 設為 null。

    你必須回傳一個合法的 JSON 物件，格式如下，且不得包含額外的 Markdown 標籤或說明文字：
    {{
      "task_type": "tw_stock_cost | us_stock_cost | tw_stock_price_only | us_stock_price_only | twse_investor | tpex_investor | us_investor | tw_listed_list_update | tw_otc_list_update | us_stock_list_update",
      "name": "任務描述名稱",
      "interval_days": 週期天數整數或 null,
      "remarks": "代號列表如 '2330' 或 'AAPL,NVDA' 或 null",
      "trigger_time": "YYYY-MM-DD HH:MM:SS 或 null"
    }}
    """
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma4:e4b",
        "prompt": f"{system_prompt}\n\n使用者輸入: {prompt}\n\nJSON 輸出:\n",
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }
    
    # 呼叫本地 Ollama
    resp = httpx.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    resp_data = resp.json()
    response_text = resp_data.get("response", "").strip()
    
    result = json.loads(response_text)
    logger.info(f"Ollama 解析成功: {result}")
    
    # 進行資料欄位格式安全校驗，防止 Ollama 隨機命名
    valid_keys = {"task_type", "name", "interval_days", "remarks", "trigger_time"}
    for k in list(result.keys()):
        if k not in valid_keys:
            result.pop(k)
            
    # 預設補齊缺漏欄位
    if "task_type" not in result:
        result["task_type"] = "tw_stock_cost"
    if "name" not in result:
        result["name"] = f"LLM 解析任務 - {result['task_type']}"
        
    return result

def parse_natural_language_task(prompt: str) -> dict:
    """
    主要對外介面：
    1. 檢查今日呼叫額度
    2. 優先嘗試 Gemini 3.5 Flash
    3. 若失敗則嘗試本地 Ollama 備援
    """
    # 檢查是否超出每日額度
    if not check_and_increment_quota():
        raise Exception(f"今日 LLM 呼叫次數已達到上限 ({DAILY_LIMIT} 次)，暫停服務。")
        
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        logger.warning("未偵測到環境變數 GEMINI_API_KEY，直接採用 Ollama 備援方案。")
        try:
            result = parse_task_with_ollama(prompt)
            record_usage_success()
            return result
        except Exception as err:
            raise Exception(f"Ollama 解析也發生錯誤: {err}")
            
    try:
        # 1. 嘗試雲端 Gemini
        result = parse_task_with_gemini(prompt, api_key)
        record_usage_success()
        return result
    except Exception as gemini_err:
        logger.error(f"雲端 Gemini 呼叫異常: {gemini_err}")
        # 2. 嘗試本地 Ollama Fallback
        try:
            result = parse_task_with_ollama(prompt)
            record_usage_success()
            return result
        except Exception as ollama_err:
            raise Exception(f"雲端 Gemini 與本地 Ollama 均無法解析。Gemini 錯誤: {gemini_err} | Ollama 錯誤: {ollama_err}")
