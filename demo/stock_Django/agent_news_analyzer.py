import os
import sys
import json
import logging
import time

# 載入全局輔助腳本路徑 (如 financial_news_analyst.md 所述)
GLOBAL_HELPER_PATH = r"c:\Users\許廷宇\.gemini\antigravity\scripts"
if GLOBAL_HELPER_PATH not in sys.path:
    sys.path.append(GLOBAL_HELPER_PATH)

try:
    from groq_helper import get_groq_response
    from groq import Groq
except ImportError:
    # 這裡保留一個保險，如果全局導入失敗則拋出明確錯誤
    raise ImportError("無法載入全局 groq_helper。請確保 groq 庫已安裝。")

logger = logging.getLogger(__name__)


class AgentNewsAnalyzer:
    """
    金融新聞情緒分析 Agent。
    基於全局設定 C:\Users\許廷宇\.gemini\agents\financial_news_analyst.md。
    調用全局 groq_helper.py 進行推論。
    """

    def __init__(self):
        # 優先讀取全局環境變數
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        if not self.api_key:
            # 如果全局變數沒抓到，嘗試載入全局 .env
            from dotenv import load_dotenv
            dotenv_path = os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", ".env")
            load_dotenv(dotenv_path)
            self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("找不到 GROQ_API_KEY。請檢查全局 .env 配置。")

        # 初始化 Groq Client (為了能使用 response_format JSON 模式)
        self.client = Groq(api_key=self.api_key)

        self.system_prompt = """你是一位高盛集團（Goldman Sachs）的高級投資分析師，負責為機構客戶撰寫簡明扼要的市場報告。
你是專門負責「財經新聞分析」的 AI Subagent (Financial News Analyst)。
請閱讀以下金融新聞，並根據新聞內文判斷相關屬性，最後請「只」輸出嚴格的 JSON 格式。

JSON 格式要求如下：
{
  "title": "新聞標題",
  "date": "發布日期",
  "content": "新聞內文的摘要或原文",
  "link": "新聞連結",
  "source": "新聞來源",
  "market": "TW 或 US",
  "positive_negative_analysis": "正面 或 負面 或 中立",
  "sentiment_score": 0.85,
  "confidence": 0.90,
  "impact_scope": "短期 或 中期 或 長期",
  "reasoning_summary": "你的判斷理由"
}

注意：
- 一律使用台灣繁體中文回覆。
- sentiment_score 介於 -1.00 到 1.00 之間。
- confidence 介於 0.00 到 1.00 之間。
"""

    def analyze_news(self, news_data: dict, retries: int = 3) -> dict:
        """
        使用 Groq API 進行分析。
        """
        title = news_data.get('標題', news_data.get('title', ''))
        date = news_data.get('發布時間', news_data.get('date', ''))
        content = news_data.get('內文', news_data.get('content', ''))
        link = news_data.get('連結', news_data.get('link', ''))
        source = news_data.get('來源', news_data.get('source', ''))

        user_prompt = f"請分析以下新聞並以 JSON 輸出：\n標題：{title}\n發布日期：{date}\n內文：{content}"

        for attempt in range(retries):
            try:
                # 直接調用 Groq SDK 以確保強制的 JSON 輸出格式
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )

                response_text = completion.choices[0].message.content
                result_json = json.loads(response_text.strip())

                # 補齊可能缺失的欄位
                for key in ["market", "sentiment_score", "confidence", "impact_scope", "reasoning_summary"]:
                    if key not in result_json:
                        result_json[key] = ""
                
                # 保留原始連結與來源
                result_json["link"] = link
                result_json["source"] = source
                if "date" not in result_json or not result_json["date"]:
                    result_json["date"] = date

                return result_json

            except Exception as e:
                logger.warning(f"分析失敗 (第 {attempt + 1} 次重試): {e}")
                if attempt == retries - 1:
                    return self._fallback_result(news_data)
                time.sleep(2)

    def _fallback_result(self, news_data: dict) -> dict:
        return {
            "title": news_data.get('標題', news_data.get('title', '')),
            "date": news_data.get('發布時間', news_data.get('date', '')),
            "content": news_data.get('內文', news_data.get('content', '')),
            "link": news_data.get('連結', news_data.get('link', '')),
            "source": news_data.get('來源', news_data.get('source', '')),
            "market": "TW",
            "positive_negative_analysis": "中立",
            "sentiment_score": 0.0,
            "confidence": 0.0,
            "impact_scope": "短期",
            "reasoning_summary": "Agent 呼叫失敗，請檢查 API Key 或網路連線。"
        }
