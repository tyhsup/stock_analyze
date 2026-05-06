import requests
import json
import logging

logger = logging.getLogger(__name__)

class AgentNewsAnalyzer:
    def __init__(self, ollama_url="http://localhost:11434/api/generate", model_name="gemma4:26b"):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.system_prompt = """你是一位高盛集團（Goldman Sachs）的高級投資分析師，負責為機構客戶撰寫簡明扼要的市場報告。
請閱讀以下金融新聞，並根據新聞內文判斷相關屬性，最後請「只」輸出嚴格的 JSON 格式，不要包含任何 markdown 標記、也不要包含其他多餘的文字。

JSON 格式要求如下：
{
  "title": "新聞標題",
  "date": "發布日期",
  "content": "新聞內文的摘要或原文",
  "link": "新聞連結",
  "source": "新聞來源",
  "market": "TW 或 US",
  "positive_negative_analysis": "正面 或 負面 或 中立",
  "sentiment_score": 0.85, // 介於 -1.00 到 1.00 之間
  "confidence": 0.90, // 介於 0.00 到 1.00 之間
  "impact_scope": "短期 或 中期 或 長期",
  "reasoning_summary": "你的判斷理由"
}
"""

    def analyze_news(self, news_data, retries=3):
        """
        news_data: dict, containing keys 'title', 'date', 'content', 'link', 'source' (if available)
        Returns: dict parsed from JSON
        """
        title = news_data.get('標題', news_data.get('title', ''))
        date = news_data.get('發布時間', news_data.get('date', ''))
        content = news_data.get('內文', news_data.get('content', ''))
        link = news_data.get('連結', news_data.get('link', ''))
        source = news_data.get('來源', news_data.get('source', ''))

        user_prompt = f"""
請分析以下新聞：
標題：{title}
發布日期：{date}
來源：{source}
連結：{link}
內文：{content}
"""
        
        payload = {
            "model": self.model_name,
            "prompt": self.system_prompt + user_prompt,
            "stream": False,
            "options": {
                "temperature": 0.2, # 低溫度以確保 JSON 格式穩定
                "top_p": 0.65
            },
            "format": "json" # Ollama 支持強制 JSON 輸出
        }

        for attempt in range(retries):
            try:
                response = requests.post(self.ollama_url, json=payload, timeout=600)
                response.raise_for_status()
                response_text = response.json().get("response", "")
                
                # 嘗試清理可能的 Markdown 標記
                if response_text.startswith("```json"):
                    response_text = response_text[7:-3]
                elif response_text.startswith("```"):
                    response_text = response_text[3:-3]
                    
                result_json = json.loads(response_text.strip())
                
                # Ensure all required keys exist
                expected_keys = ["title", "date", "content", "link", "source", "market", 
                                 "positive_negative_analysis", "sentiment_score", "confidence", 
                                 "impact_scope", "reasoning_summary"]
                
                for key in expected_keys:
                    if key not in result_json:
                        result_json[key] = news_data.get(key, "")
                        
                return result_json

            except json.JSONDecodeError as e:
                logger.warning(f"JSON Parsing Error on attempt {attempt+1}: {e}\nResponse text: {response_text}")
                if attempt == retries - 1:
                    logger.error("Max retries reached for JSON parsing.")
                    return self._fallback_result(news_data)
            except Exception as e:
                logger.warning(f"API Error on attempt {attempt+1}: {e}")
                if attempt == retries - 1:
                    logger.error("Max retries reached for API call.")
                    return self._fallback_result(news_data)

    def _fallback_result(self, news_data):
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
            "reasoning_summary": "API 或 JSON 解析失敗，無法分析情緒。"
        }
