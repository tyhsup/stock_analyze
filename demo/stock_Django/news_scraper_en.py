"""
news_scraper_en.py — Finnhub English stock news scraper for North American tickers.

Supports:
- US stock news: GET /company-news?symbol={ticker}&from={date}&to={date}
- Output interface compatible with CnyesScraper.
"""

import os
import time
import logging
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Ensure .env is loaded
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


class FinnhubScraper:
    """
    Finnhub 英文財經新聞抓取器。
    與 CnyesScraper 回傳介面相容，回傳與中文新聞相同結構的字典清單。
    """
    def __init__(self):
        self.api_key = os.getenv("FINNHUB_API_KEY", "").strip()
        if not self.api_key:
            # Fallback to root ~/.gemini/antigravity/.env if needed
            alt_env = os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", ".env")
            if os.path.exists(alt_env):
                load_dotenv(alt_env)
                self.api_key = os.getenv("FINNHUB_API_KEY", "").strip()

    def fetch_news(self, ticker: str, limit: int = 20, days_back: int = 30) -> list:
        """
        獲取美股英文新聞。

        Args:
            ticker: 美股代碼 (例如 'AAPL', 'NVDA', 'TSLA')
            limit: 限制回傳的新聞筆數
            days_back: 搜尋幾天前的新聞

        Returns:
            新聞字典陣列，包含：標題, 日期, 內容, 連結, 正負分析, 來源, 語言
        """
        start_time = time.time()
        ticker = str(ticker).strip().upper().replace('.TW', '').replace('.TWO', '')

        if not self.api_key:
            logger.error("[FinnhubScraper] FINNHUB_API_KEY 未設定，無法獲取英文新聞")
            return None

        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        url = f"{FINNHUB_BASE_URL}/company-news"
        params = {
            'symbol': ticker,
            'from': from_date,
            'to': to_date,
            'token': self.api_key
        }

        logger.info(f"[FinnhubScraper] 開始從 Finnhub 抓取 {ticker} 英文新聞, limit: {limit}")

        try:
            res = requests.get(url, params=params, timeout=12)
            if res.status_code != 200:
                logger.error(f"[FinnhubScraper] Finnhub API 錯誤 (HTTP {res.status_code}): {res.text}")
                return None

            raw_data = res.json()
            if not isinstance(raw_data, list):
                logger.warning(f"[FinnhubScraper] API 未回傳列表格式: {raw_data}")
                return []

            articles = []
            source_label = f"Finnhub-{ticker}"

            for item in raw_data[:limit]:
                headline = item.get("headline", "").strip()
                summary = item.get("summary", "").strip()
                link = item.get("url", "").strip()
                dt_ts = item.get("datetime", 0)

                # 解析時間戳記
                if dt_ts:
                    parsed_date = datetime.fromtimestamp(dt_ts).strftime('%Y-%m-%d')
                else:
                    parsed_date = datetime.now().strftime('%Y-%m-%d')

                # 若沒有摘要，以標題代替
                content = summary if len(summary) > 10 else headline

                articles.append({
                    '標題': headline,
                    '日期': parsed_date,
                    '內容': content,
                    '連結': link,
                    '正負分析': '中性',  # 預設中性，後續由 AgentNewsAnalyzer 強化
                    '來源': f"{source_label} ({item.get('source', 'Yahoo')})",
                    '語言': 'en'
                })

            elapsed = time.time() - start_time
            logger.info(f"[FinnhubScraper] 成功獲取 {len(articles)} 則 {ticker} 英文新聞，總耗時: {elapsed:.2f} 秒")
            return articles

        except Exception as e:
            logger.error(f"[FinnhubScraper] 呼叫 Finnhub 異常: {e}", exc_info=True)
            return None
