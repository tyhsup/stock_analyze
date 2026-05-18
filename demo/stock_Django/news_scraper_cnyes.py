"""
news_scraper_cnyes.py — 鉅亨網 (cnyes.com) news scraper using cnyes-cli and BeautifulSoup.

Supports:
- TW stock news: get-news-list-by-symbol TWS:{ticker}:STOCK
- US stock news: get-news-list-by-symbol USS:{ticker}:STOCK
- General search: get-news-list-by-category tw_stock --keyword {query}

If total articles < 10, automatically tries to fetch the maximum available.
"""

import logging
import time
import random
import json
import subprocess
from datetime import datetime
import requests
from bs4 import BeautifulSoup

CNYES_WWW = "https://www.cnyes.com"
logger = logging.getLogger(__name__)

# Positive/negative keywords for rough sentiment analysis (fallback)
POSITIVE_KEYWORDS = ['上漲', '獲利', '成長', '突破', '創高', '買超', '增持', 'buy', 'upgrade', 'beat', 'profit', 'growth', 'raise']
NEGATIVE_KEYWORDS = ['下跌', '虧損', '衰退', '破位', '下修', '賣超', '減持', 'sell', 'downgrade', 'miss', 'loss', 'decline']

# Import NLPService for advanced sentiment analysis
try:
    from .nlp_service import NLPService
    _nlp_service = NLPService()
except (ImportError, Exception):
    _nlp_service = None


USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0'
]


def _analyze_sentiment(text: str, content: str = None) -> str:
    """Advanced BERT sentiment analysis with keyword fallback. Prioritizes content."""
    # 優先分析內容，若內容過短則分析標題
    analysis_text = content if (content and len(content) > 50) else text
    if not analysis_text: return '中性'
    
    if _nlp_service:
        res = _nlp_service.analyze_sentiment(analysis_text)
        if res.get('label') != 'error':
            if res.get('label') == 'positive': return '正面'
            if res.get('label') == 'negative': return '負面'
            return '中性'

    # Fallback to simple keyword-based sentiment
    text_lower = analysis_text.lower()
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw.lower() in text_lower)
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw.lower() in text_lower)
    if pos > neg:
        return '正面'
    elif neg > pos:
        return '負面'
    return '中性'


def _parse_cnyes_timestamp(ts_str: str) -> str:
    """Parse cnyes datetime strings like '2025/01/15 09:30' or Unix timestamps."""
    try:
        ts_str = str(ts_str).strip()
        if not ts_str: return datetime.now().strftime('%Y-%m-%d')
        
        if ts_str.isdigit():
            return datetime.fromtimestamp(int(ts_str)).strftime('%Y-%m-%d')
        for fmt in ['%Y/%m/%d %H:%M', '%Y-%m-%d %H:%M', '%Y/%m/%d', '%Y-%m-%d', '%Y/%m/%d %H:%M:%S']:
            try:
                return datetime.strptime(ts_str, fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
    except Exception:
        pass
    return datetime.now().strftime('%Y-%m-%d')


def _to_cnyes_symbol(ticker: str, market: str = 'tw') -> str:
    """將 Ticker 轉換為鉅亨網標準 Symbol 格式。"""
    t = str(ticker).strip().upper()
    # 去除常見的字尾
    if t.endswith('.TW'):
        t = t[:-3]
    elif t.endswith('.TWO'):
        t = t[:-4]
        
    # 如果已經是標準格式，直接回傳
    if t.startswith('TWS:') or t.startswith('USS:'):
        return t
        
    # 判斷是否為台股（純數字代碼）
    if t.isdigit() or market == 'tw':
        return f"TWS:{t}:STOCK"
    else:
        return f"USS:{t}:STOCK"


def _run_cnyes_cli(cmd_args: list) -> dict:
    """呼叫 cnyes-cli 並回傳解析後的 JSON 物件。"""
    cmd = ["cnyes-cli"] + cmd_args + ["--agent"]
    try:
        cmd_str = " ".join(cmd)
        logger.info(f"執行 CLI 命令: {cmd_str}")
        res = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        if res.returncode == 0:
            return json.loads(res.stdout)
        else:
            logger.error(f"cnyes-cli 執行錯誤 (Exit {res.returncode}): {res.stderr}")
    except Exception as e:
        logger.error(f"無法執行 cnyes-cli: {e}")
    return {}


def _fetch_html_content(url: str) -> str:
    """使用 requests 抓取新聞網頁，並使用 BeautifulSoup 解析出純文字內文。"""
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
    }
    try:
        # 加上隨機微小延遲防止觸發防爬機制
        time.sleep(random.uniform(0.1, 0.3))
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            
            # 策略：尋找所有 article 標籤，並挑選內容最長且大於 100 字的那個以防 SEO 標籤干擾
            articles = soup.find_all('article')
            art = None
            if articles:
                sorted_articles = sorted(articles, key=lambda a: len(a.get_text(strip=True)), reverse=True)
                longest_art = sorted_articles[0]
                if len(longest_art.get_text(strip=True)) > 100:
                    art = longest_art
            
            # 如果沒有合適的 article 標籤，再嘗試其他 selectors
            if not art:
                art = (soup.find(attrs={"itemprop": "articleBody"}) or 
                       soup.find('section', class_=lambda c: c and 'ArticleContent' in c) or 
                       soup.find(class_='news-content') or 
                       soup.find(class_='article-content'))
                       
            if art:
                text = art.get_text(separator=' ', strip=True)
                text = " ".join(text.split())
                return text[:2000]
    except Exception as e:
        logger.warning(f"抓取內文失敗 {url}: {e}")
    return ""


class CnyesScraper:
    """
    透過 cnyes-cli (API) 與輕量 HTTP 抓取技術重構的新一代鉅亨網新聞抓取器。
    與原有 Selenium Scraper 介面 100% 相容。
    """
    def __init__(self, headless: bool = True):
        # 為了向後相容保留 headless 參數，但不做任何 Selenium 啟動
        self.headless = headless
        self.min_articles = 10

    def fetch_news(self, ticker: str, market: str = 'tw', limit: int = 1000) -> list:
        """
        獲取特定個股的新聞列表。
        
        Args:
            ticker: 股票代碼 (例如 '2330', 'AAPL')
            market: 'tw' 代表台股，'us' 代表美股
            limit: 限制回傳的新聞筆數
            
        Returns:
            新聞字典陣列，包含鍵值：標題, 日期, 內容, 連結, 正負分析, 來源
        """
        start_time = time.time()
        symbol = _to_cnyes_symbol(ticker, market)
        logger.info(f"[CnyesScraper] 開始透過 cnyes-cli 獲取個股新聞: {symbol}, limit: {limit}")
        
        # 清除 symbol 尾端的市場標記以符合舊版來源名稱標記行為
        clean_ticker = ticker.replace('.TW', '').replace('.TWO', '')
        source_label = f"鉅亨網-台股-{clean_ticker}" if market == 'tw' else f"鉅亨網-美股-{clean_ticker}"
        
        # 1. 呼叫 cnyes-cli 獲取列表
        cmd_args = ["media", "get-news-list-by-symbol", symbol, "--limit", str(limit)]
        json_data = _run_cnyes_cli(cmd_args)
        
        # 解析列表數據
        articles_data = json_data.get("results", {}).get("items", {}).get("data", [])
        if not articles_data:
            logger.warning(f"[CnyesScraper] 未能獲取 {symbol} 的新聞列表或列表為空")
            return []
            
        articles = []
        # 2. 迭代列表並透過輕量 HTTP GET 補全內文與情緒分析
        for item in articles_data[:limit]:
            title = item.get("title", "").strip()
            publish_at = item.get("publishAt", 0)
            
            # 解析日期
            parsed_date = _parse_cnyes_timestamp(str(publish_at))
            
            # 拼接網址，鉅亨網新聞網址格式為 https://news.cnyes.com/news/id/{newsId}
            news_id = item.get("newsId")
            link = f"https://news.cnyes.com/news/id/{news_id}" if news_id else ""
            
            # 抓取內文（個股 API 預設無 content）
            content = _fetch_html_content(link) if link else ""
            
            # 進行情緒分析
            sentiment = _analyze_sentiment(title, content)
            
            articles.append({
                '標題': title,
                '日期': parsed_date,
                '連結': link,
                '內容': content,
                '正負分析': sentiment,
                '來源': source_label,
            })
            
        elapsed = time.time() - start_time
        logger.info(f"[CnyesScraper] 成功獲取 {len(articles)} 篇新聞，總耗時: {elapsed:.2f} 秒")
        return articles

    def search_news(self, query: str, limit: int = 1000) -> list:
        """
        搜尋鉅亨網新聞（不限特定個股，支援關鍵字搜尋）。
        為了高效獲取並獲取完整內文，我們呼叫分類新聞 API 並在其中帶入 keyword 參數。
        由於分類新聞 API 直接包含 content 欄位，這使得我們可以直接解析，速度極快。
        """
        start_time = time.time()
        logger.info(f"[CnyesScraper] 開始搜尋關鍵字: '{query}', limit: {limit}")
        source_label = f"鉅亨網-搜尋-{query}"
        
        # 我們主要在台股新聞中進行搜尋
        cmd_args = ["media", "get-news-list-by-category", "tw_stock", "--keyword", query, "--limit", str(limit)]
        json_data = _run_cnyes_cli(cmd_args)
        
        articles_data = json_data.get("results", {}).get("items", {}).get("data", [])
        articles = []
        
        for item in articles_data[:limit]:
            title = item.get("title", "").strip()
            publish_at = item.get("publishAt", 0)
            parsed_date = _parse_cnyes_timestamp(str(publish_at))
            
            news_id = item.get("newsId")
            link = f"https://news.cnyes.com/news/id/{news_id}" if news_id else ""
            
            # 分類新聞直接含有 content（為 HTML 格式）
            content_html = item.get("content", "")
            content = ""
            if content_html:
                try:
                    content = BeautifulSoup(content_html, 'html.parser').get_text(separator=' ', strip=True)
                    content = " ".join(content.split())[:2000]
                except Exception as e:
                    logger.warning(f"解析分類新聞 HTML 失敗: {e}")
                    
            sentiment = _analyze_sentiment(title, content)
            
            articles.append({
                '標題': title,
                '日期': parsed_date,
                '連結': link,
                '內容': content,
                '正負分析': sentiment,
                '來源': source_label,
            })
            
        elapsed = time.time() - start_time
        logger.info(f"[CnyesScraper] 搜尋 '{query}' 成功返回 {len(articles)} 筆結果，總耗時: {elapsed:.2f} 秒")
        return articles

    def get_latest_news_url(self, ticker: str, market: str = 'tw') -> str:
        """回傳 UI 用來連結至鉅亨網個股新聞的 URL。"""
        return f"https://www.cnyes.com/search/news?keyword={ticker}"
