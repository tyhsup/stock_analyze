"""
news_scraper_cnyes.py — 鉅亨網 (cnyes.com) news scraper using Selenium.

Supports:
- TW stock news: /news/cat/tw_stock?keyword={ticker}
- US stock news: /news/cat/us_stock?keyword={keyword}
- General search: /news/search?keyword={query}

If total articles < 10, automatically tries to fetch the maximum available.
"""

import logging
import time
import random
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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


def _make_driver(headless: bool = True):
    """Create a configured Edge WebDriver."""
    options = webdriver.EdgeOptions()
    if headless:
        options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1280,800')
    options.add_argument('--blink-settings=imagesEnabled=false')
    
    # User-Agent Rotation
    ua = random.choice(USER_AGENTS)
    options.add_argument(f'--user-agent={ua}')
    
    return webdriver.Edge(options=options)


def _analyze_sentiment(text: str, content: str = None) -> str:
    """Advanced BERT sentiment analysis with keyword fallback. Prioritizes content."""
    # 優先分析內容，若內容過短則分析標題
    analysis_text = content if (content and len(content) > 50) else text
    if not analysis_text: return '中性'
    
    if _nlp_service:
        res = _nlp_service.analyze_sentiment(analysis_text)
        if res.get('label') != 'error':
            # label is already handled in nlp_service update to follow
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


class CnyesScraper:
    """Scrapes 鉅亨網 (cnyes.com) for stock news using Selenium."""

    def __init__(self, headless: bool = True):
        self.headless = headless
        self.min_articles = 10  # If we get fewer than this, try to fetch more

    def _fetch_article_content(self, driver, url: str) -> str:
        """進入新聞頁面抓取文章主體內容。"""
        # P0: Implement Retry Logic
        for attempt in range(2):
            try:
                driver.get(url)
                # 增加隨機等待時間以確保 JS 內容渲染完成 (2-5s)
                time.sleep(random.uniform(2.5, 4.5)) 
                
                # 使用多種方式嘗試抓取內容
                content = driver.execute_script("""
                    let art = document.querySelector('article') || 
                              document.querySelector('[itemprop="articleBody"]') ||
                              document.querySelector('section[class*="ArticleContent"]') ||
                              document.querySelector('.news-content') ||
                              document.querySelector('.article-content');
                    return art ? art.innerText : '';
                """)
                
                if not content or len(content) < 100:
                    selectors = ['article', 'div[itemprop="articleBody"]', 'section[class*="ArticleContent"]', '.news-content', '.article-content']
                    for res in selectors:
                        try:
                            element = driver.find_element(By.CSS_SELECTOR, res)
                            content = element.text.strip()
                            if len(content) > 100: break
                        except: continue

                if content:
                    content = " ".join(content.split())
                    return content[:2000]
            except Exception as e:
                logger.warning(f"無法抓取文章內容 (嘗試 {attempt+1}) {url}: {e}")
                time.sleep(2)
        return ""

    def _scroll_and_collect(self, driver, wait, max_articles: int, source_label: str) -> list:
        """
        Scroll down to load more articles via infinite scroll.
        Collects up to max_articles news items.
        """
        articles = []
        seen_urls = set()
        scroll_attempts = 0
        max_scrolls = 300 
        consecutive_no_new = 0

        while len(articles) < max_articles and scroll_attempts < max_scrolls:
            try:
                # Optimized selectors: added 'div.news-item' and similar variations found in latest Cnyes
                items = driver.find_elements(By.CSS_SELECTOR, 'a.news, article, [class*="NewsItem"], div[class*="news-item"]')

                new_found_this_cycle = 0
                for item in items:
                    try:
                        link = item.get_attribute('href')
                        if not link:
                            # Try finding 'a' child
                            try:
                                link = item.find_element(By.TAG_NAME, 'a').get_attribute('href')
                            except: pass
                        
                        if not link or link in seen_urls:
                            continue
                        seen_urls.add(link)

                        # Title
                        title = ""
                        for sel in ['h3', 'h2', 'h1', '[class*="title"]', '.text']:
                            try:
                                title_el = item.find_element(By.CSS_SELECTOR, sel)
                                title = title_el.text.strip()
                                if title: break
                            except: pass
                        
                        if not title: title = item.text.strip().split('\n')[0][:80]
                        if not title: continue

                        # Date
                        date_text = ''
                        for sel in ['time', '[class*="date"]', '[class*="time"]', 'span']:
                            try:
                                el = item.find_element(By.CSS_SELECTOR, sel)
                                date_text = el.get_attribute('datetime') or el.text
                                if date_text and (':' in date_text or '/' in date_text or '-' in date_text): break
                            except: pass

                        parsed_date = _parse_cnyes_timestamp(date_text)
                        
                        articles.append({
                            '標題': title,
                            '日期': parsed_date,
                            '連結': link,
                            '正負分析': '中性',
                            '來源': source_label,
                        })
                        new_found_this_cycle += 1

                        if len(articles) >= max_articles:
                            break
                    except Exception:
                        continue

                if len(articles) >= max_articles: break

                # 2. Infinite scroll trigger
                prev_height = driver.execute_script("return document.body.scrollHeight")
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 200);")
                time.sleep(random.uniform(0.3, 0.6))
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(random.uniform(1.2, 2.0))
                
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_found_this_cycle == 0:
                    consecutive_no_new += 1
                else:
                    consecutive_no_new = 0

                if consecutive_no_new >= 5 and new_height == prev_height:
                    break
                scroll_attempts += 1
            except Exception as e:
                logger.error(f"滾動抓取異常: {e}")
                break

        # --- 第二階段：逐一獲取內容並分析情緒 ---
        logger.info(f"開始抓取 {len(articles)} 篇新聞的詳細內容...")
        for article in articles:
            content = self._fetch_article_content(driver, article['連結'])
            article['內容'] = content
            article['正負分析'] = _analyze_sentiment(article['標題'], content)
            
        return articles

    def fetch_news(self, ticker: str, market: str = 'tw', limit: int = 1000) -> list:
        """
        Fetch stock news for a given ticker from 鉅亨網.

        Args:
            ticker: Stock ticker (e.g., '2330', 'AAPL')
            market: 'tw' for Taiwan stock, 'us' for US stock
            limit: Max articles to fetch. If result < 10, auto-expands to try more.

        Returns:
            List of news dicts with keys: 標題, 日期, 內容, 連結, 正負分析, 來源
        """
        # Build search URL
        # Ticker news on Anue is best fetched via the search endpoint for reliable filtering
        # The category URLs (cat/tw_stock_news) don't support keyword filtering reliably
        url = f"{CNYES_WWW}/search/news?keyword={ticker}"
        
        if market == 'tw':
            source_label = f"鉅亨網-台股-{ticker}"
        else:
            source_label = f"鉅亨網-美股-{ticker}"

        logger.info(f"[CnyesScraper] Fetching news for {ticker} ({market}): {url}, limit: {limit}")

        driver = _make_driver(self.headless)
        articles = []
        try:
            driver.get(url)
            wait = WebDriverWait(driver, 15)

            # Wait for page content to load
            time.sleep(3)

            articles = self._scroll_and_collect(driver, wait, limit, source_label)

            # If fewer than requested, try expanding — fetch more aggressively
            if len(articles) < limit and len(articles) < self.min_articles:
                logger.info(f"[CnyesScraper] Only {len(articles)} articles found for {ticker}, trying broader search...")
                driver.get(url)
                time.sleep(3)
                articles = self._scroll_and_collect(driver, wait, limit * 2, source_label)

            logger.info(f"[CnyesScraper] {ticker}: collected {len(articles)} articles")

        except Exception as e:
            logger.error(f"[CnyesScraper] fetch_news error for {ticker}: {e}")
        finally:
            try:
                driver.quit()
            except Exception:
                pass

        return articles[:limit] if len(articles) > limit else articles

    def search_news(self, query: str, limit: int = 1000) -> list:
        """
        Search 鉅亨網 for any keyword (not restricted to a specific stock).
        Uses the search endpoint.

        Args:
            query: Search keyword
            limit: Max articles to fetch

        Returns:
            List of news dicts
        """
        url = f"{CNYES_WWW}/news/search?keyword={query}"
        # Also try the main search
        alt_url = f"https://search.cnyes.com/?keyword={query}"
        source_label = f"鉅亨網-搜尋-{query}"

        logger.info(f"[CnyesScraper] Searching: {url}")

        driver = _make_driver(self.headless)
        articles = []
        try:
            driver.get(url)
            wait = WebDriverWait(driver, 15)
            time.sleep(3)

            articles = self._scroll_and_collect(driver, wait, limit, source_label)

            if len(articles) < self.min_articles:
                driver.get(alt_url)
                time.sleep(3)
                articles += self._scroll_and_collect(driver, wait, limit, source_label)

            # Deduplicate
            seen = set()
            unique = []
            for a in articles:
                key = a.get('連結', '') or a.get('標題', '')
                if key not in seen:
                    seen.add(key)
                    unique.append(a)
            articles = unique

            logger.info(f"[CnyesScraper] search '{query}': {len(articles)} results")

        except Exception as e:
            logger.error(f"[CnyesScraper] search_news error: {e}")
        finally:
            try:
                driver.quit()
            except Exception:
                pass

        return articles[:limit]

    def get_latest_news_url(self, ticker: str, market: str = 'tw') -> str:
        """Return the 鉅亨網 news URL for a given ticker (for direct link in UI)."""
        return f"{CNYES_WWW}/search/news?keyword={ticker}"
