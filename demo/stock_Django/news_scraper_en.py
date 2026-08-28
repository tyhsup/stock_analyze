"""
news_scraper_en.py — CNBC-CLI English stock news scraper for North American & Global market intelligence.

Replaces Finnhub REST API with local cnbc-cli-pp-cli.exe.
Supports:
- Dual-source strategy: rs search-news (Ticker-specific search) + rs get-news-feed (Top/Tech/Business RSS Feeds)
- Built-in deduplication, XML parsing, HTML unescaping, RFC 822 / ISO date normalization.
- Output interface 100% compatible with CnyesScraper and downstream AgentNewsAnalyzer.
"""

import os
import re
import json
import time
import html
import logging
import subprocess
import email.utils
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# 支援的 CNBC 新聞 RSS 分類 ID 清單
CNBC_FEED_IDS = {
    "us_top": "100003114",     # US Top News and Analysis
    "world": "100727362",      # International / World News
    "business": "10001147",    # Business News
    "tech": "19854910",        # Tech
    "finance": "10000664",     # Finance
    "economy": "20910258",     # Economy
}


def _find_cnbc_cli_bin() -> str:
    """自動探測本地 cnbc-cli-pp-cli.exe 執行檔路徑。"""
    possible_paths = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'cnbc-cli', 'bin', 'cnbc-cli-pp-cli.exe')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cnbc-cli', 'bin', 'cnbc-cli-pp-cli.exe')),
        r"e:\Infinity\mydjango\cnbc-cli\bin\cnbc-cli-pp-cli.exe",
        "cnbc-cli-pp-cli.exe"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return "cnbc-cli-pp-cli.exe"


class CnbcCliScraper:
    """
    CNBC-CLI 英文財經新聞抓取器（雙重來源：Search + RSS Feed）。
    與 CnyesScraper 回傳介面相容，回傳與中文新聞相同結構的字典清單。
    """
    def __init__(self, cli_path: Optional[str] = None):
        self.cli_bin = cli_path or _find_cnbc_cli_bin()

    def _execute_cli(self, args: list, timeout: int = 15) -> Optional[str]:
        """
        安全呼叫 cnbc-cli-pp-cli.exe 並提取 JSON 回傳內部的 XML/字串結果。
        - 嚴禁 shell=True
        - 顯式 UTF-8 編碼與錯誤替換
        - 超時與異常完整捕獲
        """
        cmd = [self.cli_bin] + args + ["--agent"]
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace'
            )
            if res.returncode != 0:
                logger.warning(f"[CnbcCliScraper] CLI 退出碼非零 ({res.returncode}): {res.stderr.strip()}")
                return None

            raw_stdout = res.stdout.strip()
            if not raw_stdout:
                return None

            # 清洗輸出，擷取有效的 JSON 區塊
            first_brace = raw_stdout.find('{')
            last_brace = raw_stdout.rfind('}')
            if first_brace == -1 or last_brace == -1:
                logger.warning("[CnbcCliScraper] CLI 輸出未包含有效 JSON 區塊")
                return None

            json_str = raw_stdout[first_brace:last_brace + 1]
            envelope = json.loads(json_str)
            return envelope.get("results", "")

        except FileNotFoundError:
            logger.error(f"[CnbcCliScraper] 找不到執行檔: {self.cli_bin}")
            return None
        except subprocess.TimeoutExpired:
            logger.error(f"[CnbcCliScraper] 執行逾時 (>{timeout}s): {args}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"[CnbcCliScraper] JSON 解析失敗: {e}")
            return None
        except Exception as e:
            logger.error(f"[CnbcCliScraper] 執行異常: {e}", exc_info=True)
            return None

    def _parse_rss_xml(self, xml_text: str, ticker: str = "", filter_ticker: bool = False) -> List[Dict]:
        """解析 RSS 2.0 Feed XML 字串為標準文章字典清單。"""
        articles = []
        if not xml_text or not isinstance(xml_text, str):
            return articles

        try:
            root = ET.fromstring(xml_text)
            channel = root.find("channel")
            if channel is None:
                return articles

            channel_title = channel.findtext("title", "CNBC News").strip()

            for item in channel.findall("item"):
                title_node = item.find("title")
                desc_node = item.find("description")
                link_node = item.find("link")
                pubdate_node = item.find("pubDate")

                title = html.unescape(title_node.text.strip()) if title_node is not None and title_node.text else ""
                desc = html.unescape(desc_node.text.strip()) if desc_node is not None and desc_node.text else ""
                link = link_node.text.strip() if link_node is not None and link_node.text else ""

                if not title:
                    continue

                # 解析 RFC 822 日期
                parsed_date = datetime.now().strftime('%Y-%m-%d')
                if pubdate_node is not None and pubdate_node.text:
                    try:
                        dt = email.utils.parsedate_to_datetime(pubdate_node.text)
                        parsed_date = dt.strftime('%Y-%m-%d')
                    except Exception:
                        pass

                content = desc if len(desc) > 10 else title

                # 若需過濾特定 ticker
                if filter_ticker and ticker:
                    pattern = rf"\b{re.escape(ticker)}\b"
                    if not (re.search(pattern, title, re.IGNORECASE) or re.search(pattern, content, re.IGNORECASE)):
                        continue

                source_label = f"CNBC-Feed ({channel_title})" if channel_title else "CNBC-Feed"

                articles.append({
                    '標題': title,
                    '日期': parsed_date,
                    '內容': content,
                    '連結': link,
                    '正負分析': '中性',
                    '來源': source_label,
                    '語言': 'en'
                })
        except ET.ParseError as e:
            logger.warning(f"[CnbcCliScraper] RSS XML 解析錯誤: {e}")
        except Exception as e:
            logger.error(f"[CnbcCliScraper] RSS 解析發生未預期異常: {e}")

        return articles

    def _parse_search_xml(self, xml_text: str, ticker: str = "") -> List[Dict]:
        """解析 CNBC Search News 回傳的 XML 為標準文章字典清單。"""
        articles = []
        if not xml_text or not isinstance(xml_text, str):
            return articles

        try:
            root = ET.fromstring(xml_text)
            for res_node in root.iter("results"):
                # 擷取標題
                title = ""
                for tag in ["headline", "{http://cnbc.com/schema/}title", "name", "{http://cnbc.com/schema/}slug"]:
                    n = res_node.find(tag)
                    if n is not None and n.text:
                        title = html.unescape(n.text.strip())
                        break

                # 擷取摘要或內容
                desc = ""
                for tag in ["{http://cnbc.com/schema/}summary", "description", "{http://cnbc.com/schema/}shorterDescription"]:
                    n = res_node.find(tag)
                    if n is not None and n.text:
                        desc = html.unescape(n.text.strip())
                        break

                # 擷取連結
                link = ""
                for tag in ["url", "{http://cnbc.com/schema/}liveURL", "contentUrl"]:
                    n = res_node.find(tag)
                    if n is not None and n.text:
                        link = n.text.strip()
                        break

                # 擷取發布日期
                parsed_date = datetime.now().strftime('%Y-%m-%d')
                for tag in ["datePublished", "{http://cnbc.com/schema/}dateLastPublished", "dateModified"]:
                    n = res_node.find(tag)
                    if n is not None and n.text:
                        try:
                            dt_str = n.text.strip()
                            parsed_date = dt_str[:10]
                            break
                        except Exception:
                            pass

                if not title:
                    continue

                content = desc if len(desc) > 10 else title

                articles.append({
                    '標題': title,
                    '日期': parsed_date,
                    '內容': content,
                    '連結': link,
                    '正負分析': '中性',
                    '來源': f"CNBC-Search ({ticker})" if ticker else "CNBC-Search",
                    '語言': 'en'
                })
        except ET.ParseError as e:
            logger.warning(f"[CnbcCliScraper] Search XML 解析錯誤: {e}")
        except Exception as e:
            logger.error(f"[CnbcCliScraper] Search 解析發生未預期異常: {e}")

        return articles

    def fetch_news(self, ticker: str, limit: int = 20, days_back: int = 30) -> list:
        """
        獲取美股英文新聞（雙重來源整合：關鍵字搜尋 + 重點 RSS Feed 聚合與去重）。

        Args:
            ticker: 美股代碼或關鍵字 (例如 'AAPL', 'NVDA', 'TSLA', 'Fed')
            limit: 限制回傳的新聞筆數
            days_back: 搜尋幾天前的新聞（相容性參數）

        Returns:
            新聞字典陣列，包含：標題, 日期, 內容, 連結, 正負分析, 來源, 語言
        """
        start_time = time.time()
        ticker = str(ticker).strip().upper().replace('.TW', '').replace('.TWO', '')

        # 安全防護：ticker 正則白名單驗證 (防禦命令注入)
        if not re.match(r"^[A-Za-z0-9\-\.\s]{1,30}$", ticker):
            logger.error(f"[CnbcCliScraper] 包含非法字元的查詢關鍵字: {ticker}")
            return []

        logger.info(f"[CnbcCliScraper] 開始從 CNBC-CLI 抓取 {ticker} 英文新聞 (雙重來源模式), limit: {limit}")

        collected: List[Dict] = []
        seen_titles = set()

        def add_articles(items: List[Dict]):
            for it in items:
                t_clean = re.sub(r'\s+', ' ', it.get('標題', '')).strip().lower()
                if t_clean and t_clean not in seen_titles:
                    seen_titles.add(t_clean)
                    collected.append(it)

        # 來源 1: 依關鍵字搜尋特定新聞
        search_xml = self._execute_cli(["rs", "search-news", "--keywords", ticker])
        if search_xml:
            search_items = self._parse_search_xml(search_xml, ticker=ticker)
            add_articles(search_items)

        # 來源 2: 查詢精選分類 RSS Feeds (US Top, Tech, Business) 搜尋特定關聯
        feed_categories = ["tech", "us_top", "business", "world"]
        for cat in feed_categories:
            if len(collected) >= limit:
                break
            feed_id = CNBC_FEED_IDS.get(cat)
            if not feed_id:
                continue

            feed_xml = self._execute_cli(["rs", "get-news-feed", "--id", feed_id])
            if feed_xml:
                # 優先抓取內容有提及 ticker 的新聞
                feed_items = self._parse_rss_xml(feed_xml, ticker=ticker, filter_ticker=True)
                add_articles(feed_items)

        # 來源 3 補底：若精確提及 ticker 的新聞數量不足，拉取 US Top / Tech 最新熱門新聞補充
        if len(collected) < limit:
            for cat in ["us_top", "tech"]:
                if len(collected) >= limit:
                    break
                feed_id = CNBC_FEED_IDS.get(cat)
                if feed_id:
                    feed_xml = self._execute_cli(["rs", "get-news-feed", "--id", feed_id])
                    if feed_xml:
                        general_feed_items = self._parse_rss_xml(feed_xml, ticker=ticker, filter_ticker=False)
                        add_articles(general_feed_items)

        final_articles = collected[:limit]
        elapsed = time.time() - start_time
        logger.info(f"[CnbcCliScraper] 成功獲取 {len(final_articles)} 則 CNBC 英文新聞，總耗時: {elapsed:.2f} 秒")
        return final_articles


# 保留類別別名以維持舊程式碼相容性
FinnhubScraper = CnbcCliScraper
