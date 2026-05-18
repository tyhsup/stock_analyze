import logging
import sys
import os

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 將當前目錄加入 path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from stock_Django.news_scraper_cnyes import CnyesScraper
except Exception as e:
    logging.error(f"無法載入 CnyesScraper: {e}")
    sys.exit(1)

def run_tests():
    logging.info("========================================")
    logging.info("開始測試重構後的 CnyesScraper")
    logging.info("========================================")
    
    scraper = CnyesScraper()
    
    # 測試 1: 抓取台股新聞 (2330)
    ticker_tw = "2330"
    logging.info(f"測試 1: 抓取台股 {ticker_tw} 新聞...")
    results_tw = scraper.fetch_news(ticker_tw, market="tw", limit=3)
    
    logging.info(f"測試 1 結果: 成功獲取 {len(results_tw)} 篇新聞")
    for idx, news in enumerate(results_tw):
        logging.info(f"新聞 {idx+1}:")
        logging.info(f"  - 標題: {news.get('標題')}")
        logging.info(f"  - 日期: {news.get('日期')}")
        logging.info(f"  - 連結: {news.get('連結')}")
        logging.info(f"  - 正負分析: {news.get('正負分析')}")
        logging.info(f"  - 來源: {news.get('來源')}")
        logging.info(f"  - 內容長度: {len(news.get('內容', ''))} 字")
        if news.get('內容'):
            logging.info(f"  - 內容摘要: {news.get('內容')[:100]}...")
            
    # 測試 2: 抓取美股新聞 (AAPL)
    ticker_us = "AAPL"
    logging.info(f"測試 2: 抓取美股 {ticker_us} 新聞...")
    results_us = scraper.fetch_news(ticker_us, market="us", limit=3)
    
    logging.info(f"測試 2 結果: 成功獲取 {len(results_us)} 篇新聞")
    for idx, news in enumerate(results_us):
        logging.info(f"新聞 {idx+1}:")
        logging.info(f"  - 標題: {news.get('標題')}")
        logging.info(f"  - 日期: {news.get('日期')}")
        logging.info(f"  - 正負分析: {news.get('正負分析')}")
        logging.info(f"  - 來源: {news.get('來源')}")
        logging.info(f"  - 內容長度: {len(news.get('內容', ''))} 字")
        
    # 測試 3: 搜尋新聞
    query = "半導體"
    logging.info(f"測試 3: 搜尋關鍵字 '{query}'...")
    results_search = scraper.search_news(query, limit=3)
    
    logging.info(f"測試 3 結果: 成功獲取 {len(results_search)} 篇新聞")
    for idx, news in enumerate(results_search):
        logging.info(f"新聞 {idx+1}:")
        logging.info(f"  - 標題: {news.get('標題')}")
        logging.info(f"  - 正負分析: {news.get('正負分析')}")
        logging.info(f"  - 內容長度: {len(news.get('內容', ''))} 字")

if __name__ == "__main__":
    run_tests()
