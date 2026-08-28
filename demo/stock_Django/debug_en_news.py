import sys
import os

sys.path.insert(0, os.path.abspath("demo"))

from stock_Django.news_scraper_en import CnbcCliScraper
from stock_Django.agent_news_analyzer import AgentNewsAnalyzer

def debug_en_news():
    scraper = CnbcCliScraper()
    analyzer = AgentNewsAnalyzer()

    print("=== Fetching CNBC news for AAPL ===")
    news_list = scraper.fetch_news("AAPL", limit=5)
    if not news_list:
        print("No news fetched from CNBC.")
        return

    for idx, item in enumerate(news_list):
        title = item.get('標題', '')
        content = item.get('內容', '')
        date = item.get('日期', '')
        
        # 1. 測試本地 FinBERT 評分
        score_res = analyzer.scorer.analyze(title, content, language='en')
        is_neutral = score_res["positive_negative_analysis"] == "中立"
        
        # 2. 長度測試
        clean_content = content.strip() if content else ""
        word_count = len(clean_content.split())
        char_count = len(clean_content)
        has_sufficient_length = word_count >= 30 or char_count >= 80

        should_upgrade = (not is_neutral) and has_sufficient_length

        print(f"\n--- Item #{idx+1} ---")
        print(f"Title: {title}")
        print(f"Date: {date}")
        print(f"Content Preview: {content[:100]}...")
        print(f"FinBERT Label: {score_res['positive_negative_analysis']} (Conf: {score_res['confidence']})")
        print(f"Word Count: {word_count}, Char Count: {char_count}")
        print(f"Is Neutral: {is_neutral}")
        print(f"Has Sufficient Length: {has_sufficient_length}")
        print(f"Should Upgrade to Gemini LLM: {should_upgrade}")

        # 執行完整分析
        full_res = analyzer.analyze_news(item, is_recent=True)
        print(f"Final Reasoning Summary: {full_res['reasoning_summary']}")

if __name__ == "__main__":
    debug_en_news()
