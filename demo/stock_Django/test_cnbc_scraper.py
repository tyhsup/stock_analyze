import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# 將 demo 加入路徑
sys.path.insert(0, os.path.abspath("demo"))

from stock_Django.news_scraper_en import CnbcCliScraper


class TestCnbcCliScraper(unittest.TestCase):
    """測試 CNBC-CLI 英文新聞抓取器"""

    def setUp(self):
        self.scraper = CnbcCliScraper()

    def test_tc01_happy_path(self):
        """TC-01: 正常路徑驗證 — 查詢美股代碼取得格式正確的新聞"""
        results = self.scraper.fetch_news("AAPL", limit=5)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0, "應至少取得 1 則 AAPL 英文新聞")

        for item in results:
            self.assertIn('標題', item)
            self.assertIn('日期', item)
            self.assertIn('內容', item)
            self.assertIn('連結', item)
            self.assertIn('正負分析', item)
            self.assertIn('來源', item)
            self.assertIn('語言', item)
            self.assertEqual(item['語言'], 'en')
            # 檢查日期格式 YYYY-MM-DD
            self.assertRegex(item['日期'], r'^\d{4}-\d{2}-\d{2}$')
            self.assertTrue(len(item['標題']) > 0)
            self.assertTrue(len(item['內容']) > 0)

    def test_tc02_command_injection_defense(self):
        """TC-02: 安全阻斷測試 — 包含注入字元的 ticker 應被安全拒絕"""
        malicious_inputs = [
            "AAPL; rm -rf /",
            "AAPL & dir",
            "AAPL | calc",
            "`whoami`",
            "AAPL'; DROP TABLE news;--"
        ]
        for bad_ticker in malicious_inputs:
            results = self.scraper.fetch_news(bad_ticker, limit=5)
            self.assertEqual(results, [], f"惡意輸入 '{bad_ticker}' 應被攔截並回傳空列表")

    def test_tc03_timeout_handling(self):
        """TC-03: 超時容錯測試 — 模擬 CLI 逾時應優雅回傳空列表而不崩潰"""
        with patch("subprocess.run") as mock_run:
            import subprocess
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["test"], timeout=1)
            results = self.scraper.fetch_news("NVDA", limit=5)
            self.assertEqual(results, [], "超時應優雅降級為空列表")

    def test_tc04_corrupted_data_robustness(self):
        """TC-04: 格式毀損容錯測試 — CLI 回傳損毀資料時不引發未捕獲異常"""
        # 測試損毀的 JSON
        with patch.object(self.scraper, "_execute_cli", return_value=None):
            results = self.scraper.fetch_news("TSLA", limit=5)
            self.assertEqual(results, [])

        # 測試損毀的 XML
        corrupted_xml = "<invalid><unclosed>xml"
        res_rss = self.scraper._parse_rss_xml(corrupted_xml, "TSLA")
        self.assertEqual(res_rss, [])
        res_search = self.scraper._parse_search_xml(corrupted_xml, "TSLA")
        self.assertEqual(res_search, [])


if __name__ == "__main__":
    unittest.main()
