from django.test import TestCase, Client
from django.urls import reverse
from unittest.mock import patch
import time

class ValuationPerformanceTest(TestCase):
    def setUp(self):
        self.client = Client()

    @patch('stock_Django.data_freshness.trigger_refresh_if_stale')
    def test_valuation_view_performance_and_status(self, mock_trigger):
        # 模擬 trigger_refresh_if_stale 使其不執行真實的 Playwright/Selenium 背景爬網
        mock_trigger.return_value = True
        
        start_time = time.time()
        response = self.client.get(reverse('valuation_detail', kwargs={'symbol': '2330.TW'}))
        elapsed_time = time.time() - start_time
        
        # 驗證回應狀態碼為 200 或 302
        self.assertIn(response.status_code, [200, 302])
        
        # 驗證回應時間低於 500ms，以驗證非同步非阻塞的效能提升
        print(f"[Performance] 2330.TW Valuation load time (mocked): {elapsed_time:.3f}s")
        self.assertLess(elapsed_time, 0.5, "Valuation page response took too long (> 500ms)")

    def test_status_api(self):
        # 測試狀態查詢 API 的功能
        response = self.client.get(reverse('valuation_refresh_status_api', kwargs={'symbol': '2330.TW'}))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('status', data)
        self.assertIn('progress', data)
        self.assertIn('message', data)

    @patch('valuation.services.etf_valuation.requests.get')
    @patch('valuation.services.etf_valuation.yf.Ticker')
    def test_etf_chart_api_fetch(self, mock_ticker, mock_get):
        # 模擬 yf.Ticker 避免它發送真實請求
        mock_ticker.side_effect = Exception("Rate limited")
        
        # 模擬 Chart API 的回傳
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "chart": {
                "result": [{
                    "meta": {
                        "regularMarketPrice": 100.0,
                        "longName": "Yuanta Taiwan 50",
                        "currency": "TWD"
                    },
                    "events": {
                        "dividends": {
                            "1700000000": {"amount": 2.5, "date": 1700000000},
                            "1710000000": {"amount": 3.0, "date": 1710000000}
                        }
                    }
                }]
            }
        }
        mock_get.return_value = mock_resp
        
        # 建立假的 loader
        class FakeLoader:
            full_symbol = "0050.TW"
            market = "tw"
            yf_session = None
            def get_market_price(self):
                return 99.0
                
        loader = FakeLoader()
        from valuation.services.etf_valuation import ETFValuationService
        result = ETFValuationService.calculate_etf_valuation("0050.TW", loader)
        
        self.assertEqual(result["name"], "Yuanta Taiwan 50")
        self.assertEqual(result["current_price"], 100.0)
        self.assertEqual(len(result["recent_dividends"]), 2)
        self.assertIsNotNone(result["dividend_yield"])

    @patch('valuation.services.etf_valuation.requests.get')
    @patch('valuation.services.etf_valuation.yf.Ticker')
    @patch('valuation.services.etf_valuation.cache')
    def test_etf_nav_fallback(self, mock_cache, mock_ticker, mock_get):
        # 1. 模擬快取未命中
        mock_cache.get.side_effect = lambda k: None
        
        # 2. 模擬 yfinance 丟出例外 (模擬限流 429)
        mock_ticker.side_effect = Exception("Rate limited")
        
        # 3. 根據 url 進行動態 Mock
        def mock_get_impl(url, *args, **kwargs):
            from unittest.mock import MagicMock
            r = MagicMock()
            if "finance.yahoo.com" in url:
                r.status_code = 200
                r.json.return_value = {
                    "chart": {"result": [{"meta": {"regularMarketPrice": 185.0}}]}
                }
            elif "cnyes.com" in url:
                r.status_code = 200
                r.text = '<html>label">每股淨值</span><span class="value">186.25</span></html>'
            else:
                r.status_code = 404
            return r
            
        mock_get.side_effect = mock_get_impl
        
        class FakeLoader:
            full_symbol = "0050.TW"
            market = "tw"
            yf_session = None
            def get_market_price(self):
                return 185.0
        
        from valuation.services.etf_valuation import ETFValuationService
        result = ETFValuationService.calculate_etf_valuation("0050.TW", FakeLoader())
        
        # 驗證是否藉由 鉅亨網 拿到淨值並計算折溢價
        self.assertEqual(result["nav_price"], 186.25)
        self.assertIsNotNone(result["discount_premium_pct"])
