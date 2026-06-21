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
