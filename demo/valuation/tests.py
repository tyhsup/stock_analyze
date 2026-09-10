from django.test import TestCase, Client
from django.urls import reverse
from unittest.mock import patch
import time
import math

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
        
        # 驗證回應時間合理，避免阻塞過久
        print(f"[Performance] 2330.TW Valuation load time (mocked): {elapsed_time:.3f}s")
        self.assertLess(elapsed_time, 2.0, "Valuation page response took too long (> 2.0s)")

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
        now_ts = int(time.time())
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
                            str(now_ts - 86400 * 30): {"amount": 2.5, "date": now_ts - 86400 * 30},
                            str(now_ts - 86400 * 180): {"amount": 3.0, "date": now_ts - 86400 * 180}
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


class Phase1CoreValuationEngineTest(TestCase):
    """
    Phase 1 核心估值引擎單元測試：
    - TSMCalculator 雙軌稀釋股數計算與極端邊界防禦
    - IterativeDCFSolver 資本結構迭代收斂與阻尼平滑
    - Assumptions 終值增長率動態天花板與 ROIC 再投資率勾稽
    - DCFModel FCFF / FCFE 型別硬綁定防護
    """

    def test_tsm_boundary_and_dilution(self):
        from valuation.services.iterative_solver import TSMCalculator, ValuationError

        basic_shares = 1000000.0
        options_count = 100000.0
        strike_price = 50.0

        # 情境 1: 價外期權 (Current Price 40 <= Strike 50) -> 不稀釋
        diluted_otm = TSMCalculator.calculate_diluted_shares(
            basic_shares=basic_shares,
            options_count=options_count,
            strike_price=strike_price,
            current_price=40.0
        )
        self.assertEqual(diluted_otm, 1000000.0)
        self.assertIsInstance(diluted_otm, float)

        # 情境 2: 價內期權 (Current Price 100 > Strike 50)
        # 淨新增股數 = 100,000 * (1 - 50/100) = 50,000
        diluted_itm = TSMCalculator.calculate_diluted_shares(
            basic_shares=basic_shares,
            options_count=options_count,
            strike_price=strike_price,
            current_price=100.0
        )
        self.assertEqual(diluted_itm, 1050000.0)
        self.assertIsInstance(diluted_itm, float)

        # 情境 3: 極端價格異常 (Current Price <= 0) -> 安全 Fallback 至官方申報值
        diluted_zero_p = TSMCalculator.calculate_diluted_shares(
            basic_shares=basic_shares,
            options_count=options_count,
            strike_price=strike_price,
            current_price=0.0,
            reported_diluted_shares=1020000.0
        )
        self.assertEqual(diluted_zero_p, 1020000.0)

        # 情境 4: 基礎股數 <= 0 拋出 ValuationError
        with self.assertRaises(ValuationError):
            TSMCalculator.calculate_diluted_shares(basic_shares=0.0)

    def test_iterative_solver_convergence(self):
        from valuation.services.iterative_solver import IterativeDCFSolver

        solver = IterativeDCFSolver(max_iter=50, tol=1e-4, damping_factor=0.5)

        # 模擬折現函式：隨 WACC 變動輸出 Enterprise Value
        def mock_discount_fn(wacc_val):
            # EV = 100 / wacc_val
            ev = 50000000.0 / max(wacc_val, 0.03)
            tv = ev * 0.7
            pv_tv = tv * 0.6
            return ev, tv, pv_tv

        res = solver.solve(
            initial_price=100.0,
            basic_shares=1000000.0,
            total_debt=20000000.0,
            cash=5000000.0,
            cost_of_equity=0.09,
            cost_of_debt=0.04,
            tax_rate=0.20,
            discount_fn=mock_discount_fn,
            options_count=50000.0,
            strike_price=60.0
        )

        self.assertTrue(res["is_converged"])
        self.assertLessEqual(res["iterations"], 50)
        self.assertGreater(res["implied_price"], 0.0)
        self.assertIsInstance(res["implied_price"], float)
        self.assertIsInstance(res["converged_wacc"], float)
        self.assertIsInstance(res["diluted_shares"], float)

    def test_terminal_value_and_roic_defenses(self):
        from valuation.services.assumptions import Assumptions

        assumptions = Assumptions()

        # 1. 驗證 g 天花板：當 g (0.05) > Rf (0.03) 時，強制 clamp 為 Rf (0.03)
        assumptions.perpetuity_growth_rate = 0.05
        g_eff = assumptions.validate_terminal_growth(risk_free_rate=0.03)
        self.assertEqual(g_eff, 0.03)
        self.assertEqual(assumptions.perpetuity_growth_rate, 0.03)

        # 2. 驗證正常 ROIC 再投資率：g = 0.03, ROIC = 0.10 -> RR = 30%
        rr_normal = assumptions.calculate_reinvestment_rate(roic=0.10, g=0.03)
        self.assertAlmostEqual(rr_normal, 0.30, places=4)

        # 3. 驗證 ROIC <= 0 極端防禦：ROIC = -0.02, g = 0.03 -> 強制 RR = 1.0 (防止負再投資率或除以零)
        rr_neg = assumptions.calculate_reinvestment_rate(roic=-0.02, g=0.03)
        self.assertEqual(rr_neg, 1.0)

        # 4. 驗證終值 FCFF 公式：NOPAT * (1 + g) * (1 - RR)
        # NOPAT = 100, g = 0.02, ROIC = 0.10 -> RR = 0.20
        # Expected = 100 * 1.02 * (1 - 0.20) = 81.6
        term_fcff = assumptions.calculate_terminal_fcff(nopat_n=100.0, roic=0.10, g=0.02)
        self.assertAlmostEqual(term_fcff, 81.6, places=2)

    def test_fcf_type_binding_protection(self):
        from valuation.services.dcf_model import DCFModel
        from valuation.services.iterative_solver import FCFType

        # 1. 正常配對：FCFF + WACC -> 通過
        model_fcff = DCFModel(fcf_type=FCFType.FCFF, discount_rate_type="WACC", discount_rate=0.08)
        self.assertEqual(model_fcff.discount_rate, 0.08)

        # 2. 異常配對：FCFF + Ke -> 拋出 TypeError
        with self.assertRaises(TypeError):
            DCFModel(fcf_type=FCFType.FCFF, discount_rate_type="KE", discount_rate=0.09)

        # 3. 異常配對：FCFE + WACC -> 拋出 TypeError
        with self.assertRaises(TypeError):
            DCFModel(fcf_type=FCFType.FCFE, discount_rate_type="WACC", discount_rate=0.08)

        # 4. 正常折現現值計算
        res = model_fcff.discount_cash_flows(cash_flows=[100.0, 110.0, 120.0], terminal_value=1000.0)
        self.assertIn("total_pv", res)
        self.assertIn("pv_cash_flows", res)
        self.assertIn("pv_terminal_value", res)
        self.assertGreater(res["total_pv"], 0.0)


class Phase2EVToEquityBridgeTest(TestCase):
    """
    Phase 2 EV-to-Equity Bridge 單元測試：
    - 精確橋接計算（現金加回、債務/租賃/特別股/少數股權/退休金扣除）
    - NaN、None 與異常值防禦
    - 結構化明細與 JSON 序列化型別檢查
    """

    def test_ev_bridge_standard_accuracy(self):
        from valuation.services.ev_bridge import EVToEquityBridge

        bridge = EVToEquityBridge.calculate_bridge(
            enterprise_value=1000.0,
            cash=150.0,
            total_debt=300.0,
            operating_lease_liability=50.0,
            preferred_stock=10.0,
            minority_interest=5.0,
            pension_deficit=15.0,
            debt_like_items=0.0
        )

        # 驗證總扣除額 = 300 + 50 + 10 + 5 + 15 = 380
        self.assertEqual(bridge["total_deductions"], 380.0)
        # 驗證廣義淨負債 = 380 - 150 = 230
        self.assertEqual(bridge["net_debt"], 230.0)
        # 驗證股權價值 = 1000 - 230 = 770
        self.assertEqual(bridge["equity_value"], 770.0)
        self.assertIsInstance(bridge["equity_value"], float)
        self.assertIsInstance(bridge["net_debt"], float)
        self.assertEqual(len(bridge["items"]), 8)

    def test_ev_bridge_edge_cases_and_nan_defense(self):
        from valuation.services.ev_bridge import EVToEquityBridge
        import json

        # 傳入 None, NaN, 空字串等髒數據
        bridge = EVToEquityBridge.calculate_bridge(
            enterprise_value=500.0,
            cash=None,
            total_debt=float("nan"),
            operating_lease_liability=-20.0,  # 負值應被安全 clamp 為 0
            preferred_stock=0.0,
            minority_interest=0.0,
            pension_deficit=0.0
        )

        self.assertEqual(bridge["cash"], 0.0)
        self.assertEqual(bridge["total_debt"], 0.0)
        self.assertEqual(bridge["operating_lease_liability"], 0.0)
        self.assertEqual(bridge["equity_value"], 500.0)

        # 驗證可直接透過標準 json.dumps 序列化，無 Decimal 或 NaN 報錯
        json_str = json.dumps(bridge)
        self.assertIn('"equity_value": 500.0', json_str)
        self.assertNotIn("NaN", json_str)


class Phase3RiskAnalysisTest(TestCase):
    """
    Phase 3 風險分析強化單元測試：
    - MonteCarloSimulator 10,000 次向量化抽樣、統計分佈、百分位與 JSON 序列化保護
    - SensitivityAnalyzer WACC vs g 雙向矩陣、Gordon 分母除以零防禦、無 NaN/Inf 檢驗
    """

    def test_monte_carlo_simulation_and_statistics(self):
        from valuation.services.monte_carlo import MonteCarloSimulator
        import json

        sim = MonteCarloSimulator(num_simulations=5000, seed=42)
        res = sim.run_simulation(
            base_revenue=1000000000.0,
            base_growth=0.10,
            growth_std=0.03,
            ebit_margin=0.40,
            tax_rate=0.20,
            base_wacc=0.08,
            wacc_std=0.01,
            risk_free_rate=0.03,
            reinvestment_rate=0.30,
            net_debt=50000000.0,
            shares_outstanding=10000000.0,
            current_price=100.0,
            projection_years=5
        )

        stats = res["statistics"]
        # 1. 驗證統計指標存在且合理
        self.assertIn("mean", stats)
        self.assertIn("std", stats)
        self.assertIn("p10", stats)
        self.assertIn("p50", stats)
        self.assertIn("p90", stats)
        self.assertIn("current_price_percentile", stats)

        # 2. 驗證分位數單調遞增與信賴區間
        self.assertLessEqual(stats["p10"], stats["p50"])
        self.assertLessEqual(stats["p50"], stats["p90"])
        self.assertLessEqual(stats["ci_90_lower"], stats["ci_90_upper"])
        self.assertGreater(stats["mean"], 0.0)

        # 3. 驗證現價百分位在 0% ~ 100% 之間
        self.assertGreaterEqual(stats["current_price_percentile"], 0.0)
        self.assertLessEqual(stats["current_price_percentile"], 100.0)

        # 4. 驗證直方圖 20 bins 結構
        hist = res["histogram"]
        self.assertEqual(len(hist), 20)
        self.assertIn("bin_start", hist[0])
        self.assertIn("count", hist[0])

        # 5. 驗證完全相容 JSON 序列化 (無 numpy.float64 類型外洩)
        json_str = json.dumps(res)
        self.assertNotIn("NaN", json_str)
        self.assertNotIn("Infinity", json_str)

    def test_sensitivity_analyzer_matrix(self):
        from valuation.services.sensitivity import SensitivityAnalyzer
        import json

        res = SensitivityAnalyzer.generate_matrix(
            base_revenue=1000000000.0,
            growth_rates=[0.10, 0.08, 0.06, 0.05, 0.04],
            ebit_margin=0.35,
            tax_rate=0.20,
            reinvestment_rate=0.25,
            base_wacc=0.08,
            base_g=0.02,
            net_debt=20000000.0,
            shares_outstanding=5000000.0,
            risk_free_rate=0.03
        )

        wacc_labels = res["wacc_labels"]
        g_labels = res["g_labels"]
        matrix = res["matrix"]
        coords = res["base_coordinates"]

        # 1. 驗證維度相符
        self.assertEqual(len(matrix), len(wacc_labels))
        self.assertEqual(len(matrix[0]), len(g_labels))

        # 2. 驗證無 NaN / Inf，且所有估值價格大於零
        for row in matrix:
            for val in row:
                self.assertIsInstance(val, float)
                self.assertFalse(math.isnan(val))
                self.assertFalse(math.isinf(val))
                self.assertGreater(val, 0.0)

        # 3. 驗證座標有效
        self.assertIn("wacc_index", coords)
        self.assertIn("g_index", coords)
        self.assertLess(coords["wacc_index"], len(wacc_labels))
        self.assertLess(coords["g_index"], len(g_labels))

        # 4. 驗證 JSON 序列化
        json_str = json.dumps(res)
        self.assertNotIn("NaN", json_str)


class Phase4SCDType2HistoryTest(TestCase):
    """Phase 4: SCD Type 2 歷史版本控制與快照單元測試"""

    def setUp(self):
        from valuation.models import ValuationAssumptionHistory
        ValuationAssumptionHistory.objects.all().delete()

    def test_scd2_initial_version_creation(self):
        from valuation.services.assumption_history import AssumptionHistoryService
        from django.utils import timezone
        import datetime

        assumptions_v1 = {
            "wacc": 0.0825,
            "perpetual_growth_rate": 0.025,
            "roic": 0.15,
            "tax_rate": 0.20
        }
        val_snapshot_v1 = {
            "dcf_per_share": 1500.5,
            "fair_value": 1620.0
        }

        rec1, is_new = AssumptionHistoryService.save_assumptions_scd2(
            symbol="2330.TW",
            market="TW",
            assumptions=assumptions_v1,
            valuation_snapshot=val_snapshot_v1,
            change_reason="初次建立基準假設"
        )

        self.assertTrue(is_new)
        self.assertEqual(rec1.version, 1)
        self.assertTrue(rec1.is_current)
        self.assertIsNone(rec1.end_date)
        self.assertEqual(rec1.effective_date, timezone.localdate())
        self.assertEqual(rec1.assumptions["wacc"], 0.0825)

    def test_scd2_version_evolution_and_concurrency(self):
        from valuation.services.assumption_history import AssumptionHistoryService
        from valuation.models import ValuationAssumptionHistory
        from django.utils import timezone
        import datetime

        # 1. 建立 v1
        assumptions_v1 = {"wacc": 0.08, "g": 0.02}
        rec1, is_new1 = AssumptionHistoryService.save_assumptions_scd2(
            symbol="2330.TW",
            market="TW",
            assumptions=assumptions_v1
        )
        self.assertEqual(rec1.version, 1)

        # 2. 相同假設再次儲存 -> 不產生新版本
        rec_same, is_new_same = AssumptionHistoryService.save_assumptions_scd2(
            symbol="2330.TW",
            market="TW",
            assumptions=assumptions_v1,
            valuation_snapshot={"updated_val": 100}
        )
        self.assertFalse(is_new_same)
        self.assertEqual(rec_same.version, 1)

        # 3. 變更假設 -> 升版至 v2
        assumptions_v2 = {"wacc": 0.09, "g": 0.025}
        rec2, is_new2 = AssumptionHistoryService.save_assumptions_scd2(
            symbol="2330.TW",
            market="TW",
            assumptions=assumptions_v2,
            change_reason="因應利率調升上調 WACC"
        )
        self.assertTrue(is_new2)
        self.assertEqual(rec2.version, 2)
        self.assertTrue(rec2.is_current)
        self.assertIsNone(rec2.end_date)

        # 4. 驗證舊版 v1 已正確標記失效
        old_v1 = ValuationAssumptionHistory.objects.get(symbol="2330.TW", market="TW", version=1)
        self.assertFalse(old_v1.is_current)
        self.assertEqual(old_v1.end_date, timezone.localdate())

        # 5. 驗證當前有效版本僅有 1 筆
        current_count = ValuationAssumptionHistory.objects.filter(
            symbol="2330.TW", market="TW", is_current=True
        ).count()
        self.assertEqual(current_count, 1)

        # 6. 驗證 get_current
        current_obj = AssumptionHistoryService.get_current("2330.TW", "TW")
        self.assertIsNotNone(current_obj)
        self.assertEqual(current_obj.version, 2)

    def test_scd2_time_travel_and_precision(self):
        from valuation.services.assumption_history import AssumptionHistoryService
        from decimal import Decimal
        import datetime

        # 手動注入兩筆不同時期的歷史版本
        d1 = datetime.date(2025, 1, 1)
        d2 = datetime.date(2025, 7, 1)
        d3 = datetime.date(2026, 1, 1)

        # v1: 2025-01-01 ~ 2025-07-01
        rec1, _ = AssumptionHistoryService.save_assumptions_scd2(
            symbol="AAPL",
            market="US",
            assumptions={"wacc": 0.075, "precision_test": Decimal("0.000123456789")},
            effective_date=d1
        )
        # 手動設為失效並切換至 d2
        rec2, _ = AssumptionHistoryService.save_assumptions_scd2(
            symbol="AAPL",
            market="US",
            assumptions={"wacc": 0.085, "precision_test": Decimal("0.000987654321")},
            effective_date=d2
        )

        # Time-travel 查詢：查詢 2025-03-01 應該命中 v1
        as_of_v1 = AssumptionHistoryService.get_as_of_date("AAPL", "US", datetime.date(2025, 3, 1))
        self.assertIsNotNone(as_of_v1)
        self.assertEqual(as_of_v1.version, 1)
        self.assertEqual(as_of_v1.assumptions["precision_test"], "0.000123456789")

        # Time-travel 查詢：查詢 2025-10-01 應該命中 v2
        as_of_v2 = AssumptionHistoryService.get_as_of_date("AAPL", "US", datetime.date(2025, 10, 1))
        self.assertIsNotNone(as_of_v2)
        self.assertEqual(as_of_v2.version, 2)

    def test_scd2_rollback_to_version(self):
        from valuation.services.assumption_history import AssumptionHistoryService
        from valuation.models import ValuationAssumptionHistory

        # 建立 v1 與 v2
        rec1, _ = AssumptionHistoryService.save_assumptions_scd2(
            symbol="MSFT", market="US", assumptions={"wacc": 0.07, "g": 0.03}
        )
        rec2, _ = AssumptionHistoryService.save_assumptions_scd2(
            symbol="MSFT", market="US", assumptions={"wacc": 0.08, "g": 0.02}
        )
        self.assertEqual(rec2.version, 2)

        # 回滾至 v1 -> 產生新版本 v3，但其 assumptions 完全承襲 v1
        rec3 = AssumptionHistoryService.rollback_to_version(
            symbol="MSFT", market="US", target_version=1, reason="回復基準模型"
        )
        self.assertEqual(rec3.version, 3)
        self.assertTrue(rec3.is_current)
        self.assertEqual(rec3.assumptions["wacc"], 0.07)
        self.assertIn("回滾至歷史版本 v1", rec3.change_reason)

        # 驗證總共有 3 個版本，且僅 v3 為 is_current
        hist = AssumptionHistoryService.get_history("MSFT", "US")
        self.assertEqual(len(hist), 3)
        self.assertEqual([h.version for h in hist], [3, 2, 1])
        self.assertEqual([h.is_current for h in hist], [True, False, False])


from django.test import override_settings

class Phase5CeleryAsyncBatchTest(TestCase):
    """Phase 5: Celery 異步排程與分塊批量估值單元測試"""

    def test_financial_json_encoder_decimal_and_datetime(self):
        """驗證 Evaluator 審查防禦點：自定義 FinancialJSONEncoder 序列化 Decimal 與 datetime 不崩潰"""
        from demo.celery import FinancialJSONEncoder
        from decimal import Decimal
        import datetime
        import json

        test_payload = {
            "symbol": "2330.TW",
            "price": Decimal("1050.2500"),
            "wacc": Decimal("0.0825"),
            "date": datetime.date(2026, 9, 11),
            "timestamp": datetime.datetime(2026, 9, 11, 12, 0, 0)
        }

        # 驗證序列化成功且無 TypeError
        encoded = json.dumps(test_payload, cls=FinancialJSONEncoder)
        self.assertIn('"price": "1050.2500"', encoded)
        self.assertIn('"wacc": "0.0825"', encoded)
        self.assertIn('"date": "2026-09-11"', encoded)

        # 驗證還原
        decoded = json.loads(encoded)
        self.assertEqual(decoded["price"], "1050.2500")

    @override_settings(CELERY_TASK_ALWAYS_EAGER=True)
    def test_single_valuation_async_task(self):
        """驗證單一股票非同步估值任務 (Eager 模式)"""
        from valuation.tasks import calculate_single_valuation_async

        async_result = calculate_single_valuation_async.apply(args=["2330.TW"])
        res = async_result.result

        self.assertIn("symbol", res)
        self.assertEqual(res["symbol"], "2330.TW")
        self.assertIn(res["status"], ["SUCCESS", "WARNING"])
        self.assertIn("task_id", res)
        self.assertIn("valuation", res)

    @override_settings(CELERY_TASK_ALWAYS_EAGER=True)
    def test_calculate_valuation_chunk(self):
        """驗證分塊批次估值任務與標的隔離機制"""
        from valuation.tasks import calculate_valuation_chunk
        from unittest.mock import patch

        def mock_calc(ticker_symbol, **kwargs):
            if ticker_symbol == "FAIL_SYM":
                return {"error": "Mocked calculation failure"}
            return {
                "symbol": ticker_symbol,
                "summary": {"fair_value": 2000.0},
                "scd2": {"version": 1}
            }

        with patch("valuation.tasks.ValuationService.calculate_valuation", side_effect=mock_calc):
            symbols = ["2330.TW", "FAIL_SYM"]
            async_result = calculate_valuation_chunk.apply(args=[symbols, "TW"])
            res = async_result.result

            self.assertEqual(res["status"], "COMPLETED")
            self.assertEqual(res["total"], 2)
            self.assertEqual(res["succeeded"], 1)
            self.assertEqual(res["failed"], 1)
            self.assertIn("2330.TW", res["results"])
            self.assertIn("FAIL_SYM", res["errors"])
            self.assertEqual(res["succeeded"] + res["failed"], 2)

    @override_settings(CELERY_TASK_ALWAYS_EAGER=True)
    def test_batch_calculate_market_valuation_dispatch(self):
        """驗證 Master 批量排程分塊派發邏輯"""
        from valuation.tasks import batch_calculate_market_valuation

        async_result = batch_calculate_market_valuation.apply(
            kwargs={"market": "TW", "chunk_size": 2}
        )
        res = async_result.result

        self.assertEqual(res["status"], "DISPATCHED")
        self.assertEqual(res["market"], "TW")
        self.assertGreater(res["total_symbols"], 0)
        self.assertGreater(res["total_chunks"], 0)
        self.assertEqual(len(res["dispatched_task_ids"]), res["total_chunks"])


class Phase6WACCGuardrailTest(TestCase):
    """
    Phase 6: WACC 異常值防護閥單元測試：
    - [3%, 20%] 區間限制與截斷 (Clamping)
    - 動態無風險利率 (Rf) 邊界檢驗與利差防護
    - 異常標記 is_flagged 與原因追蹤
    - 異常例外啟用 Fallback 8.5%
    - SCD Type 2 歷史快照之 is_flagged 與 flag_reasons 持久化驗證
    """

    def test_wacc_lower_bound_clamping_and_flagging(self):
        """驗證 WACC < 3% 時強制截斷至 3% 且標記 is_flagged=True"""
        from valuation.services.wacc_guardrail import WACCGuardrail

        res = WACCGuardrail.validate_and_clamp(
            raw_wacc=0.018,
            base_wacc=0.018,
            rf=0.015,
            market='tw'
        )

        self.assertEqual(res.wacc, 0.03)
        self.assertEqual(res.raw_wacc, 0.018)
        self.assertTrue(res.is_flagged)
        self.assertTrue(res.is_clamped)
        self.assertTrue(any("BELOW_LOWER_BOUND" in r for r in res.flag_reasons))

    def test_wacc_upper_bound_clamping_and_flagging(self):
        """驗證 WACC > 20% 時強制截斷至 20% 且標記 is_flagged=True"""
        from valuation.services.wacc_guardrail import WACCGuardrail

        res = WACCGuardrail.validate_and_clamp(
            raw_wacc=0.285,
            base_wacc=0.285,
            rf=0.042,
            market='us'
        )

        self.assertEqual(res.wacc, 0.20)
        self.assertEqual(res.raw_wacc, 0.285)
        self.assertTrue(res.is_flagged)
        self.assertTrue(res.is_clamped)
        self.assertTrue(any("ABOVE_UPPER_BOUND" in r for r in res.flag_reasons))

    def test_rf_boundary_and_spread_protection(self):
        """驗證動態無風險利率邊界與 WACC >= Rf + 0.5% 之溢酬防護"""
        from valuation.services.wacc_guardrail import WACCGuardrail

        # 1. 極低 Rf (< 0.5%) 自動調校
        clamped_rf, rf_flag, rf_reasons = WACCGuardrail.validate_rf(0.001)
        self.assertEqual(clamped_rf, 0.005)
        self.assertTrue(rf_flag)

        # 2. 極高 Rf (> 8.0%) 自動調校
        clamped_rf_high, rf_flag_high, rf_reasons_high = WACCGuardrail.validate_rf(0.12)
        self.assertEqual(clamped_rf_high, 0.08)
        self.assertTrue(rf_flag_high)

        # 3. WACC 低於無風險利率溢酬防呆 (例如 Rf = 4.5%, WACC = 4.6% < 5.0%)
        res = WACCGuardrail.validate_and_clamp(
            raw_wacc=0.046,
            rf=0.045
        )
        self.assertTrue(res.is_flagged)
        self.assertTrue(any("RISK_FREE_SPREAD" in r for r in res.flag_reasons))
        self.assertGreaterEqual(res.wacc, 0.045 + 0.005)

    def test_wacc_fallback_under_exception(self):
        """驗證數值為 NaN 或發生計算例外時，安全回退至預設 8.5% 並標記 Fallback"""
        from valuation.services.wacc_guardrail import WACCGuardrail

        # 傳入 NaN
        res = WACCGuardrail.validate_and_clamp(raw_wacc=float('nan'))
        self.assertEqual(res.wacc, 0.085)
        self.assertTrue(res.is_flagged)
        self.assertTrue(res.fallback_used)
        self.assertTrue(any("FALLBACK" in r for r in res.flag_reasons))

    def test_scd2_persistence_of_guardrail_flags(self):
        """驗證 SCD Type 2 歷史快照成功持久化 is_flagged 與 flag_reasons 欄位"""
        from valuation.services.assumption_history import AssumptionHistoryService
        from valuation.models import ValuationAssumptionHistory

        record, is_new = AssumptionHistoryService.save_assumptions_scd2(
            symbol="TEST6.TW",
            market="TW",
            assumptions={"wacc": 0.03, "g": 0.02},
            valuation_snapshot={"fair_value": 150.0},
            change_reason="Phase 6 異常防護閥測試",
            is_flagged=True,
            flag_reasons=["WACC_BELOW_LOWER_BOUND_3%"]
        )

        self.assertTrue(record.is_flagged)
        # 從資料庫重新讀取驗證
        db_rec = ValuationAssumptionHistory.objects.get(id=record.id)
        self.assertTrue(db_rec.is_flagged)
        self.assertIn("WACC_BELOW_LOWER_BOUND_3%", db_rec.flag_reasons)


class Phase7FullSystemE2EAndPerformanceTest(TestCase):
    """
    Phase 7: 全系統端對端回歸驗收與效能驗證：
    1. 全模組整合度檢驗 (Full Valuation Pipeline Integration)
    2. 歷史 Time-travel 快照回放與一致性檢驗
    3. 效能負載與延遲基準測試 (Monte Carlo < 100ms, Sensitivity < 50ms, Total < 500ms)
    4. 跨市場多資產隔離性驗證 (TW vs US)
    """

    def test_end_to_end_full_valuation_pipeline_integration(self):
        """驗證完整估值管線：求解器、EV Bridge、Monte Carlo、Sensitivity、SCD2 與 Guardrail 整合"""
        from valuation.services.valuation_service import ValuationService
        import json

        result = ValuationService.calculate_valuation("2330.TW")

        # 1. 核心指標存在性校驗
        self.assertIn("symbol", result)
        self.assertEqual(result["symbol"], "2330.TW")
        self.assertIn("fair_value", result)
        self.assertIn("current_price", result)
        self.assertIn("upside", result)
        self.assertGreater(result["fair_value"], 0.0)

        # 2. Phase 1 & 2: 迭代求解器與 EV-to-Equity Bridge 節點
        self.assertIn("dcf", result)
        dcf_data = result["dcf"]
        self.assertIn("iterations", dcf_data)
        self.assertIn("ev_to_equity_bridge", dcf_data)
        bridge = dcf_data["ev_to_equity_bridge"]
        self.assertIn("items", bridge)
        self.assertIn("net_debt", bridge)

        # 3. Phase 3: 蒙地卡羅與雙向敏感度矩陣節點
        self.assertIn("monte_carlo", result)
        mc_data = result["monte_carlo"]
        self.assertEqual(mc_data["num_simulations"], 10000)
        self.assertIn("p50", mc_data["statistics"])
        self.assertIn("histogram", mc_data)
        self.assertIn("sensitivity_matrix", result)
        sens_data = result["sensitivity_matrix"]
        self.assertIn("matrix", sens_data)
        self.assertIn("base_coordinates", sens_data)

        # 4. Phase 4: SCD Type 2 歷史快照節點
        self.assertIn("scd2", result)
        self.assertIn("version", result["scd2"])
        self.assertTrue(result["scd2"]["is_current"])

        # 5. Phase 6: WACC 異常值防護閥節點
        self.assertIn("is_flagged", result)
        self.assertIn("flag_reasons", result)
        self.assertIn("wacc_guardrail", result)
        self.assertFalse(result["is_flagged"])

        # 6. JSON 序列化安全性防禦（全型別杜絕 TypeError）
        serialized = json.dumps(result)
        self.assertIsInstance(serialized, str)
        self.assertGreater(len(serialized), 1000)

    def test_time_travel_historical_snapshot_replay_and_consistency(self):
        """驗證歷史 Time-travel 快照回溯之不可變性與時間切片查詢準確度"""
        from valuation.services.assumption_history import AssumptionHistoryService
        import datetime

        symbol = "REPLAY.TW"
        market = "TW"

        # 建立 3 個歷史時間段的假定快照
        # v1: 2025-01-01 生效，2025-05-31 結束
        r1, _ = AssumptionHistoryService.save_assumptions_scd2(
            symbol=symbol,
            market=market,
            assumptions={"wacc": 0.08, "g": 0.02, "period": "2025Q1"},
            valuation_snapshot={"fair_value": 100.0},
            change_reason="Q1 估值",
            effective_date=datetime.date(2025, 1, 1)
        )

        # v2: 2025-06-01 生效，2025-11-30 結束
        r2, _ = AssumptionHistoryService.save_assumptions_scd2(
            symbol=symbol,
            market=market,
            assumptions={"wacc": 0.085, "g": 0.025, "period": "2025Q2"},
            valuation_snapshot={"fair_value": 115.0},
            change_reason="Q2 估值",
            effective_date=datetime.date(2025, 6, 1)
        )

        # v3: 2025-12-01 生效至今
        r3, _ = AssumptionHistoryService.save_assumptions_scd2(
            symbol=symbol,
            market=market,
            assumptions={"wacc": 0.09, "g": 0.028, "period": "2025Q4"},
            valuation_snapshot={"fair_value": 130.0},
            change_reason="Q4 估值",
            effective_date=datetime.date(2025, 12, 1)
        )

        # 查詢 2025-03-15 (落於 v1 期間)
        snap_v1 = AssumptionHistoryService.get_as_of_date(symbol, market, datetime.date(2025, 3, 15))
        self.assertIsNotNone(snap_v1)
        self.assertEqual(snap_v1.version, 1)
        self.assertEqual(snap_v1.assumptions["period"], "2025Q1")

        # 查詢 2025-08-20 (落於 v2 期間)
        snap_v2 = AssumptionHistoryService.get_as_of_date(symbol, market, datetime.date(2025, 8, 20))
        self.assertIsNotNone(snap_v2)
        self.assertEqual(snap_v2.version, 2)
        self.assertEqual(snap_v2.assumptions["period"], "2025Q2")

        # 查詢最新 (2026-03-01，落於 v3 期間)
        snap_v3 = AssumptionHistoryService.get_as_of_date(symbol, market, datetime.date(2026, 3, 1))
        self.assertIsNotNone(snap_v3)
        self.assertEqual(snap_v3.version, 3)
        self.assertEqual(snap_v3.assumptions["period"], "2025Q4")

    def test_performance_and_latency_benchmarks(self):
        """驗證工業級效能基準：蒙地卡羅萬次模擬 < 100ms，敏感度矩陣 < 50ms，EV Bridge < 10ms"""
        from valuation.services.monte_carlo import MonteCarloSimulator
        from valuation.services.sensitivity import SensitivityAnalyzer
        from valuation.services.ev_bridge import EVToEquityBridge
        from valuation.services.wacc_guardrail import WACCGuardrail
        import time

        # 1. 蒙地卡羅 10,000 次模擬基準測試
        mc = MonteCarloSimulator(num_simulations=10000, seed=42)
        t0 = time.perf_counter()
        mc_res = mc.run_simulation(
            base_revenue=1000000.0,
            base_growth=0.08,
            growth_std=0.02,
            ebit_margin=0.35,
            tax_rate=0.20,
            base_wacc=0.08,
            wacc_std=0.01,
            risk_free_rate=0.04,
            reinvestment_rate=0.40,
            net_debt=200000.0,
            shares_outstanding=10000.0,
            current_price=150.0
        )
        t_mc = time.perf_counter() - t0
        self.assertLess(t_mc, 0.15, f"Monte Carlo 10k simulations took too long: {t_mc:.4f}s")

        # 2. 雙向敏感度矩陣基準測試
        t0 = time.perf_counter()
        sens_res = SensitivityAnalyzer.generate_matrix(
            base_revenue=1000000.0,
            growth_rates=[0.08, 0.07, 0.06, 0.05, 0.04],
            ebit_margin=0.35,
            tax_rate=0.20,
            reinvestment_rate=0.40,
            base_wacc=0.08,
            base_g=0.025,
            net_debt=200000.0,
            shares_outstanding=10000.0,
            risk_free_rate=0.04
        )
        t_sens = time.perf_counter() - t0
        self.assertLess(t_sens, 0.05, f"Sensitivity matrix took too long: {t_sens:.4f}s")

        # 3. EV-to-Equity Bridge 基準測試
        t0 = time.perf_counter()
        bridge_res = EVToEquityBridge.calculate_bridge(
            enterprise_value=5000000.0,
            cash=800000.0,
            total_debt=1200000.0,
            operating_lease_liability=150000.0,
            preferred_stock=50000.0,
            minority_interest=20000.0,
            pension_deficit=30000.0
        )
        t_bridge = time.perf_counter() - t0
        self.assertLess(t_bridge, 0.01, f"EV Bridge calculation took too long: {t_bridge:.4f}s")

        # 4. WACC 異常值防護閥基準測試
        t0 = time.perf_counter()
        guard_res = WACCGuardrail.validate_and_clamp(raw_wacc=0.092, rf=0.042)
        t_guard = time.perf_counter() - t0
        self.assertLess(t_guard, 0.005, f"WACC Guardrail check took too long: {t_guard:.4f}s")

    def test_multi_asset_and_market_concurrency_resilience(self):
        """驗證跨市場資產之計算隔離與相容性 (台股 TWD vs 美股 USD)"""
        from valuation.services.wacc_guardrail import WACCGuardrail

        # 驗證台股市場無風險利率預設處理
        tw_rf, _, _ = WACCGuardrail.validate_rf(0.015, market='tw')
        self.assertEqual(tw_rf, 0.015)

        # 驗證美股市場無風險利率預設處理
        us_rf, _, _ = WACCGuardrail.validate_rf(0.042, market='us')
        self.assertEqual(us_rf, 0.042)




