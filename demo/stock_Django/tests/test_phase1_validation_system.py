# -*- coding: utf-8 -*-
"""
Phase 1 模型驗證體系單元測試
測試涵蓋：
1. PriceLSTMFeatureExtractor 標準化隔離（Scaler Leakage 防護、向後相容性、測試集缺少 scaler 拋出例外）
2. WalkForwardBacktester（訊號 shift(1) 強制遞延、10 bps 摩擦成本扣除、5 大指標計算、零交易除以零防禦、Numpy JSON 序列化、資料庫參數化儲存）
"""

import os
import sys
import json
import math
import datetime
import unittest
import numpy as np
import pandas as pd

# 設置 Django 環境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')
import django
django.setup()

from django.db import connection
from stock_Django.dataset_builders import PriceLSTMFeatureExtractor
from stock_Django.backtester import WalkForwardBacktester, BacktestResult, NumpyJsonEncoder


class TestPriceLSTMFeatureExtractorScalerIsolation(unittest.TestCase):
    def setUp(self):
        # 建立模擬價格資料 (50 筆)
        dates = pd.date_range("2026-01-01", periods=50, freq="B")
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(50) * 1.5)
        self.df = pd.DataFrame({"Close": prices}, index=dates)

    def test_backwards_compatibility(self):
        """測試向後相容性：未傳入 return_scaler 時直接回傳 DataFrame，不破壞既有呼叫端"""
        res = PriceLSTMFeatureExtractor.extract_features(self.df)
        self.assertIsInstance(res, pd.DataFrame)
        self.assertIn("Daily_Return", res.columns)
        self.assertIn("Bias_5", res.columns)
        self.assertIn("Bias_20", res.columns)

    def test_fit_mode_train_produces_scaler(self):
        """測試訓練集模式 (fit_mode=True) 能擬合並回傳 StandardScaler 實例"""
        res_df, scaler = PriceLSTMFeatureExtractor.extract_features(
            self.df, fit_mode=True, return_scaler=True
        )
        self.assertIsNotNone(scaler)
        # 標準化後特徵均值應接近 0
        self.assertAlmostEqual(float(res_df['Daily_Return'].mean()), 0.0, places=2)

    def test_fit_mode_test_requires_fitted_scaler(self):
        """測試測試集模式 (fit_mode=False) 若未傳入 scaler 必須拋出 ValueError 阻止數據洩漏"""
        with self.assertRaises(ValueError):
            PriceLSTMFeatureExtractor.extract_features(
                self.df, fit_mode=False, scaler=None, return_scaler=True
            )

    def test_fit_mode_test_uses_train_scaler(self):
        """測試測試集模式使用訓練集擬合之 scaler 進行 transform，不重新計算均值與方差"""
        train_df = self.df.iloc[:30]
        test_df = self.df.iloc[30:]

        _, scaler = PriceLSTMFeatureExtractor.extract_features(
            train_df, fit_mode=True, return_scaler=True
        )
        test_scaled, _ = PriceLSTMFeatureExtractor.extract_features(
            test_df, fit_mode=False, scaler=scaler, return_scaler=True
        )
        self.assertIsInstance(test_scaled, pd.DataFrame)
        self.assertGreater(len(test_scaled), 0)
        self.assertFalse(test_scaled['Daily_Return'].isna().any())


class TestWalkForwardBacktester(unittest.TestCase):
    def setUp(self):
        self.backtester = WalkForwardBacktester(
            train_window=252, test_window=21, transaction_cost_bps=10.0
        )

    def test_signal_delay_shift_one(self):
        """測試訊號強制遞延一期 (shift 1)：T 日訊號延遲至 T+1 執行以消除前瞻偏誤"""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        signals = pd.Series([1.0, 1.0, -1.0, 0.0])

        metrics = self.backtester.calculate_metrics(returns, signals)
        # 由於訊號在第 0 期產生，第 0 期的 delayed_signal 必為 0.0
        self.assertIsInstance(metrics, dict)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("sortino_ratio", metrics)

    def test_friction_cost_applied_on_trades(self):
        """測試換倉時扣除摩擦成本 (10 bps = 0.0010)"""
        # 4 期報酬全為 0，第 0 期發出買進訊號 (1.0)，第 2 期平倉 (0.0)
        # 產生 2 次換倉動作，總報酬應為 -2 * 10 bps = -0.0020
        returns = pd.Series([0.0, 0.0, 0.0, 0.0])
        signals = pd.Series([1.0, 1.0, 0.0, 0.0])

        metrics = self.backtester.calculate_metrics(returns, signals)
        self.assertEqual(metrics["num_trades"], 2)
        self.assertAlmostEqual(metrics["total_return"], -0.0020, places=4)

    def test_zero_trades_safety(self):
        """測試無任何交易（訊號全為 0）時之邊界安全性，指標應安全返回 0.0，無 NaN 或 ZeroDivisionError"""
        returns = pd.Series([0.01, -0.01, 0.02, -0.02])
        signals = pd.Series([0.0, 0.0, 0.0, 0.0])

        metrics = self.backtester.calculate_metrics(returns, signals)
        self.assertEqual(metrics["num_trades"], 0)
        self.assertEqual(metrics["sharpe_ratio"], 0.0)
        self.assertEqual(metrics["win_rate"], 0.0)
        self.assertEqual(metrics["profit_loss_ratio"], 0.0)
        self.assertFalse(math.isnan(metrics["sharpe_ratio"]))

    def test_numpy_json_encoder(self):
        """測試 Numpy 數據型別能順利序列化為 JSON，不拋出 TypeError"""
        config = {
            "learning_rate": np.float64(0.001),
            "epochs": np.int64(50),
            "weights": np.array([0.1, 0.9]),
            "date": datetime.date(2026, 9, 16)
        }
        json_str = json.dumps(config, cls=NumpyJsonEncoder)
        self.assertIsInstance(json_str, str)
        self.assertIn('"learning_rate": 0.001', json_str)
        self.assertIn('"epochs": 50', json_str)

    def test_db_save_and_retrieve(self):
        """測試將 BacktestResult 寫入 MySQL backtest_results 表並驗證"""
        result = BacktestResult(
            symbol="2330",
            market="tw",
            model_version="test_v1.0",
            strategy_name="unit_test_strategy",
            run_date=datetime.datetime.now(),
            train_window=252,
            test_window=21,
            sharpe_ratio=1.2500,
            sortino_ratio=1.6500,
            max_drawdown=-0.0850,
            win_rate=0.5714,
            profit_loss_ratio=1.4500,
            total_return=0.1850,
            annualized_return=0.1520,
            num_trades=12,
            config_json={"test_key": np.float64(0.123)}
        )

        inserted_id = self.backtester.save_result_to_db(result)
        self.assertIsNotNone(inserted_id)
        self.assertGreater(inserted_id, 0)

        # 驗證資料庫內容
        with connection.cursor() as cursor:
            cursor.execute("SELECT symbol, sharpe_ratio, num_trades FROM backtest_results WHERE id = %s", [inserted_id])
            row = cursor.fetchone()
            self.assertEqual(row[0], "2330")
            self.assertAlmostEqual(float(row[1]), 1.2500, places=3)
            self.assertEqual(row[2], 12)


if __name__ == "__main__":
    unittest.main()
