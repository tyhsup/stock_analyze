# -*- coding: utf-8 -*-
"""
Phase 3 動態系統與時間衰減填充單元測試
測試涵蓋：
1. SentimentTimeDecay 指數衰減、首日 Null 安全、極端 delta_t 下溢歸零與負值拋錯
2. FeatureShaper 2D/3D 維度變換與 1D 升維防禦
3. DynamicModelSelector 零樣本除以零防禦 (max(samples, 1))、勝率評估
4. DynamicModelSelector 溫啟動、Memory Cache TTL 與資料庫斷連雙重降級 (Double Fallback)
5. MySQL model_registry 參數化 UPSERT 寫入驗證
"""

import os
import sys
import json
import math
import time
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

# 設置 Django 環境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')
import django
django.setup()

from django.db import connection
from stock_Django.dataset_builders import SentimentTimeDecay
from stock_Django.model_selector import (
    FeatureShaper,
    RandomForestPredictorStrategy,
    LSTMPredictorStrategy,
    DynamicModelSelector
)


class TestSentimentTimeDecay(unittest.TestCase):
    """測試情緒特徵時間衰減向前填充與邊界防禦"""

    def test_decay_factor_calculation(self):
        """測試 delta_t = 0 與正常間隔下的衰減係數"""
        # delta_t = 0 (同日) 無衰減
        factor_0 = SentimentTimeDecay.calculate_decay_factor(0.0)
        self.assertEqual(factor_0, 1.0)

        # delta_t = 5 (間隔 5 天)
        factor_5 = SentimentTimeDecay.calculate_decay_factor(5.0, lambda_decay=0.1)
        expected_5 = math.exp(-0.1 * 5.0)
        self.assertAlmostEqual(factor_5, expected_5, places=5)

    def test_extreme_delta_t_underflow_protection(self):
        """測試極端 delta_t (>100 天) 觸發下溢保護安全歸零"""
        factor_extreme = SentimentTimeDecay.calculate_decay_factor(1000.0)
        self.assertEqual(factor_extreme, 0.0)

    def test_negative_delta_t_raises_value_error(self):
        """測試異常負時間差 (時間倒流) 必須拋出 ValueError 阻止數據污染"""
        with self.assertRaises(ValueError):
            SentimentTimeDecay.calculate_decay_factor(-5.0)

    def test_series_null_safety_first_day(self):
        """測試首日為空 (Null/0.0) 時之安全性，不引發異常並保持 0.0"""
        dates = pd.date_range("2026-09-01", periods=5, freq="D")
        # 首日為 0.0，第 2 日有新聞 (1.0)，後續 3 日無新聞 (0.0)
        s = pd.Series([0.0, 1.0, 0.0, 0.0, 0.0], index=dates)
        decayed = SentimentTimeDecay.apply_time_decay_ffill(s, lambda_decay=0.1)

        self.assertEqual(decayed.iloc[0], 0.0)
        self.assertEqual(decayed.iloc[1], 1.0)
        # 第 3 日 (間隔 1 天): 1.0 * exp(-0.1 * 1)
        self.assertAlmostEqual(decayed.iloc[2], math.exp(-0.1), places=4)
        # 第 4 日 (間隔 2 天): 1.0 * exp(-0.1 * 2)
        self.assertAlmostEqual(decayed.iloc[3], math.exp(-0.2), places=4)

    def test_dataframe_batch_decay(self):
        """測試多維特徵 DataFrame 批次衰減填充"""
        dates = pd.date_range("2026-09-01", periods=3, freq="D")
        df = pd.DataFrame({
            "feat_a": [2.0, 0.0, 0.0],
            "feat_b": [0.0, 3.0, 0.0]
        }, index=dates)

        res_df = SentimentTimeDecay.apply_time_decay_ffill(df, lambda_decay=0.1)
        # feat_a 在第 1 天衰減為 2.0 * exp(-0.1)
        self.assertAlmostEqual(res_df.loc[dates[1], "feat_a"], 2.0 * math.exp(-0.1), places=4)
        # feat_b 在第 0 天為 0.0，第 1 天為 3.0，第 2 天衰減為 3.0 * exp(-0.1)
        self.assertEqual(res_df.loc[dates[0], "feat_b"], 0.0)
        self.assertEqual(res_df.loc[dates[1], "feat_b"], 3.0)
        self.assertAlmostEqual(res_df.loc[dates[2], "feat_b"], 3.0 * math.exp(-0.1), places=4)


class TestFeatureShaper(unittest.TestCase):
    """測試跨模型特徵維度塑形器"""

    def test_3d_to_2d_flatten(self):
        """測試 3D 張量展平為 2D (samples, time_steps * features)"""
        arr_3d = np.ones((10, 5, 4), dtype=np.float32)
        arr_2d = FeatureShaper.to_2d(arr_3d)
        self.assertEqual(arr_2d.shape, (10, 20))

    def test_2d_to_3d_reshape(self):
        """測試 2D 矩陣升維為 3D (samples, time_steps, feat_per_step)"""
        arr_2d = np.ones((10, 20), dtype=np.float32)
        arr_3d = FeatureShaper.to_3d(arr_2d, time_steps=5)
        self.assertEqual(arr_3d.shape, (10, 5, 4))

    def test_1d_input_defense(self):
        """測試 1D 單一樣本輸入升維保護"""
        arr_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        arr_2d = FeatureShaper.to_2d(arr_1d)
        self.assertEqual(arr_2d.shape, (1, 3))


class TestDynamicModelSelector(unittest.TestCase):
    """測試動態模型選擇器與防禦降級機制"""

    def setUp(self):
        self.selector = DynamicModelSelector(cache_ttl_seconds=300)

    def test_div_zero_protection_rolling_accuracy(self):
        """測試零樣本時分母除以零防禦，回傳預設值 0.5000"""
        acc_empty = self.selector.calculate_rolling_accuracy(np.array([]), np.array([]))
        self.assertEqual(acc_empty, 0.5000)

    def test_rolling_accuracy_normal_calculation(self):
        """測試正常預測方向命中率計算"""
        y_true = np.array([1, 1, -1, -1])
        y_pred = np.array([1, -1, -1, 1])
        # 命中 2 次 (第 0 與第 2)，勝率應為 2/4 = 0.5000
        acc = self.selector.calculate_rolling_accuracy(y_true, y_pred)
        self.assertEqual(acc, 0.5000)

    def test_memory_cache_hit(self):
        """測試本地 Memory Cache 命中，不重複查詢資料庫"""
        key = ("TEST_SYM", "TW")
        now = time.time()
        self.selector._cache[key] = {"active_model": "lstm", "timestamp": now}

        selected = self.selector.select_active_model("TEST_SYM", "TW")
        self.assertEqual(selected, "lstm")

    def test_double_fallback_on_db_failure(self):
        """測試資料庫連線失敗時觸發雙重降級 (Double Fallback)，回退至預設 random_forest"""
        # 模擬 _fetch_active_model_from_db 拋出異常或回傳 None
        with patch.object(self.selector, '_fetch_active_model_from_db', return_value=None):
            selected = self.selector.select_active_model("NON_EXISTENT", "US")
            self.assertEqual(selected, "random_forest")

    def test_evaluate_and_sync_registry_selection_logic(self):
        """測試 60 日勝率比較與模型動態切換邏輯"""
        # 構造資料使 LSTM 勝率明顯高於 RF (超過 switch_threshold=0.02)
        y_true = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1, 1])
        rf_preds = np.array([1, 1, -1, -1, 1, 1, 1, 1, -1, -1])   # 勝率低 (2/10 = 0.20)
        lstm_preds = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1, 1]) # 勝率高 (10/10 = 1.00)

        res = self.selector.evaluate_and_sync_registry(
            symbol="2330",
            market="TW",
            y_true_60d=y_true,
            rf_preds=rf_preds,
            lstm_preds=lstm_preds,
            switch_threshold=0.02
        )

        self.assertEqual(res["selected_model"], "lstm")
        self.assertGreater(res["lstm_accuracy_60d"], res["rf_accuracy_60d"])

        # 驗證資料庫是否已透過參數化寫入 model_registry 表
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT model_type, is_active, rolling_accuracy 
                FROM model_registry 
                WHERE symbol = '2330' AND market = 'TW'
                ORDER BY model_type;
            """)
            rows = cursor.fetchall()
            self.assertEqual(len(rows), 2)
            row_dict = {r[0]: {"is_active": r[1], "acc": float(r[2])} for r in rows}
            self.assertTrue(row_dict["lstm"]["is_active"])
            self.assertFalse(row_dict["random_forest"]["is_active"])


if __name__ == "__main__":
    unittest.main()
