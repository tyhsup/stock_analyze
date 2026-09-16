# -*- coding: utf-8 -*-
"""
Phase 0 資料基底工程單元測試
測試涵蓋：
1. TradingCalendarService（盤中、盤後 T+1、假日遞延、美股夏令/冬令時區、Unicode 防禦）
2. WeightedSentimentAggregator（來源權威度分級、時間指數衰減、除以零/下溢防護、向量形狀保證）
"""

import os
import sys
import math
import datetime
import unittest
import numpy as np

# 設置 Django 環境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')
import django
django.setup()

from stock_Django.trading_calendar import TradingCalendarService, sanitize_string
from stock_Django.dataset_builders import WeightedSentimentAggregator, PriceLSTMFeatureExtractor


class TestTradingCalendarService(unittest.TestCase):
    def setUp(self):
        self.cal = TradingCalendarService()

    def test_sanitize_string_unicode_and_null_byte(self):
        """測試 Unicode NFKC 正規化與截斷字元移除"""
        raw_input = "台積電\x00利多！\uff21\uff22\uff23 \U0001f9a7"
        cleaned = sanitize_string(raw_input)
        self.assertNotIn("\x00", cleaned)
        self.assertIn("台積電利多!ABC", cleaned)

    def test_taiwan_intraday_alignment(self):
        """測試台股盤中時間對齊當日 T"""
        # 2026-09-15 10:00 (週二盤中)
        ts = "2026-09-15 10:00:00"
        aligned = self.cal.align_to_trading_day(ts, market="tw")
        cls_type = self.cal.classify_news_timestamp(ts, market="tw")
        self.assertEqual(cls_type, "intraday")
        self.assertEqual(aligned, datetime.date(2026, 9, 15))

    def test_taiwan_after_hours_delay(self):
        """測試台股盤後時間（>13:30）強制遞延至 T+1 交易日，消除前瞻偏誤"""
        # 2026-09-15 14:00 (週二盤後) -> 應為 2026-09-16 (週三)
        ts = "2026-09-15 14:00:00"
        aligned = self.cal.align_to_trading_day(ts, market="tw")
        cls_type = self.cal.classify_news_timestamp(ts, market="tw")
        self.assertEqual(cls_type, "after_hours")
        self.assertEqual(aligned, datetime.date(2026, 9, 16))

    def test_taiwan_weekend_holiday_forward(self):
        """測試週末或假日新聞順延至下一個有效開盤交易日"""
        # 2026-09-12 11:00 (週六) -> 應順延至 2026-09-14 (週一)
        ts = "2026-09-12 11:00:00"
        aligned = self.cal.align_to_trading_day(ts, market="tw")
        cls_type = self.cal.classify_news_timestamp(ts, market="tw")
        self.assertEqual(cls_type, "holiday")
        self.assertEqual(aligned, datetime.date(2026, 9, 14))

    def test_us_market_dst_and_hours(self):
        """測試美股時區與夏令時間開收盤對齊"""
        # 2024-03-11 15:30 (美東夏令時間盤中，收盤為 16:00)
        ts_intraday = "2024-03-11 15:30:00"
        aligned_intra = self.cal.align_to_trading_day(ts_intraday, market="us")
        self.assertEqual(aligned_intra, datetime.date(2024, 3, 11))

        # 2024-03-11 16:30 (美東夏令時間盤後) -> 應為 2024-03-12 (T+1)
        ts_after = "2024-03-11 16:30:00"
        aligned_after = self.cal.align_to_trading_day(ts_after, market="us")
        self.assertEqual(aligned_after, datetime.date(2024, 3, 12))


class TestWeightedSentimentAggregator(unittest.TestCase):
    def test_source_weights_hierarchy(self):
        """測試使用者確認之權威度分級：CNBC/Reuters (0.9) > CNYES/MoneyDJ (0.7) > PTT (0.3)"""
        w_cnbc = WeightedSentimentAggregator.get_source_weight("cnbc")
        w_reuters = WeightedSentimentAggregator.get_source_weight("reuters")
        w_cnyes = WeightedSentimentAggregator.get_source_weight("cnyes")
        w_moneydj = WeightedSentimentAggregator.get_source_weight("moneydj")
        w_ptt = WeightedSentimentAggregator.get_source_weight("ptt")
        w_def = WeightedSentimentAggregator.get_source_weight("unknown")

        self.assertEqual(w_cnbc, 0.90)
        self.assertEqual(w_reuters, 0.90)
        self.assertEqual(w_cnyes, 0.70)
        self.assertEqual(w_moneydj, 0.70)
        self.assertEqual(w_ptt, 0.30)
        self.assertEqual(w_def, 0.50)
        self.assertTrue(w_cnbc > w_cnyes > w_ptt)

    def test_time_decay_weighting(self):
        """測試較近期的新聞權重大於歷史舊新聞"""
        dim = 768
        # 新聞 1：發布於當日開盤前（權重大），嵌入值全為 1.0
        # 新聞 2：發布於 10 天前（權重衰減），嵌入值全為 0.0
        embs = np.array([[1.0] * dim, [0.0] * dim], dtype=np.float32)
        timestamps = ["2026-09-15 08:00:00", "2026-09-05 08:00:00"]
        sources = ["cnyes", "cnyes"]
        agg = WeightedSentimentAggregator.aggregate(
            embeddings=embs,
            timestamps=timestamps,
            sources=sources,
            target_date="2026-09-15",
            decay_lambda=0.1
        )
        # 由於第一筆權重大於第二筆，聚合結果均值應 > 0.5
        self.assertGreater(agg[0], 0.50)

    def test_underflow_and_zero_division_safety(self):
        """測試極度久遠時間戳（浮點數下溢）與零權重時之除以零防禦，回退至算術平均"""
        dim = 768
        embs = np.array([[2.0] * dim, [4.0] * dim], dtype=np.float32)
        # 3 年前的新聞，e^(-0.1 * 1000) 趨近於 0
        old_timestamps = ["2020-01-01 00:00:00", "2020-01-02 00:00:00"]
        agg = WeightedSentimentAggregator.aggregate(
            embeddings=embs,
            timestamps=old_timestamps,
            target_date="2026-09-15"
        )
        self.assertFalse(np.isnan(agg).any())
        self.assertFalse(np.isinf(agg).any())
        self.assertAlmostEqual(agg[0], 3.0, places=4)

    def test_output_shape_and_dtype(self):
        """測試輸出形狀嚴格為 768 維 float32"""
        dim = 768
        embs = np.random.randn(5, dim).astype(np.float32)
        agg = WeightedSentimentAggregator.aggregate(embs)
        self.assertEqual(agg.shape, (dim,))
        self.assertEqual(agg.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
