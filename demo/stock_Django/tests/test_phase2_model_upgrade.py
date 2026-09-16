# -*- coding: utf-8 -*-
"""
Phase 2 模型升級與解釋性模組單元測試
測試涵蓋：
1. Unicode NFKC 正規化與不可見字元過濾 (clean_and_normalize_text)
2. 統一情緒分析器語言偵測、閾值防護與邊界回退 (UnifiedSentimentAnalyzer, SentimentResult)
3. RobustSHAPEncoder 序列化防護 (NaN/Inf 轉 null、NumPy 型別轉換)
4. ModelExplainer Top-K 特徵限制與前端 UI 遮擋防護 (others_shap_value 聚合)
"""

import os
import sys
import json
import math
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

# 設置 Django 環境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')
import django
django.setup()

from stock_Django.agent_news_analyzer import (
    clean_and_normalize_text,
    SentimentResult,
    UnifiedSentimentAnalyzer
)
from stock_Django.model_explainer import RobustSHAPEncoder, ModelExplainer


class TestUnicodeNormalizationAndDefense(unittest.TestCase):
    """測試文本清理與 Unicode 防護"""

    def test_nfkc_normalization(self):
        """測試全形字元轉換為標準 NFKC 半形字元"""
        full_width = "ＡＡＰＬ　股價創下歷史新高！"
        normalized = clean_and_normalize_text(full_width)
        self.assertEqual(normalized, "AAPL 股價創下歷史新高!")

    def test_control_character_removal(self):
        """測試移除控制字元與不可見字元 (保留一般空格與中文字)"""
        dirty_text = "台積電\x00營收\x08暴增\x1f。\x7f"
        cleaned = clean_and_normalize_text(dirty_text)
        self.assertEqual(cleaned, "台積電營收暴增。")

    def test_empty_and_non_string_input(self):
        """測試邊界情況：空字串、None、非字串輸入"""
        self.assertEqual(clean_and_normalize_text(""), "")
        self.assertEqual(clean_and_normalize_text(None), "")
        self.assertEqual(clean_and_normalize_text(12345), "")


class TestUnifiedSentimentAnalyzer(unittest.TestCase):
    """測試統一新聞情緒分析器邏輯"""

    def test_detect_language(self):
        """測試中英文自動語系偵測"""
        zh_text = "聯發科今日召開法說會，看好下半年旗艦晶片出貨動能。"
        en_text = "NVIDIA reports record quarterly revenue driven by strong AI demand."
        mixed_zh = "TSMC 宣布在高雄擴建 2 奈米新廠，預計明年量產。"
        
        self.assertEqual(UnifiedSentimentAnalyzer.detect_language(zh_text), 'zh-TW')
        self.assertEqual(UnifiedSentimentAnalyzer.detect_language(en_text), 'en')
        self.assertEqual(UnifiedSentimentAnalyzer.detect_language(mixed_zh), 'zh-TW')
        self.assertEqual(UnifiedSentimentAnalyzer.detect_language(""), 'zh-TW')

    def test_sentiment_result_to_dict(self):
        """測試 SentimentResult 結構與向下相容字典導出"""
        res = SentimentResult(
            label="positive",
            score=0.8523,
            confidence=0.9261,
            probabilities={"positive": 0.9261, "negative": 0.0512, "neutral": 0.0227},
            is_neutral_adjusted=False,
            language="zh-TW"
        )
        d = res.to_dict()
        self.assertEqual(d["positive_negative_analysis"], "正面")
        self.assertEqual(d["label"], "positive")
        self.assertEqual(d["sentiment_score"], 0.8523)
        self.assertEqual(d["confidence"], 0.9261)
        self.assertIn("positive", d["probabilities"])
        self.assertFalse(d["is_neutral_adjusted"])

    def test_neutral_threshold_calibration_logic(self):
        """測試 neutral_threshold=0.60 閾值校準邏輯"""
        # 建立一個不需加載權重的 mock analyzer
        analyzer = UnifiedSentimentAnalyzer.__new__(UnifiedSentimentAnalyzer)
        analyzer.neutral_threshold = 0.60
        analyzer.device = "cpu"

        # 模擬中文分析且預測機率低於 0.60 之情境
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_torch = MagicMock()
        
        # 模擬輸出 Logits 使 softmax 機率為 [0.55, 0.45] (max < 0.60)
        import torch
        logits = torch.tensor([[0.55, 0.45]])
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_model.return_value = mock_output
        mock_tokenizer.return_value.to.return_value = {"input_ids": torch.tensor([[101, 102]])}

        analyzer.tokenizer = mock_tokenizer
        analyzer.model = mock_model
        analyzer._torch = torch

        result = analyzer._analyze_zh("普通的一般性財經描述")
        self.assertEqual(result.label, "neutral")
        self.assertEqual(result.score, 0.0)
        self.assertTrue(result.is_neutral_adjusted)


class TestRobustSHAPEncoder(unittest.TestCase):
    """測試 SHAP 專用序列化器防禦機制"""

    def test_nan_inf_serialization(self):
        """測試 NaN 與正負無窮大自動轉換為 JSON null (None)"""
        data = {
            "valid_num": 3.1415,
            "nan_val": float("nan"),
            "pos_inf": float("inf"),
            "neg_inf": float("-inf"),
            "np_nan": np.float32(np.nan)
        }
        json_str = json.dumps(data, cls=RobustSHAPEncoder)
        decoded = json.loads(json_str)

        self.assertAlmostEqual(decoded["valid_num"], 3.1415, places=4)
        self.assertIsNone(decoded["nan_val"])
        self.assertIsNone(decoded["pos_inf"])
        self.assertIsNone(decoded["neg_inf"])
        self.assertIsNone(decoded["np_nan"])

    def test_numpy_types_serialization(self):
        """測試 NumPy 型別 (int64, float32, ndarray) 正確轉為原生 Python 型別"""
        data = {
            "int_val": np.int64(42),
            "float_val": np.float32(1.234),
            "arr_val": np.array([1, 2, 3], dtype=np.int32),
            "timestamp": pd.Timestamp("2026-09-16 09:30:00")
        }
        json_str = json.dumps(data, cls=RobustSHAPEncoder)
        decoded = json.loads(json_str)

        self.assertEqual(decoded["int_val"], 42)
        self.assertAlmostEqual(decoded["float_val"], 1.234, places=3)
        self.assertEqual(decoded["arr_val"], [1, 2, 3])
        self.assertEqual(decoded["timestamp"], "2026-09-16 09:30:00")


class TestModelExplainer(unittest.TestCase):
    """測試 SHAP 模型解釋器與 UI 遮擋防護"""

    def setUp(self):
        # 建立簡單線性預測函式
        self.coef = np.array([0.5, -0.3, 0.8, -0.1, 0.05, 0.12, -0.08, 0.25, 0.15, -0.22, 0.4, -0.6])
        self.predict_fn = lambda X: np.dot(X, self.coef)

        # 模擬 50 筆背景數據 (12 個特徵)
        np.random.seed(42)
        self.bg_data = np.random.randn(50, 12).astype(np.float32)
        self.feature_names = [f"Feature_{i}" for i in range(12)]
        self.input_sample = np.random.randn(1, 12).astype(np.float32)

    def test_top_k_feature_filtering(self):
        """測試 Top-K 特徵篩選與 others_shap_value 累計聚合，防止前端圖表遮擋"""
        top_k = 5
        explainer = ModelExplainer(top_k=top_k)
        res = explainer.explain_tabular_features(
            model_or_fn=self.predict_fn,
            background_data=self.bg_data,
            input_sample=self.input_sample,
            feature_names=self.feature_names
        )

        self.assertIn("top_features", res)
        self.assertEqual(len(res["top_features"]), top_k)
        self.assertIn("others_shap_value", res)
        self.assertEqual(res["total_features_count"], 12)
        
        # 檢查各特徵貢獻是否按 abs_importance 降序排列
        importances = [item["abs_importance"] for item in res["top_features"]]
        self.assertEqual(importances, sorted(importances, reverse=True))

    def test_export_feature_importance_json(self):
        """測試導出 JSON 字串包含 RobustSHAPEncoder 保護"""
        explainer = ModelExplainer(top_k=3)
        res = explainer.explain_tabular_features(
            model_or_fn=self.predict_fn,
            background_data=self.bg_data,
            input_sample=self.input_sample,
            feature_names=self.feature_names
        )
        # 人工注入一個含有 NaN 的值模擬異常情況
        res["top_features"][0]["raw_value"] = float("nan")
        
        json_output = explainer.export_feature_importance_json(res)
        self.assertIsInstance(json_output, str)
        decoded = json.loads(json_output)
        self.assertIsNone(decoded["top_features"][0]["raw_value"])


if __name__ == "__main__":
    unittest.main()
