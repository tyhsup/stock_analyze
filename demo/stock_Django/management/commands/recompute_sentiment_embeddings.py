# -*- coding: utf-8 -*-
"""
Django Management Command: recompute_sentiment_embeddings
用途：依股票代號批次清除或重新計算 stock_sentiment_embeddings 表中的語意向量快取，
      套用 Phase 0 的交易日曆時間對齊與多因子加權聚合（WeightedSentimentAggregator）。
"""

import os
import sys
import logging
from django.core.management.base import BaseCommand
from django.db import connection
import pandas as pd

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "依股票代號重新計算情緒特徵快取 (P0 加權聚合與交易日曆對齊)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbol",
            type=str,
            help="指定股票代號（例如 2330 或 2330.TW）",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            help="重算所有已有新聞資料的股票",
        )
        parser.add_argument(
            "--clear-cache",
            action="store_true",
            help="重算前先清空該股票於 stock_sentiment_embeddings 的歷史快取",
        )

    def handle(self, *args, **options):
        symbol = options.get("symbol")
        all_stocks = options.get("all")
        clear_cache = options.get("clear_cache")

        if not symbol and not all_stocks:
            self.stderr.write(self.style.ERROR("請指定 --symbol <代號> 或 --all"))
            return

        symbols_to_process = []
        if symbol:
            symbols_to_process.append(symbol.strip().upper())
        elif all_stocks:
            webbug_dir = os.getenv("WEBBUG_DIR", "E:/Infinity/webbug/")
            if os.path.exists(webbug_dir):
                for fname in os.listdir(webbug_dir):
                    if fname.endswith("_news.xlsx"):
                        sym = fname.replace("_news.xlsx", "").upper()
                        symbols_to_process.append(sym)
            else:
                self.stderr.write(self.style.WARNING(f"找不到 webbug 目錄: {webbug_dir}"))

        self.stdout.write(self.style.NOTICE(f"準備處理 {len(symbols_to_process)} 檔標的: {symbols_to_process}"))

        from stock_Django.dataset_builders import SentimentProbabilityModel

        for sym in symbols_to_process:
            clean_sym = sym.replace(".TWO", "").replace(".TW", "")
            if clear_cache:
                self.stdout.write(f"正在清除 {sym} 之歷史情緒快取...")
                with connection.cursor() as cursor:
                    cursor.execute("DELETE FROM stock_sentiment_embeddings WHERE symbol = %s", [sym])
                    cursor.execute("DELETE FROM stock_sentiment_embeddings WHERE symbol = %s", [clean_sym])
                self.stdout.write(self.style.SUCCESS(f"已清除 {sym} 歷史快取"))

            webbug_dir = os.getenv("WEBBUG_DIR", "E:/Infinity/webbug/")
            news_file = os.path.join(webbug_dir, f"{clean_sym}_news.xlsx")
            if not os.path.exists(news_file):
                self.stdout.write(self.style.WARNING(f"跳過 {sym}：找不到新聞檔案 {news_file}"))
                continue

            try:
                df_news = pd.read_excel(news_file)
                if df_news.empty:
                    self.stdout.write(self.style.WARNING(f"跳過 {sym}：新聞資料為空"))
                    continue

                # 建立覆蓋日期區間
                time_col_idx = 1 if df_news.shape[1] > 1 else 0
                dates = pd.to_datetime(df_news.iloc[:, time_col_idx], errors="coerce").dropna()
                if dates.empty:
                    self.stdout.write(self.style.WARNING(f"跳過 {sym}：無有效發布時間戳"))
                    continue

                start_d = dates.min().date()
                end_d = dates.max().date()
                date_range = pd.date_range(start_d, end_d, freq="D")
                date_df = pd.DataFrame(index=date_range)

                self.stdout.write(f"開始重新計算 {sym} 情緒特徵 ({start_d} ~ {end_d})...")
                res_df = SentimentProbabilityModel.get_sentiment_features(sym, date_df)
                self.stdout.write(self.style.SUCCESS(f"成功完成 {sym} 重算，產出維度: {res_df.shape}"))

            except Exception as e:
                self.stderr.write(self.style.ERROR(f"處理 {sym} 時發生例外錯誤: {e}"))


if __name__ == "__main__":
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "demo.settings")
    django.setup()
    from django.core.management import call_command
    call_command("recompute_sentiment_embeddings", symbol="2330", clear_cache=False)
