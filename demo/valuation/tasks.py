# -*- coding: utf-8 -*-
"""
Valuation Celery Tasks
提供非同步單一估值計算、分塊批次估值 (Chunk-based Batch Processing) 與全市場定時重算排程。
落實資料庫連線防禦、指數退避重試 (Exponential Backoff with Jitter) 與個別標的例外隔離。
"""
import logging
import random
import time
from typing import List, Dict, Any, Optional

from celery import shared_task
from django.db import connections, DatabaseError
from sqlalchemy import text

from stock_Django.mySQL_OP import OP_Fun
from valuation.services.valuation_service import ValuationService

logger = logging.getLogger(__name__)


def _get_market_symbols(market: str = 'TW') -> List[str]:
    """從資料庫獲取目標市場的有效股票代碼清單，具備安全 Fallback"""
    market = str(market).strip().upper()
    symbols: List[str] = []
    db_op = OP_Fun()

    try:
        if market == 'TW':
            query = "SELECT symbol FROM stocks_tw WHERE status = 'active' OR status IS NULL LIMIT 200"
            with db_op.engine.connect() as conn:
                rows = conn.execute(text(query)).fetchall()
                symbols = [f"{r[0]}.TW" if not (r[0].endswith('.TW') or r[0].endswith('.TWO')) else r[0] for r in rows if r and r[0]]
        else:
            query = "SELECT symbol FROM stocks_us WHERE status = 'active' OR status IS NULL LIMIT 200"
            with db_op.engine.connect() as conn:
                rows = conn.execute(text(query)).fetchall()
                symbols = [str(r[0]).strip().upper() for r in rows if r and r[0]]
    except Exception as ex:
        logger.warning(f"[Celery] 從資料庫獲取 {market} 股票代碼失敗，啟用安全 Fallback: {ex}")

    if not symbols:
        # 安全保底熱門標的
        if market == 'TW':
            symbols = ['2330.TW', '2317.TW', '2454.TW', '2382.TW', '2308.TW']
        else:
            symbols = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN']

    return sorted(list(set(symbols)))


@shared_task(bind=True, max_retries=3, name="valuation.tasks.calculate_single_valuation_async")
def calculate_single_valuation_async(self, ticker_symbol: str, **kwargs) -> Dict[str, Any]:
    """
    單一股票非同步估值計算任務。
    支援指數退避重試與完成後釋放資料庫連線。
    """
    ticker_symbol = str(ticker_symbol).strip().upper()
    logger.info(f"[Celery] 開始非同步計算股票估值: {ticker_symbol} (Task ID: {self.request.id})")

    try:
        res = ValuationService.calculate_valuation(ticker_symbol=ticker_symbol, **kwargs)
        if "error" in res:
            logger.warning(f"[Celery] 估值計算非致命警告 {ticker_symbol}: {res['error']}")
        return {
            "symbol": ticker_symbol,
            "status": "SUCCESS" if "error" not in res else "WARNING",
            "valuation": res,
            "task_id": self.request.id,
        }
    except (DatabaseError, ConnectionError) as exc:
        # 指數退避 + 隨機抖動 Jitter: 2^retry + random(1~5)
        countdown = (2 ** self.request.retries) + random.randint(1, 5)
        logger.error(f"[Celery] 資料庫/網路異常 {ticker_symbol}，將在 {countdown} 秒後重試: {exc}")
        raise self.retry(exc=exc, countdown=countdown)
    except Exception as exc:
        logger.error(f"[Celery] 股票 {ticker_symbol} 估值發生嚴重錯誤: {exc}", exc_info=True)
        return {
            "symbol": ticker_symbol,
            "status": "FAILED",
            "error": str(exc),
            "task_id": self.request.id,
        }
    finally:
        # 防禦要求：確保任務結束時正確關閉資料庫連線防止連線池枯竭
        connections.close_all()


@shared_task(bind=True, max_retries=2, name="valuation.tasks.calculate_valuation_chunk")
def calculate_valuation_chunk(self, symbols: List[str], market: str = "TW", **kwargs) -> Dict[str, Any]:
    """
    分塊批次估值運算任務 (Chunk-based Processing)。
    針對 Chunk 內每檔標的實作個別 Try-Catch 隔離，單一失敗不影響整批執行。
    """
    market = str(market).strip().upper()
    total = len(symbols)
    succeeded = 0
    failed = 0
    summary_results: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    logger.info(f"[Celery] 開始處理分塊批次任務 ({total} 檔股票, 市場: {market}, Task ID: {self.request.id})")

    try:
        for idx, sym in enumerate(symbols):
            sym = str(sym).strip().upper()
            try:
                # 呼叫核心估值計算
                val_res = ValuationService.calculate_valuation(ticker_symbol=sym, **kwargs)
                if "error" in val_res:
                    failed += 1
                    errors[sym] = str(val_res["error"])
                else:
                    succeeded += 1
                    summary_results[sym] = {
                        "fair_value": val_res.get("summary", {}).get("fair_value"),
                        "scd_version": val_res.get("scd2", {}).get("version"),
                    }
            except Exception as single_err:
                failed += 1
                errors[sym] = str(single_err)
                logger.warning(f"[Celery] 批次估值個別標的失敗 {sym}: {single_err}")

            # 微幅間隔防止外部請求觸發 429 速率限制
            if idx < total - 1:
                time.sleep(0.05)

        return {
            "status": "COMPLETED",
            "market": market,
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "results": summary_results,
            "errors": errors,
            "task_id": self.request.id,
        }
    finally:
        connections.close_all()


@shared_task(bind=True, name="valuation.tasks.batch_calculate_market_valuation")
def batch_calculate_market_valuation(self, market: str = 'TW', chunk_size: int = 20, **kwargs) -> Dict[str, Any]:
    """
    全市場批量估值 Master 排程任務。
    讀取市場股票並進行分塊 (Chunking)，透過 Celery Worker 平行派發處理。
    """
    market = str(market).strip().upper()
    symbols = _get_market_symbols(market)
    total_symbols = len(symbols)
    chunk_size = max(int(chunk_size), 5)

    chunks = [symbols[i:i + chunk_size] for i in range(0, total_symbols, chunk_size)]
    dispatched_task_ids = []

    logger.info(
        f"[Celery Master] 啟動 {market} 全市場批量估值：共 {total_symbols} 檔，拆分為 {len(chunks)} 個分塊 (每塊 {chunk_size} 檔)"
    )

    for chunk in chunks:
        # 非同步派發各分塊任務
        async_task = calculate_valuation_chunk.delay(symbols=chunk, market=market, **kwargs)
        dispatched_task_ids.append(async_task.id)

    return {
        "status": "DISPATCHED",
        "market": market,
        "total_symbols": total_symbols,
        "total_chunks": len(chunks),
        "chunk_size": chunk_size,
        "dispatched_task_ids": dispatched_task_ids,
        "master_task_id": self.request.id,
    }
