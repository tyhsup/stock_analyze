"""
data_freshness.py — Check whether MySQL data is stale and trigger updates.

Priority (as instructed):
1. OHLCV → stock_cost.py (StockCostManager)  
2. TW Investor/Chips → stock_investor.py (StockInvestorManager via Selenium)
3. US Institutional → stock_investor_us.py (Stockzoa + yfinance fallback)
4. Financial ratios → yfinance (last resort)
"""

import logging
import threading
import pandas as pd
from datetime import datetime, date, timedelta
from sqlalchemy import text
import time

logger = logging.getLogger(__name__)

# In-memory progress tracker {ticker: {status, progress, message}}
_refresh_status: dict = {}
_refresh_lock = threading.Lock()


def get_refresh_status(ticker: str) -> dict:
    with _refresh_lock:
        return _refresh_status.get(ticker.upper(), {'status': 'idle', 'progress': 0, 'message': ''})


def _set_status(ticker: str, status: str, progress: int, message: str):
    with _refresh_lock:
        _refresh_status[ticker.upper()] = {
            'status': status,
            'progress': progress,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'timestamp_ms': int(time.time() * 1000)
        }


def check_ohlcv_freshness(engine, ticker: str, is_tw: bool) -> tuple[bool, str | None]:
    """
    Returns (is_fresh, last_date_str).
    is_fresh = True if last date >= yesterday (accounting for weekends).
    """
    table = 'stock_cost' if is_tw else 'stock_cost_us'
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT MAX(Date) FROM {table} WHERE number = :num"),
                {'num': ticker}
            ).fetchone()
            if result and result[0]:
                # 統一轉換為 datetime.date，避免 datetime.datetime vs date 型別衝突
                raw = result[0]
                if hasattr(raw, 'date'):
                    last_date = raw.date() if callable(raw.date) else raw
                elif isinstance(raw, str):
                    last_date = pd.to_datetime(raw).date()
                else:
                    last_date = raw
                # Business days: fresh if last date >= last weekday
                today = date.today()
                days_back = 1
                while (today - timedelta(days=days_back)).weekday() >= 5:
                    days_back += 1
                last_biz_day = today - timedelta(days=days_back)
                return last_date >= last_biz_day, str(last_date)
    except Exception as e:
        logger.warning(f"OHLCV freshness check failed for {ticker}: {e}")
    return False, None


def check_investor_freshness(engine, ticker: str, is_tw: bool) -> tuple[bool, str | None]:
    """Check if investor/chip data is fresh."""
    if is_tw:
        table = 'stock_investor'
        date_col = '日期'
    else:
        table = 'stock_investor_us'
        date_col = 'date'

    try:
        with engine.connect() as conn:
            if is_tw:
                result = conn.execute(
                    text(f"SELECT MAX(`{date_col}`) FROM {table}")
                ).fetchone()
            else:
                result = conn.execute(
                    text(f"SELECT MAX(`{date_col}`) FROM {table} WHERE ticker = :num"),
                    {'num': ticker.upper()}
                ).fetchone()

            if result and result[0]:
                last_date = result[0] if isinstance(result[0], date) else pd.Timestamp(result[0]).date()
                # Investor data is quarterly for US, daily for TW
                threshold = timedelta(days=7) if is_tw else timedelta(days=90)
                is_fresh = (date.today() - last_date) < threshold
                return is_fresh, str(last_date)
    except Exception as e:
        logger.warning(f"Investor freshness check failed ({ticker}, tw={is_tw}): {e}")
    return False, None


def refresh_data_background(ticker: str, is_tw: bool):
    """
    Background thread: refresh all stale data for a given ticker.
    Updates _refresh_status progressively.
    """
    import pandas as pd
    from stock_Django.mySQL_OP import OP_Fun

    ticker = ticker.upper()
    _set_status(ticker, 'running', 0, '開始更新資料...')
    sql = OP_Fun()

    try:
        # --- Step 1: OHLCV (30%) ---
        _set_status(ticker, 'running', 5, f'正在更新 {ticker} 股價...')
        try:
            if is_tw:
                from stock_Django.stock_cost import StockCostManager
                mgr = StockCostManager()
                # 確保傳入的是帶後綴的代碼
                mgr.update_single_ticker(ticker)
                _set_status(ticker, 'running', 30, f'✅ 台股股價更新完成')
            else:
                from stock_Django.mySQL_OP import OP_Fun
                import yfinance as yf
                import requests
                ticker_obj = yf.Ticker(ticker, session=requests.Session())
                hist = ticker_obj.history(period="5d")
                if not hist.empty:
                    sql_op = OP_Fun()
                    hist = hist.reset_index().rename(columns={'Date':'Date','Open':'Open','High':'High','Low':'Low','Close':'Close','Volume':'Volume'})
                    hist['number'] = ticker
                    sql_op.upload_all(hist, 'stock_cost_us')
                    _set_status(ticker, 'running', 30, f'✅ 美股股價更新完成')
                else:
                    _set_status(ticker, 'running', 30, f'⚠️ 美股股價查無資料')
        except Exception as e:
            logger.error(f"OHLCV refresh error for {ticker}: {e}")
            _set_status(ticker, 'running', 30, f'⚠️ OHLCV 更新失敗: {e}')

        # --- Step 2: Investor/Chip data (70%) ---
        _set_status(ticker, 'running', 35, '正在更新法人買賣超資料...')
        try:
            if is_tw:
                # Use stock_investor Selenium scraper for TW
                from stock_Django.stock_investor import StockInvestorManager
                inv_mgr = StockInvestorManager()
                inv_mgr.update_investor_data()
                _set_status(ticker, 'running', 70, '✅ 台股三大法人資料更新完成')
            else:
                # Use stock_investor_us for US
                from stock_Django.stock_investor_us import USStockInvestorManager
                us_mgr = USStockInvestorManager()
                us_mgr.update_investor_db([ticker])
                _set_status(ticker, 'running', 70, '✅ 美股法人持股更新完成')
        except Exception as e:
            logger.error(f"Investor refresh error for {ticker}: {e}")
            _set_status(ticker, 'running', 70, f'⚠️ 法人資料更新失敗: {e}')

        # --- Step 3: Financial Data (90%) ---
        _set_status(ticker, 'running', 75, '正在更新財報基本面資料 (DB-First)...')
        try:
            import concurrent.futures
            from valuation.services.financial_data import FinancialDataLoader
            loader = FinancialDataLoader(ticker)
            # Use executor with timeout to prevent hanging on slow scrapers (Playwright)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(loader.ensure_data_freshness)
                try:
                    future.result(timeout=30) # 30s limit for scraping
                    _set_status(ticker, 'running', 90, '✅ 財報資料更新完成')
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Financial data refresh timed out for {ticker}")
                    _set_status(ticker, 'running', 90, '⚠️ 財報更新逾時 (已跳過)')
        except Exception as e:
            logger.error(f"Financial refresh error for {ticker}: {e}")
            _set_status(ticker, 'running', 90, f'⚠️ 財報更新失敗: {e}')

        _set_status(ticker, 'done', 100, f'✅ {ticker} 所有資料更新完成')

    except Exception as e:
        logger.error(f"Background refresh failed for {ticker}: {e}")
        _set_status(ticker, 'error', 0, f'❌ 更新錯誤: {e}')


def trigger_refresh_if_stale(ticker: str, is_tw: bool, engine) -> bool:
    """
    Check freshness and trigger background refresh if stale.
    Checks both price (OHLCV) and investor (Chips) data.
    Returns True if refresh was triggered.
    """
    ticker = ticker.upper()

    # Don't re-trigger if already running
    current = get_refresh_status(ticker)
    if current.get('status') == 'running':
        return False

    ohlcv_fresh, ohlcv_date = check_ohlcv_freshness(engine, ticker, is_tw)
    investor_fresh, inv_date = check_investor_freshness(engine, ticker, is_tw)

    if not ohlcv_fresh or not investor_fresh:
        reason = []
        if not ohlcv_fresh: reason.append(f"OHLCV stale (last: {ohlcv_date})")
        if not investor_fresh: reason.append(f"Investor stale (last: {inv_date})")
        
        logger.info(f"[Freshness] {ticker} data needs refresh: {', '.join(reason)}. Triggering background refresh.")
        thread = threading.Thread(
            target=refresh_data_background,
            args=(ticker, is_tw),
            daemon=True
        )
        thread.start()
        return True

    return False

# Global cooldown tracker
_news_cooldown = {}

def refresh_news_background(ticker: str, market: str, limit: int = 1000):
    """Background thread to fetch news from 鉅亨網."""
    from stock_Django.news_scraper_cnyes import CnyesScraper
    from stock_Django.news_excel import NewsExcelManager
    
    ticker = ticker.upper()
    _set_status(ticker, 'running', 10, f'正在從 鉅亨網 抓取 {ticker} 最新新聞...')
    
    try:
        scraper = CnyesScraper()
        news_mgr = NewsExcelManager()
        
        # Scrape
        results = scraper.fetch_news(ticker.replace('.TW', ''), market=market, limit=limit)
        
        if results:
            _set_status(ticker, 'running', 80, f'抓取完成，正在儲存 {len(results)} 則新聞並進行 BERT 情緒分析...')
            # NOTE: Sentiment is analyzed inside scraper.fetch_news calling _analyze_sentiment
            news_mgr.write_news(ticker, results)
            _set_status(ticker, 'done', 100, f'✅ {ticker} 新聞更新完成，共新增 {len(results)} 則資料')
        else:
            _set_status(ticker, 'done', 100, f'✅ {ticker} 抓取完成，但未發現新新聞')
            
    except Exception as e:
        logger.error(f"News refresh failed for {ticker}: {e}")
        _set_status(ticker, 'error', 0, f'❌ 新聞更新錯誤: {e}')


def trigger_news_refresh(ticker: str, limit: int = 1000) -> bool:
    """Trigger news refresh if not already running and cooldown passed."""
    ticker = ticker.upper()
    
    # 1. Check if already running
    current = get_refresh_status(ticker)
    if current.get('status') == 'running' and '新聞' in current.get('message', ''):
        return False
        
    # 2. Check Cooldown (5 minutes)
    now = time.time()
    if ticker in _news_cooldown and (now - _news_cooldown[ticker]) < 300:
        logger.info(f"[Cooldown] News refresh for {ticker} skipped (too soon).")
        return False
    
    _news_cooldown[ticker] = now
    
    is_tw = ticker.isdigit() or ".TW" in ticker
    market = 'tw' if is_tw else 'us'
    
    thread = threading.Thread(
        target=refresh_news_background,
        args=(ticker, market, limit),
        daemon=True
    )
    thread.start()
    return True
# Global cooldown tracker for investor refresh
_investor_cooldown = {}

def refresh_tw_investor_background():
    """Background thread to refresh both TWSE and TPEx investor data."""
    from stock_Django.stock_investor import StockInvestorManager
    from stock_Django.stock_investor_tpex import TPExInvestorManager
    
    label = "TW_ALL"
    _set_status(label, 'running', 10, '正在初始化台灣市場(上市+上櫃)法人更新任務...')
    
    try:
        # 1. TWSE (Listed)
        _set_status(label, 'running', 20, '1/2: 正在更新台灣上市股票法人資料 (Selenium)...')
        twse_mgr = StockInvestorManager()
        twse_mgr.update_investor_data()
        
        # 2. TPEx (OTC)
        _set_status(label, 'running', 60, '2/2: 正在更新台灣上櫃股票法人資料 (API)...')
        tpex_mgr = TPExInvestorManager()
        tpex_mgr.update_tpex_investor()
        
        _set_status(label, 'done', 100, '✅ 台灣市場法人資料更新成功 (上市+上櫃)')
    except Exception as e:
        logger.error(f"TW Investor refresh failed: {e}")
        _set_status(label, 'error', 0, f'❌ 更新失敗: {e}')


def refresh_us_investor_background(tickers: list = None):
    """Background thread to refresh US institutional data for given tickers."""
    from stock_Django.stock_investor_us import USStockInvestorManager
    
    if not tickers:
        tickers = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'AMZN', 'GOOGL', 'META', 'AMD', 'TSM', 'AVGO']
        
    label = "US_ALL"
    _set_status(label, 'running', 10, f'正在準備更新 {len(tickers)} 檔美股法人持股...')
    
    try:
        us_mgr = USStockInvestorManager()
        for i, ticker in enumerate(tickers):
            progress = 10 + int((i / len(tickers)) * 80)
            _set_status(label, 'running', progress, f'正在更新 {ticker} ({i+1}/{len(tickers)})...')
            us_mgr.update_investor_db([ticker])
            
        _set_status(label, 'done', 100, f'✅ 美股市場 {len(tickers)} 檔法人持股更新完成')
    except Exception as e:
        logger.error(f"US Investor refresh failed: {e}")
        _set_status(label, 'error', 0, f'❌ 更新失敗: {e}')
