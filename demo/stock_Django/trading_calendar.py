# -*- coding: utf-8 -*-
"""
交易日曆服務模組 (Trading Calendar Service)
負責管理台股與美股之交易日、開收盤時段，以及新聞時序 T+1 盤後對齊，消除前瞻偏誤。
"""

import datetime
import logging
import unicodedata
from typing import Dict, List, Optional, Set, Tuple, Union
import pytz
import pandas as pd
from django.db import connection

logger = logging.getLogger(__name__)

# 時區定義
TZ_TW = pytz.timezone("Asia/Taipei")
TZ_US = pytz.timezone("America/New_York")

# 市場交易時間常規定義
TW_MARKET_OPEN = datetime.time(9, 0)
TW_MARKET_CLOSE = datetime.time(13, 30)
TW_NEWS_CUTOFF = datetime.time(13, 30)  # 超過 13:30 盤後新聞強制遞延至 T+1

US_MARKET_OPEN = datetime.time(9, 30)
US_MARKET_CLOSE = datetime.time(16, 0)
US_NEWS_CUTOFF = datetime.time(16, 0)


def sanitize_string(text: Optional[str]) -> str:
    """
    字串輸入安全清洗：
    1. Unicode NFKC 正規化。
    2. 移除空字元 (\\x00) 防止 C-String 截斷。
    """
    if not text or not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    cleaned = normalized.replace("\x00", "")
    return "".join(ch for ch in cleaned if ch.isprintable() or ch in ("\n", "\t", " "))


class TradingCalendarService:
    """
    交易日曆核心服務。
    
    職責：
    1. 判定特定日期在台股或美股是否為有效交易日。
    2. 依新聞時間戳判定時序屬性（盤中、盤前、盤後、假日）。
    3. 將新聞發布時間對齊至有效交易日（盤後遞延至 T+1，假日順延）。
    4. 同步與快取 dim_trading_calendar 維度資料。
    """
    _instance = None
    _memory_cache: Dict[str, Set[datetime.date]] = {"tw": set(), "us": set()}
    _cache_loaded: Dict[str, bool] = {"tw": False, "us": False}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TradingCalendarService, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def get_market_tz(market: str = "tw") -> pytz.BaseTzInfo:
        """取得指定市場時區"""
        m = market.lower()
        if m in ("us", "nyse", "nasdaq"):
            return TZ_US
        return TZ_TW

    @staticmethod
    def get_market_hours(market: str = "tw") -> Tuple[datetime.time, datetime.time, datetime.time]:
        """取得指定市場之 (開盤, 收盤, 盤後截區線)"""
        m = market.lower()
        if m in ("us", "nyse", "nasdaq"):
            return US_MARKET_OPEN, US_MARKET_CLOSE, US_NEWS_CUTOFF
        return TW_MARKET_OPEN, TW_MARKET_CLOSE, TW_NEWS_CUTOFF

    def _ensure_cache_loaded(self, market: str = "tw"):
        """從資料庫載入交易日列表至記憶體快取"""
        m = market.lower()
        if self._cache_loaded.get(m, False):
            return

        try:
            sql = "SELECT date FROM dim_trading_calendar WHERE market = %s AND is_trading_day = 1"
            with connection.cursor() as cursor:
                cursor.execute(sql, [m])
                rows = cursor.fetchall()
                trading_days = {row[0] for row in rows}
                self._memory_cache[m] = trading_days
                self._cache_loaded[m] = True
                logger.info(f"成功載入 {len(trading_days)} 筆 {m.upper()} 交易日快取")
        except Exception as e:
            logger.warning(f"從 dim_trading_calendar 載入快取失敗 ({m}): {e}，將使用動態規則備援")

    def is_trading_day(self, target_date: Union[datetime.date, datetime.datetime, str], market: str = "tw") -> bool:
        """
        判定指定日期是否為交易日。
        優先查詢記憶體快取，快取未命中則查庫或利用 pandas_market_calendars 判斷。
        """
        m = market.lower()
        d = self._normalize_date(target_date)

        self._ensure_cache_loaded(m)
        if self._cache_loaded.get(m, False) and self._memory_cache[m]:
            return d in self._memory_cache[m]

        # 備援 1：利用 pandas_market_calendars 判斷
        try:
            import pandas_market_calendars as mcal
            cal_name = "XTAI" if m == "tw" else "NYSE"
            cal = mcal.get_calendar(cal_name)
            d_str = d.strftime("%Y-%m-%d")
            sched = cal.schedule(start_date=d_str, end_date=d_str)
            return not sched.empty
        except Exception as e:
            logger.warning(f"動態日曆排程檢查失敗: {e}，使用平日排除週末規則")
            return d.weekday() < 5

    def get_next_trading_day(self, target_date: Union[datetime.date, datetime.datetime, str], market: str = "tw") -> datetime.date:
        """取得下一個有效交易日 (T+1)"""
        m = market.lower()
        curr = self._normalize_date(target_date) + datetime.timedelta(days=1)
        max_search_days = 30
        for _ in range(max_search_days):
            if self.is_trading_day(curr, market=m):
                return curr
            curr += datetime.timedelta(days=1)
        return curr

    def get_previous_trading_day(self, target_date: Union[datetime.date, datetime.datetime, str], market: str = "tw") -> datetime.date:
        """取得前一個有效交易日 (T-1)"""
        m = market.lower()
        curr = self._normalize_date(target_date) - datetime.timedelta(days=1)
        max_search_days = 30
        for _ in range(max_search_days):
            if self.is_trading_day(curr, market=m):
                return curr
            curr -= datetime.timedelta(days=1)
        return curr

    def get_trading_days(self, market: str, start_date: Union[datetime.date, str], end_date: Union[datetime.date, str]) -> List[datetime.date]:
        """取得指定區間內之所有有效交易日清單（遞增排序）"""
        m = market.lower()
        s_date = self._normalize_date(start_date)
        e_date = self._normalize_date(end_date)
        if s_date > e_date:
            return []

        self._ensure_cache_loaded(m)
        if self._cache_loaded.get(m, False) and self._memory_cache[m]:
            days = [d for d in self._memory_cache[m] if s_date <= d <= e_date]
            return sorted(days)

        # 備援：利用 mcal
        try:
            import pandas_market_calendars as mcal
            cal_name = "XTAI" if m == "tw" else "NYSE"
            cal = mcal.get_calendar(cal_name)
            sched = cal.schedule(start_date=s_date.strftime("%Y-%m-%d"), end_date=e_date.strftime("%Y-%m-%d"))
            return [idx.date() for idx in sched.index]
        except Exception:
            # 簡單 weekday 備援
            res = []
            curr = s_date
            while curr <= e_date:
                if curr.weekday() < 5:
                    res.append(curr)
                curr += datetime.timedelta(days=1)
            return res

    def classify_news_timestamp(self, ts: Union[datetime.datetime, str], market: str = "tw") -> str:
        """
        分類新聞時間戳為：
        - 'intraday': 盤中正常交易時段
        - 'after_hours': 盤後時段（超過收盤截區線）
        - 'pre_market': 盤前時段（開盤前）
        - 'holiday': 休市日或週末
        """
        m = market.lower()
        dt = self._normalize_datetime(ts, market=m)
        d = dt.date()
        t = dt.time()

        if not self.is_trading_day(d, market=m):
            return "holiday"

        open_time, close_time, cutoff_time = self.get_market_hours(m)
        if t < open_time:
            return "pre_market"
        elif open_time <= t <= close_time:
            return "intraday"
        else:
            return "after_hours"

    def align_to_trading_day(self, news_ts: Union[datetime.datetime, str], market: str = "tw") -> datetime.date:
        """
        將新聞發布時間對齊至對應之有效交易日（防前瞻偏誤核心算式）：
        1. 盤中新聞 (09:00 ~ 13:30) 與盤前新聞 -> 當日交易日 T。
        2. 盤後新聞 (> 13:30) -> 遞延至次一交易日 T+1。
        3. 休市日或週末發布之新聞 -> 順延至開盤後之次一有效交易日。
        """
        m = market.lower()
        dt = self._normalize_datetime(news_ts, market=m)
        d = dt.date()
        t = dt.time()

        _, _, cutoff_time = self.get_market_hours(m)

        # 狀況 1：若當日非交易日（休市/週末），直接順延至次一交易日
        if not self.is_trading_day(d, market=m):
            return self.get_next_trading_day(d, market=m)

        # 狀況 2：若當日為交易日，但發布時間已逾收盤截區線，遞延至 T+1 交易日
        if t >= cutoff_time:
            return self.get_next_trading_day(d, market=m)

        # 狀況 3：盤中或盤前，歸屬當日交易日 T
        return d

    def sync_calendar_table(self, market: str = "tw", start_year: int = 2020, end_year: int = 2027) -> int:
        """
        將指定市場之歷年交易日排程同步寫入 MySQL dim_trading_calendar 維度表。
        使用 parameterized batch upsert 保證 SQL 注入安全與高性能。
        """
        import pandas_market_calendars as mcal
        m = market.lower()
        cal_name = "XTAI" if m == "tw" else "NYSE"
        open_time, close_time, cutoff_time = self.get_market_hours(m)
        tz = self.get_market_tz(m)

        logger.info(f"開始同步 {m.upper()} 交易日曆 ({start_year} - {end_year})...")
        cal = mcal.get_calendar(cal_name)
        start_str = f"{start_year}-01-01"
        end_str = f"{end_year}-12-31"

        sched = cal.schedule(start_date=start_str, end_date=end_str)
        trading_dates = set(idx.date() for idx in sched.index)

        # 產生該區間內的所有自然日紀錄
        cur_date = datetime.date(start_year, 1, 1)
        end_date = datetime.date(end_year, 12, 31)

        batch_records = []
        while cur_date <= end_date:
            is_trade = cur_date in trading_dates
            batch_records.append((
                cur_date,
                m,
                1 if is_trade else 0,
                open_time if is_trade else None,
                close_time if is_trade else None,
                cutoff_time if is_trade else None,
            ))
            cur_date += datetime.timedelta(days=1)

        upsert_sql = """
            INSERT INTO dim_trading_calendar (date, market, is_trading_day, market_open, market_close, news_cutoff)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                is_trading_day = VALUES(is_trading_day),
                market_open = VALUES(market_open),
                market_close = VALUES(market_close),
                news_cutoff = VALUES(news_cutoff);
        """

        total_inserted = 0
        chunk_size = 500
        with connection.cursor() as cursor:
            for i in range(0, len(batch_records), chunk_size):
                chunk = batch_records[i : i + chunk_size]
                cursor.executemany(upsert_sql, chunk)
                total_inserted += len(chunk)

        # 重新整理記憶體快取
        self._cache_loaded[m] = False
        self._ensure_cache_loaded(m)
        logger.info(f"成功同步 {total_inserted} 筆 {m.upper()} 交易日曆資料")
        return total_inserted

    @staticmethod
    def _normalize_date(d_val: Union[datetime.date, datetime.datetime, str]) -> datetime.date:
        """轉換各類日期格式為 datetime.date"""
        if isinstance(d_val, datetime.datetime):
            return d_val.date()
        if isinstance(d_val, datetime.date):
            return d_val
        if isinstance(d_val, str):
            clean_str = sanitize_string(d_val).split("T")[0].split(" ")[0]
            return datetime.datetime.strptime(clean_str, "%Y-%m-%d").date()
        raise ValueError(f"無法解析之日期格式: {d_val}")

    @classmethod
    def _normalize_datetime(cls, dt_val: Union[datetime.datetime, str], market: str = "tw") -> datetime.datetime:
        """轉換各類時間格式為特定市場時區感知之 datetime.datetime"""
        market_tz = cls.get_market_tz(market)
        if isinstance(dt_val, str):
            clean_str = sanitize_string(dt_val)
            parsed_dt = pd.to_datetime(clean_str)
            if pd.isna(parsed_dt):
                parsed_dt = datetime.datetime.now()
            else:
                parsed_dt = parsed_dt.to_pydatetime()
            dt_val = parsed_dt

        if isinstance(dt_val, datetime.datetime):
            if dt_val.tzinfo is None:
                # 若為 naive datetime，假設其已為該市場本地時間
                return market_tz.localize(dt_val)
            else:
                # 若已有時區資訊，轉換至該市場時區（自動適應夏令/冬令時）
                return dt_val.astimezone(market_tz)

        if isinstance(dt_val, datetime.date):
            # 預設對齊為開盤前 00:00:00
            naive = datetime.datetime.combine(dt_val, datetime.time(0, 0))
            return market_tz.localize(naive)

        raise ValueError(f"無法解析之時間戳: {dt_val}")
