import logging
import yfinance as yf
import pandas as pd
import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ETFValuationService:
    @staticmethod
    def calculate_etf_valuation(ticker_symbol: str, loader) -> Dict[str, Any]:
        """
        計算 ETF 的估值數據（折溢價與歷史配息殖利率）
        """
        full_symbol = loader.full_symbol
        market = loader.market
        
        # 預設回傳結構
        result = {
            "symbol": ticker_symbol,
            "market": market.upper(),
            "is_etf": True,
            "current_price": 0.0,
            "nav_price": None,
            "discount_premium_pct": None,
            "dividend_yield": None,
            "recent_dividends": [],
            "name": ticker_symbol,
            "error": None
        }

        try:
            logger.info(f"[ETF Valuation] Starting ETF valuation for {full_symbol}")
            ticker = yf.Ticker(full_symbol, session=loader.yf_session)
            info = ticker.info
            
            # 1. 取得 ETF 基本資訊
            result["name"] = info.get("longName") or info.get("shortName") or ticker_symbol
            
            # 2. 取得市價與淨值 (NAV)
            price = info.get("regularMarketPrice") or info.get("previousClose") or info.get("currentPrice") or info.get("open")
            if not price or price <= 0:
                # Fallback to local DB price
                price = loader.get_market_price()
            result["current_price"] = float(price) if price else 0.0
            
            nav = info.get("navPrice")
            if nav:
                result["nav_price"] = float(nav)
                if price and price > 0:
                    result["discount_premium_pct"] = float((price - nav) / nav * 100)
            else:
                logger.warning(f"[ETF Valuation] navPrice not found in yfinance for {full_symbol}")
            
            # 3. 取得歷史殖利率與配息
            # yfinance 提供了 dividends 屬性
            dividends = ticker.dividends
            if dividends is not None and not dividends.empty:
                # 排序並過濾最近幾年
                dividends = dividends.sort_index(ascending=False)
                
                # 計算最近 1 年 (365 天) 的配息總和
                now = datetime.datetime.now(datetime.timezone.utc)
                one_year_ago = now - datetime.timedelta(days=365)
                
                recent_1y_divs = dividends[dividends.index >= one_year_ago]
                total_1y_div = float(recent_1y_divs.sum())
                
                if total_1y_div > 0 and price > 0:
                    result["dividend_yield"] = float((total_1y_div / price) * 100)
                else:
                    # Fallback to info yield
                    info_yield = info.get("yield") or info.get("dividendYield")
                    if info_yield:
                        result["dividend_yield"] = float(info_yield * 100)
                
                # 整理配息歷史紀錄送給前端
                # 我們最多取前 12 筆配息紀錄，並將 Datetime 轉成 string YYYY-MM-DD
                history_list = []
                for dt, val in dividends.head(12).items():
                    history_list.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "amount": float(val)
                    })
                result["recent_dividends"] = history_list
            else:
                # Fallback yield from info
                info_yield = info.get("yield") or info.get("dividendYield")
                if info_yield:
                    result["dividend_yield"] = float(info_yield * 100)

        except Exception as e:
            logger.error(f"[ETF Valuation] Failed to calculate ETF valuation for {full_symbol}: {e}")
            result["error"] = str(e)
            # 保底價格
            if result["current_price"] <= 0:
                result["current_price"] = loader.get_market_price()

        return result
