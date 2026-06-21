import logging
import yfinance as yf
import pandas as pd
import datetime
import requests
import re
from typing import Dict, Any
from django.core.cache import cache

logger = logging.getLogger(__name__)

class ETFValuationService:
    @staticmethod
    def calculate_etf_valuation(ticker_symbol: str, loader) -> Dict[str, Any]:
        """
        計算 ETF 的估值數據，包括即時折溢價率與歷史配息殖利率。
        使用免驗證的 Chart API 獲取配息與市價，並採用「快取優先 + 多重來源備援 + 降級防禦」三階段機制獲取淨值 (NAV)。
        
        傳入參數:
          ticker_symbol (str): ETF 代號，如 "0050.TW"、"SPY"
          loader (FinancialDataLoader): 財務資料載入器實例
          
        回傳結構:
          Dict[str, Any]: 包含 symbol, name, current_price, nav_price, discount_premium_pct, dividend_yield, recent_dividends, nav_status 等資訊
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
            "nav_status": "realtime",
            "error": None
        }

        try:
            logger.info(f"[ETF Valuation] Starting ETF valuation for {full_symbol} using Chart API")
            
            # --- 第一階段：直接發送輕量級請求至免驗證 Chart API 獲取市價與歷史配息 ---
            chart_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{full_symbol}"
            params = {
                "interval": "1d",
                "period1": 0,
                "period2": 9999999999,
                "events": "div"
            }
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            }
            
            chart_data = {}
            try:
                resp = requests.get(chart_url, params=params, headers=headers, timeout=10)
                if resp.status_code == 200:
                    chart_data = resp.json().get("chart", {}).get("result", [{}])[0]
                else:
                    logger.warning(f"[ETF Valuation] Chart API status code: {resp.status_code} for {full_symbol}")
            except Exception as e_chart:
                logger.error(f"[ETF Valuation] Chart API request failed for {full_symbol}: {e_chart}")
            
            # 解析 Chart API 結果
            meta = chart_data.get("meta", {})
            result["name"] = meta.get("longName") or meta.get("shortName") or ticker_symbol
            
            # 獲取市價
            price = meta.get("regularMarketPrice") or meta.get("previousClose")
            if not price or price <= 0:
                price = loader.get_market_price()
            result["current_price"] = float(price) if price else 0.0
            
            # 整理與計算配息歷史
            events = chart_data.get("events", {})
            dividends = events.get("dividends", {})
            
            if dividends:
                # 排序配息事件由近到遠
                sorted_divs = sorted(dividends.values(), key=lambda x: x["date"], reverse=True)
                
                # 計算最近 365 天的配息總和
                now = datetime.datetime.now(datetime.timezone.utc)
                one_year_ago = now - datetime.timedelta(days=365)
                recent_1y_sum = 0.0
                history_list = []
                
                for item in sorted_divs:
                    dt = datetime.datetime.fromtimestamp(item["date"], datetime.timezone.utc)
                    if dt >= one_year_ago:
                        recent_1y_sum += float(item["amount"])
                    if len(history_list) < 12:
                        history_list.append({
                            "date": dt.strftime("%Y-%m-%d"),
                            "amount": float(item["amount"])
                        })
                
                result["recent_dividends"] = history_list
                if recent_1y_sum > 0 and price > 0:
                    result["dividend_yield"] = float((recent_1y_sum / price) * 100)
            else:
                logger.info(f"[ETF Valuation] No dividends found in Chart API for {full_symbol}")

            # --- 第二階段：高可靠淨值 (NAV) 獲取邏輯 ---
            cache_key = f"etf_nav_{full_symbol}"
            longterm_cache_key = f"etf_nav_longterm_{full_symbol}"
            
            nav_data = cache.get(cache_key)
            nav_price = None
            nav_status = "realtime"
            
            if nav_data:
                nav_price = nav_data.get("nav_price")
                nav_status = "cached"
                logger.info(f"[ETF Valuation] NAV cache hit for {full_symbol}: {nav_price}")
            else:
                logger.info(f"[ETF Valuation] NAV cache miss for {full_symbol}, fetching from online sources")
                
                # 備援 A：優先使用 yfinance 接口獲取 navPrice
                try:
                    ticker = yf.Ticker(full_symbol, session=loader.yf_session)
                    info = ticker.info
                    nav_val = info.get("navPrice")
                    if nav_val:
                        nav_price = float(nav_val)
                        # 更新快取
                        cache.set(cache_key, {"nav_price": nav_price}, 43200)       # 12 小時快取
                        cache.set(longterm_cache_key, {"nav_price": nav_price}, 2592000) # 30 天長期備份
                        logger.info(f"[ETF Valuation] Fetched NAV from yfinance for {full_symbol}: {nav_price}")
                except Exception as e_yf:
                    logger.warning(f"[ETF Valuation] yfinance NAV fetch failed for {full_symbol}: {e_yf}")
                
                # 備援 B：台股 ETF 當 yfinance 遭遇限流或失敗時，爬取鉅亨網 (Anue) 個股頁面
                if nav_price is None and (market.upper() == "TW" or ".TW" in full_symbol or ".TWO" in full_symbol):
                    try:
                        symbol_no = ticker_symbol.replace(".TW", "").replace(".TWO", "")
                        cnyes_url = f"https://www.cnyes.com/twstock/{symbol_no}"
                        cnyes_headers = {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                        }
                        cnyes_resp = requests.get(cnyes_url, headers=cnyes_headers, timeout=5)
                        if cnyes_resp.status_code == 200:
                            # 匹配鉅亨網頁面上的「每股淨值」或「昨日淨值」對應數值
                            # 例如：<span class="jsx-4238438383 label">每股淨值</span><span class="jsx-4238438383 value">186.25</span>
                            match = re.search(r'label">(?:每股)?淨值</span>\s*<span[^>]*value[^>]*>([\d\.]+)</span>', cnyes_resp.text)
                            if match:
                                nav_price = float(match.group(1))
                                cache.set(cache_key, {"nav_price": nav_price}, 43200)
                                cache.set(longterm_cache_key, {"nav_price": nav_price}, 2592000)
                                logger.info(f"[ETF Valuation] Fetched NAV from cnyes scraper for {full_symbol}: {nav_price}")
                    except Exception as e_cnyes:
                        logger.error(f"[ETF Valuation] cnyes NAV fetch failed for {full_symbol}: {e_cnyes}")
                
                # 備援 C：從長期備份快取取得歷史上一次成功的 NAV，保障折溢價率必定呈現
                if nav_price is None:
                    longterm_data = cache.get(longterm_cache_key)
                    if longterm_data:
                        nav_price = longterm_data.get("nav_price")
                        nav_status = "fallback_cached"
                        logger.warning(f"[ETF Valuation] All fetch failed. Fallback to long-term cached NAV for {full_symbol}: {nav_price}")
                    else:
                        # 備援 D：若完全沒有任何歷史快取，則使用市價估算作為保底 NAV，折溢價率會為 0
                        price_val = result["current_price"]
                        if price_val > 0:
                            nav_price = price_val
                            nav_status = "fallback_estimate"
                            logger.warning(f"[ETF Valuation] All fetch failed and no cached data. Fallback to current price for {full_symbol}: {nav_price}")

            # --- 第三階段：計算折溢價並回傳 ---
            if nav_price:
                result["nav_price"] = nav_price
                result["nav_status"] = nav_status
                if price and price > 0:
                    result["discount_premium_pct"] = float((price - nav_price) / nav_price * 100)

        except Exception as e:
            logger.error(f"[ETF Valuation] Failed to calculate ETF valuation for {full_symbol}: {e}", exc_info=True)
            result["error"] = str(e)
            if result["current_price"] <= 0:
                result["current_price"] = loader.get_market_price()

        return result
