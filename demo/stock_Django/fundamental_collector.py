# -*- coding: utf-8 -*-
import math
import logging
import pandas as pd
from datetime import datetime
from sqlalchemy import text

logger = logging.getLogger(__name__)

class FundamentalDataCollector:
    """
    基本面財務數據蒐集器 (FundamentalDataCollector)
    
    職責：
    1. 針對台股與美股，自本機 MySQL (financial_raw_tw, financial_raw_us, stock_metrics) 提取純基本面指標。
    2. 納入 11 項核心指標：PE, EPS, PB, ROE, 毛利率, 營業利益率, 淨利率, 負債比率, 自由現金流, PEG, 月營收動能。
    3. 全面落實防禦機制：除零防護、NaN / Inf 數值清洗、資料庫異常安全降級 (Fallback)。
    4. 標註數據來源與財報時效性。
    """

    @staticmethod
    def _clean_num(val, default="N/A", round_digits=2):
        """數值防禦清洗：過濾 None, NaN, Inf，確保 JSON 序列化安全與介面展示美觀"""
        if val is None:
            return default
        try:
            float_val = float(val)
            if math.isnan(float_val) or math.isinf(float_val):
                return default
            return round(float_val, round_digits)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _format_large_number(val):
        """格式化大數值 (M/B/T)"""
        if val is None:
            return "N/A"
        try:
            num = float(val)
            if math.isnan(num) or math.isinf(num):
                return "N/A"
            abs_num = abs(num)
            if abs_num >= 1e12:
                return f"{num / 1e12:.2f}T"
            elif abs_num >= 1e9:
                return f"{num / 1e9:.2f}B"
            elif abs_num >= 1e6:
                return f"{num / 1e6:.2f}M"
            elif abs_num >= 1e3:
                return f"{num / 1e3:.2f}K"
            return f"{num:.2f}"
        except (ValueError, TypeError):
            return "N/A"

    @classmethod
    def collect_fundamentals(cls, ticker: str, valuation_symbol: str, is_tw: bool, latest_price: float = None, service = None) -> dict:
        """
        核心入口：收集完整基本面指標
        """
        # 預設回傳結構 (滿足 Evaluator 異常安全防禦要求)
        result = {
            "pe": "N/A",
            "eps": "N/A",
            "pb": "N/A",
            "roe": "N/A",
            "gross_margin": "N/A",
            "operating_margin": "N/A",
            "net_margin": "N/A",
            "debt_to_equity": "N/A",
            "free_cash_flow": "N/A",
            "revenue_growth": "N/A",
            "revenue_mom": "N/A",
            "revenue_yoy": "N/A",
            "peg_ratio": "N/A",
            "book_value": "N/A",
            "dividend_yield": "N/A",
            "market_cap": "N/A",
            "data_source": "TWSE 官方統計 + 本機 financial_raw_tw" if is_tw else "SEC EDGAR + 本機 financial_raw_us",
            "data_period": "最新季度 TTM 累積",
            "data_status": "正常"
        }

        try:
            if service is None:
                from .services import StockService
                service = StockService()

            engine = service.sql_op.engine

            if is_tw:
                cls._collect_tw_fundamentals(result, valuation_symbol, latest_price, engine)
            else:
                cls._collect_us_fundamentals(result, valuation_symbol, latest_price, engine)

        except Exception as e:
            logger.error(f"FundamentalDataCollector 蒐集 {valuation_symbol} 異常: {e}", exc_info=True)
            result["data_status"] = f"資料提取異常 (已觸發防禦預設值): {str(e)[:50]}"

        return result

    @classmethod
    def _collect_tw_fundamentals(cls, result: dict, valuation_symbol: str, latest_price: float, engine):
        """台股財務指標蒐集邏輯"""
        clean_sym = valuation_symbol.replace('.TWO', '').replace('.TW', '')

        # 1. 查詢 stock_metrics 即時/快取指標 (TWSE / TPEx CLI 同步成果)
        try:
            with engine.connect() as conn:
                metric_row = conn.execute(
                    text("SELECT pe, pb, dividend_yield, updated_at FROM stock_metrics WHERE symbol = :sym"),
                    {"sym": clean_sym}
                ).fetchone()
                if not metric_row:
                    metric_row = conn.execute(
                        text("SELECT pe, pb, dividend_yield, updated_at FROM stock_metrics WHERE symbol = :sym"),
                        {"sym": valuation_symbol}
                    ).fetchone()
                    
                if metric_row:
                    if metric_row[0] is not None:
                        result["pe"] = cls._clean_num(metric_row[0])
                    if metric_row[1] is not None:
                        result["pb"] = cls._clean_num(metric_row[1])
                    if metric_row[2] is not None:
                        result["dividend_yield"] = cls._clean_num(metric_row[2])
                    if metric_row[3]:
                        result["data_source"] = f"TWSE/TPEx 官方統計 (更新: {str(metric_row[3])[:10]})"
        except Exception as e_m:
            logger.warning(f"讀取台股 {clean_sym} stock_metrics 失敗: {e_m}")

        # 2. 查詢 financial_raw_tw 季度報表數據
        try:
            df_fin = pd.read_sql(
                "SELECT * FROM financial_raw_tw WHERE symbol = %(sym)s ORDER BY year DESC, quarter DESC LIMIT 3000",
                engine,
                params={"sym": clean_sym}
            )
            if df_fin.empty:
                result["data_status"] = "無歷史季報記錄 (待爬蟲回補)"
                return

            df_pivot = df_fin.pivot_table(
                index=['year', 'quarter'], 
                columns='item_name', 
                values='amount', 
                aggfunc='first'
            ).reset_index()
            df_pivot = df_pivot.sort_values(['year', 'quarter'], ascending=[False, False])

            if len(df_pivot) < 4:
                result["data_status"] = f"部分季報 ({len(df_pivot)} 季)"

            latest_row = df_pivot.iloc[0]
            latest_year = int(latest_row.get('year', 0))
            latest_q = int(latest_row.get('quarter', 0))
            result["data_period"] = f"{latest_year}年第{latest_q}季 (累積TTM)"

            col_map = {str(c).strip(): c for c in df_pivot.columns}

            def get_match(keys):
                for k in keys:
                    if k in col_map:
                        val = latest_row[col_map[k]]
                        if pd.notna(val):
                            return float(val) * 1000
                return None

            tw_rev_keys = ['營業收入合計', '營業收入', 'Operating Revenue', '~JXP', 'Revenue', '~JXp']
            tw_gross_keys = ['營業毛利（毛損）', '營業毛利', 'Gross Profit', '~Q]l^']
            tw_op_keys = ['營業利益（損失）', '營業利益', 'Operating profit', 'Operating Income']
            tw_ni_keys = ['本期淨利（淨損）', '本期淨利', 'Net Income', 'bQ]bl^']
            tw_eps_keys = ['基本每股盈餘', '每股盈餘', 'EPS', 'EjCj']
            tw_eq_keys = ['權益總額', '權益總計', '歸屬於母公司業主之權益總計']
            tw_cap_keys = ['普通股股本', '股本']
            tw_liab_keys = ['負債總額', '負債總計', 'Total Liabilities']
            tw_ocf_keys = ['營業活動之淨現金流入（流出）', '營業活動淨現金流量', 'Operating Cash Flow']
            tw_capex_keys = ['取得不動產、廠房及設備', '資本支出', 'Capital Expenditures']

            rev = get_match(tw_rev_keys)
            gross = get_match(tw_gross_keys)
            op_profit = get_match(tw_op_keys)
            net_income = get_match(tw_ni_keys)
            equity = get_match(tw_eq_keys)
            capital = get_match(tw_cap_keys)
            liabilities = get_match(tw_liab_keys)
            ocf = get_match(tw_ocf_keys)
            capex = get_match(tw_capex_keys)

            # 毛利率、營業利益率、淨利率
            if rev and rev > 0:
                if gross is not None:
                    result["gross_margin"] = cls._clean_num((gross / rev) * 100)
                if op_profit is not None:
                    result["operating_margin"] = cls._clean_num((op_profit / rev) * 100)
                if net_income is not None:
                    result["net_margin"] = cls._clean_num((net_income / rev) * 100)

            # 負債比率 (D/E)
            if liabilities is not None and equity and equity > 0:
                result["debt_to_equity"] = cls._clean_num((liabilities / equity) * 100)

            # 自由現金流 (FCF = OCF - CapEx)
            if ocf is not None:
                fcf_val = ocf - (abs(capex) if capex else 0)
                result["free_cash_flow"] = cls._format_large_number(fcf_val)

            # 每股淨值 BPS
            bps = None
            if equity and capital and capital > 0:
                bps = equity / (capital / 10)
                result["book_value"] = cls._clean_num(bps)

            # 計算 TTM EPS 與 ROE
            def get_ttm(keys):
                for k in keys:
                    if k in col_map:
                        series = df_pivot[col_map[k]].dropna()
                        if not series.empty:
                            last_4 = series.iloc[:4]
                            mx = last_4.max()
                            others = last_4.sum() - mx
                            if len(last_4) >= 4 and others > 0 and mx > 2.5 * (others / 3):
                                return float(mx)
                            return float(last_4.sum())
                return None

            eps_ttm = get_ttm(tw_eps_keys)
            if eps_ttm:
                result["eps"] = cls._clean_num(eps_ttm)

            ni_ttm = get_ttm(tw_ni_keys)
            if ni_ttm and equity and equity > 0:
                result["roe"] = cls._clean_num((ni_ttm * 1000 / equity) * 100)

            # 營收成長率 YoY (與去年同季比)
            try:
                if len(df_pivot) >= 5 and rev:
                    prev_y_rev = None
                    for k in tw_rev_keys:
                        if k in col_map:
                            val = df_pivot.iloc[4][col_map[k]]
                            if pd.notna(val) and float(val) > 0:
                                prev_y_rev = float(val) * 1000
                                break
                    if prev_y_rev and prev_y_rev > 0:
                        growth = ((rev / prev_y_rev) - 1) * 100
                        result["revenue_growth"] = cls._clean_num(growth)
            except Exception:
                pass

            # 若未從 stock_metrics 取得 PE / PB，則以當前股價動態計算
            if latest_price and latest_price > 0:
                if result["pe"] == "N/A" and eps_ttm and eps_ttm > 0:
                    result["pe"] = cls._clean_num(latest_price / eps_ttm)
                if result["pb"] == "N/A" and bps and bps > 0:
                    result["pb"] = cls._clean_num(latest_price / bps)
                if capital and capital > 0:
                    shares = (capital / 10)
                    result["market_cap"] = cls._format_large_number(shares * latest_price)

            # 計算 PEG 比率
            pe_val = result.get("pe")
            growth_val = result.get("revenue_growth")
            if pe_val != "N/A" and growth_val != "N/A" and isinstance(growth_val, (int, float)) and growth_val > 0:
                result["peg_ratio"] = cls._clean_num(float(pe_val) / float(growth_val))

            # 查詢月營收最新動能 (MoM / YoY)
            try:
                with engine.connect() as conn:
                    rev_row = conn.execute(
                        text("""
                            SELECT revenue_mom, revenue_yoy FROM stock_monthly_revenue 
                            WHERE symbol = :sym ORDER BY date DESC LIMIT 1
                        """),
                        {"sym": clean_sym}
                    ).fetchone()
                    if rev_row:
                        if rev_row[0] is not None:
                            result["revenue_mom"] = cls._clean_num(rev_row[0])
                        if rev_row[1] is not None:
                            result["revenue_yoy"] = cls._clean_num(rev_row[1])
            except Exception:
                pass

        except Exception as e_proc:
            logger.warning(f"處理台股 {clean_sym} 財務比率異常: {e_proc}")

    @classmethod
    def _collect_us_fundamentals(cls, result: dict, valuation_symbol: str, latest_price: float, engine):
        """美股財務指標蒐集邏輯 (SEC EDGAR / Macrotrends + financial_raw_us)"""
        try:
            df_fin_us = pd.read_sql(
                "SELECT * FROM financial_raw_us WHERE symbol = %(sym)s ORDER BY year DESC, quarter DESC",
                engine,
                params={"sym": valuation_symbol}
            )

            if df_fin_us.empty:
                result["data_status"] = "無歷史季報 (DB-First 缺失，嘗試 Fallback)"
            else:
                df_pivot = df_fin_us.pivot_table(
                    index=['year', 'quarter'], 
                    columns='item_name', 
                    values='amount', 
                    aggfunc='first'
                ).reset_index()
                df_pivot = df_pivot.sort_values(['year', 'quarter'], ascending=[False, False])

                if len(df_pivot) < 4:
                    result["data_status"] = f"部分季報 ({len(df_pivot)} 季)"

                latest_row = df_pivot.iloc[0]
                latest_year = int(latest_row.get('year', 0))
                latest_q = int(latest_row.get('quarter', 0))
                result["data_period"] = f"{latest_year}-Q{latest_q} (TTM)"

                def get_first_match(series, keys):
                    for k in keys:
                        if k in series and pd.notna(series[k]):
                            return float(series[k])
                    return None

                def get_ttm_sum(df_rect, keys):
                    for k in keys:
                        if k in df_rect.columns:
                            valid_q = df_rect[k].dropna()
                            if not valid_q.empty:
                                last_4 = valid_q.iloc[:4]
                                mx = last_4.max()
                                others = last_4.sum() - mx
                                if len(last_4) >= 4 and others > 0 and mx > 2.5 * (others / 3):
                                    return float(mx)
                                return float(last_4.sum())
                    return None

                # EPS
                eps_keys = ['EarningsPerShareBasic', 'EarningsPerShareDiluted', 'NetIncomeLossPerOutstandingLimitedPartnershipUnitBasic']
                eps_ttm = get_ttm_sum(df_pivot, eps_keys)
                if eps_ttm:
                    result["eps"] = cls._clean_num(eps_ttm)

                # 股東權益與 ROE
                ni_keys = ['NetIncomeLoss', 'NetIncomeLossAvailableToCommonStockholdersBasic', 'ProfitLoss']
                eq_keys = ['StockholdersEquity', 'Total Equity']
                ni_ttm = get_ttm_sum(df_pivot, ni_keys)
                equity = get_first_match(latest_row, eq_keys)
                if ni_ttm and equity and equity > 0:
                    result["roe"] = cls._clean_num((ni_ttm / equity) * 100)

                # 每股淨值 BPS
                shares_keys = ['CommonStockSharesOutstanding', 'WeightedAverageNumberOfSharesOutstandingBasic']
                shares = get_first_match(latest_row, shares_keys)
                bps = None
                if equity and shares and shares > 0:
                    bps = equity / shares
                    result["book_value"] = cls._clean_num(bps)

                # 營收與毛利
                rev_keys = ['Revenues', 'TotalRevenues', 'OperatingRevenue', 'TotalRevenue', 'SalesRevenueNet', 'RevenueFromContractWithCustomerExcludingAssessedTax']
                cos_keys = ['CostOfGoodsAndServicesSold', 'CostOfRevenue', 'CostOfGoodsSold']
                op_keys = ['OperatingIncomeLoss', 'OperatingIncome']
                liab_keys = ['Liabilities', 'TotalLiabilities', 'LiabilitiesCurrent']
                ocf_keys = ['NetCashProvidedByUsedInOperatingActivities', 'OperatingCashFlow']
                capex_keys = ['PaymentsToAcquirePropertyPlantAndEquipment', 'CapitalExpenditure']

                rev = get_first_match(latest_row, rev_keys)
                cos = get_first_match(latest_row, cos_keys)
                op_inc = get_first_match(latest_row, op_keys)
                liab = get_first_match(latest_row, liab_keys)
                ocf = get_first_match(latest_row, ocf_keys)
                capex = get_match = get_first_match(latest_row, capex_keys)

                if rev and rev > 0:
                    if cos is not None:
                        result["gross_margin"] = cls._clean_num(((rev - cos) / rev) * 100)
                    if op_inc is not None:
                        result["operating_margin"] = cls._clean_num((op_inc / rev) * 100)
                    if ni_ttm is not None:
                        result["net_margin"] = cls._clean_num((ni_ttm / (rev * 4 if len(df_pivot) >= 4 else rev)) * 100)

                if liab is not None and equity and equity > 0:
                    result["debt_to_equity"] = cls._clean_num((liab / equity) * 100)

                if ocf is not None:
                    fcf_val = ocf - (abs(capex) if capex else 0)
                    result["free_cash_flow"] = cls._format_large_number(fcf_val)

                # 營收成長率 YoY
                try:
                    if len(df_pivot) >= 5 and rev:
                        prev_rev = get_first_match(df_pivot.iloc[4], rev_keys)
                        if prev_rev and prev_rev > 0:
                            growth = ((rev / prev_rev) - 1) * 100
                            result["revenue_growth"] = cls._clean_num(growth)
                except Exception:
                    pass

                # PE / PB
                if latest_price and latest_price > 0:
                    if eps_ttm and eps_ttm > 0:
                        result["pe"] = cls._clean_num(latest_price / eps_ttm)
                    if bps and bps > 0:
                        result["pb"] = cls._clean_num(latest_price / bps)
                    if shares and shares > 0:
                        result["market_cap"] = cls._format_large_number(shares * latest_price)

                # PEG
                pe_val = result.get("pe")
                growth_val = result.get("revenue_growth")
                if pe_val != "N/A" and growth_val != "N/A" and isinstance(growth_val, (int, float)) and growth_val > 0:
                    result["peg_ratio"] = cls._clean_num(float(pe_val) / float(growth_val))

        except Exception as e_us:
            logger.warning(f"處理美股 {valuation_symbol} 財務比率異常: {e_us}")
