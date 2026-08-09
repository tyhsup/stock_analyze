import logging
import pandas as pd
from .financial_data import FinancialDataLoader
from .wacc_calc import WACCCalculator
from .assumptions import Assumptions
from .projection import FinancialProjector
from .relative_valuation import RelativeValuator

logger = logging.getLogger(__name__)

class ValuationService:
    @staticmethod
    def calculate_valuation(
        ticker_symbol, 
        dcf_weight=0.5, 
        market_weight=0.5, 
        wacc_premium=0.0, 
        discount_convention='end_of_year', 
        tv_method='perpetuity', 
        exit_multiple=10.0, 
        debt_like_items=0.0
    ):
        """
        Unified valuation entry point using internal modular components.
        Supports Institutional Advanced Tuning parameters (WACC premium, mid-year discount, exit multiple TV, debt-like items).
        """
        ticker_symbol = ticker_symbol.upper()
        try:
            # 1. Load Financial Data
            loader = FinancialDataLoader(ticker_symbol)
            if loader.is_etf:
                from .etf_valuation import ETFValuationService
                return ETFValuationService.calculate_etf_valuation(ticker_symbol, loader)
                
            is_df, bs_df, cf_df = loader.get_full_financials()
            
            if is_df is None or bs_df is None or is_df.empty or bs_df.empty:
                return {"error": f"No financial data available in database for {ticker_symbol}."}
            
            # 2. Extract Projection Start and Market Data
            start_data = loader.extract_projection_start()
            current_price = loader.get_market_price()
            currency = loader.get_currency()
            
            if not start_data or current_price <= 0:
                return {"error": f"Insufficient data to calculate valuation for {ticker_symbol}."}
            
            # 3. Calculate WACC (支援 WACC Premium 貼水)
            wacc_calc = WACCCalculator(ticker_symbol)
            wacc_results = wacc_calc.calculate_wacc(wacc_premium=wacc_premium)
            wacc = wacc_results['WACC']
            base_wacc = wacc_results.get('Base WACC', wacc - wacc_premium)
            
            # 4. Setup Assumptions
            hist_ratios = loader.calculate_historical_ratios()
            hist_growth = loader.get_historical_growth_rates()
            
            assumptions = Assumptions()
            # Dynamic growth: use average of historical growth as baseline (limited to [2%, 15%])
            avg_hist_growth = float(hist_growth.mean()) if not hist_growth.empty else 0.05
            base_growth = max(min(avg_hist_growth, 0.15), 0.02)
            assumptions.revenue_growth_rate = [base_growth * (0.9**i) for i in range(5)]
            
            assumptions.ebit_margin = hist_ratios['ebit_margin']
            assumptions.tax_rate = hist_ratios['tax_rate']
            assumptions.cost_of_debt = wacc_results['Cost of Debt (Rd)']
            
            # Apply balance sheet ratios from history
            assumptions.ar_as_pct_revenue = hist_ratios.get('ar_as_pct_revenue', 0.1)
            assumptions.inv_as_pct_revenue = hist_ratios.get('inv_as_pct_revenue', 0.1)
            assumptions.ap_as_pct_revenue = hist_ratios.get('ap_as_pct_revenue', 0.05)
            assumptions.capex_as_pct_sales = hist_ratios.get('capex_as_pct_revenue', 0.03)
            assumptions.depreciation_as_pct_revenue = hist_ratios.get('da_as_pct_revenue', 0.03)
            
            # Helper function for DCF calculations given parameters
            def compute_dcf(proj_df, wacc_val, g_val, is_mid_year=False, use_exit_mult=False, exit_mult_val=10.0, debt_adjust=0.0):
                last_yr = proj_df.iloc[-1]
                t_rate = assumptions.tax_rate
                fcf_5 = last_yr['ebit'] * (1 - t_rate) + last_yr['depreciation'] - last_yr['capex'] - last_yr['change_in_wc']
                
                if use_exit_mult:
                    ebitda_5 = last_yr['ebit'] + last_yr['depreciation']
                    tv = ebitda_5 * exit_mult_val
                else:
                    wacc_eff = max(wacc_val, 0.03)
                    denom = wacc_eff - g_val
                    if denom < 0.01: denom = 0.01
                    tv = fcf_5 * (1 + g_val) / denom
                    
                pv_fcfs = 0.0
                for i, row in proj_df.iterrows():
                    fcf = row['ebit'] * (1 - t_rate) + row['depreciation'] - row['capex'] - row['change_in_wc']
                    t_exp = (i + 0.5) if is_mid_year else (i + 1)
                    pv_fcfs += fcf / ((1 + max(wacc_val, 0.01))**t_exp)
                
                tv_exp = 4.5 if is_mid_year else 5.0
                pv_tv = tv / ((1 + max(wacc_val, 0.01))**tv_exp)
                ev = pv_fcfs + pv_tv
                
                # Net Debt = Total Debt - Cash + Debt-like Items
                raw_net_debt = start_data.get('total_debt', 0) - start_data.get('cash', 0) + debt_adjust
                eq_val = ev - raw_net_debt
                shrs = max(start_data.get('diluted_shares', start_data.get('shares_outstanding', 1)), 1)
                implied_p = max(eq_val / shrs, 0)
                return implied_p, ev, raw_net_debt, tv, pv_tv, shrs

            # 5. Run Primary DCF Projection
            projector = FinancialProjector(start_data, assumptions)
            projections = projector.run_projection()
            
            is_mid_year = (discount_convention == 'mid_year')
            use_exit_mult = (tv_method == 'exit_multiple')
            
            implied_price_dcf, enterprise_value, net_debt, terminal_value, pv_tv, shares = compute_dcf(
                projections, 
                wacc_val=wacc, 
                g_val=assumptions.perpetuity_growth_rate, 
                is_mid_year=is_mid_year, 
                use_exit_mult=use_exit_mult, 
                exit_mult_val=exit_multiple, 
                debt_adjust=debt_like_items
            )
            
            # --- 三情境分析 (Bear / Base / Bull) ---
            # 保守情境 (Bear): WACC + 2.0%, 營收成長率 70%, g = 1.0%
            assumptions_bear = Assumptions()
            assumptions_bear.revenue_growth_rate = [g_val * 0.7 for g_val in assumptions.revenue_growth_rate]
            assumptions_bear.ebit_margin = assumptions.ebit_margin * 0.9
            assumptions_bear.tax_rate = assumptions.tax_rate
            proj_bear = FinancialProjector(start_data, assumptions_bear).run_projection()
            price_bear, _, _, _, _, _ = compute_dcf(proj_bear, wacc + 0.02, 0.01, is_mid_year, use_exit_mult, exit_multiple * 0.8, debt_like_items)

            # 樂觀情境 (Bull): WACC - 1.0%, 營收成長率 120%, g = 2.5%
            assumptions_bull = Assumptions()
            assumptions_bull.revenue_growth_rate = [g_val * 1.2 for g_val in assumptions.revenue_growth_rate]
            assumptions_bull.ebit_margin = assumptions.ebit_margin * 1.1
            assumptions_bull.tax_rate = assumptions.tax_rate
            proj_bull = FinancialProjector(start_data, assumptions_bull).run_projection()
            price_bull, _, _, _, _, _ = compute_dcf(proj_bull, max(wacc - 0.01, 0.03), 0.025, is_mid_year, use_exit_mult, exit_multiple * 1.2, debt_like_items)
            
            price_base = implied_price_dcf

            # 7. Run Relative Valuation
            rel_valuator = RelativeValuator(ticker_symbol, start_data, current_price, currency)
            hist_multiples = loader.get_historical_multiples()
            rel_results = rel_valuator.calculate_implied_fair_value(
                target_pe=hist_multiples['pe'], 
                target_ev_ebitda=hist_multiples['ev_ebitda']
            )
            
            pe_price = rel_results.get('pe_approach', {}).get('implied_price', current_price)
            ev_ebitda_price = rel_results.get('ev_ebitda_approach', {}).get('implied_price', current_price)
            implied_price_market = (pe_price + ev_ebitda_price) / 2
            
            # --- 8. 情緒溢價 (Sentiment Premium) ---
            sentiment_premium = 1.0
            try:
                from django.core.cache import cache
                cache_key = f"sentiment_premium_{ticker_symbol}"
                cached_premium = cache.get(cache_key)
                if cached_premium is not None:
                    sentiment_premium = cached_premium
                else:
                    from stock_Django.news_excel import NewsExcelManager
                    news_mgr = NewsExcelManager()
                    recent_news = news_mgr.read_news(ticker_symbol, limit=20)
                    if recent_news:
                        pos_count = sum(1 for n in recent_news if n.get('正負分析') == '正面')
                        neg_count = sum(1 for n in recent_news if n.get('正負分析') == '負面')
                        if pos_count > neg_count * 2:
                             sentiment_premium = 1.05
                        elif neg_count > pos_count * 2:
                             sentiment_premium = 0.95
                    cache.set(cache_key, sentiment_premium, 600)
            except Exception as e_s:
                logger.debug(f"Sentiment premium calculation skipped: {e_s}")

            # 9. Weighted Fair Value
            fair_value = ((implied_price_dcf * dcf_weight) + (implied_price_market * market_weight)) * sentiment_premium
            upside = (fair_value / current_price) - 1 if current_price > 0 else 0

            # Prepare projection lists
            tax_rate = assumptions.tax_rate
            years_list = [f"Year {i+1}" for i in range(len(projections))]
            revenues_list = [round(float(val) / 1000000, 2) for val in projections['revenue'].tolist()]
            fcfs_list = [round(float(row['ebit'] * (1 - tax_rate) + row['depreciation'] - row['capex'] - row['change_in_wc']) / 1000000, 2) for _, row in projections.iterrows()]
            
            discounted_fcfs_list = []
            for i, fcf_val_abs in enumerate([float(row['ebit'] * (1 - tax_rate) + row['depreciation'] - row['capex'] - row['change_in_wc']) for _, row in projections.iterrows()]):
                t_exp = (i + 0.5) if is_mid_year else (i + 1)
                val = fcf_val_abs / ((1 + wacc)**t_exp)
                discounted_fcfs_list.append(round(val / 1000000, 2))

            # twse-cli 數據
            twse_pe = hist_multiples.get('twse_pe')
            twse_pb = hist_multiples.get('twse_pb')
            twse_dy = hist_multiples.get('twse_dividend_yield')
            twse_valuation = None
            if any(v is not None for v in [twse_pe, twse_pb, twse_dy]):
                is_otc = loader.full_symbol.endswith('.TWO')
                source_name = "TPEx 櫃買中心官方" if is_otc else "TWSE 證交所官方"
                twse_valuation = {
                    "pe": round(twse_pe, 2) if twse_pe else None,
                    "pb": round(twse_pb, 2) if twse_pb else None,
                    "dividend_yield": round(twse_dy, 2) if twse_dy else None,
                    "source": source_name,
                }

            # 足球場估值數據 (Football Field Chart Data)
            football_field = {
                "dcf_range": [round(min(price_bear, price_bull), 2), round(max(price_bear, price_bull), 2)],
                "pe_range": [round(pe_price * 0.85, 2), round(pe_price * 1.15, 2)],
                "ev_ebitda_range": [round(ev_ebitda_price * 0.85, 2), round(ev_ebitda_price * 1.15, 2)],
                "target_consensus_range": [
                    round(min(current_price * 0.9, price_bear * 0.95), 2), 
                    round(max(current_price * 1.25, price_bull * 1.05), 2)
                ]
            }

            results = {
                "symbol": ticker_symbol,
                "current_price": round(current_price, 2),
                "fair_value": round(fair_value, 2),
                "upside": float(upside),
                "currency": currency,
                "institutional_tuning": {
                    "wacc_premium_pct": round(wacc_premium * 100, 2),
                    "discount_convention": discount_convention,
                    "tv_method": tv_method,
                    "exit_multiple": exit_multiple,
                    "debt_like_items": debt_like_items
                },
                "scenarios": {
                    "bear": round(price_bear, 2),
                    "base": round(price_base, 2),
                    "bull": round(price_bull, 2)
                },
                "football_field": football_field,
                "dcf": {
                    "implied_price": round(max(implied_price_dcf, 0), 2),
                    "wacc": float(wacc),
                    "base_wacc": float(base_wacc),
                    "wacc_premium": float(wacc_premium),
                    "terminal_value": round(float(terminal_value) / 1000000, 2),
                    "pv_terminal_value": round(float(pv_tv) / 1000000, 2),
                    "net_debt": float(net_debt),
                    "shares_outstanding": float(shares),
                    "projected_fcf": {
                        "years": years_list,
                        "revenues": revenues_list,
                        "fcfs": fcfs_list,
                        "discounted_fcfs": discounted_fcfs_list
                    }
                },
                "market_approach": {
                    "implied_price_avg": round(max(implied_price_market, 0), 2),
                    "pe_price": round(max(pe_price, 0), 2),
                    "ev_ebitda_price": round(max(ev_ebitda_price, 0), 2),
                    "multiples_used": {
                        "pe": round(hist_multiples['pe'], 1),
                        "ev_ebitda": round(hist_multiples['ev_ebitda'], 1)
                    }
                },
                "twse_valuation": twse_valuation,
                "assumptions": {
                    "revenue_growth_rate": assumptions.revenue_growth_rate,
                    "ebit_margin": float(assumptions.ebit_margin),
                    "tax_rate": float(assumptions.tax_rate if assumptions.tax_rate < 1 else assumptions.tax_rate / 100),
                    "wacc": float(wacc),
                    "exit_growth_rate": float(assumptions.perpetuity_growth_rate)
                }
            }
            return results
            
        except Exception as e:
            logger.error(f"Valuation failed for {ticker_symbol}: {e}", exc_info=True)
            return {"error": f"Valuation internal error: {str(e)}"}
