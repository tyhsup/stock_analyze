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
            
            # 4.1 驗證並約束永續期成長率天花板 (g <= Rf)
            risk_free_rate = assumptions.risk_free_rate
            g_capped = assumptions.validate_terminal_growth(risk_free_rate)

            # 4.2 估計 ROIC (若無足夠投資資本數據則預設合理水平 12%)
            t_rate = assumptions.tax_rate
            invested_cap = (
                start_data.get('total_debt', 0) + 
                start_data.get('share_capital', 0) + 
                start_data.get('retained_earnings', 0) - 
                start_data.get('cash', 0)
            )
            latest_ebit = start_data.get('ebit', 0)
            est_nopat = latest_ebit * (1.0 - t_rate)
            est_roic = est_nopat / invested_cap if invested_cap > 0 else 0.12

            # 5. Run Primary DCF Projection
            projector = FinancialProjector(start_data, assumptions)
            projections = projector.run_projection()
            
            is_mid_year = (discount_convention == 'mid_year')
            use_exit_mult = (tv_method == 'exit_multiple')

            # 折現與終值計算輔助函式
            def evaluate_dcf_flow(proj_df, wacc_val, g_val, use_mult, mult_val):
                last_yr = proj_df.iloc[-1]
                nopat_last = last_yr['ebit'] * (1.0 - t_rate)
                fcf_last = nopat_last + last_yr['depreciation'] - last_yr['capex'] - last_yr['change_in_wc']

                if use_mult:
                    ebitda_last = last_yr['ebit'] + last_yr['depreciation']
                    tv = ebitda_last * mult_val
                else:
                    # 筆記標準：扣除再投資率之終值 FCFF
                    term_fcf = assumptions.calculate_terminal_fcff(nopat_last, est_roic, g_val)
                    wacc_eff = max(float(wacc_val), 0.03)
                    denom = wacc_eff - g_val
                    if denom < 0.01:
                        denom = 0.01
                    tv = term_fcf / denom

                pv_fcfs = 0.0
                for i, row in proj_df.iterrows():
                    fcf = row['ebit'] * (1.0 - t_rate) + row['depreciation'] - row['capex'] - row['change_in_wc']
                    t_exp = (i + 0.5) if is_mid_year else (i + 1.0)
                    pv_fcfs += fcf / ((1.0 + max(float(wacc_val), 0.01)) ** t_exp)

                tv_exp = (len(proj_df) - 0.5) if is_mid_year else float(len(proj_df))
                pv_tv = tv / ((1.0 + max(float(wacc_val), 0.01)) ** tv_exp)
                ev = pv_fcfs + pv_tv
                return float(ev), float(tv), float(pv_tv)

            # 6. 整合 IterativeDCFSolver 與 EVToEquityBridge 進行循環論證求解與動態 TSM 稀釋計算
            from .iterative_solver import IterativeDCFSolver, TSMCalculator
            from .ev_bridge import EVToEquityBridge
            solver = IterativeDCFSolver(max_iter=50, tol=1e-4, damping_factor=0.5)

            # 提取 EV-to-Equity Bridge 核心調整項
            c_cash = float(start_data.get('cash', 0) or 0)
            t_debt = float(start_data.get('total_debt', 0) or 0)
            op_leases = float(start_data.get('operating_lease_liability', 0) or 0)
            pref_stock = float(start_data.get('preferred_stock', 0) or 0)
            min_interest = float(start_data.get('minority_interest', 0) or 0)
            pension_def = float(start_data.get('pension_deficit', 0) or 0)

            # 取得股權成本 Ke 與債務成本 Rd
            cost_of_equity = wacc_calc.calculate_cost_of_equity()
            cost_of_debt = assumptions.cost_of_debt

            # 執行迭代求解
            basic_shares = float(start_data.get('shares_outstanding', 1))
            reported_diluted = float(start_data.get('diluted_shares', basic_shares))
            opts_count = float(start_data.get('options_count', 0))
            stk_price = float(start_data.get('strike_price', 0))

            solve_res = solver.solve(
                initial_price=current_price,
                basic_shares=basic_shares,
                total_debt=t_debt,
                cash=c_cash,
                cost_of_equity=cost_of_equity,
                cost_of_debt=cost_of_debt,
                tax_rate=t_rate,
                discount_fn=lambda w: evaluate_dcf_flow(projections, w, g_capped, use_exit_mult, exit_multiple),
                options_count=opts_count,
                strike_price=stk_price,
                reported_diluted_shares=reported_diluted,
                debt_adjust=debt_like_items,
                wacc_premium=wacc_premium,
                operating_lease_liability=op_leases,
                preferred_stock=pref_stock,
                minority_interest=min_interest,
                pension_deficit=pension_def
            )

            implied_price_dcf = solve_res['implied_price']
            shares = solve_res['diluted_shares']
            wacc = solve_res['converged_wacc']

            # 重新計算收斂後之 EV, TV 與完整 EV-to-Equity Bridge
            enterprise_value, terminal_value, pv_tv = evaluate_dcf_flow(
                projections, wacc, g_capped, use_exit_mult, exit_multiple
            )
            bridge_base = EVToEquityBridge.calculate_bridge(
                enterprise_value=enterprise_value,
                cash=c_cash,
                total_debt=t_debt,
                operating_lease_liability=op_leases,
                preferred_stock=pref_stock,
                minority_interest=min_interest,
                pension_deficit=pension_def,
                debt_like_items=debt_like_items
            )
            net_debt = bridge_base['net_debt']

            # --- 三情境分析 (Bear / Base / Bull) ---
            # 保守情境 (Bear): WACC + 2.0%, 營收成長率 70%, g = 1.0%
            assumptions_bear = Assumptions()
            assumptions_bear.revenue_growth_rate = [g_val * 0.7 for g_val in assumptions.revenue_growth_rate]
            assumptions_bear.ebit_margin = assumptions.ebit_margin * 0.9
            assumptions_bear.tax_rate = assumptions.tax_rate
            proj_bear = FinancialProjector(start_data, assumptions_bear).run_projection()
            ev_bear, _, _ = evaluate_dcf_flow(proj_bear, wacc + 0.02, 0.01, use_exit_mult, exit_multiple * 0.8)
            price_bear = max((ev_bear - net_debt) / max(shares, 1), 0.0)

            # 樂觀情境 (Bull): WACC - 1.0%, 營收成長率 120%, g = min(2.5%, Rf)
            assumptions_bull = Assumptions()
            assumptions_bull.revenue_growth_rate = [g_val * 1.2 for g_val in assumptions.revenue_growth_rate]
            assumptions_bull.ebit_margin = assumptions.ebit_margin * 1.1
            assumptions_bull.tax_rate = assumptions.tax_rate
            proj_bull = FinancialProjector(start_data, assumptions_bull).run_projection()
            g_bull = min(0.025, risk_free_rate)
            ev_bull, _, _ = evaluate_dcf_flow(proj_bull, max(wacc - 0.01, 0.03), g_bull, use_exit_mult, exit_multiple * 1.2)
            price_bull = max((ev_bull - net_debt) / max(shares, 1), 0.0)
            
            price_base = implied_price_dcf

            # 7. Run Relative Valuation (若採用 Exit Multiple 法，將調校乘數注入相對估值)
            rel_valuator = RelativeValuator(ticker_symbol, start_data, current_price, currency)
            hist_multiples = loader.get_historical_multiples()
            target_ev_ebitda = exit_multiple if (tv_method == 'exit_multiple' and exit_multiple > 0) else hist_multiples['ev_ebitda']
            
            rel_results = rel_valuator.calculate_implied_fair_value(
                target_pe=hist_multiples['pe'], 
                target_ev_ebitda=target_ev_ebitda
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

            # 確保 wacc_premium_pct 正確顯示為百分比數字 (例如 1.5)，避免被二次乘 100 溢位
            wacc_premium_pct = wacc_results.get('WACC Premium Pct', float(wacc_premium))

            # --- 9. Phase 3 風險分析強化：蒙地卡羅模擬 (10,000 次) 與 WACC vs g 雙向敏感度矩陣 ---
            import numpy as np
            from .monte_carlo import MonteCarloSimulator
            from .sensitivity import SensitivityAnalyzer

            mc_sim = MonteCarloSimulator(num_simulations=10000, seed=42)
            base_rev = float(start_data.get('revenue', 0) or 1000000000)
            avg_growth = float(np.mean(assumptions.revenue_growth_rate)) if assumptions.revenue_growth_rate else 0.05
            reinvest_rate = float(assumptions.calculate_reinvestment_rate(est_roic, g_capped))

            monte_carlo_res = mc_sim.run_simulation(
                base_revenue=base_rev,
                base_growth=avg_growth,
                growth_std=0.03,
                ebit_margin=float(assumptions.ebit_margin),
                tax_rate=t_rate,
                base_wacc=float(wacc),
                wacc_std=0.012,
                risk_free_rate=float(risk_free_rate),
                reinvestment_rate=reinvest_rate,
                net_debt=net_debt,
                shares_outstanding=shares,
                current_price=current_price
            )

            sensitivity_res = SensitivityAnalyzer.generate_matrix(
                base_revenue=base_rev,
                growth_rates=assumptions.revenue_growth_rate,
                ebit_margin=float(assumptions.ebit_margin),
                tax_rate=t_rate,
                reinvestment_rate=reinvest_rate,
                base_wacc=float(wacc),
                base_g=float(g_capped),
                net_debt=net_debt,
                shares_outstanding=shares,
                risk_free_rate=float(risk_free_rate),
                is_mid_year=is_mid_year
            )

            # Phase 6: WACC 防護閥狀態與異常原因
            is_wacc_flagged = bool(wacc_results.get('is_flagged', False))
            wacc_flag_reasons = list(wacc_results.get('flag_reasons', []))
            wacc_guardrail_dict = wacc_results.get('wacc_guardrail', {})

            results = {
                "symbol": ticker_symbol,
                "current_price": round(current_price, 2),
                "fair_value": round(fair_value, 2),
                "upside": float(upside),
                "currency": currency,
                "is_flagged": is_wacc_flagged,
                "flag_reasons": wacc_flag_reasons,
                "wacc_guardrail": wacc_guardrail_dict,
                "institutional_tuning": {
                    "wacc_premium_pct": round(wacc_premium_pct, 2),
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
                    "iterations": int(solve_res.get('iterations', 1)),
                    "is_converged": bool(solve_res.get('is_converged', True)),
                    "ev_to_equity_bridge": bridge_base,
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
                },
                "monte_carlo": monte_carlo_res,
                "sensitivity_matrix": sensitivity_res
            }

            # Phase 4 & Phase 6: SCD Type 2 歷史版本控制快照 (含 WACC 異常防護閥標記)
            try:
                from valuation.services.assumption_history import AssumptionHistoryService
                assumption_snapshot = {
                    "revenue_growth_rate": assumptions.revenue_growth_rate,
                    "ebit_margin": float(assumptions.ebit_margin),
                    "tax_rate": float(assumptions.tax_rate if assumptions.tax_rate < 1 else assumptions.tax_rate / 100),
                    "wacc": float(wacc),
                    "exit_growth_rate": float(assumptions.perpetuity_growth_rate),
                    "wacc_premium": float(wacc_premium),
                    "dcf_weight": float(dcf_weight),
                    "market_weight": float(market_weight),
                    "debt_like_items": float(debt_like_items),
                    "is_flagged": is_wacc_flagged,
                    "flag_reasons": wacc_flag_reasons,
                    "raw_wacc": float(wacc_results.get('Raw WACC', wacc)),
                }
                val_snapshot = {
                    "blended_fair_value": float(fair_value),
                    "current_price": float(current_price) if current_price else None,
                    "upside": float(upside) if upside else None,
                    "implied_price_dcf": float(implied_price_dcf),
                    "implied_price_market": float(implied_price_market),
                    "ev": float(enterprise_value),
                    "equity_value": float(bridge_base.get('equity_value', 0.0)),
                }
                scd_market = getattr(loader, 'market', 'tw').upper()
                scd_rec, is_new_version = AssumptionHistoryService.save_assumptions_scd2(
                    symbol=ticker_symbol,
                    market=scd_market,
                    assumptions=assumption_snapshot,
                    valuation_snapshot=val_snapshot,
                    change_reason="估值計算自動存檔與快照",
                    is_flagged=is_wacc_flagged,
                    flag_reasons=wacc_flag_reasons,
                )
                results["scd2"] = {
                    "version": scd_rec.version,
                    "is_current": scd_rec.is_current,
                    "effective_date": scd_rec.effective_date.isoformat(),
                    "is_new_version": is_new_version,
                }
            except Exception as scd_err:
                logger.warning(f"SCD2 快照儲存失敗 (非致命): {scd_err}")

            return results
            
        except Exception as e:
            logger.error(f"Valuation failed for {ticker_symbol}: {e}", exc_info=True)
            return {"error": f"Valuation internal error: {str(e)}"}
