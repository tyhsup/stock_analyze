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
    def calculate_valuation(ticker_symbol, dcf_weight=0.5, market_weight=0.5):
        """
        Unified valuation entry point using internal modular components.
        Eliminates dependency on missing standalone main.py script.
        """
        ticker_symbol = ticker_symbol.upper()
        try:
            # 1. Load Financial Data
            loader = FinancialDataLoader(ticker_symbol)
            is_df, bs_df, cf_df = loader.get_full_financials()
            
            if is_df is None or bs_df is None or is_df.empty or bs_df.empty:
                return {"error": f"No financial data available in database for {ticker_symbol}."}
            
            # 2. Extract Projection Start and Market Data
            start_data = loader.extract_projection_start()
            current_price = loader.get_market_price()
            currency = loader.get_currency()
            
            if not start_data or current_price <= 0:
                return {"error": f"Insufficient data to calculate valuation for {ticker_symbol}."}
            
            # 3. Calculate WACC
            wacc_calc = WACCCalculator(ticker_symbol)
            wacc_results = wacc_calc.calculate_wacc()
            wacc = wacc_results['WACC']
            
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
            
            # 5. Run DCF Projection
            projector = FinancialProjector(start_data, assumptions)
            projections = projector.run_projection()
            
            # 6. Calculate DCF Implied Price
            # FCF = Net Income + D&A - CapEx - DeltaWC
            last_year = projections.iloc[-1]
            fcf_5 = last_year['net_income'] + last_year['depreciation'] - last_year['capex'] - last_year['change_in_wc']
            g = assumptions.perpetuity_growth_rate
            
            # Floor (wacc - g) at 2% to prevent extreme valuations from near-zero denominators
            denom = max(wacc - g, 0.02)
            terminal_value = fcf_5 * (1 + g) / denom
            
            pv_fcfs = 0
            for i, row in projections.iterrows():
                fcf = row['net_income'] + row['depreciation'] - row['capex'] - row['change_in_wc']
                pv_fcfs += fcf / ((1 + wacc)**(i + 1))
            
            pv_tv = terminal_value / ((1 + wacc)**5)
            enterprise_value = pv_fcfs + pv_tv
            net_debt = start_data.get('total_debt', 0) - start_data.get('cash', 0)
            equity_value = enterprise_value - net_debt
            
            # Defensive check: if Equity Value is too low or negative, fallback to a floor
            shares = max(start_data.get('shares_outstanding', 1), 1)
            implied_price_dcf = equity_value / shares
            
            # Formatting and Scaling for Display
            # For large US companies, some values might be internally in absolute units
            # Standardize for the result dictionary
            equity_value_m = equity_value / 1000000
            net_debt_m = net_debt / 1000000
            enterprise_value_m = enterprise_value / 1000000
            
            # Floor implied price at 20% of net asset value or similar if calculation goes negative
            # But the user wants a professional value, so we'll just floor at 0 for now but log it
            if implied_price_dcf <= 0:
                logger.warning(f"DCF calculation for {ticker_symbol} resulted in negative value. EV={enterprise_value}, NetDebt={net_debt}")
            
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
            
            # 8. Weighted Fair Value
            fair_value = (implied_price_dcf * dcf_weight) + (implied_price_market * market_weight)
            upside = (fair_value / current_price) - 1 if current_price > 0 else 0
            
            # Prepare projection lists for detail.html template
            # Standardize output to Millions (M) for large stocks
            years_list = [f"Year {i+1}" for i in range(len(projections))]
            revenues_list = [round(float(val) / 1000000, 2) for val in projections['revenue'].tolist()]
            fcfs_list = [round(float(row['net_income'] + row['depreciation'] - row['capex'] - row['change_in_wc']) / 1000000, 2) for _, row in projections.iterrows()]
            
            # Re-calculate discounted FCFs for display (also in Millions)
            discounted_fcfs_list = []
            for i, fcf_val_abs in enumerate([float(row['net_income'] + row['depreciation'] - row['capex'] - row['change_in_wc']) for _, row in projections.iterrows()]):
                val = fcf_val_abs / ((1 + wacc)**(i + 1))
                discounted_fcfs_list.append(round(val / 1000000, 2))

            results = {
                "symbol": ticker_symbol,
                "current_price": round(current_price, 2),
                "fair_value": round(fair_value, 2),
                "upside": float(upside),
                "currency": currency,
                "dcf": {
                    "implied_price": round(max(implied_price_dcf, 0), 2),
                    "wacc": float(wacc),
                    "terminal_value": round(float(terminal_value) / 1000000, 2),
                    "pv_terminal_value": round(float(terminal_value / ((1 + wacc)**5)) / 1000000, 2),
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
