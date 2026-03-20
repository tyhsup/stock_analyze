"""institutional_chips/views.py — Dedicated institutional investor chips dashboard."""

from django.shortcuts import render
from stock_Django import mySQL_OP, stock_chart
from stock_Django.stock_utils import StockUtils


def chips_view(request):
    """Render the institutional chips analysis page with top buyers/sellers and per-stock trends."""
    chart = stock_chart.chart_create()
    SQL_OP = mySQL_OP.OP_Fun()

    buysell_json = None
    comparison_json = None
    us_investor_json = []
    error = None

    try:
        # TW Data
        tw_data = SQL_OP.get_latest_investor_data(days=10)
        tw_data_clean = StockUtils.transfer_numeric(tw_data)

        if not tw_data_clean.empty:
            buysell_json = chart.investor_buysell_top_apex(tw_data_clean, amount=10)
            comparison_json = chart.investor_comparison_apex(tw_data_clean, amount=5, days=5)

        # US Data (Benchmark tickers for dashboard)
        from stock_Django.stock_investor_us import USStockInvestorManager
        us_mgr = USStockInvestorManager()
        benchmark_tickers = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'AMZN']
        
        for ticker in benchmark_tickers:
            us_df = us_mgr.get_latest_holders(ticker, top_n=10)
            if not us_df.empty:
                us_plot_data = chart.investor_us_apex(us_df, symbol=ticker)
                if us_plot_data:
                    us_investor_json.append(us_plot_data)

    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Chips view error: {e}")
        error = f"Failed to load chip data: {e}"

    return render(request, 'institutional_chips/index.html', {
        'buysell_json': buysell_json,
        'comparison_json': comparison_json,
        'us_investor_json': us_investor_json,
        'error': error,
    })


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from stock_Django import data_freshness
import time


@csrf_exempt
def refresh_tw_api(request):
    """Trigger background refresh for Taiwan investor data."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    # Cooldown check
    now = time.time()
    label = "TW_ALL"
    if label in data_freshness._investor_cooldown and (now - data_freshness._investor_cooldown[label]) < 300:
        return JsonResponse({'error': 'Cooldown active. Please wait 5 minutes.'}, status=429)
    
    data_freshness._investor_cooldown[label] = now
    
    import threading
    thread = threading.Thread(target=data_freshness.refresh_tw_investor_background, daemon=True)
    thread.start()
    
    return JsonResponse({'status': 'started', 'message': 'TW Background update triggered'})


@csrf_exempt
def refresh_us_api(request):
    """Trigger background refresh for USA investor data."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    now = time.time()
    label = "US_ALL"
    if label in data_freshness._investor_cooldown and (now - data_freshness._investor_cooldown[label]) < 300:
        return JsonResponse({'error': 'Cooldown active. Please wait 5 minutes.'}, status=429)
    
    data_freshness._investor_cooldown[label] = now
    
    import threading
    thread = threading.Thread(target=data_freshness.refresh_us_investor_background, daemon=True)
    thread.start()
    
    return JsonResponse({'status': 'started', 'message': 'US Background update triggered'})


def refresh_status_api(request, market):
    """Poll the status of the background update."""
    label = "TW_ALL" if market.lower() == 'tw' else "US_ALL"
    status = data_freshness.get_refresh_status(label)
    return JsonResponse(status)
