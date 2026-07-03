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
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Chips TW view error: {e}")
        error = f"Failed to load TW chip data: {e}"

    try:
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
        logging.getLogger(__name__).error(f"Chips US view error: {e}")
        if error:
            error += f" | Failed to load US chip data: {e}"
        else:
            error = f"Failed to load US chip data: {e}"

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


from django.core.cache import cache
import pandas as pd
import numpy as np

def api_us_stocks_list(request):
    """
    回傳美股股票清單供 SEC EDGAR Ticker 選擇器使用。
    僅回傳 symbol 與 name，供前端下拉選單渲染。
    """
    try:
        from market_data.models import StockUS
        stocks = StockUS.objects.all().order_by('symbol')[:500]
        data = [{'symbol': s.symbol, 'name': s.name} for s in stocks]
        return JsonResponse({'stocks': data})
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"api_us_stocks_list error: {e}")
        return JsonResponse({'stocks': [], 'error': str(e)})


def api_industry_flow(request):
    """
    API endpoint returning industry money flow (treemap data) and top 10 stocks per industry.
    """
    try:
        # 1. Parse GET parameters with fallbacks
        days = int(request.GET.get('days', 10))
        w1 = float(request.GET.get('w1', 0.5))
        w2 = float(request.GET.get('w2', 0.3))
        w3 = float(request.GET.get('w3', 0.2))
        
        # Clamp days between 1 and 30 to prevent abuse
        days = max(1, min(30, days))
        
        # 2. Normalize weights
        total_w = w1 + w2 + w3
        if total_w > 0:
            w1 /= total_w
            w2 /= total_w
            w3 /= total_w
        else:
            w1, w2, w3 = 0.5, 0.3, 0.2
            
        # 3. Check Cache
        cache_key = f"industry_flow_{days}_{w1:.2f}_{w2:.2f}_{w3:.2f}"
        cached_data = cache.get(cache_key)
        if cached_data:
            return JsonResponse(cached_data)
            
        # 4. Fetch summary from SQL OP
        from stock_Django.mySQL_OP import OP_Fun
        sql = OP_Fun()
        df = sql.get_industry_investor_summary(days)
        
        if df.empty:
            return JsonResponse({'industries': [], 'top_stocks': {}})
            
        # 5. Calculate scores for each stock
        def normalize_series(series, default=50.0):
            s_min = series.min()
            s_max = series.max()
            if s_max > s_min:
                return (series - s_min) / (s_max - s_min) * 100
            return series * 0.0 + default

        # net buy ratio = total_net_buy / total_volume
        df['net_buy_ratio'] = df.apply(
            lambda r: r['total_net_buy'] / r['total_volume'] if r['total_volume'] > 0 else 0,
            axis=1
        )
        
        # Normalize factors
        net_buy_score = normalize_series(df['net_buy_ratio'])
        consec_score = normalize_series(df['consec_buys'])
        
        # Margin settlement score: decrease in margin, increase in short
        margin_dec_score = normalize_series(-df['margin_change'])
        short_inc_score = normalize_series(df['short_change'])
        settlement_score = 0.5 * margin_dec_score + 0.5 * short_inc_score
        
        # Calculate total score
        df['score'] = w1 * net_buy_score + w2 * consec_score + w3 * settlement_score
        
        # 6. Aggregate at Industry level
        # Explicitly convert to string and strip spaces for safety
        df['industry'] = df['industry'].astype(str).str.strip()
        industry_groups = df.groupby('industry')
        
        treemap_data = []
        top_stocks_by_industry = {}
        
        for name, group in industry_groups:
            # Skip invalid or empty industry names
            if not name or name in ['0', '0.0', 'nan', 'None', '']:
                continue
                
            # Aggregate volume value and net flow
            ind_net_flow = float(group['accum_net_flow'].sum())
            ind_volume_value = float(group['accum_volume_value'].sum())
            
            # Format and append industry-level data point
            # We scale volume_value to Millions for cleaner numbers in Treemap
            treemap_data.append({
                'x': str(name),
                'y': round(ind_volume_value / 1000000.0, 2), # Unit: Millions
                'net_flow': round(ind_net_flow, 2)            # Unit: Thousands NTD
            })
            
            # Top 10 stocks ranking
            top_10 = group.sort_values(by='score', ascending=False).head(10)
            stocks_list = []
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                stocks_list.append({
                    'rank': i,
                    'number': str(row['number']),
                    'name': str(row['證券名稱']),
                    'close': float(row['Close']),
                    'net_flow': float(row['accum_net_flow']),
                    'consec_buys': int(row['consec_buys']),
                    'score': round(float(row['score']), 1)
                })
            top_stocks_by_industry[str(name)] = stocks_list
            
        result = {
            'industries': treemap_data,
            'top_stocks': top_stocks_by_industry
        }
        
        # Cache results for 10 minutes (600 seconds)
        cache.set(cache_key, result, 600)
        
        return JsonResponse(result)
        
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"api_industry_flow error: {e}")
        return JsonResponse({'error': str(e)}, status=500)


def api_master_selection(request):
    """
    大師選股 API (支援 巴菲特/彼得林區/威廉歐尼爾)
    """
    if request.method != 'GET':
        return JsonResponse({'error': 'GET required'}, status=405)
        
    market = request.GET.get('market', 'tw').lower()
    if market not in ['tw', 'us']:
        return JsonResponse({'error': 'Invalid market'}, status=400)
        
    master = request.GET.get('master', 'buffett').lower()
    if master not in ['buffett', 'lynch', 'oneil']:
        return JsonResponse({'error': 'Invalid master name'}, status=400)
        
    from valuation.models import MasterSelection
    from valuation.services.master_selection_service import MasterSelectionService
    from django.utils import timezone
    from datetime import timedelta
    
    try:
        # 查詢是否有既有紀錄
        records = MasterSelection.objects.filter(market=market, master_name=master).order_by('rank')
        
        need_update = False
        if not records.exists():
            need_update = True
        else:
            latest_record = records.order_by('-updated_at').first()
            if timezone.now() - latest_record.updated_at > timedelta(hours=24):
                need_update = True
                
        force_refresh = request.GET.get('force', 'false').lower() == 'true'
        if need_update or force_refresh:
            service = MasterSelectionService()
            service.run_selection(market, master)
            records = MasterSelection.objects.filter(market=market, master_name=master).order_by('rank')
            
        data = []
        for r in records:
            # 針對彼得林區模式特別傳回 PE 和 PEG 以利前台渲染 (雖然數值也是在原有欄位映射)
            item_data = {
                'rank': r.rank,
                'symbol': r.symbol,
                'name': r.name,
                'close_price': float(r.close_price) if r.close_price else 0.0,
                'roe': float(r.roe) if r.roe else 0.0,
                'gross_margin': float(r.gross_margin) if r.gross_margin else 0.0,
                'debt_ratio': float(r.debt_ratio) if r.debt_ratio else 0.0,
                'net_income_growth': float(r.net_income_growth) if r.net_income_growth else 0.0,
                'score': float(r.score) if r.score else 0.0
            }
            data.append(item_data)
            
        return JsonResponse({'status': 'success', 'data': data})
        
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"api_master_selection error: {e}")
        return JsonResponse({'error': str(e)}, status=500)


