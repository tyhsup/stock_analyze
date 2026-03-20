from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import JsonResponse
from .services.valuation_service import ValuationService

def valuation_root(request):
    """
    Root valuation view: redirect to last searched ticker or show blank state.
    """
    last_ticker = request.session.get('last_ticker')
    if last_ticker:
        return redirect(reverse('valuation_detail', kwargs={'symbol': last_ticker}))
    
    # If no ticker in session, show a landing page (or redirect to home but the user wants blank/input)
    # We'll use valuation_view with an empty symbol placeholder or specialized template
    return render(request, 'valuation/detail.html', {'symbol': None})

def valuation_view(request, symbol):
    """
    Renders the valuation page for a specific stock.
    """
    # Fix: Handle ticker search from the page search bar
    query_ticker = request.GET.get('ticker')
    if query_ticker and query_ticker.upper() != symbol.upper():
        # Preserve weights if present
        params = request.GET.copy()
        if 'ticker' in params: del params['ticker']
        url = reverse('valuation_detail', kwargs={'symbol': query_ticker.upper()})
        query_str = f"?{params.urlencode()}" if params else ""
        return redirect(url + query_str)

    context = {'symbol': symbol, 'ticker': symbol}
    if symbol:
        try:
            # Extract weights from GET parameters (default 50/50)
            try:
                dcf_weight_val = float(request.GET.get('dcf_weight', 50))
                market_weight_val = float(request.GET.get('market_weight', 50))
                
                # Normalize if they don't sum to 100
                total = dcf_weight_val + market_weight_val
                if total > 0:
                    dcf_weight = dcf_weight_val / total
                    market_weight = market_weight_val / total
                else:
                    dcf_weight, market_weight = 0.5, 0.5
            except (ValueError, TypeError):
                dcf_weight, market_weight = 0.5, 0.5

            context['dcf_weight_pct'] = round(dcf_weight * 100)
            context['market_weight_pct'] = round(market_weight * 100)
            
            results = ValuationService.calculate_valuation(symbol, dcf_weight=dcf_weight, market_weight=market_weight)
            
            # Feature 7: Trigger auto-refresh if data is stale
            from stock_Django.mySQL_OP import OP_Fun
            from stock_Django.data_freshness import trigger_refresh_if_stale
            is_tw = symbol.isdigit() or ".TW" in symbol.upper()
            sql = OP_Fun()
            trigger_refresh_if_stale(symbol, is_tw, sql.engine)

            # ValuationService returns an error dict when data is missing
            if 'error' in results:
                context['error'] = results['error']
                context['error_detail'] = (
                    f"Financial data for '{symbol}' has not been loaded into the database yet. "
                    f"The system attempted to auto-fetch it but may have failed. "
                    f"Please ensure the ticker format is correct (e.g., '2330.TW' for TSMC) "
                    f"and that financial data has been scraped and stored."
                )
            else:
                context['valuation'] = results
        except Exception as e:
            context['error'] = str(e)
            
    return render(request, 'valuation/detail.html', context)

def valuation_api(request, symbol):
    """
    Returns valuation data as JSON.
    """
    try:
        results = ValuationService.calculate_valuation(symbol)
        return JsonResponse(results)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
