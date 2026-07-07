from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import JsonResponse, HttpResponse
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
            
            if results.get('is_etf'):
                context['etf'] = results
                request.session['last_ticker'] = symbol.upper()
                return render(request, 'valuation/etf_detail.html', context)
            
            # Feature 7: Trigger auto-refresh if data is stale
            from stock_Django.mySQL_OP import OP_Fun
            from stock_Django.data_freshness import trigger_refresh_if_stale
            is_tw = symbol.isdigit() or ".TW" in symbol.upper()
            sql = OP_Fun()
            trigger_refresh_if_stale(symbol, is_tw, sql.engine)

            # ValuationService returns an error dict when data is missing
            if 'error' in results:
                from stock_Django.data_freshness import get_refresh_status
                status_info = get_refresh_status(symbol)
                context['is_updating'] = True
                context['refresh_status'] = status_info
                context['error'] = results['error']
                context['error_detail'] = (
                    f"Financial data for '{symbol}' has not been loaded into the database yet. "
                    f"The system is currently auto-fetching it from online sources. "
                    f"Please wait a moment while we update the database."
                )
            else:
                context['valuation'] = results
                request.session['last_ticker'] = symbol.upper()
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


def valuation_refresh_status_api(request, symbol):
    """
    API endpoint to check background sync progress of a ticker.
    """
    from stock_Django.data_freshness import get_refresh_status
    status_info = get_refresh_status(symbol)
    return JsonResponse(status_info)


def valuation_export_excel(request):
    """
    Export stock valuation model as an Excel spreadsheet with dynamic formulas.
    """
    ticker = request.GET.get('ticker')
    if not ticker:
        return JsonResponse({'error': 'Ticker parameter is required'}, status=400)
    
    ticker = ticker.upper()
    try:
        # Extract weights (similar to valuation_view)
        try:
            dcf_weight_val = float(request.GET.get('dcf_weight', 50))
            market_weight_val = float(request.GET.get('market_weight', 50))
            total = dcf_weight_val + market_weight_val
            if total > 0:
                dcf_weight = dcf_weight_val / total
                market_weight = market_weight_val / total
            else:
                dcf_weight, market_weight = 0.5, 0.5
        except (ValueError, TypeError):
            dcf_weight, market_weight = 0.5, 0.5
            
        # Get valuation data
        results = ValuationService.calculate_valuation(ticker, dcf_weight=dcf_weight, market_weight=market_weight)
        
        if 'error' in results:
            return JsonResponse({'error': f"Cannot export Excel: {results['error']}"}, status=400)
            
        # Pass weights percentages for displaying in Excel
        results['dcf_weight_pct'] = round(dcf_weight * 100)
        results['market_weight_pct'] = round(market_weight * 100)
        
        # Generate Excel spreadsheet bytes
        from .services.excel_utils import generate_valuation_excel
        excel_data = generate_valuation_excel(results)
        
        response = HttpResponse(
            excel_data,
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        response['Content-Disposition'] = f'attachment; filename={ticker}_valuation_model.xlsx'
        return response
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Excel export failed for {ticker}: {e}", exc_info=True)
        return JsonResponse({'error': f"Internal server error: {str(e)}"}, status=500)

