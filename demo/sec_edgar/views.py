import logging
from datetime import datetime
from django.shortcuts import render
from django.http import JsonResponse
from django.db import connection
from django.contrib.auth.decorators import login_required
from market_data.models import StockUS
from sec_edgar.models import SecFilingIndex, Sec13fHoldings, SecInsiderTrades, SecFinancialXbrl
from sec_edgar.services.edgar_13f_service import Edgar13FService
from sec_edgar.services.edgar_insider_service import EdgarInsiderService
from sec_edgar.services.edgar_financial_service import EdgarFinancialService

logger = logging.getLogger(__name__)

def dashboard_view(request):
    """
    SEC EDGAR 整合儀表板視圖
    """
    ticker = request.GET.get('ticker', 'AAPL').upper()
    
    # 獲取所有美股列表供下拉選單使用
    us_stocks = []
    try:
        us_stocks = StockUS.objects.all().order_by('symbol')[:200]  # 限制數量防渲染過載
    except Exception as e:
        logger.error(f"Error querying StockUS list: {e}")

    # 1. 取得 SEC 13F 官方持股數據
    sec_13f_data = []
    latest_13f_period = None
    try:
        service_13f = Edgar13FService()
        sec_13f_data = service_13f.get_top_holders_from_db(ticker, limit=50)
        if sec_13f_data:
            latest_13f_period = sec_13f_data[0]['period_of_report']
    except Exception as e:
        logger.error(f"Error fetching SEC 13F holdings from DB: {e}")

    # 2. 取得 Stockzoa 爬蟲持股數據 (備用/補充)
    stockzoa_data = []
    try:
        with connection.cursor() as cursor:
            # 依據資料庫實際欄位讀取
            cursor.execute(
                "SELECT holder_name, shares, pct_out, date FROM stock_investor_us WHERE ticker = %s ORDER BY date DESC LIMIT 50",
                [ticker]
            )
            rows = cursor.fetchall()
            for r in rows:
                stockzoa_data.append({
                    'holder_name': r[0],
                    'shares': r[1],
                    'pct_out': r[2],
                    'date': r[3]
                })
    except Exception as e:
        logger.warning(f"Failed to query stock_investor_us table: {e}")

    # 3. 取得內部人交易數據
    insider_trades = []
    try:
        insider_trades = SecInsiderTrades.objects.filter(ticker=ticker).order_by('-transaction_date', '-filing_date')[:50]
    except Exception as e:
        logger.error(f"Error querying insider trades: {e}")

    # 4. 取得標準化 XBRL 財務數據
    financial_data = []
    try:
        # 只取出重要財務概念，如 Revenues, NetIncomeLoss, OperatingIncomeLoss 等
        target_concepts = ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 
                           'NetIncomeLoss', 'OperatingIncomeLoss', 'Assets', 'Liabilities']
        financial_data = SecFinancialXbrl.objects.filter(
            ticker=ticker,
            concept__in=target_concepts
        ).order_by('-period_end', 'concept')
    except Exception as e:
        logger.error(f"Error querying financial XBRL data: {e}")

    context = {
        'current_ticker': ticker,
        'us_stocks': us_stocks,
        'sec_13f_data': sec_13f_data,
        'latest_13f_period': latest_13f_period,
        'stockzoa_data': stockzoa_data,
        'insider_trades': insider_trades,
        'financial_data': financial_data,
    }
    
    return render(request, 'sec_edgar/index.html', context)


def api_13f_holdings(request, ticker):
    """
    獲取指定股票的 13F 機構持股 API
    """
    ticker = ticker.upper()
    try:
        service = Edgar13FService()
        data = service.get_top_holders_from_db(ticker, limit=100)
        return JsonResponse({"status": "success", "data": data})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


def api_insider_trades(request, ticker):
    """
    獲取指定股票的內部人交易 API
    """
    ticker = ticker.upper()
    try:
        trades = SecInsiderTrades.objects.filter(ticker=ticker).order_by('-transaction_date')[:100]
        data = []
        for t in trades:
            data.append({
                'insider_name': t.insider_name,
                'insider_title': t.insider_title,
                'is_senior_officer': t.is_senior_officer,
                'transaction_date': t.transaction_date.strftime('%Y-%m-%d'),
                'transaction_code': t.transaction_code,
                'transaction_type': t.transaction_type,
                'shares': t.shares,
                'price_per_share': float(t.price_per_share) if t.price_per_share else 0.0,
                'total_value': float(t.total_value) if t.total_value else 0.0,
                'shares_owned_after': t.shares_owned_after,
                'filing_date': t.filing_date.strftime('%Y-%m-%d') if t.filing_date else None,
            })
        return JsonResponse({"status": "success", "data": data})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


def api_sync_13f(request):
    """
    手動/非同步觸發同步 13F 持股 API
    """
    ticker = request.GET.get('ticker', '').upper()
    if not ticker:
        return JsonResponse({"status": "error", "message": "Ticker parameter is required"}, status=400)
        
    try:
        service = Edgar13FService()
        # 同步最新一季
        result = service.sync_holdings_by_ticker(ticker, quarters_limit=1)
        return JsonResponse(result)
    except Exception as e:
        logger.error(f"Manual 13F sync failed for {ticker}: {e}")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


def api_sync_insider(request):
    """
    手動/非同步觸發同步內部人交易 API
    """
    ticker = request.GET.get('ticker', '').upper()
    if not ticker:
        return JsonResponse({"status": "error", "message": "Ticker parameter is required"}, status=400)
        
    try:
        service = EdgarInsiderService()
        result = service.sync_insiders_by_ticker(ticker, since='12mo')
        
        # 順便同步財務數據以豐富頁面呈現
        try:
            fin_service = EdgarFinancialService()
            fin_service.sync_financials_by_ticker(ticker)
        except Exception as fin_err:
            logger.warning(f"Auto financial sync during insider sync failed for {ticker}: {fin_err}")
            
        return JsonResponse(result)
    except Exception as e:
        logger.error(f"Manual insider sync failed for {ticker}: {e}")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
