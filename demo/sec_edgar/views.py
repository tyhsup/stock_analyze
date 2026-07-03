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
from sec_edgar.services.edgar_institution_service import EdgarInstitutionService

logger = logging.getLogger(__name__)

def dashboard_view(request):
    """
    SEC EDGAR 整合儀表板視圖
    """
    cik = request.GET.get('cik', '0001067983').strip()
    
    service_inst = EdgarInstitutionService()
    known_institutions = service_inst.KNOWN_INSTITUTIONS
    
    # 預設載入該機構的最新持股資料，供伺服器端初次渲染
    sec_13f_data = []
    latest_13f_period = None
    try:
        sec_13f_data = service_inst.get_institution_holdings(cik, limit=100)
        if sec_13f_data:
            latest_13f_period = sec_13f_data[0]['period_of_report']
    except Exception as e:
        logger.error(f"Error fetching initial holdings for CIK {cik} in dashboard_view: {e}")
        
    context = {
        'current_cik': cik,
        'known_institutions': known_institutions,
        'sec_13f_data': sec_13f_data,
        'latest_13f_period': latest_13f_period,
    }
    
    return render(request, 'sec_edgar/index.html', context)


def api_institution_holdings(request):
    """
    獲取指定機構的 13F 持股 API
    """
    cik = request.GET.get('cik', '0001067983').strip()
    try:
        service = EdgarInstitutionService()
        data = service.get_institution_holdings(cik, limit=200)
        return JsonResponse({"status": "success", "data": data})
    except Exception as e:
        logger.error(f"Error fetching holdings for CIK {cik}: {e}")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


def api_sync_institution_13f(request):
    """
    同步指定機構的 13F 資料 API
    """
    cik = request.GET.get('cik', '0001067983').strip()
    try:
        service = EdgarInstitutionService()
        result = service.sync_institution_13f(cik, quarters_limit=1)
        return JsonResponse(result)
    except Exception as e:
        logger.error(f"Error syncing holdings for CIK {cik}: {e}")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


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


def api_financial_data(request, ticker):
    """
    獲取指定股票的 XBRL 標準化財務數據 API
    """
    ticker = ticker.upper()
    try:
        target_concepts = [
            'Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax',
            'NetIncomeLoss', 'OperatingIncomeLoss', 'Assets', 'Liabilities',
            'EarningsPerShareBasic', 'EarningsPerShareDiluted',
        ]
        qs = SecFinancialXbrl.objects.filter(
            ticker=ticker,
            concept__in=target_concepts
        ).order_by('-period_end', 'concept')[:100]

        data = []
        for f in qs:
            data.append({
                'period_end': f.period_end.strftime('%Y-%m-%d'),
                'concept': f.concept,
                'value': float(f.value),
                'fiscal_year': f.fiscal_year,
                'fiscal_quarter': f.fiscal_quarter,
                'form_type': f.form_type,
            })
        return JsonResponse({"status": "success", "data": data})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
