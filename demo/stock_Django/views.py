from django.shortcuts import render,redirect,reverse
import json
from demo.forms import StocknumberInput, Economynews
import urllib,base64
import io,os
from io import BytesIO
import matplotlib.pyplot as plt
import mplfinance as mpf
from stock_Django import mySQL_OP

from stock_Django import stock_chart
from .stock_utils import StockUtils
from .stock_investor_us import USStockInvestorManager
from .news_excel import NewsExcelManager
from .news_scraper_cnyes import CnyesScraper
from .data_freshness import trigger_refresh_if_stale, get_refresh_status, trigger_news_refresh, check_news_freshness
from .nlp_service import NLPService
from django.http import JsonResponse
import time
import pandas as pd
import threading
import logging
from datetime import datetime
from .services import StockService
from .gemma_advisor_service import GemmaAdvisorService

logger = logging.getLogger(__name__)

# Create your views here.
def News_display(request):
    data = []
    form = Economynews(request.GET or None)
    news_mgr = NewsExcelManager()
    
    quantity_warning = False
    if form.is_valid():
        query = form.cleaned_data['news_query'].strip().upper()
        limit = form.cleaned_data['news_days']
        
        # 1. Try finding news for specific ticker first
        data = news_mgr.read_news(query, limit=limit)
        
        # 2. Check if we have enough data or if it's stale; if so, trigger a background refresh
        is_ticker = (query.isdigit() and len(query) >= 4) or \
                     ".TW" in query or \
                     (query.isalpha() and 1 <= len(query) <= 5)
        
        if is_ticker:
            is_fresh = check_news_freshness(query)
            
            if len(data) < limit or not is_fresh:
                if not is_fresh:
                    logger.info(f"[News] {query} is stale. Triggering background refresh.")
                trigger_news_refresh(query, limit=limit)
                
                if len(data) < limit:
                    quantity_warning = True

        # Fallback to general search if still empty (for non-ticker queries or failed ticker search)
        if not data:
            data = news_mgr.read_news_general(query=query, limit=limit)
            if len(data) < limit:
                quantity_warning = True
        
        # 3. Dynamic Sentiment Enhancement (Optional but high value)
        try:
            nlp = NLPService()
            for item in data:
                if item.get('正負分析') == '中性':
                    # 優先使用內容分析，否則使用標題
                    analysis_text = item.get('內容') or item.get('標題', '')
                    res = nlp.analyze_sentiment(analysis_text)
                    if res.get('label') != 'error':
                        item['正負分析'] = '正面' if res['label'] == 'positive' else '負面'
        except Exception as e_nlp:
            logger.warning(f"Dynamic NLP analysis skipped: {e_nlp}")
            
    return render(request, 'news_display.html', {
        'form': form, 
        'data': data, 
        'quantity_warning': quantity_warning
    })

def refresh_status_api(request, ticker):
    """API endpoint to check data refresh progress."""
    status = get_refresh_status(ticker)
    return JsonResponse(status)

def smart_advisor_analysis(request, ticker):
    """
    智慧顧問分析：整合技術面、情緒面與估值，調用本地 Gemma 4 進行推理。
    """
    service = StockService()
    advisor = GemmaAdvisorService()
    
    # 1. 取得基礎數據
    # 這裡儘量復用 existing logic
    
    # 技術預測
    pred_data = service._get_ai_predictions(ticker)
    
    # 估值 (調用 valuation_service)
    try:
        from valuation.services.valuation_service import ValuationService
        val_results = ValuationService.calculate_valuation(ticker)
    except:
        val_results = {"error": "估值模組未就緒"}
        
    # 情緒 (讀取最新新聞統計)
    try:
        from .news_excel import NewsExcelManager
        news_mgr = NewsExcelManager()
        news = news_mgr.read_news(ticker, limit=20)
        pos = sum(1 for n in news if n.get('正負分析') == '正面')
        neg = sum(1 for n in news if n.get('正負分析') == '負面')
        sentiment_summary = {
            'positive': pos,
            'negative': neg,
            'label': '偏多' if pos > neg else ('偏空' if neg > pos else '中性'),
            'score': pred_data.get('latest', {}).get('trend_probability', 0.5) * 100 if pred_data.get('latest') else 50
        }
    except:
        sentiment_summary = {'label': '未知'}

    # 2. 準備給 Advisor 的數據
    advisor_input = {
        'trend_label': '看漲' if sentiment_summary.get('label') == '偏多' else '盤整/看跌',
        'sentiment_summary': sentiment_summary,
        'valuation': val_results
    }
    
    # 3. 觸發 Gemma 推理 (這可能較久，故前面已將所有數據準備好)
    report = advisor.get_structured_advice(ticker, advisor_input)
    
    return JsonResponse({
        'ticker': ticker,
        'report': report,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

def news_refresh_api(request, ticker):
    """API endpoint to trigger news refresh from 鉅亨網."""
    trigger_news_refresh(ticker)
    return JsonResponse({'status': 'triggered', 'ticker': ticker})

from .services import StockService

def home(request):
    if request.method == 'POST':
        form = StocknumberInput(request.POST)
        if form.is_valid():
            number = form.cleaned_data['stock_number']
            days = form.cleaned_data['days']
            
            service = StockService()
            result = service.get_stock_data(number, int(days))
            
            # P0-A: Sentiment Analysis Integration for Dashboard
            try:
                from .news_excel import NewsExcelManager
                news_mgr = NewsExcelManager()
                cached_news = news_mgr.read_news(number, limit=20)
                if cached_news:
                    pos = sum(1 for n in cached_news if n.get('正負分析') == '正面')
                    neg = sum(1 for n in cached_news if n.get('正負分析') == '負面')
                    neu = len(cached_news) - pos - neg
                    result['sentiment_summary'] = {
                        'positive': pos,
                        'negative': neg,
                        'neutral': neu,
                        'total': len(cached_news),
                        'label': '正面' if pos > neg else ('負面' if neg > pos else '中性'),
                        'score': round((pos + 0.5 * neu) / len(cached_news) * 100, 1) if cached_news else 50
                    }
            except Exception as e_s:
                import logging
                logging.getLogger(__name__).warning(f"Dashboard sentiment failed: {e_s}")
            
            # FIXED: Relax error check to account for OTC stocks and successful valuation_symbol loading
            if not result['kline_json'] and result['historical_data'].empty:
                return render(request, 'home.html', {
                    'form': form,
                    'error': result['error'] or f"No data found for ticker '{number}'. Please ensure the ticker exists."
                })
            
            # Update session for auto-loading valuation
            request.session['last_ticker'] = result['valuation_symbol']

            return render(request, 'home.html', {
                'form': form,
                'number': number,
                **result # Unpack all service results directly into context
            })
    else:
        form = StocknumberInput()
    return render(request, 'home.html', context={'form': form})
