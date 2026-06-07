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
from .data_freshness import trigger_refresh_if_stale, get_refresh_status, get_news_refresh_status, trigger_news_refresh, check_news_freshness
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
    news_status = get_news_refresh_status(ticker)
    if news_status.get('status') in ['running', 'error'] or (news_status.get('status') == 'done' and '新聞' in news_status.get('message', '')):
        return JsonResponse(news_status)
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
    limit = request.GET.get('limit', 20)
    try:
        limit = int(limit)
    except ValueError:
        limit = 20
    trigger_news_refresh(ticker, limit=limit)
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


def gemini_advisor_analysis(request, ticker):
    """
    雲端 Gemini 投資建議 API。
    綜合分析 LSTM 預測、法人籌碼、新聞情緒、基本估值與最新歷史股價，
    使用快取機制（24 小時/1天）。
    """
    from django.core.cache import cache
    from sqlalchemy import text
    ticker = str(ticker).strip().upper()
    cache_key = f"gemini_advice_{ticker}"
    
    # 嘗試讀取快取
    cached_data = cache.get(cache_key)
    if cached_data:
        return JsonResponse({"status": "success", "report": cached_data, "cached": True})
        
    service = StockService()
    
    # 1. 取得最新股價與歷史趨勢
    is_tw = ticker.isdigit() or ".TW" in ticker or ".TWO" in ticker
    table_name = 'stock_cost' if is_tw else 'stock_cost_us'
    
    # 解析出正確的股票代碼
    valuation_symbol = ticker
    if is_tw and not (ticker.endswith('.TW') or ticker.endswith('.TWO')):
        suffix = service._resolve_tw_suffix(ticker)
        valuation_symbol = f"{ticker}{suffix}"
        
    hist_data_db, hist_date = StockUtils.load_data_c(table_name, valuation_symbol)
    if hist_data_db.empty:
        return JsonResponse({"status": "error", "message": f"找不到股票 {ticker} 的歷史價格數據。"})
        
    latest_price = float(hist_data_db['Close'].iloc[-1])
    
    # 2. LSTM 技術預測 (僅從資料庫讀取快取，避免 Cache Miss 時執行耗時的 TensorFlow 訓練)
    from datetime import datetime
    today_str = datetime.now().strftime('%Y-%m-%d')
    lstm_pred = {}
    try:
        with service.sql_op.engine.connect() as conn:
            row = conn.execute(
                text("SELECT pred_day_1, pred_day_2, pred_day_3, pred_day_4, pred_day_5, trend_probability FROM stock_ai_predictions WHERE symbol = :sys AND date = :dt"),
                {'sys': valuation_symbol, 'dt': today_str}
            ).fetchone()
            if row:
                lstm_pred = {
                    'predictions': [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])],
                    'trend_probability': float(row[5])
                }
    except Exception as e_db_pred:
        logger.warning(f"Failed to read quick AI predictions cache: {e_db_pred}")
    
    # 3. 法人籌碼
    chips_features = {}
    try:
        if is_tw:
            # 取得台股法人近 5 日買賣超數據
            clean_sym = valuation_symbol.replace('.TW', '').replace('.TWO', '')
            # 使用 pandas read_sql 載入，避開直接在 SQL 語法中傳送中文字串產生的編碼亂碼問題
            df_inv = pd.read_sql("SELECT * FROM stock_investor WHERE number = %s", service.sql_op.engine, params=(clean_sym,))
            if not df_inv.empty:
                # 尋找日期欄位
                date_cols = [c for c in df_inv.columns if '日' in c or 'Date' in c]
                if date_cols:
                    df_inv = df_inv.sort_values(by=date_cols[0], ascending=False)
                
                recent_rows = df_inv.head(5)
                
                # 動態匹配中文列名
                total_cols = [c for c in df_inv.columns if '三大' in c]
                foreign_cols = [c for c in df_inv.columns if '外資' in c or '外陸' in c]
                inv_cols = [c for c in df_inv.columns if '投信' in c]
                dealer_cols = [c for c in df_inv.columns if '自營' in c]
                
                total_net = int(recent_rows[total_cols[0]].sum()) if total_cols else 0
                foreign_net = int(recent_rows[foreign_cols[0]].sum()) if foreign_cols else 0
                investment_net = int(recent_rows[inv_cols[0]].sum()) if inv_cols else 0
                dealer_net = int(recent_rows[dealer_cols[0]].sum()) if dealer_cols else 0
                
                chips_features = {
                    "三大法人5日累計買賣超(股)": total_net,
                    "外資5日累計買賣超(股)": foreign_net,
                    "投信5日累計買賣超(股)": investment_net,
                    "自營商5日累計買賣超(股)": dealer_net
                }
        else:
            # 美股機構持股
            with service.sql_op.engine.connect() as conn:
                rows = conn.execute(
                    text("SELECT holder_name, shares, pct_out FROM stock_investor_us WHERE ticker = :tk ORDER BY date DESC LIMIT 5"),
                    {"tk": valuation_symbol}
                ).fetchall()
                if rows:
                    holders = []
                    for r in rows:
                        holders.append({"機構名稱": r[0], "持股比例": f"{float(r[2]):.2f}%"})
                    chips_features = {"前五大機構持股": holders}
    except Exception as e_chips:
        logger.warning(f"Failed to fetch chips for Gemini advice: {e_chips}")
        
    # 4. 新聞輿情情緒
    sentiment_summary = {"positive": 0, "negative": 0, "neutral": 0, "label": "中性", "score": 50.0}
    try:
        from .news_excel import NewsExcelManager
        news_mgr = NewsExcelManager()
        cached_news = news_mgr.read_news(ticker, limit=20)
        if cached_news:
            pos = sum(1 for n in cached_news if n.get('正負分析') == '正面')
            neg = sum(1 for n in cached_news if n.get('正負分析') == '負面')
            neu = len(cached_news) - pos - neg
            sentiment_summary = {
                'positive': pos,
                'negative': neg,
                'neutral': neu,
                'label': '正面' if pos > neg else ('負面' if neg > pos else '中性'),
                'score': float(round((pos + 0.5 * neu) / len(cached_news) * 100, 1))
            }
    except Exception as e_senti:
        logger.warning(f"Failed to fetch sentiment for Gemini advice: {e_senti}")
        
    # 5. 基本面估值
    valuation_features = {"fair_value": "N/A", "upside": 0.0, "rating": "N/A"}
    try:
        from valuation.services.valuation_service import ValuationService
        val_results = ValuationService.calculate_valuation(valuation_symbol)
        if val_results and "error" not in val_results:
            valuation_features = {
                "fair_value": f"{val_results.get('fair_value', 'N/A')}",
                "upside": float(val_results.get('upside', 0.0) * 100),
                "rating": val_results.get('rating', 'N/A')
            }
    except Exception as e_val:
        logger.warning(f"Failed to fetch valuation for Gemini advice: {e_val}")
        
    # 呼叫 Gemini 進行綜合預測與推薦
    try:
        from .stock_cost_AI import IntegratedStockPredModel
        ai_model = IntegratedStockPredModel(valuation_symbol)
        advice = ai_model.generate_gemini_advice(
            lstm_pred=lstm_pred,
            chips_features=chips_features,
            sentiment_summary=sentiment_summary,
            valuation_features=valuation_features,
            latest_price=latest_price
        )
        
        # 快取 24 小時 (86400 秒)
        cache.set(cache_key, advice, 86400)
        return JsonResponse({"status": "success", "report": advice, "cached": False})
    except Exception as e_ai:
        logger.error(f"Failed to generate Gemini advice: {e_ai}")
        return JsonResponse({"status": "error", "message": f"Gemini 推理模組異常: {str(e_ai)}"})

