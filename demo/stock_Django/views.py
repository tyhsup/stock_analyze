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
    force_refresh = request.GET.get("force_refresh", "").lower() == "true"
    if not force_refresh:
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
    
    # 3. 三大法人與信用交易籌碼資料
    chips_features = {}
    try:
        if is_tw:
            clean_sym = valuation_symbol.replace('.TW', '').replace('.TWO', '')
            
            # 優先讀取新表 stock_investor_tw
            try:
                df_inv = pd.read_sql(
                    "SELECT * FROM stock_investor_tw WHERE number = %s ORDER BY date DESC LIMIT 30", 
                    service.sql_op.engine, params=(clean_sym,)
                )
            except Exception as e_new_db:
                logger.warning(f"Query stock_investor_tw failed: {e_new_db}")
                df_inv = pd.DataFrame()
            
            # 若新表無資料，回退至舊表並套用欄位更名
            if df_inv.empty:
                try:
                    df_old = pd.read_sql(
                        "SELECT * FROM stock_investor WHERE number = %s", 
                        service.sql_op.engine, params=(clean_sym,)
                    )
                    if not df_old.empty:
                        df_inv = service.sql_op._fix_investor_columns(df_old)
                        num_cols = len(df_inv.columns)
                        def get_val_by_idx(idx):
                            if idx < num_cols:
                                return pd.to_numeric(df_inv.iloc[:, idx].astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
                            return pd.Series(0.0, index=df_inv.index)
                        
                        df_inv['date'] = df_inv.iloc[:, 0]
                        df_inv['number'] = df_inv.iloc[:, 1]
                        df_inv['name'] = df_inv.iloc[:, 2] if num_cols > 2 else None
                        df_inv['foreign_buy'] = get_val_by_idx(3)
                        df_inv['foreign_sell'] = get_val_by_idx(4)
                        df_inv['foreign_net'] = get_val_by_idx(5)
                        df_inv['trust_buy'] = get_val_by_idx(9)
                        df_inv['trust_sell'] = get_val_by_idx(10)
                        df_inv['trust_net'] = get_val_by_idx(11)
                        df_inv['dealer_buy'] = get_val_by_idx(13) + get_val_by_idx(16)
                        df_inv['dealer_sell'] = get_val_by_idx(14) + get_val_by_idx(17)
                        df_inv['dealer_net'] = get_val_by_idx(12)
                        df_inv['total_net'] = get_val_by_idx(19)
                        
                        df_inv['date'] = pd.to_datetime(df_inv['date']).dt.date
                        df_inv = df_inv.sort_values(by='date', ascending=False)
                except Exception as e_old_db:
                    logger.error(f"Fallback to stock_investor failed: {e_old_db}")
                    df_inv = pd.DataFrame()
            
            # 計算法人近 5 日累計買賣超
            if not df_inv.empty:
                recent_rows = df_inv.head(5)
                total_net = int(recent_rows['total_net'].sum())
                foreign_net = int(recent_rows['foreign_net'].sum())
                investment_net = int(recent_rows['trust_net'].sum())
                dealer_net = int(recent_rows['dealer_net'].sum())
                
                # 同時支援亂碼鍵名（相容舊前端）與乾淨英文鍵名
                chips_features = {
                    "三大法人5日累計買賣超(股)": total_net,
                    "外資5日累計買賣超(股)": foreign_net,
                    "投信5日累計買賣超(股)": investment_net,
                    "自營商5日累計買賣超(股)": dealer_net,
                    # 舊版亂碼欄位鍵名（極致向下相容）
                    "TjkH5֭pRW()": total_net,
                    "~5֭pRW()": foreign_net,
                    "H5֭pRW()": investment_net,
                    "5֭pRW()": dealer_net,
                    # 新增標準英文欄位
                    "total_net_5d": total_net,
                    "foreign_net_5d": foreign_net,
                    "trust_net_5d": investment_net,
                    "dealer_net_5d": dealer_net
                }
            
            # 查詢融資融券餘額數據 (stock_margin_balance) 最近 30 天
            margin_data = []
            try:
                df_margin = pd.read_sql(
                    "SELECT * FROM stock_margin_balance WHERE number = %s ORDER BY date DESC LIMIT 30",
                    service.sql_op.engine, params=(clean_sym,)
                )
                if not df_margin.empty:
                    df_margin['date'] = pd.to_datetime(df_margin['date']).dt.strftime('%Y-%m-%d')
                    margin_data = df_margin.to_dict(orient='records')
            except Exception as e_margin:
                logger.warning(f"Failed to fetch margin data: {e_margin}")
            
            # 查詢最新股權分散數據 (stock_shareholder_distribution)
            distribution_data = {}
            try:
                df_dist = pd.read_sql(
                    "SELECT * FROM stock_shareholder_distribution WHERE number = %s ORDER BY date DESC LIMIT 1",
                    service.sql_op.engine, params=(clean_sym,)
                )
                if not df_dist.empty:
                    row_dist = df_dist.iloc[0]
                    distribution_data = {
                        'date': str(row_dist['date']),
                        'retail_ratio': float(row_dist['class_1_to_5_ratio']),
                        'large_holder_ratio': float(row_dist['class_15_ratio']),
                        'total_shareholders': int(row_dist['total_shareholders'])
                    }
            except Exception as e_dist:
                logger.warning(f"Failed to fetch shareholder distribution: {e_dist}")
            
            # 整合進 chips_features 供後續分析使用
            chips_features['margin_history_30d'] = margin_data
            chips_features['shareholder_distribution'] = distribution_data
        else:
            # 美股籌碼 (Institutional Ownership)
            try:
                with service.sql_op.engine.connect() as conn:
                    # 獲取最新申報日期
                    date_row = conn.execute(
                        text("SELECT MAX(date) FROM stock_investor_us WHERE ticker = :tk"),
                        {"tk": valuation_symbol}
                    ).fetchone()
                    
                    if date_row and date_row[0]:
                        latest_date = date_row[0]
                        # 查詢該日期的前 10 大機構
                        rows = conn.execute(
                            text("""
                                SELECT holder_name, shares, pct_out, value_usd, change_pct 
                                FROM stock_investor_us 
                                WHERE ticker = :tk AND date = :dt
                                ORDER BY shares DESC 
                                LIMIT 10
                            """),
                            {"tk": valuation_symbol, "dt": latest_date}
                        ).fetchall()
                        
                        if rows:
                            holders = []
                            total_inst_pct = 0.0
                            increased_count = 0
                            decreased_count = 0
                            
                            for r in rows:
                                pct = float(r[2] or 0) * 100.0
                                change_pct = float(r[4] or 0) * 100.0
                                total_inst_pct += pct
                                
                                if change_pct > 0.01:
                                    increased_count += 1
                                elif change_pct < -0.01:
                                    decreased_count += 1
                                    
                                holders.append({
                                    "機構名稱": r[0],
                                    "持股數": f"{int(r[1]):,}",
                                    "持股比例": f"{pct:.2f}%",
                                    "市值(美元)": f"${int(r[3] or 0):,}",
                                    "持股變動": f"{change_pct:+.2f}%" if change_pct != 0 else "持平"
                                })
                            
                            chips_features = {
                                "最新申報日期": str(latest_date),
                                "前十大機構合計持股比例": f"{total_inst_pct:.2f}%",
                                "前十大機構增減倉動態": f"加倉: {increased_count} 家, 減倉: {decreased_count} 家, 持平: {len(rows) - increased_count - decreased_count} 家",
                                "前十大機構持股明細": holders
                            }
            except Exception as e_us_chips:
                logger.warning(f"Failed to fetch US chips for Gemini advice: {e_us_chips}")
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
        
    # 6. 新增：總經數據與產業別加載 (子 Agent 觀點)
    industry = "其他/未知"
    latest_macro_data = {}
    try:
        from market_data.models import MacroUS, MacroTW
        
        # 產業別獲取
        if is_tw:
            clean_sym = valuation_symbol.replace('.TW', '').replace('.TWO', '')
            with service.sql_op.engine.connect() as conn:
                row_ind = conn.execute(
                    text("SELECT 產業別 FROM stock_table_tw WHERE 有價證卷代號 = :sym"),
                    {"sym": clean_sym}
                ).fetchone()
                if row_ind and row_ind[0]:
                    industry = row_ind[0].strip()
                    
            # 台灣總經最新數據
            metrics = ['M1B_YOY', 'M2_YOY', 'CPI_YOY', 'TW_DISCOUNT_RATE', 'TW_OVERNIGHT_RATE', 'TW_CORE_CPI_YOY', 'TW_GDP_YOY', 'TW_UNRATE']
            for m in metrics:
                obj = MacroTW.objects.filter(metric=m).order_by('-date').first()
                if obj and obj.value is not None:
                    latest_macro_data[m] = float(obj.value)
        else:
            with service.sql_op.engine.connect() as conn:
                row_ind = conn.execute(
                    text("SELECT sector FROM stock_metadata WHERE symbol = :sym"),
                    {"sym": valuation_symbol}
                ).fetchone()
                if row_ind and row_ind[0]:
                    industry = row_ind[0].strip()
                    
            # 美國總經最新數據
            metrics = ['FEDFUNDS', 'US_YIELD_SPREAD', 'US_CREDIT_SPREAD', 'US_CORE_PCE_YOY', 'US_EXPORT_PRICE_YOY', 'US_MANUFACTURING_ORDERS_YOY']
            for m in metrics:
                obj = MacroUS.objects.filter(metric=m).order_by('-date').first()
                if obj and obj.value is not None:
                    latest_macro_data[m] = float(obj.value)
    except Exception as e_macro_load:
        logger.warning(f"Failed to fetch industry or macro data for advice: {e_macro_load}")

    # 呼叫 Gemini 進行綜合預測與推薦
    try:
        from .stock_cost_AI import IntegratedStockPredModel
        ai_model = IntegratedStockPredModel(valuation_symbol)
        advice = ai_model.generate_gemini_advice(
            lstm_pred=lstm_pred,
            chips_features=chips_features,
            sentiment_summary=sentiment_summary,
            valuation_features=valuation_features,
            latest_price=latest_price,
            industry=industry,
            latest_macro_data=latest_macro_data
        )
        
        # 快取 24 小時 (86400 秒)
        cache.set(cache_key, advice, 86400)
        return JsonResponse({"status": "success", "report": advice, "cached": False})
    except Exception as e_ai:
        logger.error(f"Failed to generate Gemini advice: {e_ai}")
        return JsonResponse({"status": "error", "message": f"Gemini 推理模組異常: {str(e_ai)}"})


def macrotrends_financials_api(request):
    """取得美股財務報表 API (IS, BS, CF)"""
    symbol = request.GET.get("symbol")
    stmt_type = request.GET.get("type")
    frequency = request.GET.get("frequency", "annual")
    
    if not symbol or not stmt_type:
        return JsonResponse({"error": "Missing symbol or type parameter"}, status=400)
        
    from .macrotrends_service import MacrotrendsService
    service = MacrotrendsService()
    try:
        data = service.get_financials(symbol, stmt_type, frequency)
        return JsonResponse(data, safe=False)
    except Exception as e:
        logger.error(f"Error in macrotrends_financials_api: {e}")
        return JsonResponse({"error": str(e)}, status=500)


def macrotrends_ratios_api(request):
    """取得美股財務比率 API"""
    symbol = request.GET.get("symbol")
    ratio_type = request.GET.get("type")
    
    if not symbol or not ratio_type:
        return JsonResponse({"error": "Missing symbol or type parameter"}, status=400)
        
    from .macrotrends_service import MacrotrendsService
    service = MacrotrendsService()
    try:
        data = service.get_ratios(symbol, ratio_type)
        return JsonResponse(data, safe=False)
    except Exception as e:
        logger.error(f"Error in macrotrends_ratios_api: {e}")
        return JsonResponse({"error": str(e)}, status=500)



def macro_dashboard(request):
    """渲染總體經濟 Dashboard 頁面"""
    from django.shortcuts import render
    return render(request, 'macro_dashboard.html')





def macro_data_api(request):
    """獲取台灣與美國總體經濟數據 API (已整合 JSON 序列化保護)"""
    from market_data.models import MacroUS, MacroTW
    
    # 1. 取得美國數據
    us_qs = MacroUS.objects.all().order_by('date')
    us_data = []
    for item in us_qs:
        us_data.append({
            'date': item.date.strftime('%Y-%m-%d') if item.date else None,
            'metric': item.metric,
            'value': float(item.value) if item.value is not None else None
        })
        
    # 2. 取得台灣數據
    tw_qs = MacroTW.objects.all().order_by('date')
    tw_data = []
    for item in tw_qs:
        tw_data.append({
            'date': item.date.strftime('%Y-%m-%d') if item.date else None,
            'metric': item.metric,
            'value': float(item.value) if item.value is not None else None
        })
        


    return JsonResponse({
        'status': 'success',
        'us': us_data,
        'tw': tw_data
    })









