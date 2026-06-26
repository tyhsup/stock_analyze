import logging
import pandas as pd
import numpy as np
from django.http import JsonResponse
from django.db import connection
from market_data.models import StockUS

logger = logging.getLogger(__name__)

def api_us_money_flow(request):
    """
    美股個股資金流入計算 API
    透過收盤相對位置資金流 (CLV Money Flow) 算法計算特定天數的累計資金流入率，並回傳成交量前 N 大的股票供 TreeMap 渲染。
    """
    try:
        # 解析參數
        days = int(request.GET.get('days', 5))
        limit = int(request.GET.get('limit', 200))
        
        # 限制範圍防錯
        days = max(1, min(20, days))
        limit = max(10, min(500, limit))
        
        # 1. 取得最近的交易日期 (從 stock_cost_us)
        with connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT Date FROM stock_cost_us ORDER BY Date DESC LIMIT %s", [days + 1])
            rows = cursor.fetchall()
            
        if not rows:
            return JsonResponse({"status": "success", "data": []})
            
        target_dates = [r[0] for r in rows]
        
        # 2. 獲取這幾天內所有美股的日股價與成交量數據
        # 由於 Django ORM 在大量 Time-series 資料查詢上較慢，此處使用 Raw SQL 搭配 pandas 以符合 vectorization 要求
        query = """
            SELECT number AS symbol, Date as date, Open as open, High as high, Low as low, Close as close, Volume as volume
            FROM stock_cost_us
            WHERE Date IN %s
        """
        
        df = pd.read_sql(query, con=connection, params=[tuple(target_dates)])
        
        if df.empty:
            return JsonResponse({"status": "success", "data": []})
            
        # 轉換數值型態
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # 3. 計算 CLV 與 Daily Money Flow
        # CLV = ((Close - Low) - (High - Close)) / (High - Low)
        # DMF = CLV * Close * Volume
        high_low_diff = df['high'] - df['low']
        
        # 避免分母為 0
        df['clv'] = np.where(
            high_low_diff > 0,
            ((df['close'] - df['low']) - (df['high'] - df['close'])) / high_low_diff,
            0.0
        )
        df['daily_volume_value'] = df['close'] * df['volume']
        df['dmf'] = df['clv'] * df['daily_volume_value']
        
        # 4. 依個股分群計算滾動累計值與最新一日的成交量能
        grouped = df.groupby('symbol')
        
        results = []
        for symbol, grp in grouped:
            # 排序確保時間序列順序
            grp_sorted = grp.sort_values('date')
            
            # 累計成交量能與淨資金流向
            total_vol_val = grp_sorted['daily_volume_value'].sum()
            total_dmf = grp_sorted['dmf'].sum()
            
            if total_vol_val > 0:
                net_flow_ratio = (total_dmf / total_vol_val) * 100.0
            else:
                net_flow_ratio = 0.0
                
            # 取得最新一天的資料作為基本資訊
            latest_row = grp_sorted.iloc[-1]
            
            results.append({
                'symbol': symbol,
                'volume_value': float(latest_row['daily_volume_value']),
                'net_flow_ratio': round(float(net_flow_ratio), 2),
                'close': float(latest_row['close'])
            })
            
        # 5. 轉換為 DataFrame 進行降序排序並過濾前 limit 名
        res_df = pd.DataFrame(results)
        if res_df.empty:
            return JsonResponse({"status": "success", "data": []})
            
        res_df = res_df.sort_values('volume_value', ascending=False).head(limit)
        
        # 6. 關聯美股基本資料表取得中文/英文名稱
        symbols_to_fetch = res_df['symbol'].tolist()
        stocks_info = {s.symbol: s.name for s in StockUS.objects.filter(symbol__in=symbols_to_fetch)}
        
        final_data = []
        for _, row in res_df.iterrows():
            sym = row['symbol']
            name = stocks_info.get(sym, sym)
            final_data.append({
                'symbol': sym,
                'name': name,
                'volume_value': round(row['volume_value'] / 1000000.0, 2), # 轉換為百萬美元 (M)
                'net_flow_ratio': row['net_flow_ratio'],
                'close': row['close']
            })
            
        return JsonResponse({"status": "success", "data": final_data})
        
    except Exception as e:
        logger.error(f"api_us_money_flow error: {e}", exc_info=True)
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
