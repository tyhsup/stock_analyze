import os
import sys
import json
import logging
from decimal import Decimal
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def format_row(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    將 row dict 中不支持 JSON 序列化的類型（如 datetime, date, Decimal）轉化為標準類型。
    """
    for k, v in list(row_dict.items()):
        if isinstance(v, (datetime, timedelta)):
            row_dict[k] = v.isoformat()
        elif isinstance(v, date):
            row_dict[k] = v.strftime('%Y-%m-%d')
        elif isinstance(v, Decimal):
            row_dict[k] = float(v)
    return row_dict

# ---------------------------------------------------------
# 雙模式資料庫連線獲取
# ---------------------------------------------------------
def get_db_connection():
    """
    獲取 MySQL 資料庫連線。
    如果在 Django 環境下，直接使用 Django 內建的 db connection；
    如果作為獨立腳本運行，則手動讀取 .env 並建立 mysql.connector 連線。
    """
    try:
        # 嘗試引入 Django db connection (Django 環境)
        from django.db import connection
        # 確保連線已建立
        if connection.connection is None:
            connection.ensure_connection()
        return connection
    except Exception:
        # 作為獨立腳本運行 (手動建立連線)
        import mysql.connector
        from dotenv import load_dotenv
        
        # 尋找並載入 .env
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(base_dir, 'stock_Django', '.env')
        if not os.path.exists(env_path):
            env_path = os.path.join(base_dir, '.env')
        load_dotenv(env_path)
        
        db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': os.getenv('DB_NAME', 'stock_tw_analyse'),
            'port': int(os.getenv('DB_PORT', '3306'))
        }
        
        # 建立連線 (獨立模式)
        conn = mysql.connector.connect(**db_config)
        return conn

# ---------------------------------------------------------
# 美股 yfinance 動態數據抓取與補充機制
# ---------------------------------------------------------
def fetch_and_cache_us_stock(symbol: str, period: str = "1mo") -> bool:
    """
    當本地美股股價數據缺失時，動態使用 yfinance 抓取最新股價並寫入本地 MySQL。
    """
    try:
        import yfinance as yf
        logger.info(f"[yfinance] 開始動態抓取美股 {symbol} 最新股價 (Period: {period})...")
        
        # 1. 抓取歷史數據
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            logger.warning(f"[yfinance] 無法獲取 {symbol} 的數據 (可能是 delisted 或代號錯誤)")
            return False
            
        # 2. 將 DataFrame 格式化寫入 MySQL
        conn = get_db_connection()
        is_django = hasattr(conn, 'cursor') and not hasattr(conn, 'commit')
        
        cursor = conn.cursor() if is_django else conn.cursor()
        
        insert_sql = """
            INSERT INTO stock_cost_us (number, Date, Open, High, Low, Close, Volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                Open = VALUES(Open),
                High = VALUES(High),
                Low = VALUES(Low),
                Close = VALUES(Close),
                Volume = VALUES(Volume)
        """
        
        # 批次寫入數據 (使用 executemany)
        rows_to_insert = []
        for index, row in df.iterrows():
            date_str = index.strftime('%Y-%m-%d')
            rows_to_insert.append((
                symbol,
                date_str,
                float(row['Open']),
                float(row['High']),
                float(row['Low']),
                float(row['Close']),
                int(row['Volume'])
            ))
            
        if is_django:
            # Django connection 模式
            cursor.executemany(insert_sql, rows_to_insert)
        else:
            # 原生 mysql-connector 模式
            cursor.executemany(insert_sql, rows_to_insert)
            conn.commit()
            cursor.close()
            conn.close()
            
        logger.info(f"[yfinance] 成功同步 {len(rows_to_insert)} 筆 {symbol} 美股股價數據至本地資料庫。")
        return True
        
    except Exception as e:
        logger.error(f"[yfinance] 動態抓取 {symbol} 失敗: {e}", exc_info=True)
        return False

# ---------------------------------------------------------
# Core Tools APIs (由 Orchestrator 直接調用)
# ---------------------------------------------------------
def query_taiwan_chips(symbol: str, limit: int = 10) -> Dict[str, Any]:
    """
    台股法人籌碼查詢 Tool (唯讀)。
    查詢最近 limit 天的外資、投信、自營商買賣超股數。
    Traditional Chinese: 查詢台股三大法人的買賣超數據與持股流向。
    """
    limit = min(max(limit, 1), 30)
    symbol_clean = symbol.split('.')[0].strip()
    
    conn = get_db_connection()
    is_django = hasattr(conn, 'cursor') and not hasattr(conn, 'commit')
    cursor = conn.cursor() if is_django else conn.cursor()
    
    # 優先嘗試查詢新表 stock_investor_tw
    sql_new = """
        SELECT date, number, name, foreign_net, trust_net, dealer_net, total_net
        FROM stock_investor_tw
        WHERE number = %s
        ORDER BY date DESC
        LIMIT %s
    """
    
    sql_old = """
        SELECT date, number, 證券名稱 AS name, 
               `外陸資買賣超股數(不含外資自營商)` AS foreign_net, 
               投信買賣超股數 AS trust_net, 
               自營商買賣超股數 AS dealer_net, 
               三大法人買賣超股數 AS total_net
        FROM stock_investor
        WHERE number = %s
        ORDER BY date DESC
        LIMIT %s
    """
    
    results = []
    success = False
    
    # 嘗試 1: 新表
    try:
        cursor.execute("SET NAMES utf8mb4")
        cursor.execute(sql_new, (symbol_clean, limit))
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        for row in rows:
            results.append(format_row(dict(zip(columns, row))))
        if results:
            success = True
            logger.info(f"[LocalDB] 成功從新表 stock_investor_tw 讀取 {len(results)} 筆籌碼數據。")
    except Exception as e_new:
        logger.warning(f"[LocalDB] 查詢新表 stock_investor_tw 失敗: {e_new}，嘗試 fallback 舊表...")
        
    # 嘗試 2: 舊表 (Fallback)
    if not success:
        try:
            results = []
            cursor.execute("SET NAMES utf8mb4")
            cursor.execute(sql_old, (symbol_clean, limit))
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            for row in rows:
                results.append(format_row(dict(zip(columns, row))))
            logger.info(f"[LocalDB] 成功從舊表 stock_investor 讀取 {len(results)} 筆籌碼數據。")
        except Exception as e_old:
            logger.error(f"[LocalDB] Fallback 舊表查詢同樣失敗: {e_old}")
            if not is_django:
                cursor.close()
                conn.close()
            return {"status": "error", "message": f"台股籌碼查詢失敗: {str(e_old)}"}

    if not is_django:
        cursor.close()
        conn.close()
        
    return {
        "status": "success",
        "symbol": symbol_clean,
        "market": "TW",
        "count": len(results),
        "data": results
    }

def query_us_market_data(symbol: str, limit: int = 10) -> Dict[str, Any]:
    """
    美股數據與 13F 機構籌碼查詢 Tool (唯讀)。
    查詢最近 limit 天的歷史股價，並一併查詢該股最新的 13F 機構持股概況。
    若本地股價數據缺失，自動觸發 yfinance 動態補充機制。
    """
    limit = min(max(limit, 1), 30)
    symbol_clean = symbol.upper().split('.')[0].strip()
    
    # 1. 讀取股價
    conn = get_db_connection()
    is_django = hasattr(conn, 'cursor') and not hasattr(conn, 'commit')
    cursor = conn.cursor() if is_django else conn.cursor()
    
    price_sql = """
        SELECT Date, Open, High, Low, Close, Volume
        FROM stock_cost_us
        WHERE number = %s
        ORDER BY Date DESC
        LIMIT %s
    """
    
    try:
        cursor.execute("SET NAMES utf8mb4") # 強制字元集
        cursor.execute(price_sql, (symbol_clean, limit))
        columns = [col[0] for col in cursor.description]
        price_rows = cursor.fetchall()
        
        # 1.1 觸發動態補充機制 (如果本地無股價數據)
        if not price_rows:
            logger.info(f"[LocalDB] 本地無美股 {symbol_clean} 股價數據，啟動 yfinance 動態同步...")
            if not is_django:
                cursor.close()
                conn.close()
                
            success = fetch_and_cache_us_stock(symbol_clean)
            
            # 重新取得連線與資料
            conn = get_db_connection()
            is_django = hasattr(conn, 'cursor') and not hasattr(conn, 'commit')
            cursor = conn.cursor() if is_django else conn.cursor()
            
            if success:
                cursor.execute("SET NAMES utf8mb4")
                cursor.execute(price_sql, (symbol_clean, limit))
                price_rows = cursor.fetchall()
                
        prices = [format_row(dict(zip(columns, row))) for row in price_rows]
        
        # 2. 讀取最新的 13F 機構持股籌碼
        holder_sql = """
            SELECT date, holder_name, shares, pct_out, value_usd, change_shares, change_pct
            FROM stock_investor_us
            WHERE ticker = %s
            ORDER BY date DESC, value_usd DESC
            LIMIT 10
        """
        cursor.execute(holder_sql, (symbol_clean,))
        holder_cols = [col[0] for col in cursor.description]
        holder_rows = cursor.fetchall()
        holders = [format_row(dict(zip(holder_cols, row))) for row in holder_rows]
        
        if not is_django:
            cursor.close()
            conn.close()
            
        return {
            "status": "success",
            "symbol": symbol_clean,
            "market": "US",
            "price_count": len(prices),
            "prices": prices,
            "holder_count": len(holders),
            "holders": holders
        }
        
    except Exception as e:
        logger.error(f"[LocalDB] 美股數據查詢失敗: {e}")
        return {"status": "error", "message": f"美股數據查詢失敗: {str(e)}"}


def query_earnings_data(symbol: str, limit_quarters: int = 8) -> Dict[str, Any]:
    """
    查詢該股票最近的財報原始項目數據 (台股或美股)。
    Traditional Chinese: 查詢台股或美股最近多季的財報數據。
    """
    limit_quarters = min(max(limit_quarters, 1), 16)
    symbol_clean = symbol.upper().split('.')[0].strip()
    is_tw = symbol.isdigit() or ".TW" in symbol.upper()
    table_name = "financial_raw_tw" if is_tw else "financial_raw_us"
    
    conn = get_db_connection()
    is_django = hasattr(conn, 'cursor') and not hasattr(conn, 'commit')
    cursor = conn.cursor() if is_django else conn.cursor()
    
    # 1. 先查最近有資料的 year 與 quarter
    periods_sql = f"""
        SELECT DISTINCT year, quarter 
        FROM {table_name} 
        WHERE symbol = %s 
        ORDER BY year DESC, quarter DESC 
        LIMIT %s
    """
    try:
        cursor.execute("SET NAMES utf8mb4")
        cursor.execute(periods_sql, (symbol_clean, limit_quarters))
        periods = cursor.fetchall()
        
        if not periods:
            if not is_django:
                cursor.close()
                conn.close()
            return {"status": "success", "symbol": symbol_clean, "data": []}
            
        # 2. 針對這些季度，查詢所有的財報項目
        results = []
        for year, quarter in periods:
            item_sql = f"""
                SELECT item_name, amount 
                FROM {table_name} 
                WHERE symbol = %s AND year = %s AND quarter = %s
            """
            cursor.execute(item_sql, (symbol_clean, year, quarter))
            items = cursor.fetchall()
            
            items_dict = {}
            for item_name, amount in items:
                items_dict[item_name] = float(amount) if amount is not None else 0.0
                
            results.append({
                "year": year,
                "quarter": quarter,
                "items": items_dict
            })
            
        if not is_django:
            cursor.close()
            conn.close()
            
        return {
            "status": "success",
            "symbol": symbol_clean,
            "market": "TW" if is_tw else "US",
            "data": results
        }
    except Exception as e:
        logger.error(f"[LocalDB] 財報數據查詢失敗: {e}")
        if not is_django:
            try: cursor.close()
            except: pass
            try: conn.close()
            except: pass
        return {"status": "error", "message": f"財報數據查詢失敗: {str(e)}"}


# ---------------------------------------------------------
# JSON-RPC Stdio 標準 MCP 通用服務器進入點 (雙模式支援)
# ---------------------------------------------------------
def main():
    """
    如果直接執行此腳本，則以標準 Stdio JSON-RPC 形式運作 (符合 MCP Server 標準)。
    """
    logging.basicConfig(level=logging.INFO)
    print("Local DB MCP Server 啟動中...", file=sys.stderr)
    
    try:
        for line in sys.stdin:
            request = json.loads(line)
            req_id = request.get("id")
            method = request.get("method")
            params = request.get("params", {})
            
            response = {"jsonrpc": "2.0", "id": req_id}
            
            if method == "query_taiwan_chips":
                symbol = params.get("symbol")
                limit = params.get("limit", 10)
                response["result"] = query_taiwan_chips(symbol, limit)
            elif method == "query_us_market_data":
                symbol = params.get("symbol")
                limit = params.get("limit", 10)
                response["result"] = query_us_market_data(symbol, limit)
            elif method == "query_earnings_data":
                symbol = params.get("symbol")
                limit_quarters = params.get("limit_quarters", 8)
                response["result"] = query_earnings_data(symbol, limit_quarters)
            else:
                response["error"] = {"code": -32601, "message": "Method not found"}
                
            print(json.dumps(response))
            sys.stdout.flush()
    except Exception as e:
        print(f"MCP Stdio 錯誤: {e}", file=sys.stderr)

if __name__ == '__main__':
    main()
