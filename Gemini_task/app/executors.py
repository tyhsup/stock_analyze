import os
import sys
import logging

# 動態添加 mydjango/demo 到 Python path 中，使 stock_Django module 可以正確載入
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "demo"))
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)

# 設定 logger
logger = logging.getLogger("scheduler.executors")

try:
    from stock_Django.stock_cost import StockCostManager
    from stock_Django.stock_investor import StockInvestorManager
    from stock_Django.stock_investor_tpex import TPExInvestorManager
    from stock_Django.stock_investor_us import USStockInvestorManager
except ImportError as e:
    logger.error(f"無法載入 stock_Django 中的管理器: {e}")
    raise e

def execute_tw_stock_cost(remarks: str = None):
    """
    台股股價與籌碼面更新任務執行器
    若 remarks 中包含特定的股票代碼（如 2330），則執行單一更新，否則全量更新。
    """
    manager = StockCostManager()
    ticker = (remarks or "").strip()
    
    if ticker:
        # 如果使用者輸入了多個代號，如 "2330, 2454"，逐一更新
        tickers = [t.strip() for t in ticker.replace("，", ",").split(",") if t.strip()]
        if len(tickers) > 1:
            logger.info(f"開始更新多個台股代號: {tickers}")
            success = True
            for t in tickers:
                if not manager.update_single_ticker(t):
                    success = False
            if not success:
                raise Exception(f"部分台股股價更新失敗: {tickers}")
        else:
            logger.info(f"開始更新單一台股: {tickers[0]}")
            if not manager.update_single_ticker(tickers[0]):
                raise Exception(f"更新單一台股股價失敗: {tickers[0]}")
    else:
        logger.info("開始更新全台股股價 (上市及上櫃)")
        manager.update_all_cost()

def execute_us_stock_cost(remarks: str = None):
    """
    美股股價與籌碼面更新任務執行器
    若 remarks 中包含特定的股票代碼（如 AAPL），則執行單一更新，否則全量更新。
    """
    manager = StockCostManager()
    ticker = (remarks or "").strip()
    
    if ticker:
        tickers = [t.strip().upper() for t in ticker.replace("，", ",").split(",") if t.strip()]
        if len(tickers) > 1:
            logger.info(f"開始更新多個美股代號: {tickers}")
            success = True
            for t in tickers:
                if not manager.update_single_ticker(t):
                    success = False
            if not success:
                raise Exception(f"部分美股股價更新失敗: {tickers}")
        else:
            logger.info(f"開始更新單一美股: {tickers[0]}")
            if not manager.update_single_ticker(tickers[0]):
                raise Exception(f"更新單一美股股價失敗: {tickers[0]}")
    else:
        logger.info("開始更新全美股股價")
        manager.update_us_cost()

def execute_twse_investor(remarks: str = None):
    """
    台股三大法人買賣超 (上市) 更新任務執行器
    """
    logger.info("開始更新台股三大法人買賣超 (上市)")
    manager = StockInvestorManager()
    manager.update_investor_data()

def execute_tpex_investor(remarks: str = None):
    """
    台股三大法人買賣超 (上櫃) 更新任務執行器
    """
    logger.info("開始更新台股三大法人買賣超 (上櫃)")
    manager = TPExInvestorManager()
    manager.update_all_tpex_investors()

def execute_us_investor(remarks: str = None):
    """
    美股三大法人持股更新任務執行器
    若 remarks 中指定了代號 (如 AAPL,NVDA)，則更新該清單。
    否則預設更新 ['AAPL', 'NVDA', 'TSLA']。
    """
    manager = USStockInvestorManager()
    ticker_str = (remarks or "").strip()
    
    if ticker_str:
        tickers = [t.strip().upper() for t in ticker_str.replace("，", ",").split(",") if t.strip()]
    else:
        tickers = ['AAPL', 'NVDA', 'TSLA']
        
    logger.info(f"開始更新美股法人持股: {tickers}")
    manager.update_investor_db(tickers)

def execute_tw_stock_price_only(remarks: str = None):
    """
    僅更新台灣股價更新任務執行器
    若 remarks 中包含特定的股票代碼（如 2330），則執行單一更新，否則全量更新。
    """
    execute_tw_stock_cost(remarks)

def execute_us_stock_price_only(remarks: str = None):
    """
    僅更新美國股價更新任務執行器
    若 remarks 中包含特定的股票代碼（如 AAPL），則執行單一更新，否則全量更新。
    """
    execute_us_stock_cost(remarks)

def execute_tw_listed_list_update(remarks: str = None):
    """
    台灣上市公司清單更新任務執行器
    """
    logger.info("開始更新台灣上市公司清單...")
    from stock_Django.mySQL_OP import OP_Fun
    import requests
    import pandas as pd
    from io import StringIO
    
    op = OP_Fun()
    url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
    res = requests.get(url, timeout=15)
    res.encoding = 'big5'
    
    dfs = pd.read_html(StringIO(res.text))
    if not dfs:
        raise Exception("無法讀取 HTML 表格")
    df = dfs[0]
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    
    all_stocks = []
    current_type = ""
    for _, row in df.iterrows():
        val = str(row['有價證券代號及名稱'])
        if "　" not in val and len(val) > 0:
            current_type = val
        elif current_type == "股票" and "　" in val:
            symbol, name = val.split("　", 1)
            all_stocks.append({"symbol": symbol, "name": name, "market": "sii"})
            
    if all_stocks:
        op.upsert_stocks_info(pd.DataFrame(all_stocks), market='tw')
        logger.info(f"更新台灣上市公司成功，共 {len(all_stocks)} 筆")
    else:
        raise Exception("未找到任何台灣上市公司資料")

def execute_tw_otc_list_update(remarks: str = None):
    """
    台灣上櫃公司清單更新任務執行器
    """
    logger.info("開始更新台灣上櫃公司清單...")
    from stock_Django.mySQL_OP import OP_Fun
    import requests
    import pandas as pd
    from io import StringIO
    
    op = OP_Fun()
    url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
    res = requests.get(url, timeout=15)
    res.encoding = 'big5'
    
    dfs = pd.read_html(StringIO(res.text))
    if not dfs:
        raise Exception("無法讀取 HTML 表格")
    df = dfs[0]
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    
    all_stocks = []
    current_type = ""
    for _, row in df.iterrows():
        val = str(row['有價證券代號及名稱'])
        if "　" not in val and len(val) > 0:
            current_type = val
        elif current_type == "股票" and "　" in val:
            symbol, name = val.split("　", 1)
            all_stocks.append({"symbol": symbol, "name": name, "market": "otc"})
            
    if all_stocks:
        op.upsert_stocks_info(pd.DataFrame(all_stocks), market='tw')
        logger.info(f"更新台灣上櫃公司成功，共 {len(all_stocks)} 筆")
    else:
        raise Exception("未找到任何台灣上櫃公司資料")

def execute_us_stock_list_update(remarks: str = None):
    """
    美國上市公司清單更新任務執行器
    """
    logger.info("開始更新美國上市公司清單...")
    from stock_Django.mySQL_OP import OP_Fun
    from stock_Django.scraper_us import get_us_stock_list
    
    op = OP_Fun()
    us_stocks = get_us_stock_list()
    if not us_stocks.empty:
        op.upsert_stocks_info(us_stocks, market='us')
        logger.info(f"更新美國上市公司成功，共 {len(us_stocks)} 筆")
    else:
        raise Exception("未找到任何美國上市公司資料")

# 任務執行器對照表
EXECUTORS = {
    "tw_stock_cost": execute_tw_stock_cost,
    "us_stock_cost": execute_us_stock_cost,
    "tw_stock_price_only": execute_tw_stock_price_only,
    "us_stock_price_only": execute_us_stock_price_only,
    "twse_investor": execute_twse_investor,
    "tpex_investor": execute_tpex_investor,
    "us_investor": execute_us_investor,
    "tw_listed_list_update": execute_tw_listed_list_update,
    "tw_otc_list_update": execute_tw_otc_list_update,
    "us_stock_list_update": execute_us_stock_list_update
}
