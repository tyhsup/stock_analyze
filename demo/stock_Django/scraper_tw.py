import requests
import pandas as pd
from scraper_utils import get_random_ua, smart_delay, get_session
from mySQL_OP import OP_Fun
import logging

logger = logging.getLogger(__name__)

def get_tw_stock_list():
    """從證交所獲取所有上市公司(及上櫃)代碼"""
    urls = [
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2", # 上市
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"  # 上櫃
    ]
    
    all_stocks = []
    for url in urls:
        smart_delay(1, 2)
        res = requests.get(url)
        res.encoding = 'big5'
        df = pd.read_html(res.text)[0]
        
        # 清理資料：第一列是標題，過濾出股票類型
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        
        # 只保留「有價證券代號及名稱」包含股票代號的部分 (通常是 "代號　名稱")
        # 並過濾出「股票」類別
        current_type = ""
        for _, row in df.iterrows():
            val = str(row['有價證券代號及名稱'])
            if "　" not in val and len(val) > 0:
                current_type = val
            elif current_type == "股票" and "　" in val:
                symbol, name = val.split("　", 1)
                market = "sii" if "strMode=2" in url else "otc"
                all_stocks.append({"symbol": symbol, "name": name, "market": market})
                
    return pd.DataFrame(all_stocks)

def scrape_tw_financials(symbol, year, quarter):
    """從 MOPS 抓取財報原始數據"""
    roc_year = year - 1911
    
    # MOPS 請求路徑
    is_url = "https://mops.twse.com.tw/mops/web/ajax_t164sb04" # 損益
    bs_url = "https://mops.twse.com.tw/mops/web/ajax_t164sb03" # 資產
    cf_url = "https://mops.twse.com.tw/mops/web/ajax_t164sb05" # 現流
    
    statements = {'IS': is_url, 'BS': bs_url, 'CF': cf_url}
    all_items = []
    
    # 建立 Session 以保持 Cookies
    session = get_session()
    
    # 先訪問首頁獲取初始 Cookies
    try:
        session.get("https://mops.twse.com.tw/mops/web/t100sb07_1", timeout=10)
        smart_delay(1, 2)
    except:
        pass

    # 嘗試判斷 TYPEK (sii 或 otc)
    # 實務上可以從資料庫查，這裡我們先試 sii，失敗再試 otc
    for stmt_type, url in statements.items():
        found = False
        for typek in ['sii', 'otc']:
            if found: break
            
            smart_delay(3, 5)
            payload = {
                'encodeURIComponent': '1',
                'step': '1',
                'firstin': '1',
                'off': '1',
                'keyword4': '',
                'code1': '',
                'TYPEK': typek,
                'checkbtn': '',
                'queryName': 'co_id',
                'inpuType': 'co_id',
                'TYPEK2': '',
                'co_id': symbol,
                'year': str(roc_year),
                'season': str(quarter).zfill(2)
            }
            
            headers = {
                "User-Agent": get_random_ua(),
                "Referer": "https://mops.twse.com.tw/mops/web/t164sb03",
                "Origin": "https://mops.twse.com.tw",
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
            }
            
            try:
                res = session.post(url, data=payload, headers=headers, timeout=20)
                res.encoding = 'utf-8'
                
                if "查詢無資料" in res.text:
                    continue
                if "頁面無法執行" in res.text:
                    print(f"DEBUG: Security Block for {symbol} {typek} {stmt_type}")
                    continue
                
                from io import StringIO
                dfs = pd.read_html(StringIO(res.text))
                if not dfs:
                    print(f"DEBUG: No tables for {symbol} {typek} {stmt_type}")
                    continue
                
                target_df = None
                for df in dfs:
                    if df.shape[1] >= 2:
                        target_df = df
                        break
                
                if target_df is not None:
                    found = True
                    if isinstance(target_df.columns, pd.MultiIndex):
                        target_df.columns = target_df.columns.get_level_values(-1)
                    
                    col_name = target_df.columns[0]
                    amount_name = target_df.columns[1]
                    
                    item_count = 0
                    for _, row in target_df.iterrows():
                        item = str(row[col_name]).strip()
                        amount_val = row[amount_name]
                        try:
                            if pd.isna(amount_val): continue
                            amount = float(str(amount_val).replace(',', '').replace('(', '-').replace(')', ''))
                            all_items.append({
                                'symbol': symbol, 'year': year, 'quarter': quarter,
                                'statement_type': stmt_type, 'item_name': item, 'amount': amount
                            })
                            item_count += 1
                        except: continue
                    print(f"DEBUG: Successfully scraped {stmt_type} for {symbol} ({typek}), items: {item_count}")
                    logger.info(f"Successfully scraped {stmt_type} for {symbol} ({typek})")
            except Exception as e:
                print(f"DEBUG: Error for {symbol} {typek} {stmt_type}: {e}")
                logger.error(f"Error scraping {stmt_type} for {symbol} ({typek}): {e}")
                
    return pd.DataFrame(all_items)

if __name__ == "__main__":
    op = OP_Fun()
    
    # 測試 1: 獲取股票清單並更新
    print("Fetching TW stock list...")
    stock_list = get_tw_stock_list()
    if not stock_list.empty:
        op.upsert_stocks_info(stock_list, market='tw')
        print(f"Updated {len(stock_list)} TW stocks.")
        
    # 測試 2: 抓取 2330 (台積電) 最近一季財報
    print("Testing financial scraping for 2330...")
    fin_data = scrape_tw_financials("2330", 2023, 3)
    if not fin_data.empty:
        op.bulk_upsert_raw_financials(fin_data, market='tw')
        print(f"Uploaded {len(fin_data)} items for 2330 2023Q3.")
