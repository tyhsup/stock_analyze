import requests
import pandas as pd
import time
from .scraper_utils import get_random_ua, smart_delay
from .mySQL_OP import OP_Fun
import logging

logger = logging.getLogger(__name__)

# SEC 要求 User-Agent 必須包含聯絡資訊
SEC_HEADERS = {
    "User-Agent": "MyStockApp/1.0 (contact@example.com)",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov"
}

def get_us_stock_list():
    """從 SEC 獲取 Ticker to CIK 映射"""
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": SEC_HEADERS["User-Agent"]}
    
    res = requests.get(url, headers=headers)
    data = res.json()
    
    # 轉換成 DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')
    df.rename(columns={'ticker': 'symbol', 'title': 'name', 'cik_str': 'cik'}, inplace=True)
    
    # CIK 必須補滿 10 位數
    df['cik'] = df['cik'].astype(str).str.zfill(10)
    df['market'] = 'US' # 簡化處理
    return df

def scrape_us_financials(symbol, cik):
    """
    從 SEC EDGAR API 抓取財報數據 (Fact API)
    CIK 必須是 10 位數
    """
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    
    try:
        res = requests.get(url, headers=SEC_HEADERS)
        if res.status_code != 200:
            print(f"DEBUG: SEC API Error {res.status_code} for {symbol}")
            return pd.DataFrame()
            
        data = res.json()
        
        # US GAAP 數據通常在 facts['us-gaap'] 下
        if 'us-gaap' not in data.get('facts', {}):
            return pd.DataFrame()
            
        all_items = []
        us_gaap = data['facts']['us-gaap']
        
        for concept,ConceptData in us_gaap.items():
            # 每個 concept 下有多個單位的數據 (通常是 USD)
            units = ConceptData.get('units', {})
            for unit, values in units.items():
                for entry in values:
                    # 過濾出年度 (10-K) 或季度 (10-Q) 資料
                    form = entry.get('form')
                    if form not in ['10-K', '10-Q']:
                        continue
                        
                    year = entry.get('fy')
                    quarter_val = entry.get('fp') # Q1, Q2, Q3, FY
                    
                    # 轉換 FY 為第 4 季
                    quarter = 4 if quarter_val == 'FY' else int(quarter_val[1])
                    
                    # 判斷財報類型 (更完善的映射)
                    concept_low = concept.lower()
                    if any(x in concept_low for x in ['asset', 'liabilit', 'equity', 'inventory', 'receivable', 'payable', 'cash', 'debt']):
                        # 注意：Cash 可能出現在 CF 或 BS，若有 'flow' 或 'operating' 之類則為 CF
                        if any(x in concept_low for x in ['cashflow', 'cashflows', 'operatingactivities', 'investingactivities', 'financingactivities']):
                            stmt_type = 'CF'
                        else:
                            stmt_type = 'BS'
                    elif any(x in concept_low for x in ['revenue', 'income', 'expense', 'profit', 'margin', 'tax', 'earnings']):
                        stmt_type = 'IS'
                    else:
                        stmt_type = 'IS' # 預設依然是 IS
                        
                    all_items.append({
                        'symbol': symbol,
                        'year': year,
                        'quarter': quarter,
                        'statement_type': stmt_type,
                        'item_name': concept, # 使用 XBRL tag 作為名稱
                        'amount': entry['val']
                    })
                    
        return pd.DataFrame(all_items)
    except Exception as e:
        logger.error(f"Error scraping SEC data for {symbol}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    op = OP_Fun()
    
    # 1. 獲取並更新所有 US 股票基本資訊
    print("Fetching ALL US stock tickers from SEC...")
    us_stocks = get_us_stock_list()
    if not us_stocks.empty:
        op.upsert_stocks_info(us_stocks, market='us')
        print(f"Updated {len(us_stocks)} US stock metadata entries.")
    
    # 2. 自動化抓取所有股票的財報 (最近 5 年)
    # 為了演示，我們這裡抓取前 10 支股票作為範例，實務上可移除 .head(10)
    target_stocks = us_stocks.head(10)
    print(f"Starting batch population for {len(target_stocks)} US stocks...")
    
    current_year = 2024
    start_year = current_year - 5
    
    success_count = 0
    for idx, row in target_stocks.iterrows():
        symbol = row['symbol']
        cik = row['cik']
        
        print(f"[{idx+1}/{len(target_stocks)}] Scraping {symbol} (CIK: {cik})...")
        
        # SEC Limit: 10 requests per second
        time.sleep(0.15) 
        
        df = scrape_us_financials(symbol, cik)
        if not df.empty:
            # 過濾最近 5 年
            df_filtered = df[df['year'] >= start_year]
            if not df_filtered.empty:
                op.bulk_upsert_raw_financials(df_filtered, market='us')
                print(f"  -> Uploaded {len(df_filtered)} items.")
                success_count += 1
            else:
                print(f"  -> No data in last 5 years.")
        else:
            print(f"  -> Failed to fetch data.")
            
    print(f"US Data Population complete. Successfully processed {success_count} stocks.")
