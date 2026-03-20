import os
import sys
import pandas as pd
import yfinance as yf
import json
import logging
import requests
from datetime import datetime

# Setup paths
PROJECT_ROOT = r'e:\Infinity\mydjango\demo'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Mock Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')
import django
django.setup()

from stock_Django.services import StockService

def verify_ticker(ticker_input, session):
    print(f"--- Verifying Ticker: {ticker_input} ---")
    service = StockService()
    service.session = session # Inject session
    
    # 1. Get Local Data
    try:
        local_data = service.get_stock_data(ticker_input, days=120)
        local_fin = local_data.get('financial_summary', {})
        val_symbol = local_data.get('valuation_symbol', ticker_input)
    except Exception as e:
        print(f"Error getting local data for {ticker_input}: {e}")
        return None

    # 2. Get External Data (Yahoo Finance Info)
    print(f"Fetching Yahoo data for {val_symbol}...")
    try:
        yf_ticker = yf.Ticker(val_symbol, session=session)
        yf_info = yf_ticker.info
    except Exception as e:
        print(f"Error getting Yahoo info for {val_symbol}: {e}")
        yf_info = {}

    def clean_num(v):
        if v is None or pd.isna(v): return None
        if isinstance(v, str):
            v = v.replace('兆','*1e12').replace('億','*1e8').replace('萬','*1e4')
            v = v.replace('T','*1e12').replace('B','*1e9').replace('M','*1e6')
            try:
                if '*' in v:
                    parts = v.split('*')
                    return float(parts[0]) * float(parts[1])
                return float(v.replace(',',''))
            except: return None
        return float(v)

    metrics = [
        ('marketCap', 'marketCap', 'Market Cap (Value)'),
        ('pe', 'trailingPE', 'P/E Ratio'),
        ('pb', 'priceToBook', 'P/B Ratio'),
        ('eps', 'trailingEps', 'EPS (TTM)'),
        ('roe', 'returnOnEquity', 'ROE (%)'),
        ('gross_margin', 'grossMargins', 'Gross Margin (%)'),
        ('revenue_growth', 'revenueGrowth', 'Rev Growth (%)'),
        ('fiftyTwoWeekHigh', 'fiftyTwoWeekHigh', '52W High')
    ]

    report = []
    for local_key, yf_key, label in metrics:
        local_val_raw = local_fin.get(local_key)
        yf_val_raw = yf_info.get(yf_key)
        
        # Unit adjustment
        if label == 'ROE (%)' and yf_val_raw is not None: yf_val_raw *= 100
        if label == 'Gross Margin (%)' and yf_val_raw is not None: yf_val_raw *= 100
        if label == 'Rev Growth (%)' and yf_val_raw is not None: yf_val_raw *= 100
        
        local_numeric = clean_num(local_val_raw)
        yf_numeric = clean_num(yf_val_raw)
        
        diff_pct = None
        if local_numeric is not None and yf_numeric is not None and yf_numeric != 0:
            diff_pct = abs(local_numeric - yf_numeric) / abs(yf_numeric) * 100

        report.append({
            'Metric': label,
            'Local': local_val_raw,
            'Yahoo': yf_val_raw if yf_val_raw is not None else "N/A",
            'Diff %': round(diff_pct, 2) if diff_pct is not None else "N/A"
        })

    return {
        'symbol': ticker_input,
        'val_symbol': val_symbol,
        'name': local_fin.get('short_name'),
        'metrics': report
    }

if __name__ == "__main__":
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    
    tickers = ['2330', '2317', 'AAPL', 'NVDA', '4764']
    all_results = []
    for t in tickers:
        try:
            res = verify_ticker(t, session)
            if res: all_results.append(res)
        except Exception as e:
            print(f"Failed to verify {t}: {e}")
    
    md = "# Financial Overview 比對驗證報告\n\n"
    md += f"產出時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    if not all_results:
        md += "⚠️ 無法取得任何比對數據 (可能是 429 阻擋)。\n"
    else:
        for item in all_results:
            md += f"## {item['symbol']} - {item['name']} ({item['val_symbol']})\n"
            df = pd.DataFrame(item['metrics'])
            md += df.to_markdown(index=False)
            md += "\n\n"

    report_path = os.path.join(r'C:\Users\許廷宇\.gemini\antigravity\brain\b7fe8e81-6e77-44ef-bf9a-cb6d7dedf934', 'financial_comparison_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(md)
    
    print(f"Report generated: {report_path}")
