import os
import django
import pandas as pd
import yfinance as yf
from stock_Django.stock_chart import chart_create

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')
django.setup()

def test_indicators():
    ticker = "2330.TW"
    data = yf.download(ticker, period="1y")
    if data.empty:
        print("No data found for 2330.TW")
        return
    
    # Flatten columns if multi-index (recent yfinance change)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    chart = chart_create()
    res = chart.get_ta_indicators(data.tail(120))
    
    vol_inds = res.get('volume_indicators', {})
    print(f"Volume Indicators for {ticker} (Last 120 days):")
    for name, vals in vol_inds.items():
        nan_count = sum(1 for x in vals if x is None)
        total = len(vals)
        print(f"  {name}: {total} values, {nan_count} None/NaN")

if __name__ == "__main__":
    test_indicators()
