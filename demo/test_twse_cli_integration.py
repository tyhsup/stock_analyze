"""測試 twse-cli 整合是否能正確取得台股基本指標"""
import os, sys, json

# 確保 Django 環境可用
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')

import django
django.setup()

from stock_Django.services import StockService

svc = StockService()

# 測試 1: 直接呼叫 _get_twse_cli_info
print("=" * 60)
print("TEST 1: _get_twse_cli_info('2330')")
print("=" * 60)
result = svc._get_twse_cli_info('2330')
print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
if result:
    print(f"  PE: {result.get('pe')}")
    print(f"  PB: {result.get('pb')}")
    print(f"  Dividend Yield: {result.get('dividend_yield')}")
else:
    print("  [ERROR] No data returned!")

print()

# 測試 2: 另一個台股代碼
print("=" * 60)
print("TEST 2: _get_twse_cli_info('2317')")
print("=" * 60)
result2 = svc._get_twse_cli_info('2317')
print(f"Result: {json.dumps(result2, indent=2, ensure_ascii=False)}")

print()

# 測試 3: 完整 get_stock_data 呼叫 (檢查 fin_summary)
print("=" * 60)
print("TEST 3: get_stock_data('2330', 60) - financial_summary")
print("=" * 60)
try:
    data = svc.get_stock_data('2330', 60)
    fs = data.get('financial_summary', {})
    print(f"  short_name: {fs.get('short_name')}")
    print(f"  PE: {fs.get('pe')}")
    print(f"  PB: {fs.get('pb')}")
    print(f"  Dividend Yield: {fs.get('dividend_yield')}")
    print(f"  EPS: {fs.get('eps')}")
    print(f"  ROE: {fs.get('roe')}")
    print(f"  Market Cap: {fs.get('marketCap')}")
    print(f"  52W High: {fs.get('fiftyTwoWeekHigh')}")
    print(f"  Error: {data.get('error')}")
except Exception as e:
    print(f"  [EXCEPTION] {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
