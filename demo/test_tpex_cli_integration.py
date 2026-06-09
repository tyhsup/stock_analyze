"""測試 tpex-cli 整合與上市/上櫃分流是否能正確運作"""
import os
import sys
import json

# 確保 Django 環境可用
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')

import django
django.setup()

from stock_Django.services import StockService
from django.core.cache import cache

# 清除所有快取，防止舊的 stock_data 快取干擾測試
cache.clear()
print("Django cache cleared.")

svc = StockService()

print("=" * 60)
print("TEST 1: _get_twse_cli_info('2330') (上市股票)")
print("=" * 60)
result_tse = svc._get_twse_cli_info('2330')
print(f"Result: {json.dumps(result_tse, indent=2, ensure_ascii=False)}")
if result_tse:
    print(f"  PE: {result_tse.get('pe')}")
    print(f"  PB: {result_tse.get('pb')}")
    print(f"  Dividend Yield: {result_tse.get('dividend_yield')}")
else:
    print("  [ERROR] No data returned!")

print()

print("=" * 60)
print("TEST 2: _get_tpex_cli_info('8446') (上櫃股票)")
print("=" * 60)
result_tpex = svc._get_tpex_cli_info('8446')
print(f"Result: {json.dumps(result_tpex, indent=2, ensure_ascii=False)}")
if result_tpex:
    print(f"  PE: {result_tpex.get('pe')}")
    print(f"  PB: {result_tpex.get('pb')}")
    print(f"  Dividend Yield: {result_tpex.get('dividend_yield')}")
else:
    print("  [ERROR] No data returned!")

print()

print("=" * 60)
print("TEST 3: get_stock_data('8446', 60) (上櫃股票自動路由)")
print("=" * 60)
try:
    data_tpex = svc.get_stock_data('8446', 60)
    fs_tpex = data_tpex.get('financial_summary', {})
    print(f"  Short Name: {fs_tpex.get('short_name')}")
    print(f"  PE: {fs_tpex.get('pe')}")
    print(f"  PB: {fs_tpex.get('pb')}")
    print(f"  Dividend Yield: {fs_tpex.get('dividend_yield')}")
    print(f"  Error: {data_tpex.get('error')}")
except Exception as e:
    print(f"  [EXCEPTION] {e}")

print()

print("=" * 60)
print("TEST 4: get_stock_data('2330', 60) (上市股票自動路由)")
print("=" * 60)
try:
    data_tse = svc.get_stock_data('2330', 60)
    fs_tse = data_tse.get('financial_summary', {})
    print(f"  Short Name: {fs_tse.get('short_name')}")
    print(f"  PE: {fs_tse.get('pe')}")
    print(f"  PB: {fs_tse.get('pb')}")
    print(f"  Dividend Yield: {fs_tse.get('dividend_yield')}")
    print(f"  Error: {data_tse.get('error')}")
except Exception as e:
    print(f"  [EXCEPTION] {e}")

print()
print("=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
