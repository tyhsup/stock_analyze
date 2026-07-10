import sys
import os
sys.path.append('demo')
# 設定 django 環境變數以讀取 .env 或 settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "demo.settings")

from stock_Django import mySQL_OP
from sqlalchemy import text

sql = mySQL_OP.OP_Fun()

def query_and_print(table_name, date_val, limit=5):
    print(f"=== Table: {table_name} for date {date_val} ===")
    with sql.engine.connect() as conn:
        # 這裡的 date_val 根據表不同可能是西元或民國
        query = text(f"SELECT * FROM {table_name} WHERE date = :date_val LIMIT :limit")
        res = conn.execute(query, {"date_val": date_val, "limit": limit}).fetchall()
        if not res:
            print("No records found.")
            return []
        
        # 取得欄位名稱
        meta_query = text(f"DESCRIBE {table_name}")
        cols = [row[0] for row in conn.execute(meta_query).fetchall()]
        
        for row in res:
            row_dict = dict(zip(cols, row))
            print(row_dict)
        return res

print("Checking stock_investor schemas and specific records")
with sql.engine.connect() as conn:
    # 1. DESCRIBE table to see actual types
    schema = conn.execute(text("DESCRIBE stock_investor")).fetchall()
    print("Schema of stock_investor (UTF-8 bytes representation):")
    for col in schema:
        print(f"  Field: {repr(col[0].encode('utf-8'))}, Type: {col[1]}")


    
    # 2. Check if 006201 exists in stock_investor
    res = conn.execute(text("SELECT * FROM stock_investor WHERE number = '006201' ORDER BY date DESC LIMIT 5")).fetchall()
    print("\nRecent 006201 in stock_investor:")
    cols = [col[0] for col in schema]
    for row in res:
        print(dict(zip(cols, row)))

    # 3. Check if any records exist in stock_investor on 2026-06-19
    res_date = conn.execute(text("SELECT COUNT(*) FROM stock_investor WHERE date = '2026-06-19'")).scalar()
    print(f"\nCount of records on 2026-06-19 in stock_investor: {res_date}")
    
    res_date_roc = conn.execute(text("SELECT COUNT(*) FROM stock_investor WHERE date = '115/06/19'")).scalar()
    print(f"Count of records on '115/06/19' in stock_investor: {res_date_roc}")


print("\nChecking stock_investor_tw (Show latest 10 rows)")
with sql.engine.connect() as conn:
    res = conn.execute(text("SELECT * FROM stock_investor_tw ORDER BY date DESC LIMIT 10")).fetchall()
    if res:
        meta_query = text("DESCRIBE stock_investor_tw")
        cols = [row[0] for row in conn.execute(meta_query).fetchall()]
        for row in res:
            row_dict = dict(zip(cols, row))
            print(row_dict)
    else:
        print("stock_investor_tw is empty.")

print("\nVerifying Django HTTP POST connection for stock 006201...")
import requests
try:
    # 取得 CSRF token（若 Django 有啟動 CSRF，通常 POST 需要，但若只是測試，我們先嘗試直接發送）
    session = requests.Session()
    # 先 GET 取得 cookie
    r_get = session.get("http://127.0.0.1:8000/", timeout=5)
    csrf_token = session.cookies.get('csrftoken')
    
    headers = {
        "Referer": "http://127.0.0.1:8000/"
    }
    payload = {
        "stock_number": "006201",
        "days": "30"
    }
    if csrf_token:
        payload["csrfmiddlewaretoken"] = csrf_token
        
    r_post = session.post("http://127.0.0.1:8000/", data=payload, headers=headers, timeout=10)
    print(f"POST to / succeeded. Status: {r_post.status_code}")
    print(f"Content length: {len(r_post.text)}")
    
    # 檢查是否含有關鍵字
    if "006201" in r_post.text or "元大富櫃50" in r_post.text:
        print("[SUCCESS] Found TPEx ETF 006201 (元大富櫃50) in home POST response HTML!")
    else:
        print("[FAIL] Could not find 006201 in response HTML.")
        # 列印前 2000 字元以協助排錯
        print("HTML Preview:")
        print(r_post.text[:2000])
except Exception as e:
    print(f"[FAIL] Connection to Django failed: {e}")




