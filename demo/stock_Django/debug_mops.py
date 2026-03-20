import requests
from scraper_utils import get_random_ua

def debug_mops(symbol, year, quarter):
    roc_year = year - 1911
    url = "https://mops.twse.com.tw/mops/web/ajax_t164sb03" # Balance Sheet
    
    payload = {
        'encodeURIComponent': '1',
        'step': '1',
        'firstin': '1',
        'off': '1',
        'keyword4': '',
        'code1': '',
        'TYPEK': 'all',
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
        "Referer": "https://mops.twse.com.tw/mops/web/t164sb03"
    }
    
    res = requests.post(url, data=payload, headers=headers)
    res.encoding = 'utf-8'
    
    with open("mops_debug.html", "w", encoding="utf-8") as f:
        f.write(res.text)
    
    print(f"MOPS Response length: {len(res.text)}")
    if "查詢無資料" in res.text:
        print("MOPS: No data found for this query.")
    elif "表格" in res.text or "<table" in res.text:
        print("MOPS: Tables found in response.")
    else:
        print("MOPS: Unexpected response content.")

if __name__ == "__main__":
    debug_mops("2330", 2023, 3)
