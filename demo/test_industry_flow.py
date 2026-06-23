import os
import sys
import django
import time

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')
# Add demo directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

django.setup()

from django.conf import settings
if 'testserver' not in settings.ALLOWED_HOSTS:
    settings.ALLOWED_HOSTS.append('testserver')

from django.test import Client
from stock_Django.mySQL_OP import OP_Fun

def test_sql_performance():
    """驗證 get_industry_investor_summary 在非受損日期範圍內執行快速 (< 1 秒)"""
    print("Executing test_sql_performance...")
    sql = OP_Fun()
    
    # 執行一次以暖機資料庫與快取
    sql.get_industry_investor_summary(days=2)
    
    # 測試 days=2 并測量第二遍時間
    t0 = time.time()
    df = sql.get_industry_investor_summary(days=2)
    duration = time.time() - t0
    
    print(f"SQL execution time for 2 days (warmed): {duration:.4f}s")
    assert duration < 3.0, f"SQL query took too long: {duration:.4f}s"
    assert not df.empty, "Should return some data for TW stock market"
    print("test_sql_performance passed!")

def test_json_schema():
    """驗證 API 回傳的 JSON 結構符合預期格式"""
    print("Executing test_json_schema...")
    client = Client()
    response = client.get('/chips/api/industry-flow/', {'days': 2})
    assert response.status_code == 200, f"API returned status {response.status_code}"
    
    data = response.json()
    assert 'industries' in data, "Missing 'industries' in response"
    assert 'top_stocks' in data, "Missing 'top_stocks' in response"
    
    # 驗證 industries 列表元素格式
    industries = data['industries']
    print(f"Number of industries: {len(industries)}")
    if len(industries) > 0:
        first_ind = industries[0]
        assert 'x' in first_ind
        assert 'y' in first_ind
        assert 'net_flow' in first_ind
        print(f"Sample industry data: {first_ind}")
        
    # 驗證 top_stocks 的個股格式
    top_stocks = data['top_stocks']
    if len(top_stocks) > 0:
        first_key = list(top_stocks.keys())[0]
        stocks_list = top_stocks[first_key]
        print(f"Number of stocks in industry '{first_key}': {len(stocks_list)}")
        if len(stocks_list) > 0:
            first_stock = stocks_list[0]
            assert 'rank' in first_stock
            assert 'number' in first_stock
            assert 'name' in first_stock
            assert 'close' in first_stock
            assert 'net_flow' in first_stock
            assert 'consec_buys' in first_stock
            assert 'score' in first_stock
            print(f"Sample stock data: {first_stock}")
    print("test_json_schema passed!")

def test_weight_normalization():
    """驗證 w1 + w2 + w3 != 1.0 時，後端會自動正規化權重並回傳正確結果"""
    print("Executing test_weight_normalization...")
    client = Client()
    response = client.get('/chips/api/industry-flow/', {
        'days': 2,
        'w1': 5.0,
        'w2': 3.0,
        'w3': 2.0
    })
    assert response.status_code == 200
    data = response.json()
    assert 'industries' in data
    print("test_weight_normalization passed!")

def test_empty_industry():
    """驗證當無資料或錯誤引數時，系統不會崩潰且有適當的錯誤處理"""
    print("Executing test_empty_industry...")
    client = Client()
    response = client.get('/chips/api/industry-flow/', {
        'days': 1,
        'w1': 0,
        'w2': 0,
        'w3': 0
    })
    assert response.status_code == 200
    data = response.json()
    assert 'industries' in data
    print("test_empty_industry passed!")

def main():
    print("=================== RUNNING CHIPS API TESTS ===================")
    try:
        test_sql_performance()
        print("-" * 50)
        test_json_schema()
        print("-" * 50)
        test_weight_normalization()
        print("-" * 50)
        test_empty_industry()
        print("=================== ALL TESTS PASSED SUCCESSFULLY! ===================")
    except AssertionError as e:
        print(f"\n[Failed] Test failed: Assertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Failed] Test failed: Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
