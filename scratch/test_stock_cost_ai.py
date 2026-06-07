import sys
import os
import pandas as pd
import numpy as np

# 加入根目錄至 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_cost_AI import stock_cost_AI

def mock_load_data_c(table, stock_number):
    print(f"Mocking DB load for {stock_number} with simulated data...")
    date_range = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='B')
    
    np.random.seed(42)
    close_prices = 100.0 + np.cumsum(np.random.normal(0.5, 2.0, 100))
    close_prices = np.clip(close_prices, 10.0, None)
    
    cost_data = pd.DataFrame({
        'Open': close_prices - np.random.uniform(-1.0, 1.0, 100),
        'Close': close_prices,
        'High': close_prices + np.random.uniform(0.0, 3.0, 100),
        'Low': close_prices - np.random.uniform(0.0, 3.0, 100),
        'Volume': np.random.randint(1000, 10000, 100)
    })
    
    cost_data['High'] = cost_data[['Open', 'Close', 'High']].max(axis=1)
    cost_data['Low'] = cost_data[['Open', 'Close', 'Low']].min(axis=1)
    
    cost_Date = pd.DataFrame({'Date': date_range})
    return cost_data, cost_Date

def test_prediction():
    try:
        print("Applying monkey patch to stock_cost_AI.load_data_c...")
        stock_cost_AI.load_data_c = mock_load_data_c
        
        print("Testing pred_cost on stock 2330...")
        # 預測未來 5 天
        res = stock_cost_AI.pred_cost('2330', 5)
        print("Prediction result:")
        print(res)
        
        # 檢查預測出來的值是否為常數
        closes = res['Close'].values
        if len(closes) > 1 and np.all(closes == closes[0]):
            print("WARNING: Result is a constant array!")
        else:
            print("SUCCESS: Predictions are rolling and dynamic!")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Test failed with error: {e}")

if __name__ == '__main__':
    test_prediction()
