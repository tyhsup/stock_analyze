import sys
import os
import django

# Setup Django path
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')
django.setup()

from stock_Django.stock_cost_AI import IntegratedStockPredModel
import logging

# Set logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def test_training_flow():
    symbol = "2330.TW"
    print(f"=== Testing AI Training Flow for {symbol} ===")
    
    try:
        model_service = IntegratedStockPredModel(symbol)
        print(f"Model dir: {model_service.model_dir}")
        print(f"Weights path: {model_service.weights_path}")
        
        # Force training
        success = model_service.train_incremental(epochs=2)
        if success:
            print("SUCCESS: train_incremental completed.")
            if os.path.exists(model_service.weights_path):
                print(f"SUCCESS: Model file found at {model_service.weights_path}")
            else:
                print("FAILED: Success returned True but NO FILE was saved.")
        else:
            print("FAILED: train_incremental returned False.")
            
        # Try prediction
        res = model_service.predict_5_days()
        if res:
            print(f"SUCCESS: predict_5_days returned results. Prob: {res['trend_probability']}")
        else:
            print("FAILED: predict_5_days returned None.")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    test_training_flow()
