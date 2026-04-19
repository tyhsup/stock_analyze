import sys
import os
import django
import numpy as np

# Setup Django path
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')
django.setup()

from stock_Django.stock_cost_AI import IntegratedStockPredModel
import tensorflow as tf

def test_v4_architecture():
    print("=== Testing v4.0 Model Architecture ===")
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs detected: {len(gpus)}")
    
    # Initialize model
    model_service = IntegratedStockPredModel("2330.TW")
    
    # Try to build model without data first to check architecture
    # We ignore data loading for a moment just to see if model() works
    lstm_dim = 10
    senti_dim = 768
    fin_dim = 3
    
    from stock_Django.model_architectures import StockModelArchitectures
    model = StockModelArchitectures.build_multi_input_model(lstm_dim, senti_dim, fin_dim, model_service.config)
    
    model.summary()
    print("\nModel Input Names:")
    for i in model.inputs:
        print(f" - {i.name}")
    
    print("\nVerification: SUCCESS - Model architecture built correctly.")

def test_v4_pipeline():
    print("\n=== Testing v4.0 Data Pipeline ===")
    model_service = IntegratedStockPredModel("2330.TW")
    
    # We will just try to run prepare_training_data if possible
    dfs_dict = model_service.build_all_nodes_datasets()
    if not dfs_dict:
        print("SKIP: Cannot find data in DB for pipeline test.")
        return
        
    prepared = model_service.prepare_training_data(dfs_dict)
    if prepared:
        X_ts, X_senti, X_fin, A, Y = prepared
        print(f"X_ts shape: {X_ts.shape}")
        print(f"X_senti shape: {X_senti.shape}")
        print(f"X_fin shape: {X_fin.shape}")
        print(f"A shape: {A.shape}")
        print(f"Y shape: {Y.shape}")
    else:
        print("FAILED: Data preparation returned None.")

if __name__ == "__main__":
    try:
        test_v4_architecture()
        test_v4_pipeline()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nVerification: FAILED - {e}")
