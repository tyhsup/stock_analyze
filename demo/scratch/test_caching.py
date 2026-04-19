import sys; sys.path.append('.')
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'demo.settings'
import logging
logging.basicConfig(level=logging.INFO)
import django; django.setup()
import numpy as np
import pandas as pd
from stock_Django.mySQL_OP import OP_Fun
from stock_Django.dataset_builders import SentimentProbabilityModel
import stock_Django.dataset_builders
print(f"Dataset builders file: {stock_Django.dataset_builders.__file__}")
with open(stock_Django.dataset_builders.__file__, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    print("Code check (lines 90-100):")
    for i in range(89, 100):
        print(f"{i+1}: {lines[i].strip()}")

def test_db_methods():
    print("Testing OP_Fun caching methods...")
    op = OP_Fun()
    symbol = "TEST_STOCK"
    date_str = "2026-01-01"
    embedding = np.random.rand(768).astype(np.float32)
    
    # Test Save
    op.save_sentiment_embeddings(symbol, {date_str: embedding})
    
    # Test Get
    res = op.get_sentiment_embeddings(symbol, date_str, date_str)
    print(f"Raw cache lookup result for {symbol}: {res.keys()}")
    if date_str in res:
        print(f"Successfully retrieved embedding for {date_str}")
        # Check similarity
        dist = np.linalg.norm(res[date_str] - embedding)
        print(f"Distance: {dist}")
        assert dist < 1e-5
    else:
        print(f"Failed to retrieve embedding. Available: {res.keys()}")

def test_integration():
    print("\nTesting SentimentProbabilityModel integration...")
    # Be explicit about logger
    logger = logging.getLogger('stock_Django.dataset_builders')
    logger.setLevel(logging.INFO)
    
    # Create a dummy index
    dates = pd.to_datetime(['2026-01-01', '2026-01-02'])
    date_index_df = pd.DataFrame(index=dates)
    
    # Since we don't have real news, it should return zeros but let's see if it hits the cache we just populated
    features = SentimentProbabilityModel.get_sentiment_features("TEST_STOCK", date_index_df)
    import dis
    print("Disassembly of get_sentiment_features:")
    dis.dis(SentimentProbabilityModel.get_sentiment_features)
    
    # '2026-01-01' was cached in test_db_methods
    val = features.loc['2026-01-01'].values
    print(f"2026-01-01 val sum: {np.sum(val)}")
    if np.sum(val) != 0:
        print("Cache HIT confirmed for 2026-01-01")
    else:
        print("Cache MISS or zero values for 2026-01-01")
    
    val2 = features.loc['2026-01-02'].values
    print(f"2026-01-02 values sum: {np.sum(val2)} (Expected 0 if no news)")

if __name__ == "__main__":
    try:
        test_db_methods()
        test_integration()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
