import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from stock_Django.services import StockService

@pytest.fixture
def service():
    with patch('stock_Django.services.mySQL_OP.OP_Fun'), \
         patch('stock_Django.services.stock_chart.chart_create'), \
         patch('stock_Django.services.USStockInvestorManager'), \
         patch('stock_Django.services.TPExInvestorManager'):
        return StockService()

@patch('stock_Django.services.StockUtils.load_data_c')
@patch('django.core.cache.cache.get')
@patch('django.core.cache.cache.set')
def test_get_stock_data_cache_hit(mock_cache_set, mock_cache_get, mock_load_data, service):
    mock_cache_get.return_value = {'cached': 'data'}
    
    result = service.get_stock_data("2330", 30)
    
    assert result == {'cached': 'data'}
    assert mock_cache_get.called
    assert not mock_load_data.called

@patch('stock_Django.services.StockUtils.load_data_c')
@patch('django.core.cache.cache.get')
@patch('django.core.cache.cache.set')
@patch('yfinance.Ticker')
def test_get_stock_data_db_empty_fallback_yf(mock_yf, mock_cache_set, mock_cache_get, mock_load_data, service):
    mock_cache_get.return_value = None
    mock_load_data.return_value = (pd.DataFrame(), pd.DataFrame()) # DB empty
    
    # Mock yfinance response with a real DataFrame
    df_yf = pd.DataFrame({'Open': [100.0], 'High': [105.0], 'Low': [98.0], 'Close': [102.0], 'Volume': [1000]},
                         index=[pd.Timestamp('2024-01-01')])
    mock_yf.return_value.history.return_value = df_yf
    
    result = service.get_stock_data("2330", 1)
    
    assert not result['historical_data'].empty
    assert result['historical_data'].iloc[0]['Close'] == 102.0
    assert mock_yf.called
    assert mock_cache_set.called

def test_ticker_normalization(service):
    # This just tests the logic inside get_stock_data, but we need to mock calls to avoid errors
    with patch.object(service, 'sql_op'), \
         patch.object(service, 'chart'), \
         patch.object(service, 'us_mgr'), \
         patch('stock_Django.services.StockUtils.load_data_c', return_value=(pd.DataFrame(), pd.DataFrame())), \
         patch('yfinance.Ticker') as mock_yf:
        
        mock_yf.return_value.history.return_value = pd.DataFrame()
        
        result = service.get_stock_data(" 2330.tw ", 30)
        assert result['number'] == "2330.TW"
        assert result['is_tw'] == True
        assert result['valuation_symbol'] == "2330.TW"
