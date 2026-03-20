import pytest
from django.urls import reverse
from unittest.mock import patch, MagicMock
import pandas as pd

@pytest.mark.django_db
class TestStockViews:
    @patch('stock_Django.services.StockService.get_stock_data')
    def test_home_post_success(self, mock_get_data, client):
        # Mock service response
        mock_get_data.return_value = {
            'number': '2330',
            'is_tw': True,
            'currency': 'TWD',
            'historical_data': pd.DataFrame(),
            'kline_json': '{}',
            'ta_json': {},
            'investor_json': '{}',
            'investor_tw_json': '{}',
            'investor_comparison_json': '{}',
            'us_investor_json': None,
            'investor_H': '<table></table>',
            'investor_T': '<table></table>',
            'valuation_symbol': '2330.TW',
            'error': None
        }
        
        url = reverse('home')
        response = client.post(url, {'stock_number': '2330', 'days': '30'})
        
        assert response.status_code == 200
        assert '2330' in response.content.decode('utf-8')
        assert 'last_ticker' in client.session
        assert client.session['last_ticker'] == '2330.TW'

    @patch('stock_Django.services.StockService.get_stock_data')
    def test_home_post_no_data(self, mock_get_data, client):
        mock_get_data.return_value = {
            'kline_json': None,
            'investor_json': None,
            'us_investor_json': None,
            'error': 'No data found'
        }
        
        url = reverse('home')
        response = client.post(url, {'stock_number': 'INVALID', 'days': '30'})
        
        assert response.status_code == 200
        assert 'No data found' in response.content.decode('utf-8')

    def test_home_get(self, client):
        url = reverse('home')
        response = client.get(url)
        assert response.status_code == 200
