import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from stock_Django.mySQL_OP import OP_Fun

@pytest.fixture
def mock_engine():
    with patch('stock_Django.mySQL_OP.create_engine') as mock:
        yield mock

def test_op_fun_init_singleton(mock_engine):
    # Reset singleton for test
    OP_Fun._engine = None
    
    op1 = OP_Fun()
    op2 = OP_Fun()
    
    assert op1.engine is op2.engine
    assert mock_engine.call_count == 1

def test_upload_all_empty_df():
    op = OP_Fun()
    op.engine = MagicMock()
    df = pd.DataFrame()
    
    op.upload_all(df, "test_table")
    
    # Should return early without calling to_sql
    assert not op.engine.begin.called

@patch('pandas.read_sql')
def test_get_cost_data(mock_read_sql):
    op = OP_Fun()
    op.engine = MagicMock()
    mock_read_sql.return_value = pd.DataFrame({'col': [1]})
    
    df = op.get_cost_data("test_table", stock_number="2330")
    
    assert not df.empty
    assert mock_read_sql.called
    # Verify parameterization
    args, kwargs = mock_read_sql.call_args
    assert 'params' in kwargs
    assert kwargs['params'] == {'num': '2330'}
