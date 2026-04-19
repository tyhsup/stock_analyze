import pandas as pd
import numpy as np
import logging
from sqlalchemy import text
from stock_Django.mySQL_OP import OP_Fun

logger = logging.getLogger(__name__)

class GraphBuilder:
    def __init__(self):
        self.sql_op = OP_Fun()

    def get_market_cap_neighbors(self, symbol, market='tw', limit=5):
        """
        找出同產業中市值最大的前 N 檔股票作為鄰居
        """
        try:
            if market == 'tw':
                # 台股從 stock_table_tw 找產業，從某個地方(或預設)找市值? 
                # 目前台股 metadata 主要是產業別。
                # 我們假設鄰居是同產業的其他股票。
                query_sector = "SELECT 產業別 FROM stock_table_tw WHERE 有價證卷代號 = :symbol"
                sector_df = pd.read_sql(text(query_sector), con=self.sql_op.engine, params={'symbol': symbol.split('.')[0]})
                
                if sector_df.empty:
                    return []
                
                sector = sector_df.iloc[0]['產業別']
                
                # 找出同產業的其他股票 (暫時隨機或按代碼排序，台股目前沒存市場價值)
                query_neighbors = "SELECT 有價證卷代號 FROM stock_table_tw WHERE 產業別 = :sector AND 有價證卷代號 != :symbol LIMIT :limit"
                neighbors_df = pd.read_sql(text(query_neighbors), con=self.sql_op.engine, params={'sector': sector, 'symbol': symbol.split('.')[0], 'limit': limit})
                
                return [f"{s}.TW" for s in neighbors_df['有價證卷代號'].tolist()]
            
            else:
                # 美股從 stock_metadata 找
                query_meta = "SELECT sector FROM stock_metadata WHERE symbol = :symbol"
                meta_df = pd.read_sql(text(query_meta), con=self.sql_op.engine, params={'symbol': symbol})
                
                if meta_df.empty:
                     return []
                     
                sector = meta_df.iloc[0]['sector']
                
                # 按市值 (market_cap) 排序選取前 N 檔
                query_neighbors = "SELECT symbol FROM stock_metadata WHERE sector = :sector AND symbol != :symbol ORDER BY market_cap DESC LIMIT :limit"
                neighbors_df = pd.read_sql(text(query_neighbors), con=self.sql_op.engine, params={'sector': sector, 'symbol': symbol, 'limit': limit})
                
                return neighbors_df['symbol'].tolist()
                
        except Exception as e:
            logger.error(f"Error getting neighbors for {symbol}: {e}")
            return []

    @staticmethod
    def build_adjacency_matrix(nodes):
        """
        建立全連接或產業內連接的鄰接矩陣 (目前先做全連接，因為傳入的 nodes 已經是過濾過的鄰居)
        nodes: [target, neighbor_1, ..., neighbor_k]
        """
        n = len(nodes)
        if n == 0:
            return np.array([[1.0]])
        
        # 建立一個全 1 矩陣 (包含自環)
        adj = np.ones((n, n), dtype=np.float32)
        
        # 歸一化 (Degree Normalization): D^-1 * A
        # 這裡簡單採用度數倒數
        row_sums = adj.sum(axis=1)
        adj_normalized = adj / row_sums[:, np.newaxis]
        
        return adj_normalized
