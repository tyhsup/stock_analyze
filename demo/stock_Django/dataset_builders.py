import logging
import numpy as np
import pandas as pd
from sqlalchemy import text
import torch
from transformers import AutoTokenizer, AutoModel

try:
    from stock_Django.stock_utils import StockUtils
    from stock_Django.mySQL_OP import OP_Fun
except ImportError:
    try:
        from .stock_utils import StockUtils
        from .mySQL_OP import OP_Fun
    except (ImportError, ValueError):
        from stock_utils import StockUtils
        from mySQL_OP import OP_Fun

logger = logging.getLogger(__name__)

class PriceLSTMFeatureExtractor:
    @staticmethod
    def extract_features(data_df):
        """產生 LSTM 所需之漲跌幅度與型態特徵"""
        df = data_df.copy()
        if 'Close' not in df.columns:
            return df
        # 產生日報酬率
        df['Daily_Return'] = df['Close'].pct_change()
        # 均線特徵
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        # 價格距均線的乖離率
        df['Bias_5'] = (df['Close'] - df['SMA_5']) / df['SMA_5']
        df['Bias_20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        df.fillna(0, inplace=True)
        return df

class SentimentProbabilityModel:
    _model_instance = None
    _tokenizer_instance = None
    _device = None

    @classmethod
    def _initialize_finbert(cls):
        if cls._model_instance is None:
            model_name = 'ProsusAI/finbert'
            cls._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading {model_name} onto {cls._device}")
            cls._tokenizer_instance = AutoTokenizer.from_pretrained(model_name)
            cls._model_instance = AutoModel.from_pretrained(model_name).to(cls._device)
            cls._model_instance.eval()

    @classmethod
    @torch.no_grad()
    def _get_embeddings_batch(cls, texts, batch_size=16):
        cls._initialize_finbert()
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = cls._tokenizer_instance(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(cls._device)
            outputs = cls._model_instance(**inputs)
            # 取得 [CLS] token 的隱藏層特徵 (Batch, Seq_Len, 768)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())
        if not all_embeddings:
            return np.array([])
        return np.vstack(all_embeddings)

    @classmethod
    def get_sentiment_features(cls, stock_number, date_index_df):
        """取得新聞語義特徵 (具備資料庫快取機制)"""
        embedding_dim = 768
        embedding_cols = [f'finbert_emb_{i}' for i in range(embedding_dim)]
        
        # Initialize result DataFrame
        result_df = pd.DataFrame(0.0, index=date_index_df.index, columns=embedding_cols)
        
        if date_index_df.empty:
            return result_df
            
        sql_op = OP_Fun()
        start_date = date_index_df.index.min().strftime('%Y-%m-%d')
        end_date = date_index_df.index.max().strftime('%Y-%m-%d')
        
        # 1. 嘗試從快取讀取
        cached_embeddings = sql_op.get_sentiment_embeddings(stock_number, start_date, end_date)
        
        # 2. 判斷缺失日期
        missing_dates = []
        for d in date_index_df.index:
            d_str = d.strftime('%Y-%m-%d')
            if d_str in cached_embeddings:
                result_df.loc[d, embedding_cols] = cached_embeddings[d_str]
            else:
                missing_dates.append(d)
        
        if not missing_dates:
            return result_df[embedding_cols]
            
        # 3. 僅針對缺失日期進行推論
        logger.info(f"快取未命中，正在計算 {stock_number} 共 {len(missing_dates)} 天的語義向量...")
        
        # 建立僅包含缺失日期的臨時 index df 供爬蟲使用
        # 注意: 這裡我們使用原本的邏輯去抓新聞文本
        sentiment_df = StockUtils.Sentiment_indicators(stock_number, pd.DataFrame(index=missing_dates))
        
        if sentiment_df.empty:
            return result_df[embedding_cols]

        text_col = None
        for col in ['Content', 'Title', 'text', '新聞內容', '新聞標題']:
            if col in sentiment_df.columns:
                text_col = col
                break

        if text_col:
            sentiment_df['text'] = sentiment_df[text_col].fillna("").astype(str)
            sentiment_df = sentiment_df[sentiment_df['text'].str.strip() != ""]
            
            if not sentiment_df.empty:
                embeddings = cls._get_embeddings_batch(sentiment_df['text'].tolist())
                sentiment_df['embedding'] = list(embeddings)
                
                # 以 Date 聚合 (確保它是 string 格式以便與快取邏輯一致)
                if 'Date' in sentiment_df.columns:
                    sentiment_df['Date_str'] = pd.to_datetime(sentiment_df['Date']).dt.strftime('%Y-%m-%d')
                else:
                    sentiment_df['Date_str'] = pd.to_datetime(sentiment_df.index).dt.strftime('%Y-%m-%d')
                
                daily_groups = sentiment_df.groupby('Date_str')['embedding'].apply(
                    lambda x: np.mean(np.stack(x.values), axis=0)
                )
                
                # 4. 寫回快取並更新結果
                new_cache_data = {}
                for d_str, emb in daily_groups.items():
                    new_cache_data[d_str] = emb
                    d_ts = pd.Timestamp(d_str)
                    if d_ts in result_df.index:
                        result_df.loc[d_ts, embedding_cols] = emb
                
                sql_op.save_sentiment_embeddings(stock_number, new_cache_data)
        
        return result_df[embedding_cols]

class InstitutionalFlowModel:
    @staticmethod
    def get_flow_features(stock_number, date_index_df):
        """讀取法人籌碼動向，轉化為金流特徵"""
        sql_op = OP_Fun()
        is_tw = str(stock_number).isdigit() or ".TW" in str(stock_number).upper() or ".TWO" in str(stock_number).upper()
        table_name = 'stock_investor' if is_tw else 'stock_investor_us'
        
        clean_num = str(stock_number).replace('.TW', '').replace('.TWO', '')
        if is_tw:
            inv_df = sql_op.get_cost_data(table_name=table_name, stock_number=clean_num)
        else:
            # US investor table uses 'ticker' instead of 'number'
            query = f"SELECT * FROM {table_name} WHERE ticker = :num"
            try:
                inv_df = pd.read_sql(text(query), con=sql_op.engine, params={'num': clean_num})
            except Exception as e:
                logger.error(f"Failed to fetch {table_name}: {e}")
                inv_df = pd.DataFrame()
        
        zeros = np.zeros(len(date_index_df))
        if inv_df.empty:
            return pd.DataFrame({'Net_Buy_Volume': zeros}, index=date_index_df.index)
            
        inv_df = sql_op._fix_investor_columns(inv_df)
        target_col = '三大法人買賣超股數' if '三大法人買賣超股數' in inv_df.columns else None
        
        if not target_col:
            for col in inv_df.columns:
                if '買賣超' in col or 'Net' in col:
                    target_col = col
                    break
                    
        if not target_col:
            return pd.DataFrame({'Net_Buy_Volume': zeros}, index=date_index_df.index)
            
        inv_df['日期'] = pd.to_datetime(inv_df['日期']).dt.normalize()
        inv_df.set_index('日期', inplace=True)
        
        if inv_df[target_col].dtype == object:
            inv_df[target_col] = inv_df[target_col].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce')
        
        # Merge and fill
        merged = date_index_df.copy()
        merged = merged.merge(inv_df[[target_col]], left_index=True, right_index=True, how='left')
        merged[target_col] = merged[target_col].fillna(0)
        
        features = pd.DataFrame({
            'Net_Buy_Volume': merged[target_col]
        })
        return features

class FundamentalFeatureProcessor:
    @staticmethod
    def get_fundamental_features(stock_number, date_index_df):
        """讀取財報基本面，運用 Forward-fill 給每日預測使用"""
        sql_op = OP_Fun()
        is_tw = str(stock_number).isdigit() or ".TW" in str(stock_number).upper() or ".TWO" in str(stock_number).upper()
        market = 'tw' if is_tw else 'us'
        table_name = f'financial_raw_{market}'
        clean_num = str(stock_number).replace('.TW', '').replace('.TWO', '')
        
        query = f"SELECT year, quarter, item_name, amount FROM {table_name} WHERE symbol = '{clean_num}'"
        try:
            fin_df = pd.read_sql(OP_Fun().engine, query)
        except Exception:
            fin_df = pd.DataFrame()
            
        zeros = np.zeros(len(date_index_df))
        base_features = pd.DataFrame({'EPS_Quarterly': zeros, 'Revenue_Growth': zeros}, index=date_index_df.index)
        
        if fin_df.empty:
            return base_features
            
        # 概略估算季度末日期
        fin_df['date'] = fin_df.apply(lambda row: pd.Timestamp(year=row['year'], month=int(row['quarter'])*3, day=1) + pd.offsets.MonthEnd(0), axis=1)
        
        pivot_df = fin_df.pivot_table(index='date', columns='item_name', values='amount', aggfunc='first')
        
        eps_col = next((c for c in pivot_df.columns if 'EPS' in c.upper() or '每股盈餘' in c), None)
        rev_col = next((c for c in pivot_df.columns if 'REVENUE' in c.upper() or '營業收入' in c), None)
        
        pivot_df['EPS_Quarterly'] = pivot_df[eps_col] if eps_col else 0
        pivot_df['Revenue_Growth'] = pivot_df[rev_col].pct_change().fillna(0) if rev_col else 0
            
        pivot_df = pivot_df[['EPS_Quarterly', 'Revenue_Growth']]
        
        merged = date_index_df.copy()
        combined = pd.concat([merged, pivot_df]).sort_index()
        combined['EPS_Quarterly'] = combined['EPS_Quarterly'].ffill().fillna(0)
        combined['Revenue_Growth'] = combined['Revenue_Growth'].ffill().fillna(0)
        
        final_features = combined.loc[date_index_df.index]
        return final_features[['EPS_Quarterly', 'Revenue_Growth']]
