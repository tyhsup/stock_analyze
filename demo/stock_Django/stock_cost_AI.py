import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LSTM, Dropout, Input, Concatenate
from keras.models import Model
import yfinance as yf

from .stock_utils import StockUtils
from stock_Django.mySQL_OP import OP_Fun

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
    @staticmethod
    def get_sentiment_features(stock_number, date_index_df):
        """取得新聞情緒特徵：結合 nlp_service 的分析結果"""
        sentiment_df = StockUtils.Sentiment_indicators(stock_number, date_index_df)
        if sentiment_df.empty or '正面新聞占比' not in sentiment_df.columns:
            # 右側對齊全補 0 (中立)
            zeros = np.zeros(len(date_index_df))
            return pd.DataFrame({'Pos_Ratio': zeros, 'Neg_Ratio': zeros}, index=date_index_df.index)
        
        sentiment_df.set_index('Date', inplace=True)
        date_index_df_copy = date_index_df.copy()
        
        # Merge by index
        merged = date_index_df_copy.merge(sentiment_df, left_index=True, right_index=True, how='left')
        merged['正面新聞占比'] = merged['正面新聞占比'].fillna(0)
        merged['負面新聞占比'] = merged['負面新聞占比'].fillna(0)
        
        features = pd.DataFrame({
            'Pos_Ratio': merged['正面新聞占比'],
            'Neg_Ratio': merged['負面新聞占比']
        })
        return features

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

class IntegratedStockPredModel:
    def __init__(self, stock_number):
        self.stock_number = str(stock_number)
        self.clean_number = self.stock_number.replace('.TW', '').replace('.TWO', '')
        self.model_dir = 'E:/Infinity/mydjango/demo/stock_Django/stock_model'
        os.makedirs(self.model_dir, exist_ok=True)
        self.weights_path = os.path.join(self.model_dir, f'stock_model_weights_{self.clean_number}.weights.h5')
        
        self.time_steps = 5
        self.predict_steps = 5
        self.lstm_features = 0
        self.dense_features = 0
        self.model = None
        self.max_vals = None
        self.ts_idx = None
        self.ext_idx = None

    def build_dataset(self):
        """整合 OHLCV、情緒、籌碼、基本面，建構多模態 Dataframe"""
        cost_data, Date_data = StockUtils.load_data_c('stock_cost', self.stock_number)
        if cost_data.empty:
            cost_data, Date_data = StockUtils.load_data_c('stock_cost_us', self.stock_number)
            
        if cost_data.empty:
            logger.error(f"Cannot find price data for {self.stock_number} in DB.")
            return None

        cost_data.index = pd.to_datetime(Date_data['Date']).dt.normalize()
        
        price_feat = PriceLSTMFeatureExtractor.extract_features(cost_data)
        date_index_df = pd.DataFrame(index=cost_data.index)
        
        senti_feat = SentimentProbabilityModel.get_sentiment_features(self.clean_number, date_index_df)
        flow_feat = InstitutionalFlowModel.get_flow_features(self.clean_number, date_index_df)
        fund_feat = FundamentalFeatureProcessor.get_fundamental_features(self.clean_number, date_index_df)
        
        full_df = pd.concat([price_feat, senti_feat, flow_feat, fund_feat], axis=1)
        full_df.dropna(inplace=True)
        return full_df

    def create_model(self, lstm_feat_dim, ext_feat_dim):
        """建構 Multi-Input DNN模型"""
        input_ts = Input(shape=(self.time_steps, lstm_feat_dim), name="time_series_input")
        x = LSTM(64, return_sequences=True)(input_ts)
        x = Dropout(0.2)(x)
        x = LSTM(32)(x)
        
        input_ext = Input(shape=(ext_feat_dim,), name="external_features_input")
        y = Dense(32, activation='relu')(input_ext)
        y = Dropout(0.2)(y)
        
        combined = Concatenate()([x, y])
        z = Dense(64, activation='relu')(combined)
        z = Dropout(0.2)(z)
        
        out_price = Dense(self.predict_steps, name='price_prediction')(z)
        
        model = Model(inputs=[input_ts, input_ext], outputs=out_price)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def prepare_training_data(self, full_df):
        """切割 Sliding Window 特徵，準備 Keras 輸入格式"""
        df_vals = full_df.to_numpy()
        cols = full_df.columns
        
        ext_cols = ['Pos_Ratio', 'Neg_Ratio', 'Net_Buy_Volume', 'EPS_Quarterly', 'Revenue_Growth']
        ext_idx = [full_df.columns.get_loc(c) for c in ext_cols if c in full_df.columns]
        ts_idx = [i for i in range(len(cols)) if i not in ext_idx]
        
        close_col_idx = full_df.columns.get_loc('Close')
        
        X_ts, X_ext, Y = [], [], []
        if len(df_vals) <= self.time_steps + self.predict_steps:
            return None
            
        for i in range(len(df_vals) - self.time_steps - self.predict_steps + 1):
            window = df_vals[i : i + self.time_steps]
            target = df_vals[i + self.time_steps : i + self.time_steps + self.predict_steps, close_col_idx]
            X_ts.append(window[:, ts_idx])
            X_ext.append(window[-1, ext_idx])
            Y.append(target)
            
        self.lstm_features = len(ts_idx)
        self.dense_features = len(ext_idx)
        self.ts_idx = ts_idx
        self.ext_idx = ext_idx
        
        return np.array(X_ts), np.array(X_ext), np.array(Y)

    def train_incremental(self, epochs=50, batch_size=32):
        """增量訓練邏輯 (Incremental Learning)"""
        full_df = self.build_dataset()
        if full_df is None: return False
        
        self.max_vals = full_df.max().replace(0, 1)
        norm_df = full_df / self.max_vals
        
        prepared = self.prepare_training_data(norm_df)
        if not prepared: return False
        X_ts, X_ext, Y = prepared
        
        if self.model is None:
            self.model = self.create_model(self.lstm_features, self.dense_features)
            weights_h5 = self.weights_path.replace('.weights.h5', '.h5')
            # 優先讀 .keras, 沒有就找 .h5
            if os.path.exists(self.weights_path):
                self.model.load_weights(self.weights_path)
            elif os.path.exists(weights_h5):
                self.model.load_weights(weights_h5)
                    
        logger.info(f"Training Incremental Model for {self.stock_number}...")
        self.model.fit([X_ts, X_ext], Y, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.1)
        
        self.model.save_weights(self.weights_path)
        logger.info(f"Saved optimized weights to {self.weights_path}")
        return True

    def predict_5_days(self):
        """推論未來 5 天股價，並給出看漲機率"""
        full_df = self.build_dataset()
        if full_df is None or len(full_df) < self.time_steps: return None
        
        weights_h5 = self.weights_path.replace('.weights.h5', '.h5')
        if not os.path.exists(self.weights_path) and not os.path.exists(weights_h5):
            self.train_incremental(epochs=30)
        else:
            # 即便有權重，每天也微調個 1-2 個 Epoch 保持增量學習 (Fine-tune)
            self.train_incremental(epochs=2)
            
        if not hasattr(self, 'ts_idx') or self.ts_idx is None:
            _ = self.prepare_training_data(full_df)
            
        if not hasattr(self, 'max_vals') or self.max_vals is None:
             self.max_vals = full_df.max().replace(0, 1)
             
        recent_window = full_df.iloc[-self.time_steps:].copy()
        norm_window = (recent_window / self.max_vals).to_numpy()
        
        x_ts = np.array([norm_window[:, self.ts_idx]])
        x_ext = np.array([norm_window[-1, self.ext_idx]])
        
        if self.model is None:
             self.model = self.create_model(self.lstm_features, self.dense_features)
             if os.path.exists(self.weights_path):
                 self.model.load_weights(self.weights_path)
             elif os.path.exists(weights_h5):
                 self.model.load_weights(weights_h5)
             
        pred_norm = self.model.predict([x_ts, x_ext], verbose=0)
        close_max = self.max_vals['Close']
        pred_real = pred_norm[0] * close_max
        
        last_close = full_df['Close'].iloc[-1]
        avg_pred = np.mean(pred_real)
        # 簡單趨勢轉換，正負乖離 * 5 當作偏移機率
        trend_prob = 0.5 + ((avg_pred - last_close) / last_close) * 5.0
        trend_prob = np.clip(trend_prob, 0.1, 0.9)
        
        return {
            'predictions': pred_real.tolist(),
            'trend_probability': float(trend_prob),
            'latest_feature_date': full_df.index[-1].strftime('%Y-%m-%d')
        }
