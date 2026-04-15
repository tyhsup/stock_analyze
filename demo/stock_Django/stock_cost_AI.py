import os
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from stock_Django.stock_utils import StockUtils
    from stock_Django.dataset_builders import (
        PriceLSTMFeatureExtractor, 
        SentimentProbabilityModel, 
        InstitutionalFlowModel, 
        FundamentalFeatureProcessor
    )
    from stock_Django.model_architectures import StockModelArchitectures
except ImportError:
    try:
        from .stock_utils import StockUtils
        from .dataset_builders import (
            PriceLSTMFeatureExtractor, 
            SentimentProbabilityModel, 
            InstitutionalFlowModel, 
            FundamentalFeatureProcessor
        )
        from .model_architectures import StockModelArchitectures
    except (ImportError, ValueError):
        from stock_utils import StockUtils
        from dataset_builders import (
            PriceLSTMFeatureExtractor, 
            SentimentProbabilityModel, 
            InstitutionalFlowModel, 
            FundamentalFeatureProcessor
        )
        from model_architectures import StockModelArchitectures

logger = logging.getLogger(__name__)

class IntegratedStockPredModel:
    def __init__(self, stock_number):
        self.stock_number = str(stock_number)
        self.clean_number = self.stock_number.replace('.TW', '').replace('.TWO', '')
        
        # Load configuration
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'model_config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            logger.warning("model_config.json not found. Using default configs.")
            self.config = {}

        self.model_dir = os.path.join(current_dir, 'stock_model')
        os.makedirs(self.model_dir, exist_ok=True)
        self.weights_path = os.path.join(self.model_dir, f'stock_model_weights_{self.clean_number}.weights.h5')
        
        self.time_steps = self.config.get('time_steps', 20)
        self.predict_steps = self.config.get('predict_steps', 5)
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

    def prepare_training_data(self, full_df):
        """切割 Sliding Window 特徵，準備 Keras 輸入格式"""
        df_vals = full_df.to_numpy()
        cols = full_df.columns
        
        ext_cols = [c for c in cols if 'finbert_emb_' in c] + ['Net_Buy_Volume', 'EPS_Quarterly', 'Revenue_Growth']
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

    def train_incremental(self, epochs=None, batch_size=None):
        """增量訓練邏輯 (Incremental Learning)"""
        if epochs is None:
            epochs = self.config.get('incremental_train_epochs', 20)
        if batch_size is None:
            batch_size = self.config.get('batch_size', 32)
            
        full_df = self.build_dataset()
        if full_df is None or len(full_df) < self.time_steps + self.predict_steps: 
            return False
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=self.config.get('patience', 5), 
                restore_best_weights=True
            )
        ]
        
        self.max_vals = full_df.max().replace(0, 1)
        norm_df = full_df / self.max_vals
        
        prepared = self.prepare_training_data(norm_df)
        if not prepared: return False
        X_ts, X_ext, Y = prepared
        
        if self.model is None:
            self.model = StockModelArchitectures.build_multi_input_model(self.lstm_features, self.dense_features, self.config)
            weights_h5 = self.weights_path.replace('.weights.h5', '.h5')
            if os.path.exists(self.weights_path):
                self.model.load_weights(self.weights_path)
            elif os.path.exists(weights_h5):
                self.model.load_weights(weights_h5)
                    
        logger.info(f"Training Incremental Model for {self.stock_number} (Target: {epochs} epochs)...")
        
        val_split = self.config.get('val_split', 0.1)
        self.model.fit(
            [X_ts, X_ext], Y, 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=0, 
            validation_split=val_split,
            callbacks=callbacks
        )
        
        self.model.save_weights(self.weights_path)
        logger.info(f"Saved optimized weights to {self.weights_path}")
        return True

    def predict_5_days(self):
        """推論未來 5 天股價，並給出看漲機率"""
        full_df = self.build_dataset()
        if full_df is None or len(full_df) < self.time_steps: return None
        
        weights_h5 = self.weights_path.replace('.weights.h5', '.h5')
        if not os.path.exists(self.weights_path) and not os.path.exists(weights_h5):
            self.train_incremental(epochs=self.config.get('full_train_epochs', 30))
        else:
            self.train_incremental(epochs=self.config.get('finetune_epochs', 2))
            
        if not hasattr(self, 'ts_idx') or self.ts_idx is None:
            _ = self.prepare_training_data(full_df)
            
        if not hasattr(self, 'max_vals') or self.max_vals is None:
             self.max_vals = full_df.max().replace(0, 1)
             
        recent_window = full_df.iloc[-self.time_steps:].copy()
        norm_window = (recent_window / self.max_vals).to_numpy()
        
        x_ts = np.array([norm_window[:, self.ts_idx]])
        x_ext = np.array([norm_window[-1, self.ext_idx]])
        
        if self.model is None:
             self.model = StockModelArchitectures.build_multi_input_model(self.lstm_features, self.dense_features, self.config)
             if os.path.exists(self.weights_path):
                 self.model.load_weights(self.weights_path)
             elif os.path.exists(weights_h5):
                 self.model.load_weights(weights_h5)
             
        pred_norm = self.model.predict([x_ts, x_ext], verbose=0)
        close_max = self.max_vals['Close']
        pred_real = pred_norm[0] * close_max
        
        last_close = full_df['Close'].iloc[-1]
        avg_pred = np.mean(pred_real)
        
        trend_prob = 0.5 + ((avg_pred - last_close) / (last_close + 1e-9)) * 5.0
        trend_prob = np.clip(trend_prob, 0.1, 0.9)
        
        return {
            'predictions': pred_real.tolist(),
            'trend_probability': float(trend_prob),
            'latest_feature_date': full_df.index[-1].strftime('%Y-%m-%d')
        }
