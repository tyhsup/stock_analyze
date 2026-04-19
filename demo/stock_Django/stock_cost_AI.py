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
    from stock_Django.graph_builders import GraphBuilder
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
        from .graph_builders import GraphBuilder

    except (ImportError, ValueError):
        from stock_utils import StockUtils
        from dataset_builders import (
            PriceLSTMFeatureExtractor, 
            SentimentProbabilityModel, 
            InstitutionalFlowModel, 
            FundamentalFeatureProcessor
        )
        from model_architectures import StockModelArchitectures
        from graph_builders import GraphBuilder

logger = logging.getLogger(__name__)

class IntegratedStockPredModel:
    def __init__(self, stock_number):
        self.stock_number = str(stock_number)
        self.clean_number = self.stock_number.replace('.TW', '').replace('.TWO', '')
        self.market = 'tw' if '.TW' in self.stock_number or '.TWO' in self.stock_number else 'us'
        
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
        self.max_vals_dict = {}
        self.ts_idx = None
        self.ext_idx = None
        
        self.neighbor_count = self.config.get('neighbor_count', 5)
        self.graph_builder = GraphBuilder()
        
        # Get neighbors
        self.neighbors = self.graph_builder.get_market_cap_neighbors(self.stock_number, self.market, self.neighbor_count)
        self.all_nodes = [self.stock_number] + self.neighbors
        self.num_nodes = len(self.all_nodes)
        
        # Build Adjacency Matrix
        self.adj_matrix = self.graph_builder.build_adjacency_matrix(self.all_nodes)
        
        # --- GPU Optimization ---
        self._setup_gpu()

    def _setup_gpu(self):
        """配置 GPU 資源以進行高效訓練"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Detected {len(gpus)} GPUs. Memory growth enabled.")
            else:
                logger.debug("No GPU detected, falling back to CPU.")
        except Exception as e:
            logger.warning(f"GPU setup failed: {e}")

    def build_dataset_for_symbol(self, symbol):
        """為單一隻股票建立 Dataframe"""
        clean_sym = symbol.replace('.TW', '').replace('.TWO', '')
        cost_data, Date_data = StockUtils.load_data_c('stock_cost', symbol)
        if cost_data.empty:
            cost_data, Date_data = StockUtils.load_data_c('stock_cost_us', symbol)
            
        if cost_data.empty:
            logger.error(f"Cannot find price data for {symbol} in DB.")
            return None

        cost_data.index = pd.to_datetime(Date_data['Date']).dt.normalize()
        
        price_feat = PriceLSTMFeatureExtractor.extract_features(cost_data)
        date_index_df = pd.DataFrame(index=cost_data.index)
        
        senti_feat = SentimentProbabilityModel.get_sentiment_features(clean_sym, date_index_df)
        flow_feat = InstitutionalFlowModel.get_flow_features(clean_sym, date_index_df)
        fund_feat = FundamentalFeatureProcessor.get_fundamental_features(clean_sym, date_index_df)
        
        full_df = pd.concat([price_feat, senti_feat, flow_feat, fund_feat], axis=1)
        full_df.dropna(inplace=True)
        return full_df

    def build_all_nodes_datasets(self):
        """
        取得 [Target, Neighbor 1, Neighbor 2...] 各自的 DataFrame
        回傳: dict {symbol: DataFrame}
        並且確保對齊相同的日期 Index
        """
        dfs = {}
        common_index = None

        # Build target first
        target_df = self.build_dataset_for_symbol(self.stock_number)
        if target_df is None or target_df.empty: return None
        
        dfs[self.stock_number] = target_df
        common_index = target_df.index
        
        # Build neighbors
        for n in self.neighbors:
            df_n = self.build_dataset_for_symbol(n)
            if df_n is not None and not df_n.empty:
                # 確保共同時間窗 (取最晚重疊的開始點與最早的結束點) 
                common_index = common_index.intersection(df_n.index)
                dfs[n] = df_n
        
        if len(common_index) == 0: return None
        
        # Align all DataFrames
        for k in dfs.keys():
            dfs[k] = dfs[k].loc[common_index]
            
            # Record max values for normalization
            if k not in self.max_vals_dict:
                self.max_vals_dict[k] = dfs[k].max().replace(0, 1)

        return dfs

    def prepare_training_data(self, dfs_dict):
        """為圖網路構建 (Batch, Nodes, Time, Features) Tensor (v4.0 三模態切分)"""
        target_df = dfs_dict[self.stock_number]
        cols = target_df.columns
        
        # 1. 切分模態
        senti_cols = [c for c in cols if 'finbert_emb_' in c]
        fin_cols = ['Net_Buy_Volume', 'EPS_Quarterly', 'Revenue_Growth']
        
        senti_idx = [target_df.columns.get_loc(c) for c in senti_cols if c in target_df.columns]
        fin_idx = [target_df.columns.get_loc(c) for c in fin_cols if c in target_df.columns]
        ts_idx = [i for i in range(len(cols)) if i not in senti_idx and i not in fin_idx]
        
        close_col_idx = target_df.columns.get_loc('Close')
        
        X_ts, X_senti, X_fin, Y, A = [], [], [], [], []
        num_samples = len(target_df)
        
        if num_samples <= self.time_steps + self.predict_steps:
            return None
            
        norm_arrays = {}
        for k, v_df in dfs_dict.items():
             norm_arrays[k] = (v_df / self.max_vals_dict[k]).to_numpy()
             
        for i in range(num_samples - self.time_steps - self.predict_steps + 1):
            batch_ts_nodes = []
            batch_senti_nodes = []
            batch_fin_nodes = []
            
            for node_name in self.all_nodes:
                if node_name in norm_arrays:
                    data = norm_arrays[node_name]
                else: 
                    data = np.zeros((num_samples, len(cols)))
                    
                window = data[i : i + self.time_steps]
                batch_ts_nodes.append(window[:, ts_idx])
                batch_senti_nodes.append(window[-1, senti_idx])
                batch_fin_nodes.append(window[-1, fin_idx])
                
                if node_name == self.stock_number:
                     target = data[i + self.time_steps : i + self.time_steps + self.predict_steps, close_col_idx]
                     
            X_ts.append(batch_ts_nodes)
            X_senti.append(batch_senti_nodes)
            X_fin.append(batch_fin_nodes)
            Y.append(target)
            A.append(self.adj_matrix)
            
        self.lstm_features = len(ts_idx)
        self.senti_features = len(senti_idx)
        self.fin_features = len(fin_idx)
        self.ts_idx = ts_idx
        self.senti_idx = senti_idx
        self.fin_idx = fin_idx
        
        return np.array(X_ts), np.array(X_senti), np.array(X_fin), np.array(A), np.array(Y)

    def train_incremental(self, epochs=None, batch_size=None):
        """增量訓練邏輯 (Incremental Learning)"""
        if epochs is None:
            epochs = self.config.get('incremental_train_epochs', 20)
        if batch_size is None:
            batch_size = self.config.get('batch_size', 32)
            
        dfs_dict = self.build_all_nodes_datasets()
        if dfs_dict is None or self.stock_number not in dfs_dict: 
            return False
            
        if len(dfs_dict[self.stock_number]) < self.time_steps + self.predict_steps:
             return False
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=self.config.get('patience', 5), 
                restore_best_weights=True
            )
        ]
        
        prepared = self.prepare_training_data(dfs_dict)
        if not prepared: return False
        X_ts, X_senti, X_fin, A, Y = prepared
        
        if self.model is None:
            self.model = StockModelArchitectures.build_multi_input_model(
                self.lstm_features, self.senti_features, self.fin_features, self.config
            )
            weights_h5 = self.weights_path.replace('.weights.h5', '.h5')
            try:
                if os.path.exists(self.weights_path):
                    self.model.load_weights(self.weights_path)
                elif os.path.exists(weights_h5):
                    self.model.load_weights(weights_h5)
            except (ValueError, Exception) as e:
                logger.warning(f"Weight loading failed (architecture mismatch from v3->v4?): {e}. Training from scratch.")
                    
        logger.info(f"Training Incremental Model (v4.0) for {self.stock_number} (Target: {epochs} epochs)...")
        
        val_split = self.config.get('val_split', 0.1)
        self.model.fit(
            [X_ts, X_senti, X_fin, A], Y, 
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
        dfs_dict = self.build_all_nodes_datasets()
        if dfs_dict is None or self.stock_number not in dfs_dict: return None
        target_df = dfs_dict[self.stock_number]
        if len(target_df) < self.time_steps: return None
        
        weights_h5 = self.weights_path.replace('.weights.h5', '.h5')
        if not os.path.exists(self.weights_path) and not os.path.exists(weights_h5):
            self.train_incremental(epochs=self.config.get('full_train_epochs', 30))
        else:
            self.train_incremental(epochs=self.config.get('finetune_epochs', 2))
            
        if not hasattr(self, 'ts_idx') or self.ts_idx is None:
            _ = self.prepare_training_data(dfs_dict)
            
        x_ts_nodes = []
        x_senti_nodes = []
        x_fin_nodes = []
        
        for node_name in self.all_nodes:
             if node_name in dfs_dict:
                 data = dfs_dict[node_name].iloc[-self.time_steps:].copy()
                 norm_window = (data / self.max_vals_dict[node_name]).to_numpy()
             else:
                 norm_window = np.zeros((self.time_steps, len(target_df.columns)))
                 
             x_ts_nodes.append(norm_window[:, self.ts_idx])
             x_senti_nodes.append(norm_window[-1, self.senti_idx])
             x_fin_nodes.append(norm_window[-1, self.fin_idx])
             
        # Add Batch dimension
        x_ts = np.array([x_ts_nodes])
        x_senti = np.array([x_senti_nodes])
        x_fin = np.array([x_fin_nodes])
        a_mat = np.array([self.adj_matrix])
        
        if self.model is None:
             self.model = StockModelArchitectures.build_multi_input_model(
                 self.lstm_features, self.senti_features, self.fin_features, self.config
             )
             try:
                 if os.path.exists(self.weights_path):
                     self.model.load_weights(self.weights_path)
                 elif os.path.exists(weights_h5):
                     self.model.load_weights(weights_h5)
             except (ValueError, Exception) as e:
                 logger.warning(f"Weight loading failed in predict_5_days (v4.0): {e}. Inference will use initial weights.")
             
        pred_norm = self.model.predict([x_ts, x_senti, x_fin, a_mat], verbose=0)
        close_max = self.max_vals_dict[self.stock_number]['Close']
        pred_real = pred_norm[0] * close_max
        
        last_close = target_df['Close'].iloc[-1]
        avg_pred = np.mean(pred_real)
        
        trend_prob = 0.5 + ((avg_pred - last_close) / (last_close + 1e-9)) * 5.0
        trend_prob = np.clip(trend_prob, 0.1, 0.9)
        
        return {
            'predictions': pred_real.tolist(),
            'trend_probability': float(trend_prob),
            'latest_feature_date': target_df.index[-1].strftime('%Y-%m-%d')
        }
