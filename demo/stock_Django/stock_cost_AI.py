import os
import json
import logging
import numpy as np
import pandas as pd
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    tf = None

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
        
        # v4.0 Robustness Fix: 不要用 dropna，否則只要新聞或基本面漏一天，整天數據都會消失
        # 除非股價(技術指標)本身就是空的，否則我們用 0 填充
        if 'Close' in full_df.columns:
            full_df = full_df[full_df['Close'].notnull()]
        full_df.fillna(0, inplace=True)
        
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
        
        # --- Fallback for missing TensorFlow (e.g. Python 3.14 environment) ---
        if not HAS_TENSORFLOW or tf is None:
            logger.warning(f"TensorFlow is not available. Falling back to Linear Regression for {self.stock_number}...")
            from sklearn.linear_model import LinearRegression
            closes = target_df['Close'].tail(self.time_steps).values
            X = np.arange(len(closes)).reshape(-1, 1)
            y = closes
            model = LinearRegression().fit(X, y)
            X_future = np.arange(len(closes), len(closes) + 5).reshape(-1, 1)
            pred_real = model.predict(X_future)
            last_close = closes[-1]
            avg_pred = np.mean(pred_real)
            trend_prob = 0.5 + ((avg_pred - last_close) / (last_close + 1e-9)) * 5.0
            trend_prob = np.clip(trend_prob, 0.1, 0.9)
            return {
                'predictions': [float(p) for p in pred_real],
                'trend_probability': float(trend_prob),
                'latest_feature_date': target_df.index[-1].strftime('%Y-%m-%d')
            }
        
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

    def generate_gemini_advice(
        self, 
        lstm_pred: dict, 
        chips_features: dict, 
        sentiment_summary: dict, 
        valuation_features: dict, 
        latest_price: float
    ) -> dict:
        """
        綜合分析技術面(LSTM)、籌碼面(法人)、情緒面(新聞)、基本面(估值)與最新股價，
        調用雲端 Gemini.CLI 31b (gemma-4-31b-it) 取得買賣建議與指針分數。
        """
        import subprocess
        import shutil
        
        if lstm_pred is None:
            lstm_pred = {}
        if chips_features is None:
            chips_features = {}
        if sentiment_summary is None:
            sentiment_summary = {}
        if valuation_features is None:
            valuation_features = {}
            
        # 準備資料
        pred_list = lstm_pred.get('predictions', [])
        pred_str = ", ".join([f"{p:.2f}" for p in pred_list]) if pred_list else "暫無數據"
        trend_prob = lstm_pred.get('trend_probability', 0.5)
        
        # 籌碼資料
        chips_str = json.dumps(chips_features, ensure_ascii=False)
        
        # 輿情情緒
        pos = sentiment_summary.get('positive', 0)
        neg = sentiment_summary.get('negative', 0)
        neu = sentiment_summary.get('neutral', 0)
        senti_label = sentiment_summary.get('label', '中性')
        senti_score = sentiment_summary.get('score', 50.0)
        
        # 估值數據
        fair_val = valuation_features.get('fair_value', 'N/A')
        upside = valuation_features.get('upside', 0.0)
        val_rating = valuation_features.get('rating', 'N/A')
        
        # 構建 Prompt
        prompt = (
            f"您是專業的金融分析師。請綜合分析以下提供的股票（{self.stock_number}）數據，並給出投資建議：\n\n"
            f"[技術分析與預測]\n"
            f"- 最新收盤價: {latest_price:.2f}\n"
            f"- LSTM 預測未來 5 日價格走勢: [{pred_str}]\n"
            f"- LSTM 預估看漲機率: {trend_prob * 100:.1f}%\n\n"
            f"[法人籌碼面]\n"
            f"- 近期籌碼概況: {chips_str}\n\n"
            f"[輿情情緒面]\n"
            f"- 近期新聞情緒統計: 正面 {pos} 篇, 負面 {neg} 篇, 中性 {neu} 篇\n"
            f"- 情緒強度評級: {senti_label} (情緒得分: {senti_score:.1f}/100)\n\n"
            f"[基本面與估值]\n"
            f"- 公允估值: {fair_val}\n"
            f"- 目前股價與公允值差距 (Upside): {upside}%\n"
            f"- 估值評級: {val_rating}\n\n"
            f"任務：\n"
            f"1. 綜合上述數據，分析該股票的最新投資前景。\n"
            f"2. 給予買進或賣出的建議，評等必須嚴格限制為以下五個項目之一：'強力賣出', '賣出', '觀望', '買進', '強力買進'。\n"
            f"3. 給予一個推薦分數 (score)，範圍為 0 至 100 之間（0-20: 強力賣出, 21-40: 賣出, 41-60: 觀望, 61-80: 買進, 81-100: 強力買進）。\n"
            f"4. 提供 150 字以內的簡短中文推薦理由。\n\n"
            f"請嚴格以下列 JSON 格式輸出，不要包含任何 markdown 標記（如 ```json）或額外的說明文字：\n"
            f"{{\n"
            f"  \"recommendation\": \"觀望\",\n"
            f"  \"score\": 50,\n"
            f"  \"reason\": \"理由說明\",\n"
            f"  \"details\": {{\n"
            f"    \"technical\": \"技術分析簡短評語\",\n"
            f"    \"chips\": \"籌碼分析簡短評語\",\n"
            f"    \"sentiment\": \"輿情分析簡短評語\",\n"
            f"    \"valuation\": \"基本估值簡短評語\"\n"
            f"  }}\n"
            f"}}"
        )
        
        # 換行轉換以防 Windows 參數問題
        full_prompt = prompt.replace("\n", " ").replace("\r", " ").strip()
        
        gemini_path = shutil.which("gemini")
        if not gemini_path:
            logger.error("[GeminiAdvisor] 系統中找不到 gemini CLI。")
            return self._default_advice("系統找不到 gemini CLI 執行檔")
            
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            from dotenv import load_dotenv
            dotenv_path = os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", ".env")
            load_dotenv(dotenv_path)
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            
        env = os.environ.copy()
        if gemini_api_key:
            env["GEMINI_API_KEY"] = gemini_api_key
            
        # 決定要試的模型
        custom_model = os.getenv("GEMINI_ADVISOR_MODEL")
        if custom_model:
            models_to_try = [custom_model]
        else:
            models_to_try = ["gemini-3.1-pro-preview", "gemma-4-31b-it"]

        last_error = "所有模型呼叫皆失敗"
        
        for model in models_to_try:
            try:
                logger.info(f"[GeminiAdvisor] 正在呼叫雲端 Gemini.CLI {model} 分析股票 {self.stock_number}...")
                args = [gemini_path, "-m", model, "--skip-trust", "-o", "json", "-p", full_prompt]
                
                result = subprocess.run(
                    args,
                    capture_output=True,
                    env=env,
                    shell=False
                )
                
                if result.returncode != 0:
                    stderr_msg = result.stderr.decode("utf-8", errors="replace")
                    logger.warning(f"[GeminiAdvisor] Gemini CLI {model} 執行失敗 (code: {result.returncode}), stderr: {stderr_msg}")
                    last_error = f"{model} 執行失敗: {result.returncode}"
                    continue
                    
                stdout_decoded = result.stdout.decode("utf-8", errors="replace")
                
                if "{" in stdout_decoded:
                    json_start = stdout_decoded.index("{")
                    json_data = json.loads(stdout_decoded[json_start:])
                    response_text = json_data.get("response", "").strip()
                    
                    # 移除可能存在的 markdown wrapper
                    clean_res = response_text
                    if clean_res.startswith("```"):
                        lines = clean_res.splitlines()
                        if lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines[-1].startswith("```"):
                            lines = lines[:-1]
                        clean_res = "\n".join(lines).strip()
                        
                    try:
                        parsed_response = json.loads(clean_res)
                        # 驗證必要欄位與值範圍
                        if 'recommendation' in parsed_response and 'score' in parsed_response:
                            rec = parsed_response['recommendation']
                            if rec not in ['強力賣出', '賣出', '觀望', '買進', '強力買進']:
                                parsed_response['recommendation'] = '觀望'
                            
                            # 填入具可讀性的模型名稱
                            friendly_model_name = "Gemini 3.1 Pro" if "gemini-3.1" in model else ("Gemma-4-31B" if "gemma-4" in model else model)
                            parsed_response['model_name'] = friendly_model_name
                            return parsed_response
                    except Exception as je:
                        logger.warning(f"[GeminiAdvisor] 無法解析模型 {model} 回覆的 JSON: {je}. 原始內容: {clean_res}")
                        last_error = f"{model} JSON 解析錯誤"
                else:
                    logger.warning(f"[GeminiAdvisor] 模型 {model} 輸出不符合預期 JSON 格式。原始輸出: {stdout_decoded}")
                    last_error = f"{model} 輸出格式錯誤"
                    
            except Exception as e:
                logger.error(f"[GeminiAdvisor] 呼叫 Gemini CLI ({model}) 異常: {e}")
                last_error = f"{model} 系統異常: {str(e)}"
                
        return self._default_advice(last_error)
            
    def _default_advice(self, error_msg: str) -> dict:
        return {
            "recommendation": "觀望",
            "score": 50,
            "reason": f"暫時無法取得雲端分析建議 ({error_msg})。建議投資人此時採取觀望策略，並參考下方 K 線與技術指標自行判斷。",
            "details": {
                "technical": "暫無建議",
                "chips": "暫無建議",
                "sentiment": "暫無建議",
                "valuation": "暫無建議"
            },
            "model_name": "N/A"
        }

