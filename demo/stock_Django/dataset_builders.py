import logging
import math
import datetime
import numpy as np
import pandas as pd
import os
from typing import List, Optional, Union, Dict, Any
from sqlalchemy import text
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TORCH = True
    no_grad = torch.no_grad
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModel = None
    HAS_TORCH = False
    def no_grad():
        def decorator(func):
            return func
        return decorator

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
    def extract_features(data_df, market: str = 'tw', scaler=None, fit_mode: bool = True, return_scaler: bool = False):
        """
        產生 LSTM 所需之漲跌幅度與型態特徵，整合交易日曆並落實標準化隔離 (Scaler Leakage 防護)。
        
        :param data_df: 原始價格行情 DataFrame
        :param market: 市場別 ('tw' / 'us')
        :param scaler: sklearn StandardScaler 實例 (若 fit_mode=False 必須傳入訓練集擬合之 scaler)
        :param fit_mode: True 代表訓練集 (fit_transform)，False 代表測試集 (僅 transform)
        :param return_scaler: 是否同時回傳 scaler 物件 (向後相容預設 False)
        """
        df = data_df.copy()
        if 'Close' not in df.columns:
            if return_scaler:
                return df, scaler
            return df
            
        # 整合交易日曆，過濾非交易日
        try:
            try:
                from stock_Django.trading_calendar import TradingCalendarService
            except ImportError:
                from .trading_calendar import TradingCalendarService
            cal = TradingCalendarService()
            if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                start_d = df.index.min().date()
                end_d = df.index.max().date()
                trading_days = set(cal.get_trading_days(market, start_d, end_d))
                df = df[df.index.map(lambda x: x.date() in trading_days)]
        except Exception as e:
            logger.warning(f"PriceLSTMFeatureExtractor 交易日曆對齊略過: {e}")

        # 產生日報酬率
        df['Daily_Return'] = df['Close'].pct_change()
        # 均線特徵
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        # 價格距均線的乖離率
        df['Bias_5'] = (df['Close'] - df['SMA_5']) / df['SMA_5']
        df['Bias_20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        df.fillna(0, inplace=True)

        # 標準化特徵隔離
        feature_cols = [c for c in ['Daily_Return', 'Bias_5', 'Bias_20'] if c in df.columns]
        if feature_cols:
            from sklearn.preprocessing import StandardScaler
            if fit_mode:
                if scaler is None:
                    scaler = StandardScaler()
                df[feature_cols] = scaler.fit_transform(df[feature_cols])
            else:
                if scaler is None:
                    raise ValueError("【安全警報】測試模式下（fit_mode=False）必須提供已擬合之 scaler，以防特徵洩漏！")
                df[feature_cols] = scaler.transform(df[feature_cols])

        if return_scaler:
            return df, scaler
        return df

class WeightedSentimentAggregator:
    """
    多因子情緒加權聚合器。
    
    加權公式：
    Score_d = Σ (E_i × w_time_i × w_source_i × w_conf_i) / Σ (w_time_i × w_source_i × w_conf_i)
    
    各權重項定義（使用者審查確認之分級）：
    - w_time = exp(-λ × Δt)  時間指數衰減（Δt 為距交易日天數差距，預設 λ=0.1）
    - w_source = SOURCE_WEIGHTS[source]  來源權威度：CNBC/Reuters (0.9) > CNYES/MoneyDJ (0.7) > PTT (0.3)
    - w_conf = confidence_score  模型推論置信度（預設 1.0）
    """
    
    SOURCE_WEIGHTS = {
        'cnbc': 0.90,       # 國際權威財經媒體
        'reuters': 0.90,    # 路透社
        'cnyes': 0.70,      # 台灣專業財經媒體（鉅亨網）
        'moneydj': 0.70,    # 理財網（MoneyDJ）
        'ptt': 0.30,        # 社群討論區（批踢踢）
        'default': 0.50     # 其他來源預設
    }
    
    @classmethod
    def get_source_weight(cls, source_name: Optional[str]) -> float:
        if not source_name:
            return cls.SOURCE_WEIGHTS['default']
        cleaned = str(source_name).lower().strip()
        for key, weight in cls.SOURCE_WEIGHTS.items():
            if key in cleaned:
                return weight
        return cls.SOURCE_WEIGHTS['default']
        
    @classmethod
    def aggregate(cls, embeddings: np.ndarray,
                  timestamps: Optional[List[Any]] = None,
                  sources: Optional[List[str]] = None,
                  confidences: Optional[List[float]] = None,
                  target_date: Optional[Union[datetime.date, str]] = None,
                  decay_lambda: float = 0.1,
                  embedding_dim: int = 768) -> np.ndarray:
        """
        對當日多筆新聞嵌入向量執行多因子加權聚合。
        防禦機制：
        - 零除與浮點數下溢（Underflow）/ NaN / Inf 防護，回退至算術平均。
        - 輸出強轉為 np.float32 確保序列化與儲存一致。
        """
        if len(embeddings) == 0:
            return np.zeros(embedding_dim, dtype=np.float32)
            
        n = len(embeddings)
        weights = np.ones(n, dtype=np.float64)
        
        # 1. 時間衰減權重
        if timestamps and target_date:
            try:
                t_date = pd.to_datetime(target_date).date()
                for i in range(n):
                    ts = pd.to_datetime(timestamps[i])
                    if not pd.isna(ts):
                        dt_days = max(0.0, (t_date - ts.date()).total_seconds() / 86400.0)
                        weights[i] *= math.exp(-decay_lambda * dt_days)
            except Exception as e:
                logger.warning(f"時間衰減權重計算異常: {e}")
                
        # 2. 來源權威度權重
        if sources:
            for i in range(min(n, len(sources))):
                weights[i] *= cls.get_source_weight(sources[i])
                
        # 3. 模型置信度權重
        if confidences:
            for i in range(min(n, len(confidences))):
                c = float(confidences[i]) if confidences[i] is not None else 1.0
                weights[i] *= max(0.01, min(1.0, c))
                
        # 4. 防禦：處理零除、NaN、Inf 與極小浮點數
        sum_w = float(np.sum(weights))
        if math.isnan(sum_w) or math.isinf(sum_w) or sum_w < 1e-9:
            return np.mean(embeddings, axis=0).astype(np.float32)
            
        normalized_weights = weights / sum_w
        weighted_emb = np.sum(embeddings * normalized_weights[:, np.newaxis], axis=0)
        return weighted_emb.astype(np.float32)

class SentimentTimeDecay:
    """
    時間序列情緒特徵衰減向前填充器 (Phase 3 P3)。
    實作公式：V_t = V_{t-1} * exp(-lambda * delta_t)
    具備 Evaluator 要求之防禦機制：
    - delta_t 負值檢查 (delta_t < 0 拋出 ValueError)
    - delta_t > max_gap_days (預設 100) 下溢直接歸零
    - 首日缺失/Null 安全預設為 0.0
    - 支援 Series 或 DataFrame 批次多維特徵運算
    """
    @staticmethod
    def calculate_decay_factor(delta_t: float, lambda_decay: float = 0.1, max_gap_days: float = 100.0) -> float:
        if delta_t < 0:
            raise ValueError(f"【安全防禦】時間差 delta_t 不能為負值 (收到: {delta_t})")
        if delta_t > max_gap_days:
            return 0.0
        if delta_t < 1e-9:
            return 1.0
        try:
            return float(math.exp(-lambda_decay * delta_t))
        except (OverflowError, FloatingPointError):
            return 0.0

    @classmethod
    def apply_time_decay_ffill(cls, data: Union[pd.Series, pd.DataFrame], 
                               dates: Optional[pd.DatetimeIndex] = None,
                               lambda_decay: float = 0.1,
                               max_gap_days: float = 100.0) -> Union[pd.Series, pd.DataFrame]:
        """
        對時間序列進行指數衰減向前填充。
        若某日資料全為 0 或 NaN，依據與上一筆有效新聞日之日曆天數差進行指數衰減。
        """
        if data.empty:
            return data.copy()

        out = data.copy()
        date_idx = pd.DatetimeIndex(out.index) if dates is None else pd.DatetimeIndex(dates)

        if isinstance(out, pd.Series):
            last_valid_val = None
            last_valid_date = None
            for idx, d in enumerate(date_idx):
                cur_val = out.iloc[idx]
                is_valid = pd.notna(cur_val) and float(cur_val) != 0.0
                if is_valid:
                    last_valid_val = float(cur_val)
                    last_valid_date = d
                else:
                    if last_valid_val is None or last_valid_date is None:
                        out.iloc[idx] = 0.0
                    else:
                        dt = (d - last_valid_date).total_seconds() / 86400.0
                        factor = cls.calculate_decay_factor(dt, lambda_decay, max_gap_days)
                        out.iloc[idx] = last_valid_val * factor
            return out

        # DataFrame 多欄位逐欄獨立衰減填充
        res_cols = {}
        for col in out.columns:
            res_cols[col] = cls.apply_time_decay_ffill(
                out[col], dates=date_idx, lambda_decay=lambda_decay, max_gap_days=max_gap_days
            )
        return pd.DataFrame(res_cols, index=out.index)


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
    @no_grad()
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
    def get_sentiment_features(cls, stock_number, date_index_df, apply_decay_fill: bool = True):
        """取得新聞語詞特徵 (具備資料庫快取機制與指數時間衰減向前填充)"""
        embedding_dim = 768
        embedding_cols = [f'finbert_emb_{i}' for i in range(embedding_dim)]
        
        # Helper: 套用時間衰減向前填充防護
        def _finalize(df_res):
            sub_df = df_res[embedding_cols]
            if apply_decay_fill and not sub_df.empty:
                return SentimentTimeDecay.apply_time_decay_ffill(sub_df, lambda_decay=0.1)
            return sub_df
        
        # Initialize result DataFrame
        result_df = pd.DataFrame(0.0, index=date_index_df.index, columns=embedding_cols)
        
        if date_index_df.empty:
            return result_df
            
        sql_op = OP_Fun()
        clean_num = str(stock_number).upper().replace('.TWO', '').replace('.TW', '')
        start_date = date_index_df.index.min().strftime('%Y-%m-%d')
        end_date = date_index_df.index.max().strftime('%Y-%m-%d')
        
        # 1. 嘗試從快取讀取 (批次)
        cached_embeddings = sql_op.get_sentiment_embeddings(stock_number, start_date, end_date)
        
        # 2. 向量化快取填充與缺失檢測
        if cached_embeddings:
            cache_df = pd.DataFrame.from_dict(cached_embeddings, orient='index', columns=embedding_cols)
            cache_df.index = pd.to_datetime(cache_df.index)
            # 使用 update 向量化覆寫已存在的快取數據，避免 for 迴圈
            result_df.update(cache_df)
            
        # 找出缺失快取的日期
        cached_dates_set = {pd.Timestamp(k) for k in cached_embeddings.keys()}
        missing_dates = [d for d in date_index_df.index if d not in cached_dates_set]

        
        if not missing_dates:
            return _finalize(result_df)
            
        # 3. 僅針對缺失日期讀取原始文本並進行推論
        logger.info(f"AI Cache Miss for {stock_number}. Parsing raw news for {len(missing_dates)} days...")
        
        # 嘗試讀取包含原始文本的 Excel 檔案
        webbug_dir = os.getenv('WEBBUG_DIR', 'E:/Infinity/webbug/')
        news_file = os.path.join(webbug_dir, f'{clean_num}_news.xlsx')
        
        if not os.path.exists(news_file):
            logger.warning(f"News text file not found: {news_file}. Using 0.0 embeddings fallback.")
            return _finalize(result_df)
            
        try:
            # 讀取 Excel 並解析日期與標題
            df_news = pd.read_excel(news_file)
            if df_news.empty:
                return _finalize(result_df)

            try:
                from stock_Django.trading_calendar import TradingCalendarService
            except ImportError:
                from .trading_calendar import TradingCalendarService
            cal = TradingCalendarService()
            
            market = 'tw' if clean_num.isdigit() else 'us'
            
            # 偵測各欄位 (0=標題, 1=發布時間, 3=連結)
            time_col_idx = 1 if df_news.shape[1] > 1 else 0
            text_col_idx = 0
            link_col_idx = 3 if df_news.shape[1] > 3 else (2 if df_news.shape[1] > 2 else None)
            
            df_news['Raw_Time'] = pd.to_datetime(df_news.iloc[:, time_col_idx], errors='coerce')
            df_news['Parsed_Text'] = df_news.iloc[:, text_col_idx].fillna("").astype(str)
            
            # 抽取新聞來源
            link_series = df_news.iloc[:, link_col_idx].fillna("").astype(str) if link_col_idx is not None else pd.Series([""] * len(df_news))
            def _detect_source(link_val: str, text_val: str) -> str:
                combined = (link_val + " " + text_val).lower()
                for src_key in ['cnbc', 'reuters', 'cnyes', 'moneydj', 'ptt']:
                    if src_key in combined:
                        return src_key
                return 'cnyes' if 'cnyes.com' in link_val.lower() else 'default'
            
            df_news['Source'] = [
                _detect_source(l, t) for l, t in zip(link_series, df_news['Parsed_Text'])
            ]
            
            # 透過 TradingCalendarService 對齊有效交易日 (T+1 盤後防前瞻偏誤)
            df_news['Aligned_Date'] = df_news['Raw_Time'].apply(
                lambda t: cal.align_to_trading_day(t, market=market).strftime('%Y-%m-%d') if pd.notna(t) else None
            )
            df_news = df_news.dropna(subset=['Aligned_Date'])
            
            # 過濾僅保留缺失日期的數據
            missing_date_strs = [d.strftime('%Y-%m-%d') for d in missing_dates]
            df_missing = df_news[df_news['Aligned_Date'].isin(missing_date_strs)]
            
            if df_missing.empty:
                return result_df[embedding_cols]
            
            # 進行批次推論
            texts = df_missing['Parsed_Text'].tolist()
            embeddings = cls._get_embeddings_batch(texts)
            
            if len(embeddings) > 0:
                df_missing = df_missing.copy()
                df_missing['embedding'] = list(embeddings)
                
                # 多因子加權聚合 (取代舊有算術平均)
                new_cache_data = {}
                for aligned_d, group in df_missing.groupby('Aligned_Date'):
                    group_embs = np.vstack(group['embedding'].values)
                    group_times = group['Raw_Time'].tolist()
                    group_sources = group['Source'].tolist()
                    
                    agg_emb = WeightedSentimentAggregator.aggregate(
                        embeddings=group_embs,
                        timestamps=group_times,
                        sources=group_sources,
                        target_date=aligned_d,
                        decay_lambda=0.1,
                        embedding_dim=embedding_dim
                    )
                    new_cache_data[aligned_d] = agg_emb
                    d_ts = pd.Timestamp(aligned_d)
                    if d_ts in result_df.index:
                        result_df.loc[d_ts, embedding_cols] = agg_emb
                
                sql_op.save_sentiment_embeddings(stock_number, new_cache_data)
                
        except Exception as e:
            logger.error(f"Failed to process raw news text for {stock_number}: {e}")
            
        return _finalize(result_df)

class InstitutionalFlowModel:
    @staticmethod
    def get_flow_features(stock_number, date_index_df):
        """讀取法人籌碼動向，轉化為金流特徵"""
        sql_op = OP_Fun()
        is_tw = str(stock_number).isdigit() or ".TW" in str(stock_number).upper() or ".TWO" in str(stock_number).upper()
        table_name = 'stock_investor' if is_tw else 'stock_investor_us'
        
        clean_num = str(stock_number).replace('.TWO', '').replace('.TW', '')
        if is_tw:
            inv_df = sql_op.get_cost_data(table_name=table_name, stock_number=clean_num)
        else:
            # US investor table uses 'ticker' instead of 'number'
            query = f"SELECT date, shares, change_pct FROM {table_name} WHERE ticker = :num"
            try:
                inv_df = pd.read_sql(text(query), con=sql_op.engine, params={'num': clean_num})
            except Exception as e:
                logger.error(f"Failed to fetch {table_name}: {e}")
                inv_df = pd.DataFrame()
        
        zeros = np.zeros(len(date_index_df))
        if inv_df.empty:
            return pd.DataFrame({'Net_Buy_Volume': zeros}, index=date_index_df.index)
            
        if is_tw:
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
        else:
            # 美股金流特徵提取邏輯 (季度持股變動倒推)
            inv_df['change_pct'] = pd.to_numeric(inv_df['change_pct'], errors='coerce').fillna(0.0)
            inv_df['shares'] = pd.to_numeric(inv_df['shares'], errors='coerce').fillna(0.0)
            
            shares = inv_df['shares']
            pct = inv_df['change_pct']
            
            # 使用 np.where 進行矩陣向量化計算，取代緩慢的 apply 迴圈
            inv_df['net_buy'] = np.where(
                pct <= -1.0,
                -shares,
                shares * (1.0 - 1.0 / (1.0 + pct))
            )

            inv_df['日期'] = pd.to_datetime(inv_df['date']).dt.normalize()
            
            # 按日期加總機構買賣淨額
            daily_net = inv_df.groupby('日期')['net_buy'].sum().to_frame('Net_Buy_Volume')
            
            merged = date_index_df.copy()
            merged = merged.merge(daily_net, left_index=True, right_index=True, how='left')
            merged['Net_Buy_Volume'] = merged['Net_Buy_Volume'].ffill().fillna(0.0)
            
            return merged[['Net_Buy_Volume']]

class FundamentalFeatureProcessor:
    @staticmethod
    def get_fundamental_features(stock_number, date_index_df):
        """讀取財報基本面，運用 Forward-fill 給每日預測使用"""
        sql_op = OP_Fun()
        is_tw = str(stock_number).isdigit() or ".TW" in str(stock_number).upper() or ".TWO" in str(stock_number).upper()
        market = 'tw' if is_tw else 'us'
        table_name = f'financial_raw_{market}'
        clean_num = str(stock_number).replace('.TWO', '').replace('.TW', '')
        
        query = f"SELECT year, quarter, item_name, amount FROM {table_name} WHERE symbol = '{clean_num}'"
        try:
            fin_df = pd.read_sql(sql_op.engine, query)
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

class MovingAverageTrendExtractor:
    @staticmethod
    def extract_ma_features(data_df: pd.DataFrame, window_days: int = 5) -> dict:
        """
        抽取均線線型、斜率轉折點、扣抵預警與量價確認特徵。
        全邏輯 100% 對齊前端 home.html computeMAReversalPoints 演算法。
        包含降級容錯 (Graceful Degradation) 與近 window_days 日訊號去重過濾。
        """
        if data_df is None or data_df.empty or 'Close' not in data_df.columns:
            return {
                "ma_alignment": "無數據",
                "recent_signals": [],
                "bias_20": "0.0%",
                "volume_confirmation": "無數據",
                "degradation_level": "無資料"
            }

        df = data_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df.index = pd.to_datetime(df['Date'])
            else:
                df.index = pd.to_datetime(df.index)

        df = df.sort_index()
        n_rows = len(df)
        closes = df['Close'].astype(float).values
        volumes = df['Volume'].astype(float).values if 'Volume' in df.columns else np.zeros(n_rows)

        # 1. 降級層級判定 (Graceful Degradation)
        if n_rows >= 240:
            deg_level = "完整 (5/10/20/60/240 MA)"
        elif n_rows >= 60:
            deg_level = "中短期 (5/10/20/60 MA)"
        elif n_rows >= 20:
            deg_level = "短期 (5/10/20 MA)"
        else:
            deg_level = "資料不足 (僅極短期 5MA)"

        # 2. 計算各週期 MA 與布林通道 (Bollinger Bands: 20MA ± 2σ)
        s_close = df['Close'].astype(float)
        ma5 = s_close.rolling(5).mean().values
        ma10 = s_close.rolling(10).mean().values
        ma20 = s_close.rolling(20).mean().values
        ma60 = s_close.rolling(60).mean().values if n_rows >= 60 else np.full(n_rows, np.nan)
        ma240 = s_close.rolling(240).mean().values if n_rows >= 240 else np.full(n_rows, np.nan)

        # 布林通道三軌與指標
        std20 = s_close.rolling(20).std().values
        bb_upper = ma20 + 2 * std20
        bb_lower = ma20 - 2 * std20
        bb_denom = np.where((bb_upper - bb_lower) != 0, bb_upper - bb_lower, np.nan)
        percent_b = np.where(~np.isnan(bb_denom), (closes - bb_lower) / bb_denom, 0.5)
        bandwidth = np.where(~np.isnan(ma20) & (ma20 != 0), (bb_upper - bb_lower) / ma20, np.nan)

        s_vol = pd.Series(volumes, index=df.index)
        vol_5ma = s_vol.rolling(5).mean().values if 'Volume' in df.columns else np.zeros(n_rows)

        # 3. 當前（最新）均線排列型態 (MA Alignment)
        curr_close = closes[-1]
        c_ma5 = ma5[-1] if not np.isnan(ma5[-1]) else None
        c_ma10 = ma10[-1] if not np.isnan(ma10[-1]) else None
        c_ma20 = ma20[-1] if not np.isnan(ma20[-1]) else None
        c_ma60 = ma60[-1] if not np.isnan(ma60[-1]) else None
        c_ma240 = ma240[-1] if not np.isnan(ma240[-1]) else None

        ma_alignment = "多空混沌 / 盤整"
        if c_ma5 and c_ma10 and c_ma20 and c_ma60 and c_ma240:
            if curr_close > c_ma5 > c_ma10 > c_ma20 > c_ma60 > c_ma240:
                ma_alignment = "標準強勢多頭排列 (Close > 5MA > 10MA > 20MA > 60MA > 240MA)"
            elif curr_close < c_ma5 < c_ma10 < c_ma20 < c_ma60 < c_ma240:
                ma_alignment = "標準極弱空頭排列 (Close < 5MA < 10MA < 20MA < 60MA < 240MA)"
            elif c_ma5 > c_ma10 > c_ma20 and curr_close > c_ma20:
                ma_alignment = "短中期多頭架構 (5MA > 10MA > 20MA)"
            elif c_ma5 < c_ma10 < c_ma20 and curr_close < c_ma20:
                ma_alignment = "短中期空頭架構 (5MA < 10MA < 20MA)"
        elif c_ma5 and c_ma10 and c_ma20:
            if curr_close > c_ma5 > c_ma10 > c_ma20:
                ma_alignment = "短中期多頭排列 (Close > 5MA > 10MA > 20MA)"
            elif curr_close < c_ma5 < c_ma10 < c_ma20:
                ma_alignment = "短中期空頭排列 (Close < 5MA < 10MA < 20MA)"

        # 檢測均線壓縮/糾結
        valid_mas = [m for m in [c_ma5, c_ma10, c_ma20, c_ma60] if m is not None]
        if len(valid_mas) >= 3 and c_ma20:
            spread = (max(valid_mas) - min(valid_mas)) / c_ma20
            if spread < 0.015:
                ma_alignment += " [均線高度壓縮/糾結中，蓄勢待變]"

        # 4. 布林通道最新分析結果 (Bollinger Bands Analysis)
        c_upper = bb_upper[-1] if not np.isnan(bb_upper[-1]) else None
        c_lower = bb_lower[-1] if not np.isnan(bb_lower[-1]) else None
        c_pb = percent_b[-1] if not np.isnan(percent_b[-1]) else None
        c_bw = bandwidth[-1] if not np.isnan(bandwidth[-1]) else None

        bb_zone = "無數據"
        bb_pattern = "通道平穩"
        bb_advice = "無特別訊號"

        if c_pb is not None:
            pb_pct = c_pb * 100.0
            if c_pb > 1.0:
                bb_zone = f"極強勢區 ({pb_pct:.1f}%, >+2σ)"
                bb_advice = "趨勢極強，已有多單續抱，不宜盲目追高；防範假突破拉回。"
            elif c_pb > 0.8:
                bb_zone = f"強勢區 ({pb_pct:.1f}%, +1σ ~ +2σ)"
                bb_advice = "多方主導行情，最佳進場點為回踩中軌 (20MA) 或 +1σ 時順勢加碼。"
            elif c_pb >= 0.2:
                bb_zone = f"震盪區 ({pb_pct:.1f}%, -1σ ~ +1σ)"
                bb_advice = "無明確單邊趨勢，隨機性較強，建議區間高拋低吸均值回歸或觀望。"
            elif c_pb >= 0.0:
                bb_zone = f"弱勢區 ({pb_pct:.1f}%, -2σ ~ -1σ)"
                bb_advice = "空方主導行情，最佳賣點為反彈至中軌 (20MA) 或 -1σ 時順勢做空。"
            else:
                bb_zone = f"極弱勢區 ({pb_pct:.1f}%, <-2σ)"
                bb_advice = "嚴重超賣但跌勢強勁，已有空單續抱，切忌貿然摸底抄底。"

        if c_bw is not None and n_rows >= 20:
            bw_recent = [b for b in bandwidth[-20:] if not np.isnan(b)]
            min_bw = min(bw_recent) if bw_recent else c_bw
            if c_bw <= min_bw * 1.08:
                bb_pattern = "布林緊縮蓄勢 (Squeeze，波動率大幅下降蓄勢中)"
            elif c_upper and curr_close >= c_upper * 0.995:
                bb_pattern = "貼近上軌強勢推進 (Band Walk)"
            elif c_lower and curr_close <= c_lower * 1.005:
                bb_pattern = "貼近下軌弱勢推進 (Band Walk)"

        bw_str = f"{c_bw * 100.0:.1f}%" if c_bw is not None else "N/A"
        upper_str = f"{c_upper:.2f}" if c_upper is not None else "N/A"
        middle_str = f"{c_ma20:.2f}" if c_ma20 is not None else "N/A"
        lower_str = f"{c_lower:.2f}" if c_lower is not None else "N/A"

        bollinger_analysis = {
            "bb_tri_tracks": f"上軌: {upper_str} / 中軌: {middle_str} / 下軌: {lower_str}",
            "percent_b": f"%B 指標: {bb_zone}",
            "bandwidth": f"通道帶寬: {bw_str}",
            "pattern_state": f"布林型態: {bb_pattern}",
            "strategy_advice": f"實戰策略建議: {bb_advice}"
        }

        # 5. 20日乖離率 (BIAS)
        bias_20_str = "N/A"
        if c_ma20:
            bias_20_val = ((curr_close - c_ma20) / c_ma20) * 100.0
            bias_eval = "健康區間"
            if bias_20_val > 8.0:
                bias_eval = "正乖離偏大 (防短線獲利回吐)"
            elif bias_20_val < -8.0:
                bias_eval = "負乖離偏大 (注意超賣反彈)"
            bias_20_str = f"{bias_20_val:+.1f}% ({bias_eval})"

        # 6. 量價配合確認 (Volume Confirmation)
        curr_vol = volumes[-1]
        c_vol_5ma = vol_5ma[-1] if not np.isnan(vol_5ma[-1]) else 0
        vol_ratio = (curr_vol / c_vol_5ma) if c_vol_5ma > 0 else 1.0
        vol_confirmation = f"當前成交量為 5日均量之 {vol_ratio:.1f}倍"
        if vol_ratio >= 1.5:
            vol_confirmation += " (帶量動能充沛)"
        elif vol_ratio < 0.8:
            vol_confirmation += " (量縮震盪觀望)"

        # 6.5 RSI(14) 與 MACD(12, 26, 9) 技術指標計算與狀態解讀
        delta = s_close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / np.where(avg_loss != 0, avg_loss, np.nan)
        rsi_series = 100.0 - (100.0 / (1.0 + rs))
        rsi_values = rsi_series.values

        ema12 = s_close.ewm(span=12, adjust=False).mean()
        ema26 = s_close.ewm(span=26, adjust=False).mean()
        macd_dif = (ema12 - ema26).values
        macd_signal = (ema12 - ema26).ewm(span=9, adjust=False).mean().values
        macd_hist = (macd_dif - macd_signal) * 2.0

        # RSI 狀態分析
        c_rsi = rsi_values[-1] if n_rows >= 14 and not np.isnan(rsi_values[-1]) else None
        rsi_zone = "無數據 (數據不足 14 筆)"
        rsi_advice = "數據觀察中"
        if c_rsi is not None:
            if c_rsi >= 70.0:
                rsi_zone = f"超買熱區 ({c_rsi:.1f}, >=70)"
                rsi_advice = "短線過熱風險增加，注意頂位背離或獲利回吐賣壓。"
            elif c_rsi >= 50.0:
                rsi_zone = f"多方控盤偏強區 ({c_rsi:.1f}, 50~70)"
                rsi_advice = "動能維持多頭對抗，站穩中軸 50 以上保持震盪看多。"
            elif c_rsi >= 30.0:
                rsi_zone = f"空方控盤偏弱區 ({c_rsi:.1f}, 30~50)"
                rsi_advice = "動能受制於中軸 50 以下，宜等待回彈過 50 確證。"
            else:
                rsi_zone = f"超賣反彈區 ({c_rsi:.1f}, <30)"
                rsi_advice = "指標進入極度超賣區，短線隨時可能引發技術性跌深反彈。"

        rsi_analysis = {
            "rsi_value": f"{c_rsi:.1f}" if c_rsi is not None else "N/A",
            "rsi_zone": rsi_zone,
            "rsi_advice": rsi_advice
        }

        # MACD 狀態分析
        c_dif = macd_dif[-1] if n_rows >= 26 and not np.isnan(macd_dif[-1]) else None
        c_sig = macd_signal[-1] if n_rows >= 26 and not np.isnan(macd_signal[-1]) else None
        c_hist = macd_hist[-1] if n_rows >= 26 and not np.isnan(macd_hist[-1]) else None
        p_hist = macd_hist[-2] if n_rows >= 27 and not np.isnan(macd_hist[-2]) else c_hist

        macd_zone = "無數據 (數據不足 26 筆)"
        macd_hist_state = "柱狀體持平"
        macd_advice = "動能觀察中"

        if c_dif is not None and c_sig is not None and c_hist is not None:
            zone_str = "零軸上方多頭分區 (DIF > 0)" if c_dif > 0 else "零軸下方空頭分區 (DIF < 0)"
            macd_zone = f"{zone_str} (DIF: {c_dif:.2f}, Signal: {c_sig:.2f})"
            if c_hist > 0:
                if c_hist >= p_hist:
                    macd_hist_state = f"紅柱發散擴大 (Hist: +{c_hist:.2f})"
                    macd_advice = "多頭攻擊動能持續擴張，趨勢強勁。"
                else:
                    macd_hist_state = f"紅柱收斂縮短 (Hist: +{c_hist:.2f})"
                    macd_advice = "多頭漲勢放緩，動能高檔收斂，宜留意多頭回檔風險。"
            else:
                if c_hist <= p_hist:
                    macd_hist_state = f"綠柱發散擴大 (Hist: {c_hist:.2f})"
                    macd_advice = "空頭賣壓沉重發散，跌勢尚未見底。"
                else:
                    macd_hist_state = f"綠柱收斂縮短 (Hist: {c_hist:.2f})"
                    macd_advice = "空頭賣壓漸見收斂，可能醞釀止跌築底或反彈。"

        macd_analysis = {
            "macd_zone": macd_zone,
            "hist_state": macd_hist_state,
            "macd_advice": macd_advice
        }

        # 7. 掃描近 window_days 天發生的轉折訊號 (對齊 home.html computeMAReversalPoints, Bollinger, RSI & MACD)
        start_idx = max(0, n_rows - window_days)
        detected_signals = []

        dates = df.index.strftime('%Y-%m-%d').values

        for i in range(start_idx, n_rows):
            dt_str = dates[i]
            v_ratio = (volumes[i] / vol_5ma[i]) if (vol_5ma[i] > 0) else 1.0
            vol_tag = " (帶量確證)" if v_ratio >= 1.5 else ""

            # E. RSI(14) 轉折與超買/超賣突破訊號
            if i >= 1 and not np.isnan(rsi_values[i]) and not np.isnan(rsi_values[i-1]):
                if rsi_values[i-1] <= 30 and rsi_values[i] > 30:
                    detected_signals.append((i, 5, f"{dt_str} RSI 自超賣區向上回彈突破 30{vol_tag}"))
                elif rsi_values[i-1] >= 70 and rsi_values[i] < 70:
                    detected_signals.append((i, 5, f"{dt_str} RSI 自超買區向下拉回跌破 70"))
                elif rsi_values[i-1] <= 50 and rsi_values[i] > 50:
                    detected_signals.append((i, 4, f"{dt_str} RSI 向上突破 50 中軸分水嶺"))
                elif rsi_values[i-1] >= 50 and rsi_values[i] < 50:
                    detected_signals.append((i, 4, f"{dt_str} RSI 向下跌破 50 中軸分水嶺"))

            # F. MACD(12, 26, 9) 黃金交叉 / 死亡交叉 / 柱狀體翻轉
            if i >= 1 and not np.isnan(macd_dif[i]) and not np.isnan(macd_signal[i]) and not np.isnan(macd_dif[i-1]) and not np.isnan(macd_signal[i-1]):
                if macd_dif[i-1] <= macd_signal[i-1] and macd_dif[i] > macd_signal[i]:
                    cross_tag = " (零軸上方強勢金叉)" if macd_dif[i] > 0 else " (零軸下方反彈金叉)"
                    detected_signals.append((i, 6, f"{dt_str} MACD DIF/Signal 黃金交叉{cross_tag}{vol_tag}"))
                elif macd_dif[i-1] >= macd_signal[i-1] and macd_dif[i] < macd_signal[i]:
                    cross_tag = " (高檔死叉)" if macd_dif[i] > 0 else " (破位死叉)"
                    detected_signals.append((i, 6, f"{dt_str} MACD DIF/Signal 死亡交叉{cross_tag}"))

            if i >= 1 and not np.isnan(macd_hist[i]) and not np.isnan(macd_hist[i-1]):
                if macd_hist[i-1] <= 0 and macd_hist[i] > 0:
                    detected_signals.append((i, 5, f"{dt_str} MACD 柱狀體轉正翻紅"))
                elif macd_hist[i-1] >= 0 and macd_hist[i] < 0:
                    detected_signals.append((i, 5, f"{dt_str} MACD 柱狀體轉負翻綠"))

            # 布林開口突破發散與轉折
            if i >= 1 and not np.isnan(bandwidth[i]) and not np.isnan(bandwidth[i-1]):
                if bandwidth[i] > bandwidth[i-1] * 1.12:
                    if closes[i] > bb_upper[i] and closes[i-1] <= bb_upper[i-1]:
                        detected_signals.append((i, 6, f"{dt_str} 布林通道開口發散突破上軌{vol_tag}"))
                    elif closes[i] < bb_lower[i] and closes[i-1] >= bb_lower[i-1]:
                        detected_signals.append((i, 6, f"{dt_str} 布林通道開口發散跌破下軌"))

            # A. 季線斜率轉折
            if i >= 2 and not np.isnan(ma60[i]) and not np.isnan(ma60[i-1]) and not np.isnan(ma60[i-2]):
                slope_curr = ma60[i] - ma60[i-1]
                slope_prev = ma60[i-1] - ma60[i-2]
                if slope_prev <= 0 and slope_curr > 0:
                    detected_signals.append((i, 3, f"{dt_str} 季線翻揚{vol_tag}"))
                elif slope_prev >= 0 and slope_curr < 0:
                    detected_signals.append((i, 3, f"{dt_str} 季線下彎"))

            # B. 年線斜率轉折
            if i >= 1 and not np.isnan(ma240[i]) and not np.isnan(ma240[i-1]):
                slope240 = ma240[i] - ma240[i-1]
                if closes[i] > ma240[i] and closes[i-1] <= ma240[i-1] and slope240 > 0:
                    detected_signals.append((i, 7, f"{dt_str} 年線翻揚{vol_tag}"))
                elif closes[i] < ma240[i] and closes[i-1] >= ma240[i-1] and slope240 < 0:
                    detected_signals.append((i, 7, f"{dt_str} 年線下彎"))

            # C. 多週期黃金 / 死亡交叉
            # 5MA / 20MA
            if i >= 1 and not np.isnan(ma5[i]) and not np.isnan(ma5[i-1]) and not np.isnan(ma20[i]) and not np.isnan(ma20[i-1]):
                if ma5[i-1] <= ma20[i-1] and ma5[i] > ma20[i]:
                    detected_signals.append((i, 4, f"{dt_str} 5MA/20MA 黃金交叉{vol_tag}"))
                elif ma5[i-1] >= ma20[i-1] and ma5[i] < ma20[i]:
                    detected_signals.append((i, 4, f"{dt_str} 5MA/20MA 死亡交叉"))

            # 5MA / 10MA (短線)
            if i >= 1 and not np.isnan(ma5[i]) and not np.isnan(ma5[i-1]) and not np.isnan(ma10[i]) and not np.isnan(ma10[i-1]):
                if ma5[i-1] <= ma10[i-1] and ma5[i] > ma10[i]:
                    detected_signals.append((i, 2, f"{dt_str} 短線 5MA/10MA 黃金交叉"))
                elif ma5[i-1] >= ma10[i-1] and ma5[i] < ma10[i]:
                    detected_signals.append((i, 2, f"{dt_str} 短線 5MA/10MA 死亡交叉"))

            # 20MA / 60MA (中線)
            if i >= 1 and not np.isnan(ma20[i]) and not np.isnan(ma20[i-1]) and not np.isnan(ma60[i]) and not np.isnan(ma60[i-1]):
                if ma20[i-1] <= ma60[i-1] and ma20[i] > ma60[i]:
                    detected_signals.append((i, 5, f"{dt_str} 中線 20MA/60MA 黃金交叉{vol_tag}"))
                elif ma20[i-1] >= ma60[i-1] and ma20[i] < ma60[i]:
                    detected_signals.append((i, 5, f"{dt_str} 中線 20MA/60MA 死亡交叉"))

            # D. 20MA 扣抵值前瞻預警
            if i >= 21:
                c_curr = closes[i]
                c_prev = closes[i-1]
                c_k20 = closes[i-20]
                c_k21 = closes[i-21]
                if c_curr > c_k20 and c_prev <= c_k21:
                    detected_signals.append((i, 3, f"{dt_str} 20MA 扣抵翻揚預警 (預示 20MA 即將上揚)"))
                elif c_curr < c_k20 and c_prev >= c_k21:
                    detected_signals.append((i, 3, f"{dt_str} 20MA 扣抵下彎預警 (預示 20MA 即將下彎)"))

        # 按 權重與日期 排序
        detected_signals.sort(key=lambda x: (x[0], x[1]), reverse=True)
        recent_signal_texts = [s[2] for s in detected_signals[:5]]

        return {
            "ma_alignment": ma_alignment,
            "recent_signals": recent_signal_texts,
            "bias_20": bias_20_str,
            "volume_confirmation": vol_confirmation,
            "degradation_level": deg_level,
            "bollinger_analysis": bollinger_analysis,
            "rsi_analysis": rsi_analysis,
            "macd_analysis": macd_analysis
        }

