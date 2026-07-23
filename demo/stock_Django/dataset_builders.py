import logging
import numpy as np
import pandas as pd
import os
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
    def get_sentiment_features(cls, stock_number, date_index_df):
        """取得新聞語詞特徵 (具備資料庫快取機制)"""
        embedding_dim = 768
        embedding_cols = [f'finbert_emb_{i}' for i in range(embedding_dim)]
        
        # Initialize result DataFrame
        result_df = pd.DataFrame(0.0, index=date_index_df.index, columns=embedding_cols)
        
        if date_index_df.empty:
            return result_df
            
        sql_op = OP_Fun()
        clean_num = str(stock_number).upper().replace('.TW', '').replace('.TWO', '')
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
            return result_df[embedding_cols]
            
        # 3. 僅針對缺失日期讀取原始文本並進行推論
        logger.info(f"AI Cache Miss for {stock_number}. Parsing raw news for {len(missing_dates)} days...")
        
        # 嘗試讀取包含原始文本的 Excel 檔案
        webbug_dir = os.getenv('WEBBUG_DIR', 'E:/Infinity/webbug/')
        news_file = os.path.join(webbug_dir, f'{clean_num}_news.xlsx')
        
        if not os.path.exists(news_file):
            logger.warning(f"News text file not found: {news_file}. Using 0.0 embeddings fallback.")
            return result_df[embedding_cols]
            
        try:
            # 讀取 Excel 並解析日期與標題
            df_news = pd.read_excel(news_file)
            if df_news.empty:
                return result_df[embedding_cols]
                
            # 假設欄位順序: Index 0=標題, Index 1=發布時間
            df_news['Parsed_Date'] = pd.to_datetime(df_news.iloc[:, 1], errors='coerce').dt.strftime('%Y-%m-%d')
            df_news['Parsed_Text'] = df_news.iloc[:, 0].fillna("").astype(str)
            
            # 過濾僅保留缺失日期的數據
            missing_date_strs = [d.strftime('%Y-%m-%d') for d in missing_dates]
            df_missing = df_news[df_news['Parsed_Date'].isin(missing_date_strs)]
            
            if df_missing.empty:
                return result_df[embedding_cols]
            
            # 進行批次推論
            texts = df_missing['Parsed_Text'].tolist()
            embeddings = cls._get_embeddings_batch(texts)
            
            if len(embeddings) > 0:
                df_missing = df_missing.copy()
                df_missing['embedding'] = list(embeddings)
                
                # 按日聚合 取平均
                daily_groups = df_missing.groupby('Parsed_Date')['embedding'].apply(
                    lambda x: np.mean(np.stack(x.values), axis=0) if len(x) > 0 else np.zeros(embedding_dim)
                )
                
                # 4. 寫回快取並更新結果
                new_cache_data = {}
                for d_str, emb in daily_groups.items():
                    new_cache_data[d_str] = emb
                    d_ts = pd.Timestamp(d_str)
                    if d_ts in result_df.index:
                        result_df.loc[d_ts, embedding_cols] = emb
                
                sql_op.save_sentiment_embeddings(stock_number, new_cache_data)
                
        except Exception as e:
            logger.error(f"Failed to process raw news text for {stock_number}: {e}")
            
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
        clean_num = str(stock_number).replace('.TW', '').replace('.TWO', '')
        
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

        # 7. 掃描近 window_days 天發生的轉折訊號 (對齊 home.html computeMAReversalPoints & Bollinger)
        start_idx = max(0, n_rows - window_days)
        detected_signals = []

        dates = df.index.strftime('%Y-%m-%d').values

        for i in range(start_idx, n_rows):
            dt_str = dates[i]
            v_ratio = (volumes[i] / vol_5ma[i]) if (vol_5ma[i] > 0) else 1.0
            vol_tag = " (帶量確證)" if v_ratio >= 1.5 else ""

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
            "bollinger_analysis": bollinger_analysis
        }

