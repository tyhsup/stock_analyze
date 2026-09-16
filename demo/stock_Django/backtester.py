# -*- coding: utf-8 -*-
"""
Walk-Forward 滾動回測引擎模組 (WalkForwardBacktester)
提供嚴格防前瞻偏誤（訊號遞延一期 shift(1)）、特徵標準化隔離與摩擦成本扣除之回測基礎架構。
"""

import json
import math
import logging
import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import numpy as np
import pandas as pd
from django.db import connection

logger = logging.getLogger(__name__)


class NumpyJsonEncoder(json.JSONEncoder):
    """自定義 JSON 序列化器，防止 Numpy 數值與日期型別拋出 TypeError"""
    def default(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            if math.isnan(obj) or math.isinf(obj):
                return 0.0
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        return super().default(obj)


@dataclass
class BacktestResult:
    """回測評估結果數據容器"""
    symbol: str
    market: str
    model_version: str
    strategy_name: str
    run_date: datetime.datetime
    train_window: int
    test_window: int
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_loss_ratio: float
    total_return: float
    annualized_return: float
    num_trades: int
    oos_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    config_json: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """轉為標準字典格式，並確保浮點數格式化以防 UI 數值溢出"""
        return {
            "symbol": self.symbol,
            "market": self.market,
            "model_version": self.model_version,
            "strategy_name": self.strategy_name,
            "run_date": self.run_date.isoformat() if isinstance(self.run_date, (datetime.date, datetime.datetime)) else str(self.run_date),
            "train_window": self.train_window,
            "test_window": self.test_window,
            "sharpe_ratio": round(float(self.sharpe_ratio), 4),
            "sortino_ratio": round(float(self.sortino_ratio), 4),
            "max_drawdown": round(float(self.max_drawdown), 4),
            "win_rate": round(float(self.win_rate), 4),
            "profit_loss_ratio": round(float(self.profit_loss_ratio), 4),
            "total_return": round(float(self.total_return), 4),
            "annualized_return": round(float(self.annualized_return), 4),
            "num_trades": int(self.num_trades),
            "config_json": self.config_json,
        }


class WalkForwardBacktester:
    """
    Walk-Forward 滾動視窗回測引擎。
    
    核心特性：
    1. 滾動視窗：train_window (預設 252 天) 與 test_window (預設 21 天)。
    2. 防前瞻偏誤：所有訊號強制執行 shift(1) 遞延，T 日生成之訊號僅能在 T+1 日執行。
    3. 摩擦成本：每次持倉換位時扣除交易摩擦成本（預設 10 bps）。
    4. 數值防禦：包含除以零防護 (epsilon=1e-8)、無交易時安全預設值、Numpy 型別序列化保護。
    """
    def __init__(self, train_window: int = 252, test_window: int = 21,
                 transaction_cost_bps: float = 10.0,
                 risk_free_rate: float = 0.015,
                 epsilon: float = 1e-8):
        self.train_window = int(train_window)
        self.test_window = int(test_window)
        self.tc = float(transaction_cost_bps) / 10000.0  # 10 bps = 0.0010
        self.risk_free_rate = float(risk_free_rate)
        self.epsilon = float(epsilon)

    def calculate_metrics(self, returns: pd.Series, raw_signals: pd.Series) -> Dict[str, float]:
        """
        計算策略指標（含訊號強制遞延、摩擦成本扣除與防禦性邊界檢查）。
        
        :param returns: 基準資產單期報酬率序列 (如 Close.pct_change())
        :param raw_signals: 模型產出之原始倉位訊號 (1=多, -1=空, 0=平倉)
        :return: 包含 sharpe, sortino, max_drawdown, win_rate, profit_loss_ratio, total_return, annualized_return, num_trades 之字典
        """
        metrics = {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_loss_ratio": 0.0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "num_trades": 0
        }

        if returns.empty or raw_signals.empty:
            return metrics

        # 1. 訊號強制遞延一期 (防前瞻偏誤核心防線)
        delayed_signals = raw_signals.shift(1).fillna(0.0)

        # 2. 摩擦成本計算：倉位變動處 (diff != 0) 扣除摩擦成本
        position_change = delayed_signals.diff().fillna(delayed_signals.iloc[0] if not delayed_signals.empty else 0.0)
        trades_count = int((position_change != 0).sum())
        trade_penalty = (position_change != 0).astype(float) * self.tc

        # 3. 策略淨報酬率序列
        strategy_returns = (returns * delayed_signals) - trade_penalty
        strategy_returns = strategy_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        n_periods = len(strategy_returns)
        if n_periods == 0:
            return metrics

        metrics["num_trades"] = trades_count

        # 4. 累積報酬率與年化報酬率
        cum_growth = (1.0 + strategy_returns).cumprod()
        total_ret = float(cum_growth.iloc[-1] - 1.0) if not cum_growth.empty else 0.0
        metrics["total_return"] = total_ret

        if n_periods > 0 and total_ret > -1.0:
            ann_ret = float((1.0 + total_ret) ** (252.0 / max(1, n_periods)) - 1.0)
        else:
            ann_ret = -1.0
        metrics["annualized_return"] = ann_ret

        # 若無任何實質倉位變動或所有訊號為 0
        if trades_count == 0 or (delayed_signals == 0).all():
            return {k: round(v, 4) for k, v in metrics.items()}

        # 5. Sharpe Ratio (年化)
        rf_daily = self.risk_free_rate / 252.0
        excess_returns = strategy_returns - rf_daily
        mean_excess = excess_returns.mean()
        std_ret = strategy_returns.std()
        if std_ret > self.epsilon:
            sharpe = float((mean_excess / (std_ret + self.epsilon)) * np.sqrt(252.0))
        else:
            sharpe = 0.0
        metrics["sharpe_ratio"] = sharpe

        # 6. Sortino Ratio (年化，僅考量下行波動)
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 1 else 0.0
        if downside_std > self.epsilon:
            sortino = float((mean_excess / (downside_std + self.epsilon)) * np.sqrt(252.0))
        else:
            sortino = float(sharpe) if sharpe > 0 else 0.0
        metrics["sortino_ratio"] = sortino

        # 7. Max Drawdown
        running_max = cum_growth.cummax()
        drawdown = (cum_growth - running_max) / (running_max + self.epsilon)
        metrics["max_drawdown"] = float(drawdown.min())

        # 8. 勝率與盈虧比 (僅統計有持倉的活躍交易日)
        active_returns = strategy_returns[delayed_signals != 0]
        if len(active_returns) > 0:
            wins = active_returns[active_returns > 0]
            losses = active_returns[active_returns < 0]
            metrics["win_rate"] = float(len(wins) / len(active_returns))
            
            avg_gain = float(wins.mean()) if len(wins) > 0 else 0.0
            avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0
            if avg_loss > self.epsilon:
                metrics["profit_loss_ratio"] = float(avg_gain / (avg_loss + self.epsilon))
            else:
                metrics["profit_loss_ratio"] = float(avg_gain / self.epsilon) if avg_gain > 0 else 0.0

        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}

    def run_walk_forward(self, df: pd.DataFrame,
                           model_predict_fn: Callable[[pd.DataFrame, pd.DataFrame, Any], Tuple[pd.Series, pd.Series]],
                           symbol: str = "2330",
                           market: str = "tw",
                           model_version: str = "v1.0",
                           strategy_name: str = "walk_forward_default",
                           feature_cols: Optional[List[str]] = None,
                           save_to_database: bool = True) -> BacktestResult:
        """
        執行完整 Walk-Forward 滾動回測流程。
        
        :param df: 包含特徵與 'Daily_Return' 或 'Close' 的時間序列 DataFrame (需以日期為索引)
        :param model_predict_fn: 預測函式，簽章為 fn(train_df, test_df, scaler) -> (predictions, signals)
        :param symbol: 標的代號
        :param market: 市場別 ('tw' / 'us')
        :param model_version: 模型版本
        :param strategy_name: 策略名稱
        :param feature_cols: 需標準化之特徵欄位清單
        :param save_to_database: 是否自動將結果儲存至 MySQL backtest_results 表
        :return: BacktestResult 實例
        """
        n = len(df)
        min_required = self.train_window + self.test_window
        if n < min_required:
            raise ValueError(f"數據長度 ({n}) 不足，至少需要 {min_required} 筆資料 (train={self.train_window}, test={self.test_window})")

        from sklearn.preprocessing import StandardScaler

        if feature_cols is None:
            feature_cols = [c for c in ['Daily_Return', 'Bias_5', 'Bias_20'] if c in df.columns]

        return_col = 'Daily_Return'
        if return_col not in df.columns:
            if 'Close' in df.columns:
                df = df.copy()
                df[return_col] = df['Close'].pct_change().fillna(0.0)
            else:
                raise ValueError("DataFrame 必須包含 'Daily_Return' 或 'Close' 欄位以計算報酬")

        all_oos_records = []
        all_signals = []
        all_returns = []

        logger.info(f"開始 {symbol} Walk-Forward 滾動回測: 總樣本數={n}, 訓練窗口={self.train_window}, 測試窗口={self.test_window}")

        # 滾動視窗迴圈
        for start_idx in range(0, n - min_required + 1, self.test_window):
            train_end = start_idx + self.train_window
            test_end = min(train_end + self.test_window, n)

            train_df = df.iloc[start_idx:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()

            if test_df.empty:
                break

            # 嚴格特徵標準化隔離 (Scaler Leakage 防護)
            scaler = StandardScaler()
            if feature_cols:
                train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
                test_df[feature_cols] = scaler.transform(test_df[feature_cols])

            # 調用模型預測
            preds, signals = model_predict_fn(train_df, test_df, scaler)

            test_df = test_df.copy()
            test_df['predicted'] = preds.values if hasattr(preds, 'values') else preds
            test_df['signal'] = signals.values if hasattr(signals, 'values') else signals

            all_oos_records.append(test_df[['predicted', 'signal', return_col]])
            all_signals.append(test_df['signal'])
            all_returns.append(test_df[return_col])

        if not all_oos_records:
            raise RuntimeError("Walk-Forward 未產出任何樣本外預測記錄")

        oos_df = pd.concat(all_oos_records)
        combined_signals = pd.concat(all_signals)
        combined_returns = pd.concat(all_returns)

        # 計算全樣本外累積指標
        metrics = self.calculate_metrics(combined_returns, combined_signals)

        config_data = {
            "train_window": self.train_window,
            "test_window": self.test_window,
            "transaction_cost_bps": self.tc * 10000.0,
            "risk_free_rate": self.risk_free_rate,
            "feature_cols": feature_cols,
            "total_oos_periods": len(oos_df)
        }

        result = BacktestResult(
            symbol=symbol,
            market=market,
            model_version=model_version,
            strategy_name=strategy_name,
            run_date=datetime.datetime.now(),
            train_window=self.train_window,
            test_window=self.test_window,
            sharpe_ratio=metrics["sharpe_ratio"],
            sortino_ratio=metrics["sortino_ratio"],
            max_drawdown=metrics["max_drawdown"],
            win_rate=metrics["win_rate"],
            profit_loss_ratio=metrics["profit_loss_ratio"],
            total_return=metrics["total_return"],
            annualized_return=metrics["annualized_return"],
            num_trades=metrics["num_trades"],
            oos_predictions=oos_df,
            config_json=config_data
        )

        if save_to_database:
            self.save_result_to_db(result)

        return result

    def save_result_to_db(self, result: BacktestResult) -> int:
        """
        將回測指標安全寫入 MySQL backtest_results 資料表。
        使用參數化查詢防止 SQL 注入，並透過 NumpyJsonEncoder 防止 JSON 序列化失敗。
        """
        sql = """
            INSERT INTO backtest_results (
                symbol, market, model_version, strategy_name, run_date,
                train_window, test_window, sharpe_ratio, sortino_ratio,
                max_drawdown, win_rate, profit_loss_ratio, total_return,
                annualized_return, num_trades, config_json
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s
            );
        """
        config_str = json.dumps(result.config_json, cls=NumpyJsonEncoder, ensure_ascii=False)
        params = (
            str(result.symbol),
            str(result.market),
            str(result.model_version),
            str(result.strategy_name),
            result.run_date.strftime("%Y-%m-%d %H:%M:%S") if isinstance(result.run_date, datetime.datetime) else str(result.run_date),
            int(result.train_window),
            int(result.test_window),
            float(result.sharpe_ratio),
            float(result.sortino_ratio),
            float(result.max_drawdown),
            float(result.win_rate),
            float(result.profit_loss_ratio),
            float(result.total_return),
            float(result.annualized_return),
            int(result.num_trades),
            config_str
        )

        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            last_id = cursor.lastrowid
            logger.info(f"成功儲存 {result.symbol} 回測結果至 backtest_results (ID: {last_id})")
            return last_id
