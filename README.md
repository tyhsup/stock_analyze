# Smart Quantitative Stock Analysis & AI Valuation Decision Platform

## Overview

This project is an institutional-grade quantitative research and artificial intelligence valuation platform designed for both the Taiwan (TWSE / TPEx) and US (NYSE / NASDAQ) equity markets. Built on the Django web framework with a localized MySQL database architecture, the system provides an end-to-end investment research pipeline that integrates financial engineering indicators, institutional order flow tracking, deep learning time-series price forecasting, and Large Language Model (LLM) qualitative intelligence.

In financial markets where information is often fragmented, subjective, or obscured by black-box algorithms, this platform delivers a transparent, backtested, and explainable decision station to support data-driven investment strategies.

---

## Key System Modules & Features

### 1. Dual-Theme Interactive Quantitative Dashboard
* Seamless toggle between Light and Dark themes with contrast optimization for professional readability.
* High-performance financial visualization powered by ApexCharts.js: interactive candlestick charts, multi-period moving averages (SMA), Bollinger Bands with volatility squeeze tags, Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), Hilbert Transform Phasor components (HT_PHASOR), and volume profile overlays.

### 2. Institutional Chip Tracking & Order Flow Analytics
* Daily tracking of net buying/selling positions across Foreign Investors, Investment Trusts, and Proprietary Dealers.
* Visual analysis of institutional holding ratios and ownership concentration to identify institutional capital allocation trends.

### 3. Blended Fair Value Valuation Engine
* Combines fundamental Discounted Cash Flow (DCF) models with relative valuation multipliers (PE, PB, PS) to derive balanced intrinsic value ranges.
* Dynamic computation of Weighted Average Cost of Capital (WACC), perpetual growth rates, and margins of safety to eliminate single-model valuation bias.

### 4. Macroeconomic Observatory
* Symmetrical macroeconomic layout covering both Taiwan and the United States, structured according to institutional asset management decision frameworks.
* Three core monitoring pillars:
  * **Liquidity & Money Supply**: M1B / M2 YoY growth trends.
  * **Inflation Dynamics**: CPI and Core CPI YoY trajectory.
  * **Policy & Yield Spreads**: Federal Funds Rate, 10Y-2Y Treasury yield spread, and Taiwan foreign exchange reserves versus the benchmark index.

---

## Advanced Quantitative & AI Architecture (Phases 0–3)

To elevate the system from basic feature processing to production-grade quantitative capabilities, the platform incorporates four layers of structural enhancements:

### 1. Data Foundation & Governance (`Phase 0`)
* **Cross-Market Trading Calendar Governance (`TradingCalendarService`)**: Implemented the `dim_trading_calendar` table spanning Taiwan and US exchanges. Automatically handles Eastern Daylight Time (EDT) and Eastern Standard Time (EST) transitions, enforces strict market close cut-offs (13:30 for TW, 16:00 for US), and rolls post-market announcements to the subsequent trading day ($T+1$) to eliminate lookahead bias.
* **Hierarchical Authority-Weighted Sentiment Aggregator (`WeightedSentimentAggregator`)**: Replaced arithmetic averaging with source-tier authority weighting (CNBC / Reuters: 0.90, CNYES / MoneyDJ: 0.70, Community Forums: 0.30) combined with exponential time decay $w = \exp(-0.1 \cdot \Delta t)$, maximizing signal-to-noise ratio in news sentiment embeddings.

### 2. Validation & Backtesting Framework (`Phase 1`)
* **Walk-Forward Backtesting Engine (`WalkForwardBacktester`)**: Implements an out-of-sample rolling validation framework using a 252-day training window (1 trading year) and a 21-day test window (1 trading month). All generated trading signals enforce a mandatory $T+1$ execution delay (`shift(1)`) and incorporate a 10 bps (0.0010) unilateral transaction friction cost.
* **Feature Normalization Parameter Isolation (`PriceLSTMFeatureExtractor`)**: Enforces strict separation between training (`fit_mode=True`) and testing (`fit_mode=False`) data. The test pipeline is restricted to transforming data using the scaler fitted on training data; missing scalers trigger exceptions to prevent mean and variance leakage across periods.
* **Metric Persistence**: Logs Sharpe Ratio, Sortino Ratio (downside risk with zero-division defense), Maximum Drawdown, Win Rate, and Profit/Loss Ratio into the `backtest_results` table.

### 3. Model Upgrades & Explainability (`Phase 2`)
* **Unified Bilingual Sentiment Analyzer (`UnifiedSentimentAnalyzer`)**: Routes Traditional Chinese text to `IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment` and dynamically loads `ProsusAI/finbert` for English text. Employs a calibrated neutral threshold of `0.60` to resolve over-filtering issues on concise financial dispatches.
* **SHAP Feature Attribution & Layout Protection (`ModelExplainer`)**: Generates Shapley additive feature attributions for tabular models. Features a custom `RobustSHAPEncoder` that sanitizes `NaN` and `Inf` values into valid JSON `null` representations. Restricts front-end display to Top-K prominent features and aggregates remaining signals into `others_shap_value` to prevent UI occlusion.
* **NLP A/B Benchmark Evaluation (`NLPMultiModelABTester`)**: Retains legacy model artifacts as a benchmark control group. Empirical testing demonstrates a 4.17x inference acceleration (75.79 ms/sample vs. 315.93 ms/sample) with explicit memory and GPU cache reclamation.

### 4. Dynamic System & Temporal Continuity (`Phase 3`)
* **Dynamic Model Selector & Router (`DynamicModelSelector`)**: Leverages the Strategy Pattern to dynamically switch between a time-series LSTM and a robust `RandomForestClassifier` based on rolling 60-day Out-of-Sample (OOS) directional hit rates. Features in-memory warm starting, local TTL caching (300 seconds), and a double-fallback mechanism to prevent database connection exhaustion and UI latency.
* **Feature Shaper (`FeatureShaper`)**: Seamlessly adapts inputs between 3D temporal tensors `(samples, time_steps, features)` for LSTM and flattened 2D arrays `(samples, features)` for Random Forest.
* **Exponential Time Decay Forward-Fill (`SentimentTimeDecay`)**: For non-trading days and news gaps, applies continuous exponential decay: $V_t = V_{t-1} \cdot \exp(-0.1 \cdot \Delta t)$. Includes extreme gap underflow zeroing ($\Delta t > 100$), negative time-delta exception traps, and column-wise independent tracking.
* **Model Registry Table (`model_registry`)**: Tracks model versions, active statuses, and rolling accuracy metrics with parameterized MySQL `ON DUPLICATE KEY UPDATE` operations under `utf8mb4_unicode_ci`.

---

## Technology Stack

* **Backend Framework**: Python 3.11+ / 3.14, Django 5.x
* **Frontend Technologies**: HTML5, JavaScript, Bootstrap 5, Vanilla CSS, ApexCharts.js
* **Database Management**: MySQL 8.0+, SQLAlchemy, mysql-connector-python with connection pooling
* **Quantitative Analysis & Backtesting**: Pandas (vectorized time-series), NumPy, Scikit-learn, SHAP
* **Deep Learning & NLP**: PyTorch, HuggingFace Transformers (Erlangshen-Roberta, FinBERT)
* **Data Ingestion & APIs**: aiohttp (asynchronous HTTP crawlers), yfinance, TWSE / TPEx Open Data CLIs, FRED API

---

## Core Database Schema

* `stocks_tw` / `stocks_us`: Metadata and listing status of Taiwan and US equities.
* `stock_cost` / `stock_cost_us`: Daily historical OHLCV price and volume data.
* `stock_investor` / `stock_investor_us`: Institutional trading breakdown and ownership ratios.
* `financial_raw_tw` / `financial_raw_us`: Quarterly and annual financial statement filings.
* `macro_tw` / `macro_us`: Time-series macroeconomic indicators (FRED and Central Bank of Taiwan).
* `dim_trading_calendar`: Multilateral trading calendar dimension table with market timezones.
* `backtest_results`: Quantitative performance metrics and configuration records from walk-forward backtests.
* `model_registry`: Version control, active routing flags, and rolling OOS accuracy records for prediction models.

---

## Installation & Setup Guide

### Prerequisites
* Python 3.11 or higher
* MySQL Server 8.0 or higher (create a target database named `stock_tw_analyse`)
* C++ Build Tools (required for specific machine learning and technical analysis dependencies)

### Setup Steps

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd mydjango
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   Create a `.env` file in the `demo/stock_Django/` directory with your database credentials and API keys:
   ```env
   DB_HOST=localhost
   DB_USER=your_db_user
   DB_PASSWORD=your_db_password
   DB_NAME=stock_tw_analyse
   GEMINI_API_KEY=your_gemini_api_key
   ```

4. **Apply Database Migrations**
   ```bash
   cd demo
   python manage.py migrate
   ```

5. **Execute Regression Test Suite**
   ```bash
   python -m unittest discover -s stock_Django/tests -p "test_*.py" -v
   ```
   *(All 41 unit and integration tests across Phases 0 through 3 must pass before running in production).*

6. **Start the Development Server**
   ```bash
   python manage.py runserver
   ```
   Access the dashboard at `http://127.0.0.1:8000/` in your web browser.

---

## Quality Assurance & Verification

The platform adheres to test-driven development and zero-destruction principles across all major updates:
* **Phase 0 (Data Foundation)**: `test_phase0_data_foundation.py` (9 tests: calendar boundaries, timezone conversion, weighted aggregation).
* **Phase 1 (Validation Architecture)**: `test_phase1_validation_system.py` (9 tests: scaler leakage isolation, signal shift delay, friction cost deduction, database persistence).
* **Phase 2 (Model Upgrades & Explainability)**: `test_phase2_model_upgrade.py` (10 tests: Unicode NFKC normalization, threshold calibration, RobustSHAP serialization, Top-K UI bounds).
* **Phase 3 (Dynamic System)**: `test_phase3_dynamic_system.py` (13 tests: time-decay forward fill, dimension shaper, div-zero safety, double fallback, model registry sync).

**Full regression suite: 41 out of 41 tests pass (100% pass rate in ~5.4 seconds)**, ensuring end-to-end stability, zero regression, and robust backward compatibility.
