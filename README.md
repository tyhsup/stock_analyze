# 智慧台美股量化分析與 AI 估值決策平台

## 專案概述

本專案為一套整合台股（TWSE / TPEx）與美股（NYSE / NASDAQ）的端到端量化分析與人工智慧估值決策系統。系統以後端 Django 框架為核心，結合本地端 MySQL 資料庫進行高頻快取與持久化治理，並整合量化技術指標、三大法人籌碼追蹤、深度學習價格時序預測以及大語言模型（LLM）新聞定性洞察。

面對金融市場中分散破碎的資訊環境，本平台旨在提供一套邏輯嚴謹、具備回測驗證支持與模型可解釋性的研究工作站，協助研究員與投資人進行客觀、數據導向的投資決策。

---

## 核心系統架構與關鍵功能

### 1. 雙主題互動式量化儀表板
* 支援淺色與深色主題切換，介面針對高對比閱讀體驗進行最佳化。
* 整合 ApexCharts 圖表庫，提供高互動性之技術面與籌碼面圖表：包含 K 線圖、移動平均線、布林通道（Bollinger Bands）、相對強弱指標（RSI）、平滑異同移動平均線（MACD）、希爾伯特變換向量軌跡（HT_PHASOR）以及成交量分佈。

### 2. 三大法人與機構籌碼追蹤
* 視覺化外資、投信與自營商的每日買賣超動態與累積持股水位。
* 提供機構籌碼集中度分析圖表，直觀呈現大額資金的流向與持倉集中趨勢。

### 3. 多維混合內在價值估值模型
* 結合現金流折現模型（Discounted Cash Flow, DCF）與市場法乘數估值（PE、PB、PS 相對估值法），提供綜合內在價值區間。
* 自動揭示折現率（WACC）、永續成長率與安全邊際假設，消除單一估值模型之主觀偏誤。

### 4. 總體經濟跨維度觀測站
* 參照資產管理決策架構，對稱呈現台美兩地之核心總體經濟指標。
* 涵蓋三大板塊：貨幣流動性（M1B / M2 年增率）、通膨監測（CPI 與核心 CPI 年增率）以及政策利率與利差（Fed Funds Rate、10Y-2Y 公債殖利率利差、台灣外匯存底與加權指數對比）。

---

## 近期量化與 AI 系統強化模組

為了使系統由基礎特徵工程晉升為具備量化實戰級別之決策體系，本專案近期完成了四大維度的深層升級：

### 1. 數據基礎治理（時區與交易日曆對齊）
* **跨市場交易日曆治理 (`TradingCalendarService`)**：建立 `dim_trading_calendar` 資料表，收錄台美兩地完整交易日曆，自動處理美東夏令日光節約時間（EDT / EST）轉換，並嚴格落實盤中與盤後截斷點（13:30 / 16:00），將盤後發布之新聞自動滾動至次一交易日（T+1），防止前瞻偏誤。
* **多因子權威度加權聚合 (`WeightedSentimentAggregator`)**：捨棄傳統算術平均，建立來源權威度階層架構（CNBC / Reuters: 0.90，鉅亨網 / MoneyDJ: 0.70，社群論壇: 0.30），結合時間指數衰減 $w = \exp(-0.1 \cdot \Delta t)$，提升情緒特徵之訊噪比。

### 2. 嚴謹驗證體系（避免過擬合與數據洩漏）
* **滾動向前走步回測引擎 (`WalkForwardBacktester`)**：落實標準訓練視窗（252 個交易日）與測試視窗（21 個交易日）循環滾動驗證，所有交易訊號強制執行 $T+1$ 遞延（`shift(1)`），並計入單邊 10 bps（0.0010）之交易摩擦成本。
* **特徵標準化隔離防護 (`PriceLSTMFeatureExtractor`)**：嚴格區分訓練集（`fit_mode=True`）與測試集（`fit_mode=False`），測試階段僅允許使用訓練期擬合之 Scaler 參數進行轉換，若未傳入 Scaler 則主動拋出例外，徹底杜絕均值與變異數跨樣本洩漏。
* **量化指標持久化**：建立 `backtest_results` 表，記錄 Sharpe Ratio、Sortino Ratio、最大回撤（Max Drawdown）、勝率與盈虧比。

### 3. 模型升級與特徵可解釋性
* **雙語統一情緒分析器 (`UnifiedSentimentAnalyzer`)**：中文文本自動路由至 `IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment`，英文文本延遲載入 `ProsusAI/finbert`，並校準中立防護閾值（0.60），解決客觀財經新聞過度判定為中立之問題。
* **SHAP 特徵貢獻度歸因 (`ModelExplainer`)**：導入 SHAP 特徵重要性分析，並實作 `RobustSHAPEncoder` 將數值中的 `NaN` 與 `Inf` 安全序列化為 JSON `null`；同時硬性限制前端展示為 Top-K 重要特徵，其餘特徵匯總為 `others_shap_value`，避免前端視覺遮擋與版面溢出。
* **A/B 基準評測框架 (`NLPMultiModelABTester`)**：保留原自訓練模型作為對照組，實測顯示新版雙語推論延遲降低至 75.79 ms/sample，取得 4.17 倍之推論加速。

### 4. 動態系統與時序連續性
* **動態模型選擇與路由 (`DynamicModelSelector`)**：採納策略模式（Strategy Pattern），在 LSTM 時序模型與 Random Forest（scikit-learn）之間依據近 60 日樣本外（OOS）方向勝率動態切換；具備記憶體溫啟動（Warm Start）、本地快取（TTL 300 秒）與資料庫斷連雙重降級（Double Fallback）保護機制。
* **特徵維度塑形器 (`FeatureShaper`)**：自動在 3D 時序張量與 2D 表格特徵間安全轉換，防止維度不匹配異常。
* **時序衰減向前填充 (`SentimentTimeDecay`)**：針對非交易日與無新聞日，採用指數衰減向前填充：$V_t = V_{t-1} \cdot \exp(-0.1 \cdot \Delta t)$，具備 $\Delta t > 100$ 天下溢直接歸零與負值例外攔截，保留市場情緒之衰減記憶。
* **模型註冊表 (`model_registry`)**：記錄各個股模型版本、啟用狀態與滾動勝率，支援高併發參數化 UPSERT。

---

## 技術棧 (Technology Stack)

* **後端架構**：Python 3.11+ / 3.14、Django 5.x
* **前端介面**：HTML5、JavaScript、Bootstrap 5、Vanilla CSS、ApexCharts.js
* **資料庫管理**：MySQL 8.0+、mysql-connector-python、SQLAlchemy
* **數據分析與回測**：Pandas（向量化資料處理）、NumPy、Scikit-learn、SHAP
* **深度學習與自然語言處理**：PyTorch、HuggingFace Transformers（Erlangshen-Roberta、FinBERT）
* **自動化資料獲取**：aiohttp（異步網路爬蟲）、yfinance、台灣證券交易所與櫃買中心數據接口、FRED API

---

## 核心資料庫結構清單

* `stocks_tw` / `stocks_us`：台美股個股基本資料與上市狀態。
* `stock_cost` / `stock_cost_us`：每日歷史價量數據（開高低收與成交量）。
* `stock_investor` / `stock_investor_us`：三大法人與機構每日籌碼明細。
* `financial_raw_tw` / `financial_raw_us`：每季財務報表原始數據。
* `macro_tw` / `macro_us`：台美總體經濟時間序列資料。
* `dim_trading_calendar`：跨市場交易日曆維度表。
* `backtest_results`：Walk-Forward 滾動回測量化績效指標表。
* `model_registry`：動態模型註冊與滾動勝率追蹤表。

---

## 安裝與快速開始

### 環境需求
* Python 3.11 或更新版本
* MySQL Server 8.0 或更新版本（需先建立資料庫 `stock_tw_analyse`）
* C++ 依賴環境（部分機器學習與 TA-Lib 套件編譯需求）

### 部署步驟

1. **取得專案原始碼**
   ```bash
   git clone <repository-url>
   cd mydjango
   ```

2. **安裝必要依賴套件**
   ```bash
   pip install -r requirements.txt
   ```

3. **環境變數設定**
   請於 `demo/stock_Django/` 目錄下建立 `.env` 檔案，填入 MySQL 帳號密碼與相關金鑰：
   ```env
   DB_HOST=localhost
   DB_USER=your_db_user
   DB_PASSWORD=your_db_password
   DB_NAME=stock_tw_analyse
   GEMINI_API_KEY=your_gemini_api_key
   ```

4. **資料庫結構遷移與初始化**
   ```bash
   cd demo
   python manage.py migrate
   ```

5. **執行全套回歸驗證測試**
   ```bash
   python -m unittest discover -s stock_Django/tests -p "test_*.py" -v
   ```
   *(目前涵蓋 Phase 0 至 Phase 3 共 41 項單元測試，全數通過後方可上線運作)*

6. **啟動伺服器**
   ```bash
   python manage.py runserver
   ```
   啟動完成後，於瀏覽器造訪 `http://127.0.0.1:8000/` 即可進入系統工作台。

---

## 軟體品質與測試驗證

本專案採行嚴格的測試驅動與回歸保護流程：
* **Phase 0 測試**：`test_phase0_data_foundation.py`（9 項，交易日曆與時區加權）
* **Phase 1 測試**：`test_phase1_validation_system.py`（9 項，Walk-Forward 回測與 Scaler 隔離）
* **Phase 2 測試**：`test_phase2_model_upgrade.py`（10 項，Unicode 防禦、0.60 閾值校準與 RobustSHAP 序列化）
* **Phase 3 測試**：`test_phase3_dynamic_system.py`（13 項，時間衰減向前填充、特徵塑形器、雙重降級與除零防護）

**全套 41 項單元與整合測試保持 100% 通過（OK，耗時約 5.4 秒）**，確保系統演進過程具備零破壞性與向下相容性。
