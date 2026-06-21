# 台美股智慧分析與 AI 估值平台 (Smart Stock Analysis & AI Valuation Platform)

## 專案概述

本專案為個人專題研究作品，旨在建立一個整合台股與美股的智慧型投資分析平台。透過後端 Django 框架與 MySQL 資料庫，系統能自動化同步市場數據，並結合 TA-Lib 技術指標、三大法人籌碼流向、LSTM 股價預測模型以及大語言模型（LLM）的新聞情緒分析，為使用者提供一站式的個股研究、合理價估值與智慧投資決策輔助。

---

## 系統核心功能

1. **雙主題互動式儀表板 (Dual-Theme Dashboard)**
   * 提供明亮（Light）與暗色（Dark）雙主題介面，且全網頁圖表與文字對比度皆經過最佳化。
   * 首頁整合多達 12 個 ApexCharts 互動式圖表，包含 K 線圖、成交量、技術指標（RSI、MACD、HT_PHASOR 軌跡圖）、三大法人持股統計、法人集中度圖表以及 Gemini 智慧指針圖。

2. **三大法人籌碼追蹤 (Institutional Chip Tracking)**
   * 視覺化外資、投信、自營商的買進、賣出及買賣超股數。
   * 繪製法人持股比率與集中度圓餅圖，幫助使用者掌握市場主力資金流向。

3. **合理價估值計算機 (Fair Value Calculator)**
   * 混合折現現金流模型（DCF）與市場乘數法（Relative Valuation）。
   * 自動計算個股合理價（Fair Value）與潛在上漲空間（Upside），並動態呈現估值所採用的假設條件。

4. **AI 財經新聞智能洞察 (AI Sentiment & Insights)**
   * 串接鉅亨網（CNYES）新聞 API，由大語言模型（Gemini / Llama 3）對新聞進行結構化摘要與情緒分析。
   * 自動生成「AI 智能洞察」卡片，提供短中長期的市場影響分析，並繪製新聞情緒分佈圖。

5. **自動化數據更新排程器 (Automated Data Scheduler)**
   * 獨立的任務排程管理系統，支援定時或手動執行資料同步任務。
   * 串接 TWSE、TPEX 與美股數據同步工具，將取得的股價、籌碼與財報資料持久化儲存於本地 MySQL 資料庫，避免外部 API 的流量限制。

---

## 系統技術架構

本專案採用現代 Web 技術與數據科學工具鏈進行開發：

* **前端技術 (Frontend)**
  * **基礎結構**：HTML5、JavaScript、Bootstrap 5、Vanilla CSS。
  * **資料視覺化**：ApexCharts.js (繪製高品質互動式金融圖表)。
  * **圖示庫**：FontAwesome、Bootstrap Icons。

* **後端技術 (Backend)**
  * **核心框架**：Django 5.x (Python 3.11+)。
  * **排程控制**：獨立任務排程管理器（結合前端控制介面與背景執行緒）。

* **資料庫管理 (Database)**
  * **主資料庫**：MySQL (用於儲存台美股歷史股價、法人籌碼與公司財報)。
  * **資料庫連接**：Django ORM 與 mysql-connector-pooling 連接池最佳化。

* **數據抓取與爬蟲 (Data Scraping)**
  * **外部 API**：yfinance (美股數據)、鉅亨網新聞 API。
  * **爬蟲與同步工具**：aiohttp (非同步請求與防阻擋機制)、TWSE/TPEX 命令行同步工具 (CLI)。

* **資料分析與 AI 模型 (Data Analysis & AI)**
  * **技術指標**：TA-Lib (計算 RSI、MACD、HT_PHASOR 等金融指標)。
  * **數據處理**：Pandas (向量化運算)、NumPy。
  * **機器學習**：TensorFlow/Keras (LSTM 時間序列股價預測)、Scikit-learn。
  * **自然語言處理**：透過 API 呼叫大語言模型進行新聞文本情緒標記與智能洞察生成。

---

## 資料庫設計簡述

系統主要資料表包含：
* `stocks_tw` / `stocks_us`：台美股股票清單與基本資訊。
* `stock_cost` / `stock_cost_us`：每日歷史交易數據（開盤價、最高價、最低價、收盤價、成交量）。
* `stock_investor` / `stock_investor_us`：三大法人每日交易籌碼細節與持股集中度。
* `financial_raw_tw` / `financial_raw_us`：台美股公司歷年季度財務報表原始數據。
* `valuation_valuationresult`：個股合理價估值結果與模型假設參數。

---

## 安裝與運行指南

### 環境需求
* Python 3.11+
* MySQL Server (建立名為 `stock_tw_analyse` 的資料庫)
* 安裝 TA-Lib C++ 函式庫 (Windows 環境下建議下載預編譯之 `.whl` 檔案進行安裝)

### 執行步驟

1. **複製專案倉庫**
   ```bash
   git clone <repository-url>
   cd mydjango
   ```

2. **安裝 Python 依賴套件**
   ```bash
   pip install -r requirements.txt
   ```

3. **設定環境變數**
   在 `demo/stock_Django/` 目錄下建立 `.env` 檔案，填入您的 MySQL 連線資訊與 API 金鑰：
   ```env
   DB_HOST=localhost
   DB_USER=your_db_user
   DB_PASSWORD=your_db_password
   DB_NAME=stock_tw_analyse
   GROQ_API_KEY=your_groq_api_key
   ```

4. **執行資料庫遷移**
   ```bash
   python manage.py migrate
   ```

5. **啟動 Django 開發伺服器**
   ```bash
   python manage.py runserver
   ```
   啟動後，即可透過瀏覽器造訪 `http://127.0.0.1:8000/` 使用本平台。
