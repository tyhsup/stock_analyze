---
trigger: always_on
---

# Advanced Domain-Specific Rules
-財報分析相關的數值請使用常用商業單位M/B/T
-程式碼變數命名上請使用英文
-技術指標的配色使用強烈的對比色
-假如有變動變數名稱, 需要確認所有有引用的function內的變數名稱有更新成新的變數名稱
-每次修改完後須開啟瀏覽器驗證修改內容是否有成功執行
-每次測試完要關閉瀏覽
-每次測試前要確認前一次輸入的參數有被清除
-Django 伺服器位置為 http://127.0.0.1:8000/
-嚴格限制瀏覽器子代理不具任何程式修改能力
-當你執行『系統更新驗證』任務時，請遵循 Observe-Act-Verify 模式。在驗證階段，請優先使用隨機資料庫數據而非預設值。若發現圖表未渲染或情緒分析文字異常，你必須調用系統日誌工具讀取 Python Traceback，並將錯誤代碼與截圖路徑一併回報。
-每次執行完測試後, 將瀏覽器子代理清除, 下次根據測試條件重構瀏覽器子代理

## Scraping & Automation
- Implement 2-5s random delays and rotation of User-Agents.
- Use `aiohttp` for concurrent requests; handle 429 errors by pausing execution.
- Use streaming writes for large file downloads (M3U8/Media).

## Database Management
- Strictly enforce Parameterized Queries to prevent SQL Injection.
- Use `executemany` for bulk inserts (batch size > 100).
- Credentials must be sourced from `.env` only.
- Ensure all DB connections are wrapped in context managers (`with` statement).

## Financial Analysis
- Prioritize `pandas` vectorization over Python loops for time-series data.
- Perform sanity checks on financial ratios before model execution.
- Use `decimal.Decimal` for currency-sensitive calculations.
- Cache external API responses locally to respect rate limits.
- 收集來的資料儲存在本地端資料庫(e.g.:mySQL), 避免yfinance或爬蟲網頁的流量限制

## Agent Behavior
- Before running any `pip install`, verify compatibility with the current environment.
- If a script involves database deletion or table dropping, ASK for manual confirmation.
# Yahoo Finance & MySQL Specific Rules

## yfinance Best Practices
- Initialize `yf.Ticker` with a persistent `requests.Session`.
- Always verify `df.empty` after calling `.history()` before performing calculations.
- Prefer `yf.download()` for multiple tickers over sequential `.history()` calls.
- Log specific error messages for symbols that return no data (e.g., delisted stocks).

## MySQL Development Standards
- Use `mysql.connector.pooling` for efficient connection management.
- All write operations for time-series data MUST use `executemany` with batching.
- Implement `ON DUPLICATE KEY UPDATE` to handle stock data overlaps.
- NEVER hardcode DB credentials; read from `.env` using `os.getenv`.
- Ensure a `Unique Index` exists on (ticker, date) for stock price tables.

## Data Analysis Consistency
- Use `pandas` `.copy()` when modifying slices of financial DataFrames to avoid `SettingWithCopyWarning`.
- Standardize column names to lowercase (e.g., 'adj_close', 'volume') immediately after ingestion.
- Store monetary values as `DECIMAL(18, 4)` in MySQL for precision.
# Local Python, yfinance & MySQL Rules

## MySQL Local Security & Performance
- **Credentials**: Only use low-privilege users from `.env`. NEVER hardcode 'root'.
- **Connection**: Prefer `localhost` over `127.0.0.1` for local socket speed.
- **Pooling**: Use `mysql.connector.pooling` for all DB interactions.
- **Safety**: Prompt for `mysqldump` before any `DROP` or `ALTER` command.
- **DataType**: Map Python `float` to MySQL `DECIMAL(18, 4)` for financial accuracy.

## yfinance Data Handling
- **Resilience**: Wrap `yf.Ticker` calls in try-except to handle network timeouts.
- **Validation**: Check `if df.empty` before initiating any SQL `INSERT`.
- **Formatting**: Ensure DataFrame index is converted to string 'YYYY-MM-DD' for MySQL DATE columns.
- **Batching**: Use `cursor.executemany` for any data chunk larger than 50 rows.

## Python Coding Style
- **Type Hints**: Mandatory for all database and fetching functions.
- **Docstrings**: Explain the SQL query logic and expected return structure in Traditional Chinese.
- **Logging**: Log SQL execution time for any query taking > 0.5s.

# Development Workflow Rules

- **Git Protocol**: 每當我表示「功能開發完成」、「Bug 修復好了」或「今天先到這邊」時，請主動分析當前變更，並詢問我：「是否需要現在幫您同步到 GitHub？」。
- **Message Quality**: 自動生成的 Git Commit 訊息必須具體描述修改了哪些 Python 模組或資料庫表格。