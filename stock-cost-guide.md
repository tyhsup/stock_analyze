# 領域特化開發規範 (Advanced Domain-Specific Rules) - V2.1.0

## 一、 專案架構與環境規範 (Project Architecture & Environment)

* **Django 專案目錄**：Django 專案主體與核心 App 固定位於 `demo/` 資料夾（`e:\Infinity\mydjango\demo`），所有 `manage.py` 命令與服務啟動皆以 `demo/` 為根目錄。
* **虛擬環境**：必須使用專案根目錄之 Python 虛擬環境 (`.venv\Scripts\python.exe` / `.venv/bin/python`)，執行 `pip install` 前須先驗證與當前 Python 環境相容性。
* **Django 伺服器位址**：開發測試伺服器位址為 `http://127.0.0.1:8000/`。
* **程式碼命名**：程式碼變數與函式命名統一使用英文，符合 PEP 8 規範。變數名稱變更時必須確認全專案引用處皆已同步更新。

---

## 二、 財報與 UI 估值規範 (Financial Analysis & UI Standards)

* **商業計量單位**：財報與估值模型數值請統一使用常用商業單位：M（百萬）、B（十億）、T（兆）。
* **卡片名稱與多市場支援**：Fair Value Calculator（合理價值計算器）之官方指標卡片統一命名為 **`Exchange Official Metrics`**（交易所官方指標），右側搭配動態來源 Badge（例如 `TWSE 證交易所官方`、`TPEx 櫃買中心官方`），以涵蓋台股上市、上櫃及美股市場。
* **圖表與指標配色**：技術指標與估值圖表配色須使用強烈對比色，確保深色/淺色模式下之視覺辨識度。
* **瀏覽器驗證機制**：每次修改 UI 或視圖邏輯後，須開啟瀏覽器（使用 Browser Subagent）進行 Observe-Act-Verify 驗證。測試完成後必須閉合瀏覽器並清除測試參數。瀏覽器代理嚴格限制不具備任何程式碼修改能力。

---

## 三、 雙數據源分工與爬蟲規範 (Data Sources & Scraping)

### 1. 雙數據源分工 (Dual Data Source Architecture)
* **`twse-cli-v2` (台股官方數據源)**：專門負責臺灣證券交易所（TWSE）與櫃買中心（TPEx）官方歷史估值乘數（P/E、P/B、殖利率）及籌碼資料擷取，為台股估值之權威數據源。
* **`yfinance` (全球與備援數據源)**：負責美股歷史股價、財務報表與國際指標擷取，並作為台股數據獲取失敗時之備援機制。

### 2. 爬蟲與流量保護 (Scraping Resilience)
* 爬蟲作業需實作 2 ~ 5 秒隨機延遲並輪替 User-Agent。
* 異步請求統一採用 `aiohttp`；遇到 `429 Too Many Requests` 時必須暫停執行。
* 外部 API 回應須具備本地快取機制，避免觸發 Rate Limit。
* 針對大檔案或串流媒體下載，統一採用 Streaming Writes。

---

## 四、 資料庫管理與持久化規範 (Database & Persistence)

* **預設資料庫**：專案使用 MySQL 資料庫 `stock_tw_analyse`（設定於 `demo/demo/settings.py`），憑證須由 `.env` 讀取，嚴禁硬編碼。
* **ORM 與 Raw SQL 並重原則**：
  * **標準 CRUD**：優先使用 Django ORM (`models.Model`)，提升程式碼可維護性與安全性。
  * **巨量寫入與複雜查詢**：針對巨量時間序列數據或複雜多表 JOIN，允許使用原生 SQL 或 `executemany`（Batch Size > 100）。
* **SQL 安全防護**：原生 SQL 強制使用參數化查詢（Parameterized Queries），嚴禁字串拼接。連線須使用 Context Manager (`with` 語句) 或連線池 (`mysql.connector.pooling`)。
* **資料型態與精度**：金額與股價等敏感數值於 Python 端使用 `decimal.Decimal`，MySQL 端儲存為 `DECIMAL(18, 4)`。股票價格表須對 `(ticker, date)` 建立 Unique Index。
* **破壞性操作防範**：若腳本涉及刪除資料庫或資料表（DROP Table/Database），必須事先詢問使用者取得手動確認，並提示執行 `mysqldump` 備份。

---

## 五、 四 Agent 開會與反模擬機制 (4-Agent Council & Anti-Simulation)

* **強制開會觸發**：涉及功能開發、缺陷修復、架構重構或規格制定時，必須觸發四 Agent（Commander、Planner、Generator、Evaluator）開會機制。
* **MCP 工具自動化**：開會流程必須調用 `multi-agent-council` MCP Server（`council_orchestrator` / `planner_consult` / `generator_code` / `evaluator_review`）。
* **反模擬驗證 (Anti-Simulation Enforcement)**：開會紀錄必須原封不動引用 MCP Server 回傳之 HMAC-SHA256 防偽簽章 `[VERIFIED_<ROLE>|nonce=...|sig=...|ts=...]`，嚴禁主模型自行編造角色簽章。
* **編碼驗證迴圈 (Code Verification Loop)**：Generator 產出代碼後，必須由 Evaluator 進行獨立審查；若審查退回，Generator 必須修正問題點後重新提交，直到 Evaluator 審查通過。

---

## 六、 開發工作流規範 (Development Workflow Rules)

* **Git 協定 (Git Protocol)**：每當使用者表示「功能開發完成」、「Bug 修復好了」或「今天先到這邊」時，主動分析當前變更，並詢問：「是否需要現在幫您同步到 GitHub？」。
* **Commit 品質 (Message Quality)**：自動生成的 Git Commit 訊息必須具體描述修改了哪些 Python 模組、HTML 模板或資料庫表格。
