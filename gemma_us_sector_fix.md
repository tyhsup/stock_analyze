# 技術報告：關於 v3.0 結構化關聯建模 (GNN) 美股產業資訊缺失之解決方案

**致：** 專案開發團隊
**日期：** 2023年10月27日
**主題：** 解決美股節點關聯性缺失（Sector 欄位缺失）之技術路徑

---

## 1. 前言 (Introduction)
在目前的 v3.0 GNN 建模架構中，節點間的邊（Edges）主要依賴於「產業別 (Sector)」的同質性來構建鄰接矩陣。由於 `stocks_us` 資料庫目前僅具備基本識別資訊，導致美股節點無法透過產業標籤進行結構化關聯，造成美股與台股在模型架構上的不對稱性。本報告旨在提供資料補全、替代建模方法及資料庫架構優化之建議。

---

## 2. 資料補全建議 (Data Enrichment Strategies)

為了恢復產業標籤，建議導入自動化 ETL (Extract, Transform, Load) 流程，從以下低成本來源提取 `Sector` 與 `Industry` 資訊。

### 方案 A：使用 `yfinance` 庫 (首選推薦)
`yfinance` 是目前最穩定且免費的 Python 接口，能直接從 Yahoo Finance 抓取公司基本面資訊。
*   **實作邏輯**：
    1.  遍歷 `stocks_us` 中的 `symbol`。
    2.  調用 `ticker.info` 屬性。
    3.  提取 `sector` 與 `industry` 欄位。
*   **優點**：完全免費、開發成本極低、包含完整的產業層級（Sector $\rightarrow$ Industry $\rightarrow$ Group）。
*   **代碼範例**：
    ```python
    import yfinance as yf
    def update_stock_sector(symbol):
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info.get('sector', None), info.get('industry', None)
    ```

### 方案 B：使用 SEC EDGAR (高可靠度方案)
若對資料準確度有極高要求（例如需符合法規定義），可透過 SEC 的官方 API 抓取公司註冊文件。
*   **實作邏輯**：透過 `CIK` 編號查詢公司的 `Form 10-K` 文件，利用 NLP 技術（或簡單的 Regex）從 "Business" 章節中提取產業描述。
*   **優算點**：官方來源，具備法律效力。
*   **缺點**：解析成本高，需要處理非結構化文本。

---

## 3. 替代方案：非產業別之數據驅動關聯 (Data-Driven Alternatives)

若因時效性無法立即補全資料，建議採用「特徵空間相似度」來構建鄰接矩陣，將「先驗知識 (Prior Knowledge)」轉化為「數據驅動 (Data-Driven)」的關聯。

### 方案一：收益率相關性矩陣 (Return Correlation Matrix)
*   **邏輯**：利用歷史價格序列的 **Pearson Correlation** 或 **Spearman Rank Correlation** 作為邊的權重。
*   **實作**：
    $$A_{ij} = \text{corr}(R_i, R_j)$$
    其中 $R$ 為過去 $N$ 日的對數收益率。
*   **優點**：捕捉動態的市場連動性，能捕捉到產業變動之外的隱性關聯（例如供應鏈上下游）。

### 方案二：ETF 成分股重疊度 (ETF Component Overlap)
*   **邏輯**：如果兩檔美股同時出現在同一個 ETF（如 QQQ, SPY）的成分股清單中，則視為具有結構性關聯。
*   **實作**：
    1.  建立一個「ETF-Stock」的二部圖 (Bipartite Graph)。
    2.  計算兩股票共同擁有的 ETF 數量或權重比例。
*   **優點**：這是一種「隱性產業標籤」，能有效模擬機構投資人的持倉邏輯。

### 方案三：基本面特徵相似度 (Fundamental Feature Similarity)
*   **邏輯**：利用 P/E, P/S, Debt/Equity 等數值特徵，計算節點間的 **Euclidean Distance** 或 **Cosine Similarity**。
*   **優點**：即使沒有產業名稱，也能將「財務結構相似」的公司連結起來。

---

## 4. 資料庫 Schema 建議 (Database Schema Evolution)

為了支援後續的擴展性（例如未來可能加入 ESG 分數、市值規模等），不建議直接在 `stocks_us` 增加大量欄位，建議採用 **Metadata 擴展模式**。

### 建議架構：增加 `stock_metadata` 表

#### 1. 原有表：`stocks_us` (保持輕量化)
僅保留唯一識別碼與基本靜態資訊。
| Column | Type | Description |
| :------ | :--- | :--- |
| `symbol` (PK) | VARCHAR | 美股代碼 |
| `name` | VARCHAR | 公司名稱 |
| `market` | VARCHAR | 市場代碼 (e.g., NASDAQ) |

#### 2. 新增表：`stock_metadata` (存放維度資訊)
將產業、規模、屬性等變動或高維度資訊分離。
| Column | Type | Description |
| :--- | :--- | :--- |
| `symbol` (FK) | VARCHAR | 關聯 `stocks_us.symbol` |
| `sector` | VARCHAR | 產業大類 (e.g., Technology) |
| `industry` | VARCHAR | 產業細分 (e.g., Software - Infrastructure) |
| `market_cap_range`| VARCHAR | 市值區間 (e.g., Large-cap) |
| `last_updated` | TIMESTAMP | 資料更新時間 (用於 ETL 追蹤) |

#### 3. 新增表：`stock_relations` (選配，用於儲存計算後的邊)
若採用方案二或三，可將計算後的關聯權重存入此表，避免每次 GNN 訓練都要重新計算。
| Column | Type | Description |
| :--- | :--- | :--- |
| `source_symbol` | VARCHAR | 起點節點 |
| `target_symbol` | VARCHAR | 終點節點 |
| `relation_type` | VARCHAR | 關聯類型 (e.g., 'correlation', 'etf_overlap') |
| `weight` | FLOAT | 權重值 |

---

## 5. 結論與行動建議 (Conclusion & Action Plan)

1.  **短期 (Immediate)**：開發一個基於 `yfinance` 的 Python Script，針對 `stocks_us` 進行一次性補全，並建立 `stock_metadata` 表。
2.  **中期 (Mid-term)**：在 GNN 訓練流程中，同時引入 **ETF Overlap** 作為一種邊的特徵，以強化模型對美股結構化資訊的理解。
3.  **長期 (Long-term)**：建立自動化 ETL Pipeline，定期（如每週）更新 `stock_metadata` 中的產業與財務特徵。

**報告人：** [您的姓名/職位]
**狀態：** 待審閱 (Pending Review)