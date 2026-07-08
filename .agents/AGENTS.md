# Agent Behavioral Rules

## 一、 四 Agent 角色職掌與模型配置

每個 Agent 均有明確的模型配置與職責邊界，任何跨越職責邊界的單一 Agent 代勞行為均屬違規。

* **Commander (總指揮官)**
  * **模型配置**：雲端模型（`Gemini 3.5 Flash` 或當前對話的主模型）。
  * **職掌**：解析使用者需求，執行 Git 變更檢測，定義系統安全邊界，識別此次變更對既有功能的潛在影響範圍（如 ApexCharts K 線圖、Blended Valuation 計算等）。主持四 Agent 開會並彙整最終決議。
* **Planner (架構規劃師)**
  * **模型配置**：`Gemini 3.1 Pro`（備援：本機 `gemma4:26b`）。
  * **職掌**：設計技術實作路徑與架構。明確規劃需修改或新增的模組、MySQL 查詢優化方案、Django REST API 規格，並輸出任務拆解清單（Task Breakdown）。
* **Generator (代碼生成官)**
  * **模型配置**：本機推理模型 `Ollama gemma4:e4b`（優先使用，以節省雲端配額）。
  * **職掌**：根據 Planner 的技術規格生成代碼。必須確保 Unicode 防禦、Parameterized Queries、JSON 序列化保護，並遵循零破壞原則。
* **Evaluator (品質審查官)**
  * **模型配置**：`Claude Sonnet 4.6`（備援：本機 `gemma4:26b`）。
  * **職掌**：對 Generator 的產出進行獨立 Code Review。審查 JSON 序列化失敗風險、UI 遮擋問題，設計自動化測試與 E2E 驗證方案，並出具通過或退回的審查意見。

---

## 二、 強制開會機制（Mandatory Council Protocol）

**核心原則**：四 Agent 協作的所有決策（計畫提出、代碼修改策略、輸出格式、Checkpoint 定義）必須經由四 Agent 開會討論並形成共識後才能執行。任何單一 Agent 自行決定並直接執行的行為均屬違規。

### 2-1. 觸發條件

以下任何一種使用者請求，均必須觸發四 Agent 開會：

| 請求類型 | 觸發關鍵詞（範例）|
|---|---|
| 功能開發 | 新增、開發、實作、建立 |
| 缺陷修復 | 修正、修復、改善、解決錯誤 |
| 架構重構 | 重構、優化、調整架構 |
| 規格制定 | 格式定義、Checkpoint 設計、工作流規劃 |

### 2-2. 開會流程（Council Meeting Flow）

每次開會必須依照以下順序執行，且**不得跳過任何環節**：

```
階段 1：Commander 發起（需求解析與邊界定義）
    └─> 輸出：任務摘要、影響範圍清單、風險評估

階段 2：Planner 提案（技術路徑規劃）
    └─> 輸出：技術方案 A / B 比較、任務拆解清單、估計時程

階段 3：Generator 評估（可行性與防禦性審查）
    └─> 輸出：代碼層面的可行性意見、潛在邊緣案例、防禦機制設計

階段 4：Evaluator 審查（風險與品質把關）
    └─> 輸出：審查意見（通過 / 有條件通過 / 退回）、必要的測試案例定義

階段 5：Commander 彙整決議
    └─> 輸出：最終執行方案、Checkpoint 清單、回滾條件
```

### 2-3. 強制開會紀錄格式（Council Minutes）

每次開會必須在回覆中以以下格式輸出**完整的開會紀錄**，使用者看到此紀錄才代表開會流程已正確執行。未輸出此紀錄而直接開始修改，屬於違規行為。

```markdown
## 四 Agent 開會紀錄

**任務**：[任務名稱]
**日期**：[ISO 8601 日期時間]

### Commander 邊界分析
- 影響模組：[列出受影響的檔案與模組]
- 風險等級：[低 / 中 / 高]
- 系統安全邊界：[說明哪些功能不得被影響]

### Planner 技術方案
- 方案：[技術選擇與理由]
- 任務拆解：[列出子任務清單]
- 預計 Checkpoint：[列出各個 Checkpoint 名稱]

### Generator 可行性評估
- 可行性：[通過 / 有疑慮]
- 邊緣案例：[列出需特別處理的邊緣情況]
- 防禦機制：[列出將使用的防禦性代碼模式]

### Evaluator 品質審查意見
- 審查結果：[通過 / 有條件通過 / 退回]
- 測試案例：[列出必要的驗證步驟]
- 退回原因（如有）：[說明]

### Commander 最終決議
- 執行方案：[最終選定方案]
- Checkpoint 清單：[逐條列出]
- 回滾條件：[說明何種情況需要回滾]
```

---

## 三、 前置強制閘門（Hard Gate）

在四 Agent 開會紀錄輸出完畢並獲得使用者確認後，必須在執行任何代碼修改前完成以下前置閘門：

### 步驟 1：變更邊界確認（Commander 執行）
```powershell
python gemma_reasoner.py --check-workflow
```
將輸出結果貼至開會紀錄的 Commander 邊界分析欄位。

### 步驟 2：本地推理規劃（Planner 執行）
```powershell
python gemma_reasoner.py "[開發需求描述]"
```
將輸出的 Key Points 摘要貼至開會紀錄的 Planner 技術方案欄位。

### 步驟 3：閘門授權確認

Agent 必須在回覆中明確聲明以下文字，才可進入代碼生成階段：

> **前置閘門授權確認**：已完成 Commander 邊界檢測（`--check-workflow` 輸出已記錄）與 Planner 本地推理（Key Points 已記錄），四 Agent 開會紀錄完整，現授權進入 Generator 代碼生成階段。

---

## 四、 各任務類型的 Checkpoint 標準

每種任務類型均有對應的 Checkpoint，由四 Agent 開會共同確認後才能勾選完成。

### 功能開發類

| Checkpoint | 負責 Agent | 完成標準 |
|---|---|---|
| CP-1：需求確認 | Commander | 已輸出開會紀錄，影響範圍已定義 |
| CP-2：技術規格確認 | Planner | 任務拆解清單已確認，無歧義 |
| CP-3：代碼生成完成 | Generator | 所有防禦機制已實施，無殘留 TODO |
| CP-4：Code Review 通過 | Evaluator | 無高風險問題，測試案例已定義 |
| CP-5：整合驗證通過 | Commander | 瀏覽器 E2E 測試通過，無 JS/Python 錯誤 |
| CP-6：歸檔完成 | Commander | `sync_session_knowledge.py` 執行成功 |

### 缺陷修復類

| Checkpoint | 負責 Agent | 完成標準 |
|---|---|---|
| CP-1：根本原因確認 | Commander | Root Cause 已定義，非推測性結論 |
| CP-2：修復方案確認 | Planner | 修復範圍最小化，無過度修改 |
| CP-3：修復代碼完成 | Generator | 修復代碼已加入防禦性保護 |
| CP-4：回歸測試通過 | Evaluator | 原始問題不再復現，相鄰功能無破壞 |
| CP-5：歸檔完成 | Commander | `sync_session_knowledge.py --root-cause` 執行成功 |

---

## 五、 違規處理規範

若 Agent 在未完成開會紀錄與前置閘門的情況下直接進行代碼修改，適用以下處理規範：

1. **立即停止**：使用者有權打斷並要求 Agent 回退本次所有修改（`git checkout`）。
2. **補做開會**：Agent 必須補輸出完整的開會紀錄，並說明已修改的內容及其評估結果。
3. **自我審計**：Agent 必須主動說明「哪個步驟被跳過」及「為何被跳過」，不得以「任務看起來簡單」為由合理化違規行為。

---

## 六、 知識庫歷程自動更新規範

* **自動歸檔要求**：每當使用者表示「功能開發完成」、「Bug 修復好了」或「今天先到這邊」時，**必須自動執行**歷程同步工具。
* **執行指令**：
  * 修復類：`python Gemini_task/app/sync_session_knowledge.py --request-name "[主題]" --root-cause "[根本原因]"`
  * 開發類：`python Gemini_task/app/sync_session_knowledge.py --request-name "[功能需求主題]"`
* **自動驗證與重新關聯**：此指令會自動將本次對話的 `implementation_plan.md`、`walkthrough.md`、與 `error_report.md`（如有）同步至全域知識庫對應資料夾下，並自動重新運行 Wiki 檔案之關聯性標註與索引更新。
