# Agent Behavioral Rules

## 知識庫歷程自動更新規範
* **自動歸檔要求**：每當使用者表示「功能開發完成」、「Bug 修復好了」或「今天先到這邊」時，或當您（Agent）準備結束當前會話並提交工作成果前，**必須自動執行** 歷程同步工具。
* **執行指令**：
  * 對於修復類任務：
    `python Gemini_task/app/sync_session_knowledge.py --request-name "[需求或修復主題名稱]" --root-cause "[簡述修補的根本原因]"`
  * 對於一般功能開發：
    `python Gemini_task/app/sync_session_knowledge.py --request-name "[功能需求主題名稱]"`
* **自動驗證與重新關聯**：此指令會自動將本次對話的 `implementation_plan.md`、`walkthrough.md`、與 `error_report.md`（如有）同步至全域知識庫對應資料夾下，並自動重新運行 Wiki 檔案之關聯性標註與索引更新。
