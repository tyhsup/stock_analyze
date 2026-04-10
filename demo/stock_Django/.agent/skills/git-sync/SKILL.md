---
# Git Sync & Version Control Skill

## Description
當使用者要求「同步」、「更新 Github」、「備份程式碼」或執行 Git 相關操作時，使用此技能自動完成從本地到遠端的推送流程。

## Instructions
1. **狀態檢查**：先執行 `git status` 了解目前有哪些檔案變更（包含 Python 腳本與資料庫設定）。
2. **自動暫存**：執行 `git add .` 將所有變更加入暫存區。
3. **智慧 Commit**：
   - 根據具體的程式碼變更（例如：修改了 `stock_investor-ver2.py` 的抓取邏輯或更新了 `mySQL_OP.py`），生成一段專業的英文 Commit Message。
   - 格式建議：`feat: [功能說明]`, `fix: [修復說明]`, 或 `docs: [文件更新]`。
4. **執行推送**：執行 `git push origin main`。
5. **結果回報**：完成後告訴使用者「已成功將變更推送到 GitHub」，並簡述這次 commit 的內容。

## Constraints
- 如果遇到 `merge conflict`（合併衝突），請停止自動化並徵詢使用者的處理方式。
- 嚴禁使用 `git push --force`，除非使用者明確要求。

---
