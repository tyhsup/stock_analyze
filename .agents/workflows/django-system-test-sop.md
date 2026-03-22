---
description: 
---

階段一：服務生命週期管理 (Service Management)
目標： 確保環境中只有一個最新版的 Django Server 在運行。

狀態檢查 (Check)：

執行系統指令 netstat -ano | findstr :8000 (Windows) 或調用埠口掃描工具。

分支決策：

若有 PID 佔用：執行 taskkill /F /PID <PID> 強制關閉。

若無：跳至下一步。

重啟服務 (Execute)：

在專案根目錄執行 python manage.py runserver。

觀察指標：等待 Terminal 出現 Starting development server at http://127.0.0.1:8000/ 字樣。

逾時處理：若 15 秒內無回應，讀取 error log 並嘗試 python manage.py migrate（防止因資料庫結構更新導致的啟動失敗）。

階段二：動態測試樣本提取 (Data Sampling)
目標： 避免使用固定的寫死（Hard-coded）測試值，確保資料庫連線正常。

資料獲取 (Query)：

透過 Python 工具連接 MySQL 執行：SELECT stock_id FROM stock_table ORDER BY RAND() LIMIT 1;。

變量存儲 (Context)：

將選中的 stock_id 存入 Agent 的當前會話上下文（例如：2330.TW）。

階段三：瀏覽器端功能審計 (UI/UX Audit)
目標： 模擬真人行為，驗證前端渲染與後端資料流。

導航 (Action)：

調用 Browser Tool 開啟 http://127.0.0.1:8000/。

交互 (Interaction)：

在搜尋框輸入步驟二取得的 stock_id ,天數設定120天並點擊analyze。

監控 Network Tab：確保沒有任何 4xx 或 5xx 的 API 請求。

確認Updating Data 進度表有按照完成事項進度更新

嚴格限制瀏覽器代理只做測試不做任何程式修改

深度驗證 (Validation)：

使用現有的skills進行以下檢查 :

圖表檢查：使用選取器檢查 <canvas> 或 <svg>。執行 JavaScript document.querySelector('canvas').getContext('2d') 確保 Canvas 不是空白。

技術指標檢查 : 點擊Indicators內的所有選項, 確認所有技術指標都有顯示(包括非預設的技術指標)

情緒分析檢查：點擊News按鈕, 輸入stock_id(symbol or keyword)以及天數(Max results)後搜尋新聞數量並確認情緒分析結果。

Institutional 調查 : 點擊Institutional 按鈕進入頁面後確認圖表是否正常顯示, 是否需要優化。

估值計算檢查 : 點擊Fair Value Calculator, 進入頁面後確認估值計算結果是否有異常。

檢核標準：文字長度 > 10 字符，且不包含「加載中」或「發生錯誤」等字眼。圖表必須能顯示, 估值計算結果若與現價落差太大需確認是否是研發資本投入過多等狀況造成。

資料時效性：抓取網頁上的「資料日期」，與系統當前日期進行 date_diff。若資料日期不匹配, 定義為異常。

每次輸入新參數前須將舊參數去除, 避免錯誤輸入。

階段四：異常處理與回報 (Reporting)
目標： 結構化輸出結果，方便開發者快速定位問題。

成功路徑：Agent 需輸出一則簡述，包含測試的股票代碼、Server PID、以及「所有指標正常」的綠色勾號。

失敗路徑：

若 UI 報錯：Agent 應自動截圖 (Snapshot) 並分析 HTML 源碼中的錯誤訊息。

最後關閉瀏覽器代理

若資料未更新：Agent 應檢查 MySQL_OP.py 的最後執行日誌。