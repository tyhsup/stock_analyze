import os
import sys
import json
import subprocess
import requests

def get_groq_key():
    paths = [
        "demo/stock_Django/.env",
        "../demo/stock_Django/.env",
        "../../demo/stock_Django/.env",
        "e:/Infinity/mydjango/demo/stock_Django/.env"
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("GROQ_API_KEY="):
                        return line.split("=")[1].strip()
    return os.getenv("GROQ_API_KEY", "")

def check_workflow_boundaries():
    """
    執行 Git 狀態邊界檢測，判斷是否即將進行關鍵變更，引導主代理遵循四 Agent 推理工作流。
    """
    print("\n[Workflow Boundary Checker] 正在分析當前專案變更狀態...")
    try:
        res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd="e:/Infinity/mydjango")
        lines = res.stdout.strip().splitlines()
        
        if not lines:
            print(">> 狀態：當前無任何待修改或未追蹤的檔案。安全。")
            return
            
        critical_modifications = []
        config_modifications = []
        
        for line in lines:
            status, path = line[:2].strip(), line[3:].strip()
            # 判斷是否為關鍵的 Python 程式邏輯變更
            if path.endswith(".py") and not "test" in path and not "scratch" in path:
                critical_modifications.append(path)
            # 判斷是否為配置、Markdown 或同步腳本變更
            elif path.endswith(".md") or "sync" in path:
                config_modifications.append(path)
                
        print("\n--- 邊界分析報告 ---")
        if critical_modifications:
            print("[警告] 檢測到關鍵 Python 模組正在進行/即將進行修改：")
            for p in critical_modifications:
                print(f"  *  {p}")
            print("\n>> 強制指引：")
            print("   本變更涉及系統核心邏輯（如 Views, Orchestrator 或 MCP 通訊）。")
            print("   請務必在進行實體代碼修改前，先執行「gemma_reasoner.py」進行本地推理規劃，")
            print("   並確保設計符合Blended Valuation與零破壞原則！\n")
            
        if config_modifications:
            print("[提示] 檢測到 Markdown 設定檔、自訂技能或同步腳本修改：")
            for p in config_modifications:
                print(f"  *  {p}")
            print("\n>> 防錯指引：")
            print("   即使為輕量級設定檔或輔助腳本變更，仍有以下潛在風險：")
            print("   1. 檔案讀寫 open(...) 時，必須加上 errors='ignore' 或 errors='replace' 以防 Unicode 錯誤。")
            print("   2. 自訂技能 SKILL.md 必須使用全大寫檔名，且必須包含 name 與 description 欄位。\n")
            
        if not critical_modifications and not config_modifications:
            print(">> 狀態：檢測到非核心檔案變更（測試或臨時檔案）。建議常規除錯。")
            
    except Exception as e:
        print(f"無法執行 Git 狀態檢測: {e}")

def generate_four_agent_template():
    """
    印出四 Agent (Commander, Planner, Generator, Evaluator) 推理與決策提示範本。
    """
    template = """
================================================================================
四 Agent (Commander/Planner/Generator/Evaluator) 推理規劃提示範本
================================================================================
請將以下結構發送給 Groq Llama-3.3-70b 進行深度分析：

【Role: Commander (總指揮官)】
任務：請解析以下開發需求，定義系統邊界，並指出可能影響的既有前後端功能（Blended Valuation 表單、ApexCharts 繪圖等）：
需求內容：[在此填入您的需求]

【Role: Planner (架構規劃師)】
任務：請設計技術實作路徑，重點說明：
1. 哪些 Python 檔案需要修改，是否有新增的 App？
2. 資料庫連線池與 MySQL 查詢優化（有無 parameters 參數防 SQL 注入）？
3. 本地 API 是否符合 Django REST 規範？

【Role: Generator (代碼生成官)】
任務：請給出符合傳統繁體中文排版（中英文半形空格）、專有名詞官方大小寫的精準 Python/HTML 代碼。
* [!] 防錯要點 1：若進行檔案讀寫，open() 必須加入 errors="ignore" 以防止 Unicode 解碼崩潰！
* [!] 防錯要點 2：若涉及 Gemini 二輪 Function Calling，必須在 contents model 中完整原樣回傳第一輪 parts (含 thoughtSignature)，防止 HTTP 400 錯誤！

【Role: Evaluator (品質審查官)】
任務：請對 Generator 產出的代碼進行嚴格 Code Review，列出可能發生的：
1. 資料型態 JSON 序列化失敗風險（datetime / Decimal）。
2. 對原有圖表 ApexCharts 產生的樣式遮擋與排版影響。
3. 提供自動化單元測試與手動 E2E 驗證的具體測試方案。
================================================================================
"""
    print(template)

def query_gemma(prompt: str) -> str:
    key = get_groq_key()
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    
    # 融入歷史踩坑教訓的 System Prompt
    system_prompt = (
        "你是一個專門運行於本機端、協助金融分析平台開發的 AI 核心推理助理。你的職責是進行精準的代碼審查與架構分析。\n"
        "當在分析與生成代碼時，你必須主動確保以下「專案歷史踩坑防禦要點」被嚴格執行：\n"
        "1. 【Unicode 防禦】：在讀取、同步或輸出本地檔案時，Python 的 open() 函數必須明確加上 errors='ignore' 或 errors='replace'，防止系統預設編碼 (如 cp950) 與 utf-8 衝突引發解碼崩潰。\n"
        "2. 【Gemini API 400 兼容】：如果涉及 Gemini API 的 Function Calling，第二輪 functionResponse 發送時，必須在 contents 的 model 角色中，將第一輪取得的 parts（包含推理思維 thoughtSignature）完整且原封不動地發回給 Google 官方 API，否則會直接觸發 HTTP 400 錯誤。\n"
        "3. 【JSON 序列化保護】：從本地 MySQL 查詢數據返回給 API 時，必須將 datetime.date/datetime.datetime 以及 decimal.Decimal 格式化為標準的 string 與 float，防止 JSON 序列化失敗。\n"
        "4. 【零破壞原則】：新增的功能（如對話懸浮面板、導出按鈕）必須完全在獨立的 App 下開發，絕不遮擋或干擾既有的 ApexCharts K線圖與 blended valuation 計算。\n"
        "請以冷靜、務實且符合台灣繁體中文規範（中英文數字間保留半形空格）回答，禁止使用 Emoji 與情緒化修飾詞。"
    )
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2
    }
    
    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=30,
            proxies={"http": None, "https": None}
        )
        if response.status_code == 200:
            res_data = response.json()
            try:
                return res_data["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                return f"Error: Unexpected response structure from Groq API: {res_data}"
        else:
            return f"Error: Groq API returned status code {response.status_code}. Details: {response.text}"
    except Exception as e:
        return f"Status: ExecutionError\nRoot Cause: {str(e)}\nSuggested Fix: Check network connection and API key."

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--check-workflow":
            check_workflow_boundaries()
        elif arg == "--generate-template":
            generate_four_agent_template()
        else:
            res = query_gemma(arg)
            with open("gemma_reasoner_out.txt", "w", encoding="utf-8") as f:
                f.write(res)
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except AttributeError:
                pass
            print(res)
    else:
        print("Usage:")
        print("  python gemma_reasoner.py --check-workflow      # 執行 Git 變更邊界安全檢測")
        print("  python gemma_reasoner.py --generate-template   # 生成四 Agent 推理規劃模板")
        print("  python gemma_reasoner.py \"<prompt>\"            # 常規本地推理（內建專案防錯防禦機制）")
