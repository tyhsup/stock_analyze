import os
import sys
import json
import shutil
import logging
import requests
import subprocess
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# 載入 Local DB MCP Server 工具
from . import mcp_local_db_server

class FinancialOrchestrator:
    """
    輕量化金融 AI Orchestrator。
    負責解析使用者自然語言意圖，並在多個特化 Subagents 之間進行任務分發與調度。
    支持 Gemini 原生 Tool Use (Function Calling) 以串接本機 MySQL 與 yfinance 數據。
    三級備援推理機制：優先直連 API，次之 CLI，最後本地 Ollama。
    """
    def __init__(self, model_name: str = "gemini-3.5-flash", max_depth: int = 3):
        self.model_name = model_name
        self.max_depth = max_depth
        
        # 讀取與既有系統相同的 API Key
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            from dotenv import load_dotenv
            dotenv_path = os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", ".env")
            if os.path.exists(dotenv_path):
                load_dotenv(dotenv_path)
                self.gemini_api_key = os.getenv("GEMINI_API_KEY")
            else:
                project_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_Django", ".env")
                if os.path.exists(project_env):
                    load_dotenv(project_env)
                    self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        self.gemini_path = shutil.which("gemini")
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # 定義 Tool Calling 宣告 (符合 Google API 規格)
        self.tools_declaration = [
            {
                "functionDeclarations": [
                    {
                        "name": "query_taiwan_chips",
                        "description": "查詢台股股票代號最新的法人籌碼，包含外資、投信、自營商買賣超股數。",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "symbol": {
                                    "type": "STRING",
                                    "description": "台股股票代號 (例如：'2330', '2303')"
                                },
                                "limit": {
                                    "type": "INTEGER",
                                    "description": "查詢天數，預設為 10 天，最多可查詢 30 天。"
                                }
                            },
                            "required": ["symbol"]
                        }
                    },
                    {
                        "name": "query_us_market_data",
                        "description": "查詢美股股票代號最新的歷史股價與 13F 機構持股，若數據缺失會自動從外網同步補充並快取。",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "symbol": {
                                    "type": "STRING",
                                    "description": "美股股票代號 (例如：'AAPL', 'TSLA')"
                                },
                                "limit": {
                                    "type": "INTEGER",
                                    "description": "查詢天數，預設為 10 天。"
                                }
                            },
                            "required": ["symbol"]
                        }
                    }
                ]
            }
        ]

    def load_financial_skill(self) -> str:
        """
        讀取專案專用的金融 AI 助理技能規範 SKILL.md 的內容。
        """
        skill_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".agents", "skills", "financial-ai-assistant", "SKILL.md")
        if os.path.exists(skill_path):
            try:
                with open(skill_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"[Orchestrator] 載入 SKILL.md 失敗: {e}")
        return ""

    def _call_gemini_api_direct(self, system_prompt: str, user_prompt: str, tools: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """
        優先使用 requests 直接呼叫 Gemini 官方 API (HTTP POST)。
        """
        if not self.gemini_api_key:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            return None
            
        model = self.model_name
        if "gemini-3.5-flash" in model or "gemini-3.1" in model:
            # 在某些 API 環境中，若 v1beta 不接受 3.5-flash，可在此處向下相容
            pass

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.gemini_api_key}"
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{system_prompt}\n\n使用者輸入：{user_prompt}"}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 2048
            }
        }
        
        if tools:
            payload["tools"] = tools

        try:
            logger.info(f"[Orchestrator] 直接調用 Gemini 官方 API ({model})...")
            response = requests.post(url, json=payload, headers=headers, timeout=20)
            if response.status_code == 200:
                res_json = response.json()
                # 檢查是否為 Tool Call 或者是 Text Response
                candidates = res_json.get("candidates", [])
                if not candidates:
                    return None
                
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    return None
                
                # 如果是 Tool Call，回傳完整的 parts JSON 字串
                if "functionCall" in parts[0]:
                    return json.dumps({"type": "function_call", "calls": parts})
                
                return parts[0].get("text", "").strip()
            else:
                logger.error(f"[Orchestrator] Gemini API 請求失敗, HTTP {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logger.error(f"[Orchestrator] Gemini API 呼叫異常: {e}")
            return None

    def _call_gemini_cli(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        備援方案 1: 使用 gemini CLI
        """
        if not self.gemini_path:
            return None

        full_prompt = f"{system_prompt}\n\n使用者輸入：{user_prompt}"
        full_prompt = full_prompt.replace("\r", " ").replace("\n", " ").strip()

        env = os.environ.copy()
        if self.gemini_api_key:
            env["GEMINI_API_KEY"] = self.gemini_api_key

        args = [self.gemini_path, "-m", self.model_name, "--skip-trust", "-p", full_prompt]
        
        try:
            logger.info(f"[Orchestrator] 調用 gemini CLI ({self.model_name})...")
            result = subprocess.run(args, capture_output=True, env=env, shell=False, timeout=20)
            if result.returncode != 0:
                return None
            
            stdout_decoded = result.stdout.decode("utf-8", errors="replace").strip()
            try:
                json_data = json.loads(stdout_decoded)
                return json_data.get("response", "").strip()
            except json.JSONDecodeError:
                return stdout_decoded
        except Exception as e:
            logger.error(f"[Orchestrator] gemini CLI 異常: {e}")
            return None

    def _call_ollama_fallback(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        備援方案 2: 本地 Ollama
        """
        logger.warning("[Orchestrator] 啟動本地 Ollama (gemma4) 作為最終備援...")
        combined_prompt = f"{system_prompt}\n\n使用者輸入：{user_prompt}"
        payload = {
            "model": "gemma4-cpu",
            "prompt": combined_prompt,
            "stream": False,
            "options": { "temperature": 0.2, "num_ctx": 4096 }
        }
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return None
        except Exception as e:
            logger.error(f"[Orchestrator] Ollama 呼叫異常: {e}")
            return None

    def ask_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        統一的三級推理入口
        """
        skill_content = self.load_financial_skill()
        if skill_content:
            system_prompt = f"{system_prompt}\n\n[金融 AI 助理核心技能與雲端智慧建議指引規範]\n{skill_content}"

        res = self._call_gemini_api_direct(system_prompt, user_prompt)
        if res:
            return res
        res = self._call_gemini_cli(system_prompt, user_prompt)
        if res:
            return res
        return self._call_ollama_fallback(system_prompt, user_prompt)

    def ask_llm_with_tools(self, system_prompt: str, user_prompt: str) -> str:
        """
        實現 Function Calling 交互，呼叫本地資料庫並回傳給 LLM 產出最終回覆。
        """
        skill_content = self.load_financial_skill()
        if skill_content:
            system_prompt = f"{system_prompt}\n\n[金融 AI 助理核心技能與雲端智慧建議指引規範]\n{skill_content}"

        # 1. 帶上 tools 進行第一次調用
        first_res = self._call_gemini_api_direct(system_prompt, user_prompt, tools=self.tools_declaration)
        
        if not first_res:
            return "抱歉，目前無法存取金融數據庫分析服務。"

        # 2. 判斷是否觸發了 Function Call
        try:
            parsed = json.loads(first_res)
            if isinstance(parsed, dict) and parsed.get("type") == "function_call":
                # 觸發了 Tool Calling！
                calls = parsed.get("calls", [])
                func_call = calls[0].get("functionCall", {})
                func_name = func_call.get("name")
                args = func_call.get("args", {})
                
                logger.info(f"[Orchestrator] 檢測到 Tool Call: {func_name}，參數: {args}")
                
                # 3. 在本地執行 SQL 查詢
                tool_output = {}
                if func_name == "query_taiwan_chips":
                    symbol = args.get("symbol")
                    limit = args.get("limit", 10)
                    tool_output = mcp_local_db_server.query_taiwan_chips(symbol, limit)
                elif func_name == "query_us_market_data":
                    symbol = args.get("symbol")
                    limit = args.get("limit", 10)
                    tool_output = mcp_local_db_server.query_us_market_data(symbol, limit)
                
                # 4. 將數據回傳給 Gemini 進行最終分析
                if not self.gemini_api_key:
                    self.gemini_api_key = os.getenv("GEMINI_API_KEY")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.gemini_api_key}"
                headers = {"Content-Type": "application/json"}
                
                second_payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": f"{system_prompt}\n\n使用者輸入：{user_prompt}"}]
                        },
                        {
                            "role": "model",
                            "parts": calls
                        },
                        {
                            "role": "function",
                            "parts": [
                                {
                                    "functionResponse": {
                                        "name": func_name,
                                        "response": {"output": tool_output}
                                    }
                                }
                            ]
                        }
                    ],
                    "tools": self.tools_declaration
                }
                logger.info("[Orchestrator] 將本地查詢數據回傳給 Gemini 進行整合分析...")
                sec_resp = requests.post(url, json=second_payload, headers=headers, timeout=25)
                if sec_resp.status_code == 200:
                    candidates = sec_resp.json().get("candidates", [])
                    if candidates:
                        return candidates[0].get("content", {}).get("parts", [])[0].get("text", "").strip()
                else:
                    logger.error(f"[Orchestrator] 第二輪 API 呼叫失敗，HTTP {sec_resp.status_code}: {sec_resp.text}")
                    os.makedirs('scratch', exist_ok=True)
                    with open('scratch/django_error.log', 'w', encoding='utf-8') as ef:
                        ef.write(f"HTTP {sec_resp.status_code}: {sec_resp.text}\nAPI Key: {self.gemini_api_key}\n")
                
                return f"已成功查詢到數據，但在生成最終報告時發生錯誤。數據摘要：{json.dumps(tool_output, ensure_ascii=False)[:300]}..."
        except json.JSONDecodeError:
            pass
        except Exception as err:
            logger.error(f"[Orchestrator] Function Calling 處理失敗: {err}", exc_info=True)
            os.makedirs('scratch', exist_ok=True)
            import traceback
            with open('scratch/django_error.log', 'w', encoding='utf-8') as ef:
                ef.write(f"Exception: {err}\n{traceback.format_exc()}\n")
            
        return first_res

    def route_intent(self, query: str) -> Dict[str, Any]:
        """
        Stage 1: 意圖路由 (Intent Routing)
        """
        system_prompt = (
            "你是一個金融分析意圖路由專家。請分析使用者的輸入，並將其分類至以下四個代理之一：\n"
            "1. 'CHIP_AGENT'：用於查詢台股法人籌碼、外資/投信/自營商持股比例或 concentration。\n"
            "2. 'VALUATION_AGENT'：用於要求進行股票估值計算（如 DCF、乘數模型）、調整估值假設參數，或申請下載 Excel 模型。\n"
            "3. 'RESEARCH_AGENT'：用於查詢個股新聞、要求進行 AI 情緒分析、起草財報季報分析報告或覆蓋初始報告。\n"
            "4. 'GENERAL_AGENT'：其他一般對話、問候、財務知識問答、或無法歸入上述三類的通用任務。\n\n"
            "請嚴格輸出以下 JSON 格式（不要有 markdown 標記，如 ```json，也不要有其他解釋字眼）：\n"
            "{\"agent\": \"CHIP_AGENT\" 或 \"VALUATION_AGENT\" 或 \"RESEARCH_AGENT\" 或 \"GENERAL_AGENT\", \"reason\": \"原因分析\"}"
        )
        
        res_text = self.ask_llm(system_prompt, query)
        
        default_route = {"agent": "GENERAL_AGENT", "reason": "預設一般對話"}
        if not res_text:
            return default_route
            
        clean_text = res_text.strip()
        if clean_text.startswith("```"):
            lines = clean_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_text = "\n".join(lines).strip()
            
        try:
            return json.loads(clean_text)
        except Exception as e:
            logger.warning(f"[Orchestrator] 無法解析意圖 JSON: {e}，原始回覆: {res_text}")
            for agent_name in ["CHIP_AGENT", "VALUATION_AGENT", "RESEARCH_AGENT"]:
                if agent_name in res_text:
                    return {"agent": agent_name, "reason": "關鍵字匹配"}
            return default_route

    def dispatch_to_agent(self, agent_name: str, query: str, history: List[Dict[str, str]], depth: int) -> Dict[str, Any]:
        """
        Stage 2: 委派給 Subagent 執行。
        """
        if depth > self.max_depth:
            return {
                "response": "抱歉，系統在處理您的複雜請求時已達到最大代理跳轉深度上限，已暫停處理。請嘗試拆分您的問題。",
                "route_log": [f"Error: Exceeded max depth of {self.max_depth}"]
            }
            
        route_log = f"[Depth {depth}] 任務已指派給 {agent_name}。"
        logger.info(route_log)
        
        if agent_name == "CHIP_AGENT":
            # 籌碼與數據代理 (第二階段正式串接 Tool Use)
            system_prompt = (
                "你是一個專業的金融數據與籌碼分析代理。請基於提供的工具 (Tools)，幫使用者查詢台股三大法人籌碼或美股 13F 機構持股數據。\n"
                "查詢到數據後，請為使用者進行結構化、條列式的中文分析：\n"
                "1. 對於台股，請分析外資、投信與自營商的買賣超趨勢，計算法人持股動態。\n"
                "2. 對於美股，請條列最新的前三大機構持股比例與股數變動狀況。\n"
                "3. 以專業、務實的態度總結籌碼流向對股價的潛在影響。\n"
                "請一律使用繁體中文，中英文之間保留半形空格，專有名詞使用官方大小寫。"
            )
            response_text = self.ask_llm_with_tools(system_prompt, query)
            
        elif agent_name == "VALUATION_AGENT":
            response_text = (
                f"【估值建模代理】已接收您的請求：「{query}」。\n\n"
                "我們已完成第三階段開發！您現在可以直接在『合理價值計算機』網頁的估值總覽卡片中，"
                "點擊「Export Excel Valuation Model」按鈕，下載具備動態折現公式的活體財務模型。\n\n"
                "若要稽核您在外自建的模型公式，請直接點擊導覽列或前往 [Excel 模型稽核頁面](file:///agents/audit/) 上傳您的 Excel 檔案。"
            )
        elif agent_name == "RESEARCH_AGENT":
            response_text = (
                f"【股權研究代理】已接收您的請求：「{query}」。\n\n"
                "個股覆蓋報告與財報分析功能已上線！\n"
                "請在對話框中直接輸入以下快捷指令以獲取深度投研分析：\n"
                "*   `/earnings [股票代號]` (例如 `/earnings 2330`)：分析近 6 季財報結構、利潤率走勢與亮點風險。\n"
                "*   `/initiate [股票代號]` (例如 `/initiate AAPL`)：整合股價、法人籌碼與財報起草個股首次覆蓋報告。"
            )
        else:
            # 一般代理 - 直接進行對話
            system_prompt = (
                "你是一個專業的金融 AI 助理。請以冷靜、務實且極具台灣金融業專業水準的態度，回答使用者的金融知識、理財或專案操作問題。\n"
                "請一律使用繁體中文，中英文之間請保留半形空格，字元與標點規範需遵守台灣習慣。不要有冗長廢話，直接精準切入主題。"
            )
            response_text = self.ask_llm(system_prompt, query) or "抱歉，目前 AI 助理無法取得回覆，請稍後再試。"

        return {
            "response": response_text,
            "route_log": [route_log]
        }

    def handle_earnings_command(self, ticker: str) -> Dict[str, Any]:
        """
        處理 /earnings 季報分析指令 (透過本地 MySQL 查詢與 LLM 分析)。
        """
        logger.info(f"[Orchestrator] 觸發快捷指令 /earnings，代號: {ticker}")
        earnings_data = mcp_local_db_server.query_earnings_data(ticker, limit_quarters=6)
        
        if earnings_data.get("status") == "error" or not earnings_data.get("data"):
            msg = (
                f"【財務報表分析】抱歉，未能在本地資料庫中檢測到 {ticker} 的歷史季報數據。\n\n"
                f"建議：請先前往『合理價值計算機』頁面搜尋 `{ticker}`，"
                f"系統將會自動從線上同步該個股的財務報表，更新成功後即可在此使用快捷分析指令。"
            )
            return {
                "agent": "RESEARCH_AGENT",
                "reason": "指令匹配",
                "response": msg,
                "route_log": ["[LocalDB] 查無財報數據，提示線上同步。"]
            }
            
        system_prompt = (
            "你是一個專業的股權研究分析師（Equity Research Analyst）。請針對提供的最近多季原始財報數據（Revenue, EBIT, Net Income 等），\n"
            "為使用者撰寫一份深入的財報季度更新報告（Earnings Analysis Report）。\n"
            "請以專業、條列式且邏輯嚴謹的台灣繁體中文回答，包含以下面向：\n"
            "1. 營收與淨利增長趨勢分析。\n"
            "2. 毛利率與營業利益率（EBIT Margin）走勢分析。\n"
            "3. 資產負債表結構穩定度評估。\n"
            "4. 簡明條列該股的投資亮點與潛在風險。\n"
            "請一律使用繁體中文，中英文之間保留半形空格，格式採用 Markdown 編排，字詞需符合台灣商業習慣。"
        )
        user_prompt = f"個股代號：{ticker}\n歷史財報數據摘要：\n{json.dumps(earnings_data, ensure_ascii=False)}"
        
        response_text = self.ask_llm(system_prompt, user_prompt) or "抱歉，分析過程中發生異常，請稍後重試。"
        
        return {
            "agent": "RESEARCH_AGENT",
            "reason": "指令匹配",
            "response": response_text,
            "route_log": [f"[LocalDB] 成功讀取 {len(earnings_data['data'])} 季財報進行 LLM 深度分析。"]
        }

    def handle_initiate_command(self, ticker: str) -> Dict[str, Any]:
        """
        處理 /initiate 初始覆蓋報告指令。
        """
        logger.info(f"[Orchestrator] 觸發快捷指令 /initiate，代號: {ticker}")
        earnings_data = mcp_local_db_server.query_earnings_data(ticker, limit_quarters=4)
        
        is_tw = ticker.isdigit() or ".TW" in ticker.upper()
        if is_tw:
            market_data = mcp_local_db_server.query_taiwan_chips(ticker, limit=10)
        else:
            market_data = mcp_local_db_server.query_us_market_data(ticker, limit=10)
            
        if (not earnings_data.get("data") or len(earnings_data.get("data", [])) == 0) and market_data.get("status") == "error":
            msg = (
                f"【個股覆蓋報告】抱歉，未能在本地數據庫中查到 {ticker} 的完整數據。\n\n"
                f"請先在『合理價值計算機』查詢此個股，以啟動自動化線上同步。"
            )
            return {
                "agent": "RESEARCH_AGENT",
                "reason": "指令匹配",
                "response": msg,
                "route_log": ["[LocalDB] 查無歷史數據。"]
            }
            
        system_prompt = (
            "你是一個高級股權研究總監（Director of Equity Research）。請基於提供的股票最新市場股價、法人籌碼流向與歷史財務數據，\n"
            "為該個股起草一份正式的個股首次覆蓋報告（Initiation of Coverage Report）。\n"
            "報告結構必須極度專業，包含以下模組：\n"
            "1. 評級與投資摘要（Rating & Investment Summary）。\n"
            "2. 核心投資亮點（Investment Thesis）。\n"
            "3. 法人籌碼結構分析。\n"
            "4. 合理估值敏感度評估（WACC 與永續成長率對合理價格的影響）。\n"
            "5. 潛在投資風險提示。\n"
            "請一律使用繁體中文，中英文之間保留半形空格，格式採用 Markdown 編排，不要廢話，直接給予專業投行報告風格內容。"
        )
        user_prompt = (
            f"個股代號：{ticker}\n"
            f"行情與籌碼數據：\n{json.dumps(market_data, ensure_ascii=False)}\n\n"
            f"財務數據：\n{json.dumps(earnings_data, ensure_ascii=False)}"
        )
        
        response_text = self.ask_llm(system_prompt, user_prompt) or "抱歉，分析過程中發生異常，請稍後重試。"
        
        return {
            "agent": "RESEARCH_AGENT",
            "reason": "指令匹配",
            "response": response_text,
            "route_log": ["[LocalDB] 成功讀取財報與籌碼行情，交由 LLM 生成覆蓋報告。"]
        }

    def chat(self, query: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        外部調用主入口。
        """
        history = history or []
        query_clean = query.strip()
        
        # 1. 優先攔截快捷斜線指令
        if query_clean.startswith("/"):
            parts = query_clean.split(maxsplit=1)
            cmd = parts[0].lower()
            ticker = parts[1].strip() if len(parts) > 1 else ""
            
            if cmd == "/earnings" and ticker:
                return self.handle_earnings_command(ticker)
            elif cmd == "/initiate" and ticker:
                return self.handle_initiate_command(ticker)
            elif cmd in ["/debug-model", "/debug_model"]:
                return {
                    "agent": "VALUATION_AGENT",
                    "reason": "指令匹配",
                    "response": "【Excel 模型稽核】請前往 [Excel 模型稽核頁面](file:///agents/audit/) 上傳您的 `.xlsx` 財務模型檔案，AI 代理將會為您在網頁上呈現公式、硬編碼與數值勾稽的稽核結果。",
                    "route_log": ["[Instruction] 導向模型稽核網頁。"]
                }
        
        route_res = self.route_intent(query)
        target_agent = route_res.get("agent", "GENERAL_AGENT")
        reason = route_res.get("reason", "")
        
        execution_res = self.dispatch_to_agent(target_agent, query, history, depth=1)
        
        return {
            "agent": target_agent,
            "reason": reason,
            "response": execution_res.get("response"),
            "route_log": execution_res.get("route_log", [])
        }
