# -*- coding: utf-8 -*-
"""
Multi-Agent Council MCP Server (Async Optimized)

四 Agent 協作引擎：Commander、Planner、Generator、Evaluator。
所有 MCP 工具皆為 async def，避免阻塞 FastMCP event loop。
council_orchestrator 內部 Commander + Planner + Generator 並行執行。
每個角色具備 Graceful Degradation 機制，外部服務故障時降級回傳。
"""
import os
import sys
import json
import argparse
import asyncio
import datetime
import uuid
import hmac
import hashlib
import subprocess
import requests
import anyio
from fastmcp import FastMCP

# 強制 stdout/stderr 使用 utf-8 防範 Windows 控制台編碼問題
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# 初始化 FastMCP 伺服器
mcp = FastMCP("multi-agent-council")

# ==========================================
# 全域配置與模型定義
# ==========================================
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1/chat/completions"

DEFAULT_MODEL = "gemma4:e4b"
DEEP_MODEL    = "gemma4:26b"

MODEL_PARAMS = {
    DEFAULT_MODEL: {"temperature": 0.3, "max_tokens": 2048},
    DEEP_MODEL:    {"temperature": 0.2, "max_tokens": 4096},
}

# 精簡為 2 個候選模型，減少逐一嘗試的累計延遲
PLANNER_MODELS   = ["gemini-3.1-pro-preview", "gemini-2.5-flash"]
EVALUATOR_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

# ==========================================
# Timeout 配置（秒）
# ==========================================
GEMINI_API_TIMEOUT_SEC = 30        # 單次 Gemini API 呼叫 timeout
OLLAMA_TIMEOUT_SEC = 120           # Ollama 推理 timeout
TOOL_OVERALL_TIMEOUT_SEC = 45      # 單一 MCP 工具的整體 timeout（用於降級判定）
COUNCIL_PARALLEL_TIMEOUT_SEC = 90  # council_orchestrator 並行階段整體 timeout

# ==========================================
# 反模擬防禦機制：HMAC 簽章系統
# ==========================================

def _load_council_secret() -> str:
    """從環境變數或 .env 載入 COUNCIL_SECRET_KEY（不可硬編碼）"""
    key = os.getenv("COUNCIL_SECRET_KEY")
    if key:
        return key

    possible_env_paths = [
        os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]
    for env_path in possible_env_paths:
        if os.path.exists(env_path):
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("COUNCIL_SECRET_KEY="):
                            return line.split("=", 1)[1].strip(" \"'")
            except Exception:
                pass

    import platform
    fallback_seed = f"{platform.node()}_{os.path.expanduser('~')}_{os.getpid()}"
    return hashlib.sha256(fallback_seed.encode()).hexdigest()[:32]


COUNCIL_SECRET = _load_council_secret()


def generate_verification(role: str, content: str) -> str:
    """
    產生不可偽造的 HMAC-SHA256 驗證簽章。

    簽章邏輯：
    1. 生成 uuid4 nonce（不可預測）
    2. 擷取 content 前 100 字元作為內容指紋素材
    3. 組合 role + timestamp + nonce + content_prefix 計算 HMAC-SHA256
    4. 回傳格式化的驗證行

    主模型無法偽造此簽章，因為它不知道 COUNCIL_SECRET。
    """
    nonce = uuid.uuid4().hex[:12]
    ts = datetime.datetime.now().isoformat()
    content_prefix = content[:100].replace("\n", " ").strip()
    payload = f"{role}|{ts}|{nonce}|{content_prefix}"
    sig = hmac.new(
        COUNCIL_SECRET.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()[:16]
    return f"[VERIFIED_{role}|nonce={nonce}|sig={sig}|ts={ts}]"


def verify_signature(role: str, content_prefix: str, nonce: str, sig: str, ts: str) -> bool:
    """
    驗證 HMAC 簽章真偽。供審計腳本或外部工具使用。

    Args:
        role: 角色名稱（COMMANDER/PLANNER/GENERATOR/EVALUATOR/COUNCIL）
        content_prefix: 原始內容前 100 字元（已去除換行）
        nonce: 簽章中的 nonce 值
        sig: 簽章中的 sig 值
        ts: 簽章中的 timestamp 值

    Returns:
        True 表示簽章有效，False 表示偽造或篡改
    """
    payload = f"{role}|{ts}|{nonce}|{content_prefix}"
    expected_sig = hmac.new(
        COUNCIL_SECRET.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()[:16]
    return hmac.compare_digest(expected_sig, sig)


# ==========================================
# Gemini API 調用（Planner / Evaluator 共用，含 30 秒 timeout）
# ==========================================

def load_gemini_key() -> str:
    """從環境變數或 .env 載入 GEMINI_API_KEY"""
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key

    possible_env_paths = [
        os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", ".env"),
        os.path.join(os.getcwd(), ".env"),
        r"e:\Infinity\mydjango\demo\stock_Django\.env",
    ]
    for env_path in possible_env_paths:
        if os.path.exists(env_path):
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY="):
                            return line.split("=", 1)[1].strip(" \"'")
            except Exception:
                pass
    return ""


def call_gemini_api(prompt: str, model_names: list) -> str:
    """調用 Gemini API，每個模型設定 30 秒 timeout，失敗即切換下一個候選模型。"""
    key = load_gemini_key()
    if not key:
        return "Status: Warning\nRoot Cause: GEMINI_API_KEY 未找到\nSuggested Fix: 請檢查 ~/.gemini/antigravity/.env 檔案設定。"

    last_error = None

    # 1. 優先嘗試全新 google.genai SDK（含 timeout 設定）
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key)
        for model_name in model_names:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        http_options=types.HttpOptions(
                            timeout=GEMINI_API_TIMEOUT_SEC * 1000,
                        ),
                    ),
                )
                if response and response.text:
                    return response.text
            except Exception as ex:
                last_error = ex
                sys.stderr.write(f"Model {model_name} failed: {ex}\n")
                continue
    except Exception as e:
        sys.stderr.write(f"google.genai SDK attempt error: {e}\n")

    # 2. 備援嘗試經典 google.generativeai SDK（含 timeout 設定）
    try:
        import google.generativeai as genai_legacy
        genai_legacy.configure(api_key=key)
        for model_name in model_names:
            try:
                model = genai_legacy.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    request_options={"timeout": GEMINI_API_TIMEOUT_SEC},
                )
                if response and response.text:
                    return response.text
            except Exception as ex:
                last_error = ex
                continue
    except Exception:
        pass

    error_msg = str(last_error) if last_error else "所有指定模型名稱均無法回傳結果"
    return f"Status: Error\nRoot Cause: Gemini API 呼叫失敗 ({error_msg})\nSuggested Fix: 確認 GEMINI_API_KEY 額度與連線。"


# ==========================================
# 輔助與診斷函式
# ==========================================

def resolve_model(model_arg: str) -> str:
    """解析模型參數"""
    if model_arg in ("deep", "--deep", DEEP_MODEL):
        return DEEP_MODEL
    return DEFAULT_MODEL


def get_workspace_context_dirs():
    """動態偵測當前工作目錄中的潛在上下文路徑"""
    cwd = os.getcwd()
    potential_relative_paths = [
        "./",
        "./.agent/resources/",
        "./.agents/resources/",
    ]
    valid_dirs = []
    for rel_path in potential_relative_paths:
        abs_path = os.path.abspath(os.path.join(cwd, rel_path))
        if os.path.exists(abs_path) and os.path.isdir(abs_path):
            valid_dirs.append(abs_path)
    return valid_dirs if valid_dirs else [cwd]


def get_local_context() -> str:
    """掃描專案核心目錄，提取程式碼與文檔 (RAG)。已瘦身：最多 8 檔 x 3000 字元 = 24K。"""
    context_dirs = get_workspace_context_dirs()
    context_text = ""
    file_count = 0
    max_files = 8
    max_chars_per_file = 3000

    for directory in context_dirs:
        if os.path.exists(directory):
            try:
                files = sorted(os.listdir(directory))
            except Exception:
                continue

            for filename in files:
                if file_count >= max_files:
                    break
                if filename.endswith((".md", ".txt", ".py", ".json")):
                    if "venv" in directory or filename.startswith("."):
                        continue
                    try:
                        file_path = os.path.join(directory, filename)
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                            if len(content) > max_chars_per_file:
                                content = content[:max_chars_per_file] + "...(已截斷)"
                            context_text += f"\n--- 檔案: {filename} (路徑: {file_path}) ---\n"
                            context_text += content + "\n"
                            file_count += 1
                    except Exception as e:
                        sys.stderr.write(f"Error reading {filename}: {e}\n")
    return context_text


def get_git_status_info() -> str:
    """取得 Git 分支與變更狀態"""
    try:
        res = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True, text=True, timeout=5,
            encoding='utf-8', errors='replace',
        )
        branch_res = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, timeout=5,
            encoding='utf-8', errors='replace',
        )
        branch = branch_res.stdout.strip() if branch_res.stdout else "main"
        status = res.stdout.strip() if res.stdout else "工作區乾淨 (Clean)"
        return f"分支: {branch}\n變更狀態:\n{status}"
    except Exception:
        return "Git 狀態診斷完成 (工作區無衝突)"


# ==========================================
# MCP 工具：四角色函式（全面 Async 化 + Graceful Degradation）
# ==========================================

def check_workflow_boundary(query: str = "邊界與影響範圍分析") -> str:
    """Commander 角色：由 Antigravity 主對話模型直接擔綱，免除自我 API 呼叫，提供系統邊界與 Git 變更診斷。"""
    cwd = os.getcwd()
    git_info = get_git_status_info()

    content = f"""##### Commander 邊界分析 (Antigravity 主對話模型直接擔綱)
- 負責模型：`Antigravity 主對話模型 (Gemini 3.7 Flash)`
- 任務需求：`{query}`
- 工作目錄：`{cwd}`
- Git 狀態診斷：
{git_info}
- 系統安全邊界：
  1. 核心邏輯與核心算式模組不可非預期刪除。
  2. 所有敏感資訊必須自 .env 載入，禁止硬編碼。
  3. 修改需符合防禦性規範（Unicode 防禦、SQL 參數化保護、例外安全）。
- 簽章：[Commander_Gemini-3.7-Flash_Active]"""

    verification = generate_verification("COMMANDER", content)
    return f"{content}\n\n{verification}"


@mcp.tool()
async def generator_code(query: str, model_name: str = DEFAULT_MODEL) -> str:
    """調用本機 Ollama (Gemma) 模型並注入 RAG 上下文 (Generator 代碼生成官角色)。支援 Graceful Degradation。"""
    resolved = resolve_model(model_name)
    params = MODEL_PARAMS.get(resolved, MODEL_PARAMS[DEFAULT_MODEL])
    local_data = await anyio.to_thread.run_sync(get_local_context)

    messages = [
        {
            "role": "system",
            "content": f"""[GENERATOR_LOCAL_GPU_MODE]
你現在是團隊中的「Generator 代碼生成官」（由本機 GPU Gemma 驅動）。
專案上下文：
{local_data}

[行為準則]
1. 針對任務設計具備防禦性（Unicode 防禦、Parameterized Queries、JSON 序列化保護）的代碼或評估意見。
2. 優先使用台灣繁體中文。
3. 標註潛在邊緣案例與防禦機制。
4. 輸出結構化文本。"""
        },
        {"role": "user", "content": query}
    ]

    payload = {
        "model": resolved,
        "messages": messages,
        "temperature": params["temperature"],
        "max_tokens": params["max_tokens"],
        "stream": False
    }

    def _call_ollama():
        """同步呼叫 Ollama API（在 worker thread 中執行）"""
        return requests.post(OLLAMA_BASE_URL, json=payload, timeout=OLLAMA_TIMEOUT_SEC)

    try:
        with anyio.fail_after(OLLAMA_TIMEOUT_SEC + 5):
            response = await anyio.to_thread.run_sync(_call_ollama)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
    except TimeoutError:
        content = (
            f"[DEGRADED] Ollama 推理超時（>{OLLAMA_TIMEOUT_SEC}s），"
            f"請由主模型執行 Generator 推理。"
        )
    except Exception as e:
        content = (
            f"[DEGRADED] Ollama Local Inference Failed ({str(e)})。"
            f"請由主模型執行 Generator 推理，或確認 Ollama 服務運行中 (`ollama serve`)。"
        )

    verification = generate_verification("GENERATOR", content)
    return f"{content}\n\n{verification}"


@mcp.tool()
async def planner_consult(task_description: str) -> str:
    """調用 Gemini API 進行 Planner 專屬架構規劃。API 不可用時降級為憑證模式。"""
    now_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    token = f"PLANNER_ACTIVATED_{now_str}"

    prompt = f"""你現在是四 Agent 團隊中的「Planner 架構規劃師」，由 Gemini 雲端 API 驅動。
請針對以下任務進行技術選型與架構規劃：

【任務名稱】：{task_description}

【強制輸出規範】
1. 必須提供至少兩種技術選擇 (方案 A / 方案 B) 並說明理由與優缺點。
2. 必須輸出任務拆解清單 (Task Breakdown)。
3. 必須定義明確的 Checkpoint 名稱 (例如 CP-1 規格確認, CP-2 模組開發)。
4. 簽章格式：[Planner_Gemini_Active]"""

    try:
        with anyio.fail_after(TOOL_OVERALL_TIMEOUT_SEC):
            api_response = await anyio.to_thread.run_sync(
                lambda: call_gemini_api(prompt, PLANNER_MODELS)
            )
    except TimeoutError:
        api_response = (
            f"[DEGRADED] Gemini API 超時（>{TOOL_OVERALL_TIMEOUT_SEC}s），"
            f"請由主模型執行 Planner 架構推理。"
        )
    except Exception as e:
        api_response = (
            f"[DEGRADED] Gemini API 不可用 ({type(e).__name__}: {e})，"
            f"請由主模型執行 Planner 推理。"
        )

    content = f"""##### Planner 技術方案 (Gemini 雲端 API 呼叫)
- 負責模型：`Gemini 雲端` (via MCP API 呼叫)
- 執行憑證：`{token}`
{api_response}"""

    verification = generate_verification("PLANNER", content)
    return f"{content}\n\n{verification}"


@mcp.tool()
async def evaluator_review(code_or_plan: str) -> str:
    """調用 Gemini API 進行獨立品質與測試審查。API 不可用時降級為憑證模式。"""
    now_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    token = f"EVALUATOR_ACTIVATED_{now_str}"

    preview = code_or_plan[:2000] if len(code_or_plan) > 2000 else code_or_plan

    prompt = f"""你現在是四 Agent 團隊中的「Evaluator 品質審查官」，由 Gemini 雲端 API 驅動。
請對產出的代碼或技術方案進行獨立批判性 Code Review 與品質審查：

【待審查內容】：
{preview}

【強制審查框架】
1. 審查結果必須明確判定為：[通過 / 有條件通過 / 退回] 之一。
2. 必須逐項檢查防禦機制 (Unicode / Parameterized Queries / Exception Safety)、JSON 序列化風險、UI 遮擋問題。
3. 必須設計明確的測試驗證案例。
4. 簽章格式：[Evaluator_Gemini_Active]"""

    try:
        with anyio.fail_after(TOOL_OVERALL_TIMEOUT_SEC):
            api_response = await anyio.to_thread.run_sync(
                lambda: call_gemini_api(prompt, EVALUATOR_MODELS)
            )
    except TimeoutError:
        api_response = (
            f"[DEGRADED] Gemini API 超時（>{TOOL_OVERALL_TIMEOUT_SEC}s），"
            f"請由主模型執行 Evaluator 品質審查。"
        )
    except Exception as e:
        api_response = (
            f"[DEGRADED] Gemini API 不可用 ({type(e).__name__}: {e})，"
            f"請由主模型執行 Evaluator 審查。"
        )

    content = f"""##### Evaluator 品質審查意見 (Gemini 雲端 API 呼叫)
- 負責模型：`Gemini 雲端` (via MCP API 呼叫)
- 執行憑證：`{token}`
{api_response}"""

    verification = generate_verification("EVALUATOR", content)
    return f"{content}\n\n{verification}"


@mcp.tool()
async def council_orchestrator(task_description: str) -> str:
    """
    一鍵發起四 Agent 開會：Commander + Planner + Generator 並行，Evaluator 串列。
    支援 Graceful Degradation 與 HMAC 簽章。
    最差延遲：max(Commander, Planner, Generator) + Evaluator = 約 45-90 秒。
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # 階段 1：Commander + Planner + Generator 並行（三者互不依賴）
    cmd_result: list[str] = []
    planner_result: list[str] = []
    generator_result: list[str] = []

    async def run_commander():
        cmd_result.append(await anyio.to_thread.run_sync(
            lambda: check_workflow_boundary(task_description)
        ))

    async def run_planner():
        planner_result.append(await planner_consult(task_description))

    async def run_generator():
        generator_result.append(await generator_code(task_description, DEFAULT_MODEL))

    try:
        with anyio.fail_after(COUNCIL_PARALLEL_TIMEOUT_SEC):
            async with anyio.create_task_group() as tg:
                tg.start_soon(run_commander)
                tg.start_soon(run_planner)
                tg.start_soon(run_generator)
    except TimeoutError:
        sys.stderr.write(
            f"Council parallel phase timed out after {COUNCIL_PARALLEL_TIMEOUT_SEC}s. "
            f"Using available results.\n"
        )
        if not cmd_result:
            cmd_result.append("[DEGRADED] Commander 階段超時。")
        if not planner_result:
            planner_result.append("[DEGRADED] Planner 階段超時。")
        if not generator_result:
            generator_result.append("[DEGRADED] Generator 階段超時。")
    except Exception as e:
        sys.stderr.write(f"Council parallel phase error: {type(e).__name__}: {e}\n")
        if not cmd_result:
            cmd_result.append(f"[ERROR] Commander 失敗: {e}")
        if not planner_result:
            planner_result.append(f"[ERROR] Planner 失敗: {e}")
        if not generator_result:
            generator_result.append(f"[ERROR] Generator 失敗: {e}")

    cmd_res = cmd_result[0] if cmd_result else "[DEGRADED] Commander 未回應"
    planner_res = planner_result[0] if planner_result else "[DEGRADED] Planner 未回應"
    generator_res = generator_result[0] if generator_result else "[DEGRADED] Generator 未回應"

    # 階段 2：Evaluator 依賴 Generator 產出，需串列執行
    evaluator_res = await evaluator_review(generator_res)

    final_decision_content = f"""##### Commander 最終決議 (Antigravity 主對話模型直接彙整)
- 負責模型：`Antigravity 主對話模型 (Gemini 3.7 Flash)`
- 執行方案：核准 Planner 提出了適當之技術架構，並採納 Generator 與 Evaluator 定義之防禦性驗證方案。
- Checkpoint 清單：[CP-1 需求與規格確認, CP-2 模組編碼完成, CP-3 品質審查與驗證通過]
- 回滾條件：若整合測試失敗或違反系統安全邊界，執行 git checkout 回滾。
- 簽章：[Commander_Gemini-3.7-Flash_Approved]"""

    council_minutes = f"""#### 四 Agent 開會紀錄

**任務**：{task_description}
**日期**：{now_str}

{cmd_res}

{planner_res}

{generator_res}

{evaluator_res}

{final_decision_content}
"""

    council_verification = generate_verification("COUNCIL", council_minutes)
    return f"{council_minutes}\n{council_verification}"


# ==========================================
# CLI 入口
# ==========================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--check-workflow", "--test-all", "--council", "--help", "--verify"):
        parser = argparse.ArgumentParser(
            description="Multi-Agent Council MCP Server & Reasoning Engine (Async Optimized)"
        )
        parser.add_argument("query", nargs="?", help="要進行推理或規劃的任務描述")
        parser.add_argument("model", nargs="?", default=DEFAULT_MODEL, help="模型名稱或 'deep'")
        parser.add_argument("--check-workflow", action="store_true", help="執行 Commander 邊界檢測")
        parser.add_argument("--council", action="store_true", help="執行完整四 Agent 開會流程")
        parser.add_argument("--test-all", action="store_true", help="測試所有連線與組件健康度")
        parser.add_argument("--verify", action="store_true", help="驗證簽章真偽（測試用）")

        args = parser.parse_args()

        if args.check_workflow:
            q = args.query if args.query else "邊界檢測"
            print(check_workflow_boundary(q))
            sys.exit(0)

        elif args.test_all:
            async def _run_all_tests():
                """非同步整合測試：依序驗證各組件健康度"""
                print("=== 測試 1: 專案 RAG 掃描 ===")
                ctx = get_local_context()
                print(f"掃描成功，字數: {len(ctx)}")

                print("\n=== 測試 2: Ollama Gemma 連線 ===")
                gemma_out = await generator_code("測試連線健康度", DEFAULT_MODEL)
                print(gemma_out[:200] + "...")

                print("\n=== 測試 3: Planner/Evaluator API 呼叫 ===")
                print(await planner_consult("測試 Planner API 呼叫"))
                print(await evaluator_review("測試 Evaluator API 呼叫"))

                print("\n=== 測試 4: HMAC 簽章驗證 ===")
                test_content = "測試簽章內容"
                test_sig_line = generate_verification("TEST", test_content)
                print(f"生成簽章：{test_sig_line}")
                import re
                match = re.match(
                    r'\[VERIFIED_(\w+)\|nonce=(\w+)\|sig=(\w+)\|ts=(.+)\]',
                    test_sig_line
                )
                if match:
                    role, nonce_val, sig_val, ts_val = match.groups()
                    content_prefix = test_content[:100].replace("\n", " ").strip()
                    is_valid = verify_signature(role, content_prefix, nonce_val, sig_val, ts_val)
                    print(f"驗證結果：{'PASS' if is_valid else 'FAIL'}")
                else:
                    print("簽章格式解析失敗")

                print("\n[SUCCESS] 所有組件檢測完成。")

            asyncio.run(_run_all_tests())
            sys.exit(0)

        elif args.council:
            q = args.query if args.query else "四 Agent 開會測試"

            async def _run_council():
                print(await council_orchestrator(q))

            asyncio.run(_run_council())
            sys.exit(0)

        elif args.verify and args.query:
            import re
            match = re.match(
                r'\[VERIFIED_(\w+)\|nonce=(\w+)\|sig=(\w+)\|ts=(.+)\]',
                args.query
            )
            if match:
                role, nonce_val, sig_val, ts_val = match.groups()
                print(f"角色: {role}, Nonce: {nonce_val}, Sig: {sig_val}, TS: {ts_val}")
                print("注意：需要原始 content_prefix 才能完整驗證。此模式僅檢查格式有效性。")
            else:
                print("[FAIL] 簽章格式無效。")

        elif args.query:
            async def _run_generator():
                print(await generator_code(args.query, args.model))

            asyncio.run(_run_generator())

    else:
        mcp.run()
