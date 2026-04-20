"""
NotebookLM MCP Server v2.0 - 主入口
整合 RPC Client、서비스層與 FastMCP 工具定義。
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from .core.auth_manager import AuthManager
from .core.localization import Localization
from .core.rpc_client import NotebookLMRPCClient
from .services.notebook_service import NotebookService
from .services.research_service import ResearchService
from .services.studio_service import StudioService

# 設定 logger 輸出至 stderr（MCP 相容）
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# === 全域服務實例 ===
_rpc_client: Optional[NotebookLMRPCClient] = None
_notebook_svc: Optional[NotebookService] = None
_research_svc: Optional[ResearchService] = None
_studio_svc: Optional[StudioService] = None
_auth_manager: Optional[AuthManager] = None
_localization: Optional[Localization] = None

mcp = FastMCP(
    name="NotebookLM MCP Server v2.0",
    instructions=(
        "NotebookLM 知識庫管理工具。支援筆記本管理、來源新增、"
        "深層研究、音訊/影片/簡報生成。所有輸出預設使用繁體中文。"
    ),
)


def _get_services() -> tuple:
    """取得或初始化服務實例"""
    global _rpc_client, _notebook_svc, _research_svc, _studio_svc
    global _auth_manager, _localization

    if _rpc_client is None:
        auth = _get_auth_manager()
        creds = auth.load_credentials()
        if not creds:
            raise RuntimeError(
                "尚未認證。請先執行：python -m notebooklm_mcp_v2.setup_auth"
            )
        cookies, csrf_token = creds
        _localization = Localization(default_language="zh-TW")
        _rpc_client = NotebookLMRPCClient(
            cookies=cookies,
            csrf_token=csrf_token,
            language="zh-TW",
        )
        _notebook_svc = NotebookService(_rpc_client)
        _research_svc = ResearchService(_rpc_client)
        _studio_svc = StudioService(_rpc_client, _localization)

    return _notebook_svc, _research_svc, _studio_svc


def _get_auth_manager() -> AuthManager:
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


# ============================================================
# MCP 工具定義
# ============================================================

# --- 認證與健康檢查 ---

@mcp.tool()
async def healthcheck() -> Dict[str, Any]:
    """檢查 MCP Server 連線狀態與認證情況"""
    auth = _get_auth_manager()
    creds = auth.load_credentials()
    return {
        "status": "healthy" if creds else "needs_auth",
        "authenticated": bool(creds),
        "version": "2.0.0",
        "language": "zh-TW",
        "message": (
            "已驗證，可正常使用" if creds
            else "請先執行 setup_auth 完成認證"
        ),
    }


@mcp.tool()
async def refresh_auth(
    chrome_profile_dir: str,
) -> Dict[str, Any]:
    """
    從 Chrome Profile 重新提取 cookies 並刷新 CSRF Token。

    Args:
        chrome_profile_dir: Chrome 使用者資料目錄路徑（需先關閉 Chrome）
    """
    try:
        auth = _get_auth_manager()
        cookies = auth.extract_cookies_from_chrome(chrome_profile_dir)
        if not cookies:
            return {"status": "error", "message": "無法提取 cookies，請確認 Chrome Profile 路徑"}

        csrf_token = await auth.fetch_csrf_token(cookies)
        auth.save_credentials(cookies, csrf_token)

        # 重置服務實例以使用新認證
        global _rpc_client, _notebook_svc, _research_svc, _studio_svc
        _rpc_client = None
        _notebook_svc = None
        _research_svc = None
        _studio_svc = None

        return {
            "status": "success",
            "cookies_count": len(cookies),
            "message": "認證資訊已更新並加密儲存",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# --- 筆記本管理 ---

@mcp.tool()
async def notebook_list() -> Dict[str, Any]:
    """列出所有 NotebookLM 筆記本"""
    nb_svc, _, _ = _get_services()
    result = await nb_svc.list_notebooks()
    if result.error:
        return {"status": "error", "message": result.error}
    return {"status": "success", "data": result.data}


@mcp.tool()
async def notebook_get(notebook_id: str) -> Dict[str, Any]:
    """
    取得筆記本詳情，包含所有來源清單。

    Args:
        notebook_id: 筆記本 UUID
    """
    nb_svc, _, _ = _get_services()
    result = await nb_svc.get_notebook(notebook_id)
    if result.error:
        return {"status": "error", "message": result.error}
    return {"status": "success", "data": result.data}


@mcp.tool()
async def notebook_create(title: str) -> Dict[str, Any]:
    """
    建立新筆記本。

    Args:
        title: 筆記本標題
    """
    nb_svc, _, _ = _get_services()
    result = await nb_svc.create_notebook(title)
    if result.error:
        return {"status": "error", "message": result.error}
    return {"status": "success", "data": result.data, "message": f"已建立筆記本：{title}"}


@mcp.tool()
async def notebook_query(
    notebook_id: str,
    query: str,
    source_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    對筆記本進行 AI 查詢（使用 NotebookLM 串流端點）。

    Args:
        notebook_id: 目標筆記本 ID
        query: 查詢問題
        source_ids: 可選，限定查詢的來源 ID 清單
    """
    nb_svc, _, _ = _get_services()
    result = await nb_svc.query(notebook_id, query, source_ids)
    if result.error:
        return {"status": "error", "message": result.error}
    return {"status": "success", "data": result.data}


@mcp.tool()
async def notebook_query_expert(
    notebook_id: str,
    query: str,
    mode: str = "technical",
    source_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    技術文件專家模式查詢（自動注入技術指示詞）。

    Args:
        notebook_id: 目標筆記本 ID
        query: 查詢問題
        mode: technical / api / comparison
        source_ids: 可選，限定查詢的來源 ID 清單
    """
    from .core.localization import Localization
    loc = Localization()
    expert_query = loc.inject_expert_prompt(query, mode=mode)

    nb_svc, _, _ = _get_services()
    result = await nb_svc.query(notebook_id, expert_query, source_ids)
    if result.error:
        return {"status": "error", "message": result.error}
    return {
        "status": "success",
        "mode": mode,
        "data": result.data,
    }


# --- 來源管理（Librarian 功能）---

@mcp.tool()
async def source_add_url(
    notebook_id: str,
    url: str,
) -> Dict[str, Any]:
    """
    新增 URL 來源至筆記本（自動偵測 YouTube）。

    Args:
        notebook_id: 目標筆記本 ID
        url: 要新增的 URL（支援一般網頁與 YouTube）
    """
    nb_svc, _, _ = _get_services()
    result = await nb_svc.add_source_url(notebook_id, url)
    if result.error:
        return {"status": "error", "message": result.error}
    return {"status": "success", "data": result.data, "message": f"已新增來源：{url}"}


@mcp.tool()
async def source_add_text(
    notebook_id: str,
    title: str,
    content: str,
) -> Dict[str, Any]:
    """
    新增純文字來源至筆記本。

    Args:
        notebook_id: 目標筆記本 ID
        title: 來源標題
        content: 文字內容
    """
    nb_svc, _, _ = _get_services()
    result = await nb_svc.add_source_text(notebook_id, title, content)
    if result.error:
        return {"status": "error", "message": result.error}
    return {"status": "success", "data": result.data, "message": f"已新增文字來源：{title}"}


@mcp.tool()
async def source_add_batch(
    notebook_id: str,
    urls: List[str],
) -> Dict[str, Any]:
    """
    批次新增多個 URL 來源。

    Args:
        notebook_id: 目標筆記本 ID
        urls: URL 清單（最多 20 個）
    """
    nb_svc, _, _ = _get_services()
    results = []
    errors = []

    for url in urls[:20]:
        result = await nb_svc.add_source_url(notebook_id, url)
        if result.error:
            errors.append({"url": url, "error": result.error})
        else:
            results.append({"url": url, "status": "success"})
        await asyncio.sleep(0.5)  # 避免速率限制

    return {
        "status": "success" if not errors else "partial",
        "added": len(results),
        "failed": len(errors),
        "errors": errors,
    }


# --- 研究功能（Librarian 深層研究）---

@mcp.tool()
async def research_start(
    notebook_id: str,
    query: str,
    mode: str = "fast",
    source_type: str = "web",
) -> Dict[str, Any]:
    """
    啟動研究任務。

    Args:
        notebook_id: 目標筆記本 ID
        query: 研究主題
        mode: "fast"（10-30秒）或 "deep"（3-5分鐘，僅支援 web）
        source_type: "web" 或 "drive"
    """
    _, research_svc, _ = _get_services()
    if mode == "deep":
        result = await research_svc.start_deep_research(notebook_id, query)
    else:
        result = await research_svc.start_fast_research(
            notebook_id, query, source_type
        )
    if result.error:
        return {"status": "error", "message": result.error}
    return {
        "status": "success",
        "mode": mode,
        "data": result.data,
        "message": f"已啟動{mode}研究：{query}",
    }


@mcp.tool()
async def research_status(notebook_id: str) -> Dict[str, Any]:
    """輪詢研究任務進度"""
    _, research_svc, _ = _get_services()
    result = await research_svc.poll_research(notebook_id)
    if result.error:
        return {"status": "error", "message": result.error}
    parsed = ResearchService.parse_research_results(result.data)
    return {"status": "success", "research": parsed}


@mcp.tool()
async def research_wait_and_import(
    notebook_id: str,
    max_sources: int = 10,
    max_wait: int = 300,
) -> Dict[str, Any]:
    """
    等待研究完成並自動匯入所有發現的來源。

    Args:
        max_sources: 最多匯入幾個來源
        max_wait: 最長等待秒數
    """
    _, research_svc, _ = _get_services()
    wait_result = await research_svc.wait_for_research(
        notebook_id, max_wait=max_wait
    )
    if wait_result.error:
        return {"status": "error", "message": wait_result.error}

    parsed = ResearchService.parse_research_results(wait_result.data)
    sources = parsed.get("sources", [])[:max_sources]
    task_id = parsed.get("task_id", "")

    if not sources or not task_id:
        return {
            "status": "success",
            "message": "研究完成但無可匯入的來源",
            "research": parsed,
        }

    import_result = await research_svc.import_sources(
        notebook_id, task_id, sources
    )
    if import_result.error:
        return {"status": "error", "message": import_result.error}

    return {
        "status": "success",
        "imported": len(sources),
        "message": f"已匯入 {len(sources)} 個來源",
        "sources": [s.get("title", s.get("url", "")) for s in sources],
    }


# --- Studio 多模態生成 ---

@mcp.tool()
async def studio_create_audio(
    notebook_id: str,
    source_ids: List[str],
    focus_prompt: str = "",
    format_name: str = "deep_dive",
    length_name: str = "default",
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    生成音訊導讀（Podcast）。預設繁體中文。

    Args:
        source_ids: 參與生成的來源 ID
        focus_prompt: 聚焦指示（例：請著重討論 GNN 模型架構）
        format_name: deep_dive / brief / critique / debate
        length_name: short / default / long
        language: 語言代碼，預設 zh-TW
    """
    _, _, studio_svc = _get_services()
    result = await studio_svc.create_audio_overview(
        notebook_id=notebook_id,
        source_ids=source_ids,
        focus_prompt=focus_prompt,
        format_name=format_name,
        length_name=length_name,
        language=language,
    )
    if result.error:
        return {"status": "error", "message": result.error}
    return {
        "status": "success",
        "message": "音訊導讀生成任務已提交，請使用 studio_poll_status 確認進度",
        "data": result.data,
    }


@mcp.tool()
async def studio_create_slides(
    notebook_id: str,
    source_ids: List[str],
    focus_prompt: str = "",
    format_name: str = "detailed",
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    生成簡報（Slide Deck）。預設繁體中文。

    Args:
        source_ids: 參與生成的來源 ID
        focus_prompt: 聚焦指示
        format_name: detailed / presenter
        language: 語言代碼，預設 zh-TW
    """
    _, _, studio_svc = _get_services()
    result = await studio_svc.create_slide_deck(
        notebook_id=notebook_id,
        source_ids=source_ids,
        focus_prompt=focus_prompt,
        format_name=format_name,
        language=language,
    )
    if result.error:
        return {"status": "error", "message": result.error}
    return {
        "status": "success",
        "message": "簡報生成任務已提交",
        "data": result.data,
    }


@mcp.tool()
async def studio_poll_status(notebook_id: str) -> Dict[str, Any]:
    """查詢 Studio 生成進度與下載連結"""
    _, _, studio_svc = _get_services()
    result = await studio_svc.poll_studio_status(notebook_id)
    if result.error:
        return {"status": "error", "message": result.error}
    artifacts = StudioService.parse_studio_status(result.data)
    return {
        "status": "success",
        "artifacts": artifacts,
        "count": len(artifacts),
    }


@mcp.tool()
async def studio_wait(
    notebook_id: str,
    max_wait: int = 600,
) -> Dict[str, Any]:
    """
    等待所有 Studio 任務完成並返回下載連結。

    Args:
        max_wait: 最長等待秒數（影片生成需較長時間）
    """
    _, _, studio_svc = _get_services()
    result = await studio_svc.wait_for_studio(
        notebook_id, max_wait=max_wait
    )
    if result.error:
        return {"status": "error", "message": result.error}
    artifacts = StudioService.parse_studio_status(result.data)
    return {
        "status": "success",
        "artifacts": artifacts,
        "message": f"所有 Studio 任務已完成，共 {len(artifacts)} 個輸出",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
