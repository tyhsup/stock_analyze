"""
研究服務：Fast Research / Deep Research / 結果輪詢 / 來源匯入。
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ..core.rpc_client import NotebookLMRPCClient, RPCResponse

logger = logging.getLogger(__name__)


class ResearchService:
    """NotebookLM 深層研究自動化服務"""

    def __init__(self, rpc: NotebookLMRPCClient):
        self.rpc = rpc

    async def start_fast_research(
        self,
        notebook_id: str,
        query: str,
        source_type: str = "web",
    ) -> RPCResponse:
        """
        啟動快速研究（約 10-30 秒，~10 個來源）。

        Args:
            notebook_id: 目標筆記本 ID
            query: 研究主題
            source_type: "web" 或 "drive"
        """
        source_code = 1 if source_type == "web" else 2
        return await self.rpc.execute(
            self.rpc.RPC.FAST_RESEARCH,
            [[query, source_code], None, 1, notebook_id],
            notebook_id=notebook_id,
        )

    async def start_deep_research(
        self,
        notebook_id: str,
        query: str,
    ) -> RPCResponse:
        """
        啟動深度研究（約 3-5 分鐘，~40 個來源，含 AI 報告）。
        僅支援 Web 來源。
        """
        return await self.rpc.execute(
            self.rpc.RPC.DEEP_RESEARCH,
            [None, [1], [query, 1], 5, notebook_id],
            notebook_id=notebook_id,
        )

    async def poll_research(
        self, notebook_id: str
    ) -> RPCResponse:
        """輪詢研究進度（status: 1=進行中, 2=已完成）"""
        return await self.rpc.execute(
            self.rpc.RPC.POLL_RESEARCH,
            [None, None, notebook_id],
            notebook_id=notebook_id,
        )

    async def wait_for_research(
        self,
        notebook_id: str,
        poll_interval: int = 15,
        max_wait: int = 300,
    ) -> RPCResponse:
        """
        等待研究完成，自動輪詢直到狀態為 completed。

        Args:
            poll_interval: 輪詢間隔（秒）
            max_wait: 最長等待時間（秒）
        """
        elapsed = 0
        while elapsed < max_wait:
            result = await self.poll_research(notebook_id)
            if result.error:
                return result

            # 檢查狀態
            if result.data:
                try:
                    status = self._extract_status(result.data)
                    if status == 2:
                        logger.info("研究任務已完成")
                        return result
                    logger.info(
                        f"研究進行中... ({elapsed}s/{max_wait}s)"
                    )
                except Exception:
                    pass

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        return RPCResponse(
            error=f"研究等待逾時（{max_wait}s）",
            data=result.data if result else None,
        )

    @staticmethod
    def _extract_status(data: Any) -> int:
        """從輪詢回應中提取狀態碼"""
        if isinstance(data, list) and data:
            # 結構: [[[task_id, [...], status], ...]]
            first = data
            while isinstance(first, list) and first:
                if (
                    isinstance(first, list)
                    and len(first) >= 3
                    and isinstance(first[-1], int)
                ):
                    return first[-1]
                first = first[0]
        return 0

    async def import_sources(
        self,
        notebook_id: str,
        task_id: str,
        sources: List[Dict[str, Any]],
    ) -> RPCResponse:
        """
        將研究發現的來源匯入筆記本。

        Args:
            sources: 來源列表，每項需包含 url, title, type
        """
        source_params = []
        for src in sources:
            src_type = src.get("type", "web")
            if src_type == "web":
                source_params.append([
                    None, None,
                    [src["url"], src.get("title", "")],
                    None, None, None, None, None, None, None, 2,
                ])
            else:
                source_params.append([
                    [src.get("doc_id", ""), src.get("mime", ""), None,
                     src.get("title", "")],
                    None, None, None, None, None, None, None, None, None, 1,
                ])

        return await self.rpc.execute(
            self.rpc.RPC.IMPORT_RESEARCH,
            [None, [1], task_id, notebook_id, source_params],
            notebook_id=notebook_id,
        )

    @staticmethod
    def parse_research_results(
        data: Any,
    ) -> Dict[str, Any]:
        """
        解析研究結果，提取來源清單與摘要。

        Returns:
            {
                "task_id": str,
                "query": str,
                "status": int,
                "sources": [{"url": str, "title": str, "description": str}],
                "summary": str,
            }
        """
        result: Dict[str, Any] = {
            "task_id": "",
            "query": "",
            "status": 0,
            "sources": [],
            "summary": "",
        }

        if not data or not isinstance(data, list):
            return result

        try:
            # 嘗試解析嵌套結構
            entry = data
            while isinstance(entry, list) and isinstance(entry[0], list):
                entry = entry[0]

            if len(entry) >= 3:
                result["task_id"] = entry[0] if isinstance(entry[0], str) else ""
                result["status"] = (
                    entry[-1] if isinstance(entry[-1], int) else 0
                )

                # 內部資料區塊
                if len(entry) >= 2 and isinstance(entry[1], list):
                    inner = entry[1]
                    if len(inner) >= 2 and isinstance(inner[0], str):
                        result["query"] = inner[0]

                    # 來源陣列
                    if len(inner) >= 4 and isinstance(inner[3], list):
                        for src in inner[3]:
                            if isinstance(src, list) and len(src) >= 3:
                                result["sources"].append({
                                    "url": src[0] if src[0] else "",
                                    "title": src[1] if src[1] else "",
                                    "description": (
                                        src[2] if len(src) > 2 else ""
                                    ),
                                })

                    # 摘要
                    if len(inner) >= 5 and isinstance(inner[4], str):
                        result["summary"] = inner[4]

        except (IndexError, TypeError) as e:
            logger.warning(f"研究結果解析異常: {e}")

        return result
