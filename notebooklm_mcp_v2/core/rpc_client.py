"""
NotebookLM RPC Client v2.0
基於 batchexecute 協議的高效 RPC 通訊模組。
RPC ID 參考：jacob-bd/notebooklm-mcp-cli API_REFERENCE.md
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RPCResponse(BaseModel):
    """RPC 回應封裝"""
    data: Any = None
    error: Optional[str] = None
    raw_text: Optional[str] = None


class NotebookLMRPCClient:
    """
    batchexecute RPC Client for NotebookLM。
    透過直接呼叫 Google 內部 RPC 端點實現毫秒級操作。
    """

    BASE_URL = "https://notebooklm.google.com"
    BATCHEXECUTE_URL = (
        "https://notebooklm.google.com/_/LabsTailwindUi/data/batchexecute"
    )
    QUERY_URL = (
        "https://notebooklm.google.com/_/LabsTailwindUi/data/"
        "google.internal.labs.tailwind.orchestration.v1."
        "LabsTailwindOrchestrationService/GenerateFreeFormStreamed"
    )

    # === RPC ID 對照表 ===
    class RPC:
        LIST_NOTEBOOKS = "wXbhsf"
        GET_NOTEBOOK = "rLM1Ne"
        CREATE_NOTEBOOK = "CCqFvf"
        UPDATE_NOTEBOOK = "s0tc2d"
        DELETE_NOTEBOOK = "WWINqb"
        ADD_SOURCE = "izAoDd"
        GET_SOURCE_DETAILS = "hizoJc"
        CHECK_SOURCE_FRESHNESS = "yR9Yof"
        SYNC_DRIVE_SOURCE = "FLmJqe"
        DELETE_SOURCE = "tGMBJ"
        FAST_RESEARCH = "Ljjv0c"
        DEEP_RESEARCH = "QA9ei"
        POLL_RESEARCH = "e3bVqc"
        IMPORT_RESEARCH = "LBwxtb"
        CREATE_STUDIO = "R7cb6c"
        POLL_STUDIO = "gArtLc"
        DELETE_STUDIO = "V5N4be"
        RENAME_STUDIO = "rc3d8d"
        REVISE_SLIDES = "KmcKPe"
        GET_CONVERSATIONS = "hPTbtc"

    def __init__(
        self,
        cookies: Dict[str, str],
        csrf_token: str,
        language: str = "zh-TW",
    ):
        self.cookies = cookies
        self.csrf_token = csrf_token
        self.language = language
        self._request_counter = 0
        self._build_label: Optional[str] = None

        self.client = httpx.AsyncClient(
            headers={
                "Content-Type": (
                    "application/x-www-form-urlencoded;charset=utf-8"
                ),
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Referer": self.BASE_URL,
                "Origin": self.BASE_URL,
                "X-Same-Domain": "1",
            },
            cookies=self.cookies,
            timeout=60.0,
            follow_redirects=True,
        )

    async def _get_request_id(self) -> int:
        """產生遞增的 request ID"""
        self._request_counter += 1
        return self._request_counter * 100000

    async def execute(
        self,
        rpc_id: str,
        params: List[Any],
        notebook_id: Optional[str] = None,
    ) -> RPCResponse:
        """
        執行 batchexecute RPC 呼叫。

        Args:
            rpc_id: RPC 方法 ID（見 RPC 類別常數）
            params: RPC 參數列表
            notebook_id: 若提供，將設定 source-path
        """
        f_req = [[[rpc_id, json.dumps(params), None, "generic"]]]
        req_id = await self._get_request_id()

        query_params: Dict[str, Any] = {
            "rpcids": rpc_id,
            "hl": self.language,
            "_reqid": req_id,
            "rt": "c",
        }
        if notebook_id:
            query_params["source-path"] = f"/notebook/{notebook_id}"

        data = {
            "f.req": json.dumps(f_req),
            "at": self.csrf_token,
        }

        try:
            response = await self.client.post(
                self.BATCHEXECUTE_URL,
                data=data,
                params=query_params,
            )

            if response.status_code == 401:
                return RPCResponse(
                    error="認證已過期，請重新執行 login"
                )
            if response.status_code != 200:
                return RPCResponse(
                    error=f"HTTP {response.status_code}: {response.text[:200]}"
                )

            return self._parse_response(response.text)

        except httpx.TimeoutException:
            return RPCResponse(error="RPC 請求逾時（60s）")
        except Exception as e:
            logger.error(f"RPC 執行失敗 [{rpc_id}]: {e}")
            return RPCResponse(error=str(e))

    def _parse_response(self, raw_text: str) -> RPCResponse:
        """
        解析 Google batchexecute 回應格式。
        回應前綴為 )]}' 反 XSSI 標記，後接多段 JSON。
        """
        text = raw_text
        if text.startswith(")]}'"):
            text = text[4:].strip()

        try:
            # 提取第一個完整的 JSON 陣列
            # 格式：<byte_count>\n<json_array>
            lines = text.split("\n")
            json_parts = []
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                # 跳過純數字行（byte count）
                if line.isdigit():
                    i += 1
                    continue
                if line:
                    json_parts.append(line)
                i += 1

            if not json_parts:
                return RPCResponse(error="空回應", raw_text=raw_text[:500])

            combined = "\n".join(json_parts)
            outer = json.loads(combined)

            # 標準結構: [["wrb.fr", rpc_id, "result_json", ...]]
            if outer and isinstance(outer, list):
                for entry in outer:
                    if (
                        isinstance(entry, list)
                        and len(entry) > 2
                        and isinstance(entry[0], str)
                        and entry[0] == "wrb.fr"
                    ):
                        if entry[2]:
                            result = json.loads(entry[2])
                            return RPCResponse(data=result)

                # 備用解析：巢狀陣列格式
                if (
                    isinstance(outer[0], list)
                    and isinstance(outer[0][0], list)
                ):
                    entry = outer[0][0]
                    if len(entry) > 2 and entry[2]:
                        result = json.loads(entry[2])
                        return RPCResponse(data=result)

            return RPCResponse(
                data=outer,
                error=None,
                raw_text=raw_text[:500],
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失敗: {e}")
            return RPCResponse(
                error=f"回應解析失敗: {e}",
                raw_text=raw_text[:500],
            )

    async def query_notebook(
        self,
        notebook_id: str,
        query: str,
        source_ids: Optional[List[str]] = None,
        conversation_id: Optional[str] = None,
    ) -> RPCResponse:
        """
        對筆記本發送查詢（使用 streaming 端點，非 batchexecute）。
        """
        sources_param = []
        if source_ids:
            sources_param = [
                [[[sid]]] for sid in source_ids
            ]

        params = [
            sources_param,
            query,
            None,
            [2, None, [1]],
            conversation_id,
        ]

        f_req = [None, json.dumps(params)]

        data = {
            "f.req": json.dumps(f_req),
            "at": self.csrf_token,
        }

        query_params = {
            "hl": self.language,
            "_reqid": await self._get_request_id(),
            "rt": "c",
            "source-path": f"/notebook/{notebook_id}",
        }

        try:
            response = await self.client.post(
                self.QUERY_URL,
                data=data,
                params=query_params,
            )

            if response.status_code != 200:
                return RPCResponse(
                    error=f"Query HTTP {response.status_code}"
                )

            # 串流回應解析
            return self._parse_streaming_response(response.text)

        except Exception as e:
            logger.error(f"Query 執行失敗: {e}")
            return RPCResponse(error=str(e))

    def _parse_streaming_response(self, raw_text: str) -> RPCResponse:
        """解析串流查詢回應，提取最終答案"""
        text = raw_text
        if text.startswith(")]}'"):
            text = text[4:].strip()

        chunks = []
        for line in text.split("\n"):
            line = line.strip()
            if not line or line.isdigit():
                continue
            try:
                parsed = json.loads(line)
                chunks.append(parsed)
            except json.JSONDecodeError:
                continue

        # 合併所有文字片段為最終回應
        answer_parts = []
        for chunk in chunks:
            if isinstance(chunk, list):
                self._extract_text_from_chunk(chunk, answer_parts)

        answer = "".join(answer_parts).strip()
        return RPCResponse(data={"answer": answer, "chunks": len(chunks)})

    @staticmethod
    def _extract_text_from_chunk(
        chunk: Any, parts: List[str]
    ) -> None:
        """遞迴提取 chunk 中的文字片段"""
        if isinstance(chunk, str) and len(chunk) > 5:
            parts.append(chunk)
        elif isinstance(chunk, list):
            for item in chunk:
                NotebookLMRPCClient._extract_text_from_chunk(item, parts)

    async def close(self) -> None:
        """關閉 HTTP 連線"""
        await self.client.aclose()
