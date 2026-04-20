"""
筆記本與來源管理服務
封裝 RPC Client 的 Notebook/Source CRUD 操作。
"""

import logging
from typing import Any, Dict, List, Optional

from ..core.rpc_client import NotebookLMRPCClient, RPCResponse

logger = logging.getLogger(__name__)


class NotebookService:
    """筆記本與來源管理的高階服務層"""

    def __init__(self, rpc: NotebookLMRPCClient):
        self.rpc = rpc

    async def list_notebooks(self) -> RPCResponse:
        """列出所有筆記本"""
        return await self.rpc.execute(
            self.rpc.RPC.LIST_NOTEBOOKS,
            [None, 1, None, [2]],
        )

    async def get_notebook(self, notebook_id: str) -> RPCResponse:
        """取得筆記本詳情（含來源清單）"""
        return await self.rpc.execute(
            self.rpc.RPC.GET_NOTEBOOK,
            [notebook_id, None, [2], None, 0],
            notebook_id=notebook_id,
        )

    async def create_notebook(self, title: str) -> RPCResponse:
        """建立新筆記本"""
        return await self.rpc.execute(
            self.rpc.RPC.CREATE_NOTEBOOK,
            [title, None, None, [2],
             [1, None, None, None, None, None, None, None, None, None, [1]]],
        )

    async def rename_notebook(
        self, notebook_id: str, new_title: str
    ) -> RPCResponse:
        """重新命名筆記本"""
        return await self.rpc.execute(
            self.rpc.RPC.UPDATE_NOTEBOOK,
            [notebook_id, [[None, None, None, [None, new_title]]]],
            notebook_id=notebook_id,
        )

    async def delete_notebook(self, notebook_id: str) -> RPCResponse:
        """刪除筆記本（不可逆操作）"""
        return await self.rpc.execute(
            self.rpc.RPC.DELETE_NOTEBOOK,
            [[notebook_id], [2]],
        )

    async def configure_chat(
        self,
        notebook_id: str,
        goal: int = 1,
        custom_prompt: Optional[str] = None,
        response_length: int = 1,
    ) -> RPCResponse:
        """
        設定筆記本的聊天行為。

        Args:
            goal: 1=預設, 2=自訂(需 custom_prompt), 3=學習指南
            response_length: 1=預設, 4=較長, 5=較短
        """
        goal_param = [goal]
        if goal == 2 and custom_prompt:
            goal_param = [goal, custom_prompt]

        return await self.rpc.execute(
            self.rpc.RPC.UPDATE_NOTEBOOK,
            [
                notebook_id,
                [[None, None, None, None, None, None, None,
                  [goal_param, [response_length]]]],
            ],
            notebook_id=notebook_id,
        )

    # === 來源管理 ===

    async def add_source_url(
        self, notebook_id: str, url: str
    ) -> RPCResponse:
        """
        新增 URL 來源（自動偵測 YouTube）。
        """
        is_youtube = "youtube.com" in url or "youtu.be" in url

        if is_youtube:
            source_data = [
                None, None, None, None, None, None, None,
                [url], None, None, 1,
            ]
        else:
            source_data = [
                None, None, [url], None, None, None, None,
                None, None, None, 1,
            ]

        return await self.rpc.execute(
            self.rpc.RPC.ADD_SOURCE,
            [[[source_data]], notebook_id, [2],
             [1, None, None, None, None, None, None, None, None, None, [1]]],
            notebook_id=notebook_id,
        )

    async def add_source_text(
        self, notebook_id: str, title: str, content: str
    ) -> RPCResponse:
        """新增純文字來源"""
        source_data = [
            None, [title, content], None, 2,
            None, None, None, None, None, None, 1,
        ]
        return await self.rpc.execute(
            self.rpc.RPC.ADD_SOURCE,
            [[[source_data]], notebook_id, [2],
             [1, None, None, None, None, None, None, None, None, None, [1]]],
            notebook_id=notebook_id,
        )

    async def get_source_details(
        self, source_id: str
    ) -> RPCResponse:
        """取得來源詳細資訊"""
        return await self.rpc.execute(
            self.rpc.RPC.GET_SOURCE_DETAILS,
            [[source_id], [2], [2]],
        )

    async def delete_source(self, source_id: str) -> RPCResponse:
        """刪除來源（不可逆操作）"""
        return await self.rpc.execute(
            self.rpc.RPC.DELETE_SOURCE,
            [[[source_id]], [2]],
        )

    async def query(
        self,
        notebook_id: str,
        query_text: str,
        source_ids: Optional[List[str]] = None,
    ) -> RPCResponse:
        """對筆記本發送查詢"""
        return await self.rpc.query_notebook(
            notebook_id=notebook_id,
            query=query_text,
            source_ids=source_ids,
        )
