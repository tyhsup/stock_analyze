"""
Studio 服務：音訊導讀、影片、簡報、心智圖生成與狀態輪詢。
預設語言：繁體中文 (zh-TW)
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ..core.localization import Localization
from ..core.rpc_client import NotebookLMRPCClient, RPCResponse

logger = logging.getLogger(__name__)


class StudioService:
    """NotebookLM Studio 多模態內容生成服務"""

    def __init__(
        self,
        rpc: NotebookLMRPCClient,
        localization: Optional[Localization] = None,
    ):
        self.rpc = rpc
        self.loc = localization or Localization(default_language="zh-TW")

    async def create_audio_overview(
        self,
        notebook_id: str,
        source_ids: List[str],
        focus_prompt: str = "",
        format_name: str = "deep_dive",
        length_name: str = "default",
        language: Optional[str] = None,
    ) -> RPCResponse:
        """
        生成音訊導讀（Podcast）。

        Args:
            notebook_id: 目標筆記本 ID
            source_ids: 參與生成的來源 ID 清單
            focus_prompt: 聚焦指示（如「請著重說明 API 設計原則」）
            format_name: deep_dive / brief / critique / debate
            length_name: short / default / long
            language: BCP-47 語言代碼，預設使用 zh-TW
        """
        lang = self.loc.get_language_code(language)
        fmt = self.loc.AUDIO_FORMATS.get(format_name, 1)
        length = self.loc.AUDIO_LENGTHS.get(length_name, 2)

        nested_sources = [[[sid]] for sid in source_ids]
        flat_sources = [[sid] for sid in source_ids]

        params = [
            [2],
            notebook_id,
            [
                None, None,
                1,             # STUDIO_TYPE_AUDIO
                nested_sources,
                None, None,
                [
                    None,
                    [
                        focus_prompt,
                        length,
                        None,
                        flat_sources,
                        lang,
                        None,
                        fmt,
                    ],
                ],
            ],
        ]

        return await self.rpc.execute(
            self.rpc.RPC.CREATE_STUDIO,
            params,
            notebook_id=notebook_id,
        )

    async def create_video_overview(
        self,
        notebook_id: str,
        source_ids: List[str],
        focus_prompt: str = "",
        format_name: str = "explainer",
        visual_style: str = "classic",
        language: Optional[str] = None,
    ) -> RPCResponse:
        """
        生成影片導讀。

        Args:
            format_name: explainer / brief / cinematic
            visual_style: auto_select / classic / whiteboard / kawaii / anime ...
        """
        lang = self.loc.get_language_code(language)
        fmt = self.loc.VIDEO_FORMATS.get(format_name, 1)
        style_code = self.loc.VIDEO_STYLES.get(visual_style, 3)

        nested_sources = [[[sid]] for sid in source_ids]
        flat_sources = [[sid] for sid in source_ids]

        params = [
            [2],
            notebook_id,
            [
                None, None,
                3,             # STUDIO_TYPE_VIDEO
                nested_sources,
                None, None, None, None,
                [
                    None, None,
                    [
                        flat_sources,
                        lang,
                        focus_prompt,
                        None,
                        fmt,
                        style_code,
                        None,  # custom style prompt（僅 visual_style="custom" 時使用）
                    ],
                ],
            ],
        ]

        return await self.rpc.execute(
            self.rpc.RPC.CREATE_STUDIO,
            params,
            notebook_id=notebook_id,
        )

    async def create_slide_deck(
        self,
        notebook_id: str,
        source_ids: List[str],
        focus_prompt: str = "",
        format_name: str = "detailed",
        length_name: str = "default",
        language: Optional[str] = None,
    ) -> RPCResponse:
        """
        生成簡報（Slide Deck）。

        Args:
            format_name: detailed / presenter
            length_name: short / default
        """
        lang = self.loc.get_language_code(language)
        fmt = self.loc.SLIDE_FORMATS.get(format_name, 1)
        length = 1 if length_name == "short" else 3

        nested_sources = [[[sid]] for sid in source_ids]

        params = [
            [2],
            notebook_id,
            [
                None, None,
                8,             # STUDIO_TYPE_SLIDE_DECK
                nested_sources,
                None, None, None, None, None, None, None, None,
                None, None, None, None,
                [[focus_prompt, lang, fmt, length]],
            ],
        ]

        return await self.rpc.execute(
            self.rpc.RPC.CREATE_STUDIO,
            params,
            notebook_id=notebook_id,
        )

    async def poll_studio_status(
        self, notebook_id: str
    ) -> RPCResponse:
        """輪詢 Studio 生成狀態（status: 1=進行中, 3=已完成）"""
        return await self.rpc.execute(
            self.rpc.RPC.POLL_STUDIO,
            [
                [2],
                notebook_id,
                'NOT artifact.status = "ARTIFACT_STATUS_SUGGESTED"',
            ],
            notebook_id=notebook_id,
        )

    async def wait_for_studio(
        self,
        notebook_id: str,
        poll_interval: int = 15,
        max_wait: int = 600,
    ) -> RPCResponse:
        """
        等待 Studio 生成完成，自動輪詢。

        Args:
            max_wait: 最長等待秒數（影片生成可能需要 5-10 分鐘）
        """
        elapsed = 0
        while elapsed < max_wait:
            result = await self.poll_studio_status(notebook_id)
            if result.error:
                return result

            artifacts = self.parse_studio_status(result.data)
            if artifacts:
                all_done = all(a.get("status") == 3 for a in artifacts)
                if all_done:
                    logger.info("Studio 生成完成")
                    return result

                in_progress = [
                    a["title"] for a in artifacts if a.get("status") != 3
                ]
                logger.info(
                    f"Studio 生成中：{in_progress} ({elapsed}s/{max_wait}s)"
                )

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        return RPCResponse(
            error=f"Studio 等待逾時（{max_wait}s）",
            data=result.data if result else None,
        )

    @staticmethod
    def parse_studio_status(data: Any) -> List[Dict[str, Any]]:
        """
        解析 Studio 狀態回應，提取所有 artifacts 的狀態與下載連結。

        Returns:
            [{
                "id": str,
                "title": str,
                "type": int,   # 1=Audio, 3=Video, 7=Infographic, 8=Slides
                "status": int, # 1=進行中, 3=已完成
                "audio_url": str | None,
                "video_url": str | None,
            }]
        """
        artifacts = []
        if not data or not isinstance(data, list):
            return artifacts

        try:
            # 結構: [[artifact_data_array]]
            outer = data
            if isinstance(outer[0], list):
                outer = outer[0]
            if isinstance(outer[0], list):
                outer = outer[0]

            for item in outer:
                if not isinstance(item, list) or len(item) < 5:
                    continue

                artifact: Dict[str, Any] = {
                    "id": item[0] if isinstance(item[0], str) else "",
                    "title": item[1] if len(item) > 1 else "",
                    "type": item[2] if len(item) > 2 else 0,
                    "status": item[4] if len(item) > 4 else 0,
                    "audio_url": None,
                    "video_url": None,
                }

                # 提取下載 URL（位於 index 8）
                if len(item) > 8 and isinstance(item[8], list):
                    url_data = item[8]
                    if url_data:
                        url = url_data[0] if isinstance(url_data[0], str) else ""
                        art_type = artifact["type"]
                        if art_type == 1:
                            artifact["audio_url"] = url
                        elif art_type == 3:
                            artifact["video_url"] = url

                artifacts.append(artifact)

        except (IndexError, TypeError) as e:
            logger.warning(f"Studio 狀態解析異常: {e}")

        return artifacts

    async def delete_artifact(
        self, artifact_id: str
    ) -> RPCResponse:
        """刪除 Studio 輸出物（不可逆）"""
        return await self.rpc.execute(
            self.rpc.RPC.DELETE_STUDIO,
            [[2], artifact_id],
        )
