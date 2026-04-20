"""
繁體中文在地化模組
處理所有輸出的語言設定與 RPC 參數中的語言代碼注入。
"""

from typing import Optional


class Localization:
    """
    管理 NotebookLM MCP v2.0 的語言在地化設定。
    預設使用繁體中文（zh-TW），可動態切換。
    """

    # BCP-47 語言代碼對照
    SUPPORTED_LANGUAGES = {
        "zh-TW": "繁體中文",
        "zh-CN": "簡體中文",
        "en": "English",
        "ja": "日本語",
        "ko": "한국어",
        "es": "Español",
        "fr": "Français",
        "de": "Deutsch",
    }

    # Studio 類型代碼
    STUDIO_TYPES = {
        "audio": 1,
        "video": 3,
        "infographic": 7,
        "slide_deck": 8,
    }

    # 音訊格式代碼
    AUDIO_FORMATS = {
        "deep_dive": 1,
        "brief": 2,
        "critique": 3,
        "debate": 4,
    }

    # 音訊長度代碼
    AUDIO_LENGTHS = {
        "short": 1,
        "default": 2,
        "long": 3,
    }

    # 影片格式代碼
    VIDEO_FORMATS = {
        "explainer": 1,
        "brief": 2,
        "cinematic": 3,
    }

    # 影片視覺風格代碼
    VIDEO_STYLES = {
        "auto_select": 1,
        "custom": 2,
        "classic": 3,
        "whiteboard": 4,
        "kawaii": 5,
        "anime": 6,
        "watercolor": 7,
        "retro_print": 8,
        "heritage": 9,
        "paper_craft": 10,
    }

    # 簡報格式代碼
    SLIDE_FORMATS = {
        "detailed": 1,
        "presenter": 2,
    }

    # 研究來源類型
    RESEARCH_SOURCES = {
        "web": 1,
        "drive": 2,
    }

    # 聊天設定代碼
    CHAT_GOALS = {
        "default": 1,
        "custom": 2,
        "learning_guide": 3,
    }

    RESPONSE_LENGTHS = {
        "default": 1,
        "longer": 4,
        "shorter": 5,
    }

    def __init__(self, default_language: str = "zh-TW"):
        if default_language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"不支援的語言代碼: {default_language}。"
                f"支援的語言: {list(self.SUPPORTED_LANGUAGES.keys())}"
            )
        self.default_language = default_language

    def get_language_code(
        self, override: Optional[str] = None
    ) -> str:
        """取得語言代碼，支援覆蓋"""
        lang = override or self.default_language
        if lang not in self.SUPPORTED_LANGUAGES:
            return self.default_language
        return lang

    def get_language_display(
        self, lang_code: Optional[str] = None
    ) -> str:
        """取得語言的顯示名稱"""
        code = lang_code or self.default_language
        return self.SUPPORTED_LANGUAGES.get(code, code)

    @staticmethod
    def inject_expert_prompt(
        base_query: str,
        mode: str = "technical",
    ) -> str:
        """
        為技術文件專家模式注入指示詞。

        Args:
            base_query: 原始查詢文字
            mode: 專家模式（technical / api / comparison）
        """
        prompts = {
            "technical": (
                "請以技術文件專家的角度回答以下問題。"
                "請專注於：API 定義、函數簽名、參數說明、"
                "回傳值格式、錯誤處理模式。"
                "若有程式碼範例，請附上完整的可執行程式碼區塊。\n\n"
            ),
            "api": (
                "請以 API 參考手冊的格式回答。"
                "針對每個端點列出：HTTP 方法、路徑、"
                "必填/選填參數、請求/回應範例。\n\n"
            ),
            "comparison": (
                "請進行版本比較分析。"
                "列出各版本間的差異、新增功能、"
                "棄用 API、遷移步驟。\n\n"
            ),
        }
        prefix = prompts.get(mode, prompts["technical"])
        return prefix + base_query
