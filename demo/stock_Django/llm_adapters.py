# -*- coding: utf-8 -*-
"""
LLM 雙軌適配器模組 (Dual-Track LLM Adapters)
支援容器化與本地混合環境：
- 主軌道：GeminiNativeAdapter (官方 REST API / Python SDK，具指數退避與配額保護)
- 備援軌道：OllamaFallbackAdapter (host.docker.internal 本地 GPU/CPU 推理服務)
- 熔斷保護：LLMCircuitBreaker (CLOSED / OPEN / HALF_OPEN 三態狀態機)
- 防禦機制：Unicode 清理、Markdown Wrapper 剝離、JSON 結構安全校驗與零崩潰保護
"""

import os
import sys
import abc
import time
import json
import re
import logging
import unicodedata
from typing import Optional, Dict, Any, List, Tuple

import requests
from requests.exceptions import Timeout, ConnectionError, RequestException

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. 抽象介面 (Abstract Base Class)
# ==============================================================================

class AbstractLLMAdapter(abc.ABC):
    """LLM 適配器抽象基類"""

    @abc.abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> Optional[str]:
        """生成純文字回應"""
        pass

    @abc.abstractmethod
    def generate_json(self, prompt: str, **kwargs) -> Optional[Dict[str, Any]]:
        """生成結構化 JSON 回應"""
        pass

    @abc.abstractmethod
    def get_status(self) -> str:
        """返回適配器當前狀態"""
        pass


# ==============================================================================
# 2. 熔斷器 (Circuit Breaker)
# ==============================================================================

class LLMCircuitBreaker:
    """
    三態熔斷器機制：
    - CLOSED：正常運行，計數失敗次數
    - OPEN：連續失敗達閥值，直接跳轉備援，拒絕請求主服務
    - HALF_OPEN：冷卻時間過後，允許單次探測請求
    """
    STATE_CLOSED = "CLOSED"
    STATE_OPEN = "OPEN"
    STATE_HALF_OPEN = "HALF_OPEN"

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = self.STATE_CLOSED
        self.last_state_change = time.time()

    def can_execute(self) -> bool:
        now = time.time()
        if self.state == self.STATE_CLOSED:
            return True
        elif self.state == self.STATE_OPEN:
            if now - self.last_state_change > self.recovery_timeout:
                logger.info("[CircuitBreaker] 冷卻時間已達，熔斷器狀態由 OPEN 轉為 HALF_OPEN 進行探測。")
                self.state = self.STATE_HALF_OPEN
                self.last_state_change = now
                return True
            return False
        elif self.state == self.STATE_HALF_OPEN:
            return True
        return True

    def record_success(self):
        if self.state != self.STATE_CLOSED:
            logger.info("[CircuitBreaker] 探測呼叫成功，熔斷器恢復為 CLOSED 狀態。")
        self.failure_count = 0
        self.state = self.STATE_CLOSED
        self.last_state_change = time.time()

    def record_failure(self):
        self.failure_count += 1
        now = time.time()
        if self.state == self.STATE_HALF_OPEN:
            logger.warning("[CircuitBreaker] HALF_OPEN 探測失敗，立即重置為 OPEN 狀態。")
            self.state = self.STATE_OPEN
            self.last_state_change = now
        elif self.failure_count >= self.failure_threshold:
            logger.warning(f"[CircuitBreaker] 連續失敗達 {self.failure_count} 次，熔斷器跳閘轉為 OPEN 狀態。")
            self.state = self.STATE_OPEN
            self.last_state_change = now


# ==============================================================================
# 3. 雲端 Gemini 原生適配器 (GeminiNativeAdapter)
# ==============================================================================

class GeminiNativeAdapter(AbstractLLMAdapter):
    """
    使用 Google Generative Language REST API 直接與 Gemini 雲端通訊。
    不依賴宿主機外部 gemini CLI 執行檔，完全相容 Docker Linux 容器環境。
    """
    DEFAULT_MODELS = [
        "gemini-3.1-pro-preview",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
        "gemma-4-31b-it",
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        self.circuit_breaker = LLMCircuitBreaker(failure_threshold=3, recovery_timeout=60.0)

    def _get_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        # 動態重載環境變數
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        return self.api_key

    def _call_gemini_rest(self, prompt: str, model_name: str, response_json: bool = False, timeout: int = 45) -> Optional[str]:
        api_key = self._get_api_key()
        if not api_key:
            logger.error("[GeminiNative] 缺失 GEMINI_API_KEY，無法呼叫雲端 API。")
            return None

        # 清理並正規化 Unicode 文字防禦編碼問題
        cleaned_prompt = unicodedata.normalize('NFC', prompt)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        
        generation_config = {
            "temperature": 0.2,
            "maxOutputTokens": 2048,
        }
        if response_json:
            generation_config["responseMimeType"] = "application/json"

        payload = {
            "contents": [{
                "parts": [{"text": cleaned_prompt}]
            }],
            "generationConfig": generation_config
        }

        # 最多重試 2 次 (指數退避)
        for attempt in range(2):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            text_out = parts[0].get("text", "").strip()
                            self.circuit_breaker.record_success()
                            return text_out
                    logger.warning(f"[GeminiNative] 回應中無有效內容: {data}")
                    return None
                elif resp.status_code in (429, 503):
                    logger.warning(f"[GeminiNative] 模型 {model_name} 遭遇暫時性錯誤 (HTTP {resp.status_code})，準備重試...")
                    time.sleep(1.5 * (attempt + 1))
                else:
                    logger.error(f"[GeminiNative] 呼叫失敗 HTTP {resp.status_code}: {resp.text[:200]}")
                    break
            except (Timeout, ConnectionError) as e:
                logger.warning(f"[GeminiNative] 網路超時或連線異常 (嘗試 {attempt + 1}/2): {e}")
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"[GeminiNative] 未知錯誤: {e}")
                break

        self.circuit_breaker.record_failure()
        return None

    def generate_text(self, prompt: str, model: Optional[str] = None, **kwargs) -> Optional[str]:
        if not self.circuit_breaker.can_execute():
            logger.info("[GeminiNative] 熔斷器處於 OPEN 狀態，跳過主通道。")
            return None

        preferred_model = model or os.getenv("GEMINI_ADVISOR_MODEL")
        models_to_try = [preferred_model] if preferred_model else self.DEFAULT_MODELS

        for m in models_to_try:
            if not m:
                continue
            res = self._call_gemini_rest(prompt, model_name=m, response_json=False)
            if res:
                return res
        return None

    def generate_json(self, prompt: str, model: Optional[str] = None, **kwargs) -> Optional[Dict[str, Any]]:
        if not self.circuit_breaker.can_execute():
            logger.info("[GeminiNative] 熔斷器處於 OPEN 狀態，跳過主通道。")
            return None

        preferred_model = model or os.getenv("GEMINI_ADVISOR_MODEL")
        models_to_try = [preferred_model] if preferred_model else self.DEFAULT_MODELS

        for m in models_to_try:
            if not m:
                continue
            res_str = self._call_gemini_rest(prompt, model_name=m, response_json=True)
            if res_str:
                parsed = self._extract_json(res_str)
                if parsed:
                    parsed["_model_source"] = f"Gemini ({m})"
                    return parsed
        return None

    @staticmethod
    def _extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
        """安全解析 JSON，自動剝離 Markdown 圍欄字串"""
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        # 搜尋首個 { 與最後一個 }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_substr = cleaned[start:end + 1]
            try:
                return json.loads(json_substr)
            except Exception as e:
                logger.warning(f"[GeminiNative] JSON 解析失敗: {e}")
        return None

    def get_status(self) -> str:
        return f"GeminiNative (Circuit: {self.circuit_breaker.state})"


# ==============================================================================
# 4. 本地 Ollama 備援適配器 (OllamaFallbackAdapter)
# ==============================================================================

class OllamaFallbackAdapter(AbstractLLMAdapter):
    """
    連線至宿主機 Ollama 服務 (預設 http://host.docker.internal:11434)。
    當雲端 API 失敗、斷網或配額超限時提供零延遲本地備援。
    """
    DEFAULT_MODELS = [
        "gemma4:e4b",
        "gemma4:26b",
        "qwen2.5:7b",
        "deepseek-r1:8b"
    ]

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")).rstrip("/")

    def _resolve_available_model(self) -> Optional[str]:
        """動態偵測 Ollama 當前已安裝模型"""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                tags = resp.json().get("models", [])
                installed = [t.get("name", "") for t in tags]
                # 依優先序比對
                for pref in self.DEFAULT_MODELS:
                    for inst in installed:
                        if pref in inst:
                            return inst
                if installed:
                    return installed[0]
        except Exception as e:
            logger.debug(f"[OllamaFallback] 探測已安裝模型失敗: {e}")
        return self.DEFAULT_MODELS[0]

    def generate_text(self, prompt: str, model: Optional[str] = None, timeout: int = 40, **kwargs) -> Optional[str]:
        target_model = model or self._resolve_available_model()
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3}
        }
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            logger.warning(f"[OllamaFallback] 請求失敗 HTTP {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            logger.warning(f"[OllamaFallback] 無法連線至宿主機 Ollama ({self.base_url}): {e}")
        return None

    def generate_json(self, prompt: str, model: Optional[str] = None, timeout: int = 45, **kwargs) -> Optional[Dict[str, Any]]:
        target_model = model or self._resolve_available_model()
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": target_model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2}
        }
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                raw_response = resp.json().get("response", "").strip()
                parsed = GeminiNativeAdapter._extract_json(raw_response)
                if parsed:
                    parsed["_model_source"] = f"Ollama Local ({target_model})"
                    return parsed
        except Exception as e:
            logger.warning(f"[OllamaFallback] JSON 生成失敗 ({self.base_url}): {e}")
        return None

    def get_status(self) -> str:
        return f"OllamaFallback (Endpoint: {self.base_url})"


# ==============================================================================
# 5. 雙軌統一協調客戶端 (DualTrackLLMClient)
# ==============================================================================

class DualTrackLLMClient:
    """
    雙軌 LLM 協調客戶端：
    優先嘗試 GeminiNativeAdapter，若失敗或熔斷則自動無縫切換至 OllamaFallbackAdapter。
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DualTrackLLMClient, cls).__new__(cls)
            cls._instance._init_adapters()
        return cls._instance

    def _init_adapters(self):
        self.primary = GeminiNativeAdapter()
        self.fallback = OllamaFallbackAdapter()

    def generate_text(self, prompt: str, **kwargs) -> str:
        # 1. 主通道嘗試
        result = self.primary.generate_text(prompt, **kwargs)
        if result:
            return result

        # 2. 備援通道嘗試
        logger.info("[DualTrackClient] 主通道未回應，啟用本地 Ollama 備援通道...")
        result = self.fallback.generate_text(prompt, **kwargs)
        if result:
            return result

        return "外部智慧模型目前處於連線維護中，建議參考客觀技術與基本面指標。"

    def generate_json(self, prompt: str, required_keys: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        # 1. 主通道嘗試
        result = self.primary.generate_json(prompt, **kwargs)
        if result and self._validate_keys(result, required_keys):
            return result

        # 2. 備援通道嘗試
        logger.info("[DualTrackClient] 主通道未取得有效結構，切換至本地 Ollama 備援...")
        result = self.fallback.generate_json(prompt, **kwargs)
        if result and self._validate_keys(result, required_keys):
            return result

        # 3. 確定性預設降級安全結構 (確保前端完全不崩潰)
        logger.error("[DualTrackClient] 雙軌通道皆未回傳有效 JSON，返回安全降級結構。")
        return {
            "recommendation": "觀望",
            "score": 50,
            "reason": "目前雲端與本地 AI 模型連線冷卻中，已啟動安全保護機制，建議以客觀均線與財報為準。",
            "details": {
                "technical": "技術面均線與動能維持中性判讀",
                "chips": "籌碼面資料以最新交易所統計為準",
                "sentiment": "輿情情緒面維持中立評級",
                "fundamental": "基本面獲利指標正常載入",
                "macro": "總經政策環境平穩"
            },
            "_model_source": "System Fallback Rule"
        }

    @staticmethod
    def _validate_keys(data: Dict[str, Any], required_keys: Optional[List[str]]) -> bool:
        if not required_keys:
            return True
        return all(k in data for k in required_keys)


# 全域單例取得函式
def get_llm_client() -> DualTrackLLMClient:
    return DualTrackLLMClient()
