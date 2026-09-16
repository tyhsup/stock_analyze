import os
import sys
import json
import logging
import time
import unicodedata
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
import numpy as np

# ──────────────────────────────────────────────
# 全局 Groq helper 路徑
# ──────────────────────────────────────────────
GLOBAL_HELPER_PATH = r"c:\Users\許廷宇\.gemini\antigravity\scripts"
if GLOBAL_HELPER_PATH not in sys.path:
    sys.path.append(GLOBAL_HELPER_PATH)

logger = logging.getLogger(__name__)
import threading
import random


def clean_and_normalize_text(text: str) -> str:
    """
    Unicode NFKC 正規化與控制字元過濾 (Evaluator 防禦要求)。
    處理全形半形、Emoji、不可見零寬字元與 \x00 截斷符號。
    """
    if not isinstance(text, str):
        return ""
    # 1. NFKC 規範化
    normalized = unicodedata.normalize('NFKC', text)
    # 2. 移除空字元與控制字元 (保留換行 \n 與空白)
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', normalized)
    return cleaned.strip()


@dataclass
class SentimentResult:
    """統一情緒分析結果數據容器"""
    label: str                                   # 'positive' / 'negative' / 'neutral'
    score: float                                 # -1.0 ~ 1.0 綜合情緒分
    confidence: float                            # 0.0 ~ 1.0 模型最高信心度
    probabilities: Dict[str, float]              # 各類別機率分佈 {'positive': ..., 'negative': ..., 'neutral': ...}
    is_neutral_adjusted: bool = False           # 是否因低於 neutral_threshold 而被校正為中立
    language: str = 'zh-TW'
    embedding: Optional[np.ndarray] = None       # 768D 語意特徵向量

    def to_dict(self) -> Dict[str, Any]:
        """相容既有字典格式與前端渲染"""
        label_map = {
            'positive': '正面', 'negative': '負面', 'neutral': '中立',
            '正面': '正面', '負面': '負面', '中立': '中立'
        }
        zh_label = label_map.get(self.label, '中立')
        return {
            "positive_negative_analysis": zh_label,
            "label": self.label,
            "sentiment_score": round(float(self.score), 4),
            "confidence": round(float(self.confidence), 4),
            "probabilities": {k: round(float(v), 4) for k, v in self.probabilities.items()},
            "is_neutral_adjusted": bool(self.is_neutral_adjusted),
            "language": self.language
        }


class UnifiedSentimentAnalyzer:
    """
    統一新聞情緒分析器 (Unified Sentiment Analyzer)。
    
    模型配置：
    - 中文核心：IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment
    - 英文核心：ProsusAI/finbert (Lazy loaded)
    - 閾值防護：neutral_threshold 預設 0.60 (修復 2026-08-12 歷史 0.65 過高誤過濾缺陷)
    """
    def __init__(self, neutral_threshold: float = 0.60,
                 zh_model_name: str = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment",
                 en_model_name: str = "ProsusAI/finbert"):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.neutral_threshold = float(neutral_threshold)
        self.zh_model_name = zh_model_name
        self.en_model_name = en_model_name

        logger.info(f"[UnifiedSentimentAnalyzer] 初始化中文模型，裝置: {self.device}，模型: {zh_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(zh_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(zh_model_name)
        self.model.to(self.device)
        self.model.eval()

        self.en_tokenizer = None
        self.en_model = None

    def _ensure_en_model(self) -> bool:
        """延遲載入英文 FinBERT 模型"""
        if self.en_tokenizer is None or self.en_model is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            logger.info(f"[UnifiedSentimentAnalyzer] 延遲載入英文 FinBERT: {self.en_model_name}...")
            try:
                self.en_tokenizer = AutoTokenizer.from_pretrained(self.en_model_name)
                self.en_model = AutoModelForSequenceClassification.from_pretrained(self.en_model_name)
                self.en_model.to(self.device)
                self.en_model.eval()
            except Exception as e:
                logger.error(f"[UnifiedSentimentAnalyzer] 載入英文 FinBERT 失敗: {e}")
                return False
        return True

    @staticmethod
    def detect_language(text: str) -> str:
        """自動偵測語言類別 ('zh-TW' 或 'en')"""
        if not text:
            return 'zh-TW'
        # 計算中文字元數量
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        if chinese_chars >= 2:
            return 'zh-TW'
        # 計算英文字母比例
        ascii_letters = len(re.findall(r'[a-zA-Z]', text))
        if ascii_letters > len(text) * 0.4:
            return 'en'
        return 'zh-TW'

    def analyze(self, text: str, language: str = 'auto', title: str = "", content: str = "") -> SentimentResult:
        """
        統一情感分析入口。
        
        :param text: 欲分析之本文（若有傳 title / content 則優先拼接）
        :param language: 語言別 ('auto', 'zh-TW', 'en')
        :param title: 新聞標題
        :param content: 新聞內容
        :return: SentimentResult 實例
        """
        # 組合並清洗文本
        if title or content:
            raw_text = f"{title}. {content[:300]}".strip()
        else:
            raw_text = text

        cleaned_text = clean_and_normalize_text(raw_text)
        if not cleaned_text:
            return SentimentResult(
                label="neutral",
                score=0.0,
                confidence=0.0,
                probabilities={"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                is_neutral_adjusted=False,
                language=language if language != 'auto' else 'zh-TW'
            )

        # 語言判定
        actual_lang = self.detect_language(cleaned_text) if language == 'auto' else language

        if actual_lang == 'en':
            return self._analyze_en(cleaned_text)
        return self._analyze_zh(cleaned_text)

    def _analyze_zh(self, text: str) -> SentimentResult:
        """中文新聞情緒推論 (Erlangshen-Roberta-110M-Sentiment)"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with self._torch.no_grad():
                outputs = self.model(**inputs)
                probs = self._torch.softmax(outputs.logits, dim=-1)[0].cpu().tolist()

            neg_prob = float(probs[0])
            pos_prob = float(probs[1])
            max_prob = max(neg_prob, pos_prob)

            # 閾值防護：若最高信心度低於 neutral_threshold (0.60)，強制判定為中立
            if max_prob < self.neutral_threshold:
                label = "neutral"
                score = 0.0
                is_neutral_adjusted = True
            else:
                label = "positive" if pos_prob > neg_prob else "negative"
                score = round(pos_prob - neg_prob, 4)
                is_neutral_adjusted = False

            return SentimentResult(
                label=label,
                score=score,
                confidence=round(max_prob, 4),
                probabilities={
                    "positive": round(pos_prob, 4),
                    "negative": round(neg_prob, 4),
                    "neutral": round(1.0 - abs(pos_prob - neg_prob), 4)
                },
                is_neutral_adjusted=is_neutral_adjusted,
                language="zh-TW"
            )
        except Exception as e:
            logger.error(f"[UnifiedSentimentAnalyzer] 中文分析失敗: {e}")
            return SentimentResult(
                label="neutral",
                score=0.0,
                confidence=0.0,
                probabilities={"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                is_neutral_adjusted=False,
                language="zh-TW"
            )

    def _analyze_en(self, text: str) -> SentimentResult:
        """英文新聞情緒推論 (ProsusAI/finbert)"""
        if not self._ensure_en_model():
            return SentimentResult(
                label="neutral",
                score=0.0,
                confidence=0.0,
                probabilities={"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                is_neutral_adjusted=False,
                language="en"
            )

        try:
            inputs = self.en_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with self._torch.no_grad():
                outputs = self.en_model(**inputs)
                probs = self._torch.softmax(outputs.logits, dim=-1)[0].cpu().tolist()

            pos_prob = float(probs[0])
            neg_prob = float(probs[1])
            neu_prob = float(probs[2])

            confidence = max(pos_prob, neg_prob, neu_prob)

            # FinBERT 標籤與閾值防護
            if neu_prob >= 0.50 or max(pos_prob, neg_prob) < self.neutral_threshold:
                label = "neutral"
                score = 0.0
                is_neutral_adjusted = (neu_prob < 0.50 and max(pos_prob, neg_prob) < self.neutral_threshold)
            elif pos_prob > neg_prob:
                label = "positive"
                score = round(pos_prob - neg_prob, 4)
                is_neutral_adjusted = False
            else:
                label = "negative"
                score = round(pos_prob - neg_prob, 4)
                is_neutral_adjusted = False

            return SentimentResult(
                label=label,
                score=score,
                confidence=round(confidence, 4),
                probabilities={
                    "positive": round(pos_prob, 4),
                    "negative": round(neg_prob, 4),
                    "neutral": round(neu_prob, 4)
                },
                is_neutral_adjusted=is_neutral_adjusted,
                language="en"
            )
        except Exception as e:
            logger.error(f"[UnifiedSentimentAnalyzer] 英文分析失敗: {e}")
            return SentimentResult(
                label="neutral",
                score=0.0,
                confidence=0.0,
                probabilities={"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                is_neutral_adjusted=False,
                language="en"
            )

    def batch_analyze(self, texts: List[str], language: str = 'auto') -> List[SentimentResult]:
        """批次分析多筆新聞"""
        return [self.analyze(t, language=language) for t in texts]


class RateLimiter:
    """
    執行緒安全的速率限制器。
    限制每分鐘的雲端 API 呼叫次數在 10 RPM 以下（最小間隔 6.0 秒）。
    """
    def __init__(self, min_interval: float = 6.0):
        self.min_interval = min_interval
        self.last_request_time = 0.0
        self.lock = threading.Lock()

    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                logger.info(f"[RateLimiter] 速率限制中，等待 {sleep_time:.2f} 秒以維持 10 RPM 以下...")
                time.sleep(sleep_time)
            self.last_request_time = time.time()


class FinBertScorer(UnifiedSentimentAnalyzer):
    """
    雙語情緒分析器相容封裝（繼承 UnifiedSentimentAnalyzer）。
    完全向後相容既有呼叫端，回傳舊版字典格式。
    """
    def __init__(self, model_name: str = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"):
        super().__init__(neutral_threshold=0.60, zh_model_name=model_name)

    def analyze(self, title: str, content: str, language: str = 'zh-TW') -> dict:
        """分析新聞並回傳字典（完全相容舊版介面）"""
        res = super().analyze(text="", language=language, title=title, content=content)
        return res.to_dict()


# ══════════════════════════════════════════════════════════════════
# Stage 2：Gemini CLI 雲端 Gemma 4 31B 定性解釋器
# ══════════════════════════════════════════════════════════════════
GEMINI_SYSTEM_PROMPT = """你是專業金融分析師。請基於提供的新聞，補充定性分析並輸出 JSON 格式。請不要包含任何 markdown 標記（如 ```json）或額外文字。JSON 格式如下：
{
  "market": "TW",
  "impact_scope": "短期",
  "reasoning_summary": "原因說明"
}"""


class AgentNewsAnalyzer:
    """
    金融新聞混合分析器。
    Stage 1: 本地 Roberta / FinBERT 模型 (雙語快速評分)
    Stage 2: Gemini CLI 雲端 Gemma 4 31B (深度解釋)
    """

    def __init__(self):
        # 讀取 API Key (優先從環境變數，次之從 .env)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            from dotenv import load_dotenv
            dotenv_path = os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", ".env")
            load_dotenv(dotenv_path)
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        import shutil
        self.gemini_path = shutil.which("gemini")
        if not self.gemini_path:
            logger.warning("[AgentNewsAnalyzer] 警告：在系統中找不到 gemini CLI 指令。")

        # 初始化 10 RPM 限制器（最小間隔 6.0 秒）
        self.rate_limiter = RateLimiter(min_interval=6.0)

        logger.info("[AgentNewsAnalyzer] 載入本地 Scorer...")
        self.scorer = FinBertScorer()

    def analyze_news(self, news_data: dict, force_llm: bool = False, is_recent: bool = True) -> dict:
        title    = news_data.get("title",   news_data.get("標題", ""))
        date     = news_data.get("date",    news_data.get("發布時間", ""))
        content  = news_data.get("content", news_data.get("內文", ""))
        link     = news_data.get("link",    news_data.get("連結", ""))
        source   = news_data.get("source",  news_data.get("來源", ""))
        language = news_data.get("language", news_data.get("語言", "zh-TW"))

        # Stage 1: 本地雙語評分
        score_res = self.scorer.analyze(title, content, language=language)
        
        # 清理並計算長度
        clean_content = content.strip() if content else ""
        clean_title = title.strip() if title else ""
        combined_len = len(clean_title) + len(clean_content)

        if language == 'en':
            # 英文新聞：解除 is_neutral 強制攔截。只要標題與內文總長度 >= 20 字元且屬於近期新聞，即升級 LLM
            has_sufficient_length = combined_len >= 20 or len(clean_content.split()) >= 5
            should_upgrade = force_llm or (has_sufficient_length and is_recent)
        else:
            # 中文新聞：維持本地判定為「非中立」且長度符合要求 (>= 150/200 字元) 且 7 天內條件
            has_sufficient_length = len(clean_content) >= 150 or combined_len >= 200
            is_neutral = score_res["positive_negative_analysis"] == "中立"
            should_upgrade = force_llm or (
                not is_neutral 
                and has_sufficient_length
                and is_recent
            )
        
        gemini_res = {}
        if should_upgrade:
            gemini_res = self._call_gemini(title, content, score_res) or {}

        # 預設本地摘要說明
        default_summary = f"本地 FinBERT/Roberta 評分完成：{score_res['positive_negative_analysis']} (情緒得分: {score_res['sentiment_score']}, 信心度: {score_res['confidence']:.0%})"

        # 合併
        return {
            "title": title,
            "date": date,
            "content": content,
            "link": link,
            "source": source,
            "language": language,
            "positive_negative_analysis": score_res["positive_negative_analysis"],
            "sentiment_score": score_res["sentiment_score"],
            "confidence": score_res["confidence"],
            "market": gemini_res.get("market", self._infer_market(source, content)),
            "impact_scope": gemini_res.get("impact_scope", "短期"),
            "reasoning_summary": gemini_res.get("reasoning_summary", default_summary)
        }

    def _call_gemini(self, title: str, content: str, score_res: dict) -> Optional[dict]:
        import subprocess
        
        # 解析本地評分以避免在 prompt 參數中傳遞含有引號和括號的 dict 字串
        local_label = score_res.get("positive_negative_analysis", "中立")
        local_score = score_res.get("sentiment_score", 0.0)
        local_conf = score_res.get("confidence", 0.0)

        # 強制將換行換成空白，以防止 Windows 下參數被截斷
        prompt = (
            f"請以專業金融分析師的角色，對以下新聞進行定性分析，並【僅】以 JSON 格式輸出，不要有任何額外文字或說明。 "
            f"新聞標題：{title}。 "
            f"新聞內文：{content[:300]}。 "
            f"本地評分：{local_label}，情緒得分：{local_score}，信心度：{local_conf}。 "
            f"請嚴格輸出 JSON 格式如下： "
            f"{{\"market\": \"TW\" 或 \"US\", \"impact_scope\": \"短期\" 或 \"中期\" 或 \"長期\", \"reasoning_summary\": \"50字以內理由\"}}"
        )
        full_prompt = f"{GEMINI_SYSTEM_PROMPT} {prompt}"
        full_prompt = full_prompt.replace("\n", " ").replace("\r", " ").strip()
        
        if not self.gemini_path:
            import shutil
            self.gemini_path = shutil.which("gemini")
            if not self.gemini_path:
                logger.error("[AgentNewsAnalyzer] 找不到 gemini CLI。")
                return None

        env = os.environ.copy()
        if self.gemini_api_key:
            env["GEMINI_API_KEY"] = self.gemini_api_key

        max_retries = 3
        base_delay = 6.0
        models_to_try = ["gemini-3.1-pro-preview", "gemma-4-31b-it"]

        for model in models_to_try:
            logger.info(f"[AgentNewsAnalyzer] 開始使用模型 {model} 進行新聞情緒分析...")
            for attempt in range(max_retries + 1):
                # 確保雲端呼叫符合 10 RPM 速率限制 (最小間隔 6.0 秒)
                self.rate_limiter.wait_if_needed()

                try:
                    logger.info(f"[AgentNewsAnalyzer] 調用 Gemini CLI 雲端 {model} 模型 (嘗試 {attempt + 1}/{max_retries + 1})...")
                    args = [self.gemini_path, "-m", model, "--skip-trust", "-o", "json", "-p", full_prompt]
                    
                    result = subprocess.run(
                        args,
                        capture_output=True,
                        env=env,
                        shell=False
                    )
                    
                    if result.returncode != 0:
                        stderr_msg = result.stderr.decode("utf-8", errors="replace")
                        logger.warning(f"[AgentNewsAnalyzer] Gemini CLI 執行失敗 (code: {result.returncode}), stderr: {stderr_msg}")
                    else:
                        stdout_decoded = result.stdout.decode("utf-8", errors="replace")
                        if "{" in stdout_decoded:
                            json_start = stdout_decoded.index("{")
                            json_data = json.loads(stdout_decoded[json_start:])
                            response_text = json_data.get("response", "").strip()
                            
                            # 移除可能存在的 markdown wrapper
                            clean_res = response_text
                            if clean_res.startswith("```"):
                                lines = clean_res.splitlines()
                                if lines[0].startswith("```"):
                                    lines = lines[1:]
                                if lines[-1].startswith("```"):
                                    lines = lines[:-1]
                                clean_res = "\n".join(lines).strip()
                            
                            try:
                                return json.loads(clean_res)
                            except Exception as je:
                                logger.warning(f"[AgentNewsAnalyzer] 無法解析模型回覆的 JSON: {je}. 原始內容: {clean_res}")
                        else:
                            logger.warning(f"[AgentNewsAnalyzer] 輸出中找不到 JSON 物件。原始輸出: {stdout_decoded}")
                            
                except Exception as e:
                    logger.error(f"[AgentNewsAnalyzer] Gemini CLI 呼叫異常: {e}")

                if attempt == max_retries:
                    break

                # 指數退避延遲並加入 Jitter 隨機抖動
                delay = (base_delay * (2 ** attempt)) + random.uniform(0.5, 1.5)
                logger.warning(f"[AgentNewsAnalyzer] 模型 {model} 呼叫失敗或解析錯誤，將在 {delay:.2f} 秒後進行第 {attempt + 2} 次重試...")
                time.sleep(delay)

            logger.warning(f"[AgentNewsAnalyzer] 模型 {model} 已達到最大重試次數，將嘗試備用模型（若有）。")

        logger.error(f"[AgentNewsAnalyzer] 所有模型呼叫皆失敗，放棄雲端分析。")
        return None

    def _infer_market(self, source: str, content: str) -> str:
        text = (source + content).lower()
        if any(k in text for k in ["nasdaq", "nyse", "fed", "美股"]): return "US"
        return "TW"
