import os
import sys
import json
import logging
import time
from typing import Optional

# ──────────────────────────────────────────────
# 全局 Groq helper 路徑
# ──────────────────────────────────────────────
GLOBAL_HELPER_PATH = r"c:\Users\許廷宇\.gemini\antigravity\scripts"
if GLOBAL_HELPER_PATH not in sys.path:
    sys.path.append(GLOBAL_HELPER_PATH)

# Groq library imports are no longer mandatory as we use Gemini CLI

logger = logging.getLogger(__name__)
import threading
import random

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


class FinBertScorer:
    """
    使用 IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment 進行中文情緒分析。
    - 優先使用 CUDA GPU。
    - 雖然名稱不是 FinBERT，但這是目前本地執行中文情緒分析最穩定、快速的選擇。
    - 輸出映射：
        0: Negative -> 負面
        1: Positive -> 正面
    - 若信心度低於門檻，自動判定為「中立」。
    """

    def __init__(self, model_name: str = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[Scorer] 初始化，裝置: {self.device}，模型: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self._torch = torch
        self.neutral_threshold = 0.65  # 若最高信心度低於此值，視為中立

    def analyze(self, title: str, content: str) -> dict:
        """
        分析新聞，回傳量化情緒指標。
        """
        combined = f"{title}. {content[:200]}".strip()
        if not combined:
            return {"positive_negative_analysis": "中立", "sentiment_score": 0.0, "confidence": 0.0}

        try:
            inputs = self.tokenizer(
                combined,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with self._torch.no_grad():
                outputs = self.model(**inputs)
                probs = self._torch.softmax(outputs.logits, dim=-1)[0].cpu().tolist()

            # Erlangshen 輸出：[Negative_prob, Positive_prob]
            neg_prob = probs[0]
            pos_prob = probs[1]

            confidence = max(neg_prob, pos_prob)
            
            if confidence < self.neutral_threshold:
                label = "中立"
                sentiment_score = 0.0
            else:
                label = "正面" if pos_prob > neg_prob else "負面"
                sentiment_score = round(pos_prob - neg_prob, 4)

            return {
                "positive_negative_analysis": label,
                "sentiment_score": sentiment_score,
                "confidence": round(confidence, 4)
            }

        except Exception as e:
            logger.error(f"[Scorer] 推論失敗: {e}")
            return {"positive_negative_analysis": "中立", "sentiment_score": 0.0, "confidence": 0.0}


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
    Stage 1: 本地 Roberta 模型 (快速評分)
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
        title   = news_data.get("title",   news_data.get("標題", ""))
        date    = news_data.get("date",    news_data.get("發布時間", ""))
        content = news_data.get("content", news_data.get("內文", ""))
        link    = news_data.get("link",    news_data.get("連結", ""))
        source  = news_data.get("source",  news_data.get("來源", ""))

        # Stage 1: 本地快速評分
        score_res = self.scorer.analyze(title, content)
        
        # 升級判斷 (重要新聞篩選機制)：
        # 僅在本地判定為「非中立」且「長度大於 400 字」且「新聞屬於 7 天內 (is_recent=True)」時升級至雲端
        is_neutral = score_res["positive_negative_analysis"] == "中立"
        should_upgrade = force_llm or (
            not is_neutral 
            and len(content) > 400 
            and is_recent
        )
        
        gemini_res = {}
        if should_upgrade:
            gemini_res = self._call_gemini(title, content, score_res) or {}

        # 合併
        return {
            "title": title,
            "date": date,
            "content": content,
            "link": link,
            "source": source,
            "positive_negative_analysis": score_res["positive_negative_analysis"],
            "sentiment_score": score_res["sentiment_score"],
            "confidence": score_res["confidence"],
            "market": gemini_res.get("market", self._infer_market(source, content)),
            "impact_scope": gemini_res.get("impact_scope", "短期"),
            "reasoning_summary": gemini_res.get("reasoning_summary", f"本地評分完成 (信心度 {score_res['confidence']:.0%})")
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

        for attempt in range(max_retries + 1):
            # 1. 確保雲端呼叫符合 10 RPM 速率限制 (最小間隔 6.0 秒)
            self.rate_limiter.wait_if_needed()

            try:
                logger.info(f"[AgentNewsAnalyzer] 調用 Gemini CLI 雲端 Gemma 4 31B 模型 (嘗試 {attempt + 1}/{max_retries + 1})...")
                args = [self.gemini_path, "-m", "gemma-4-31b-it", "--skip-trust", "-o", "json", "-p", full_prompt]
                
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
            logger.warning(f"[AgentNewsAnalyzer] 雲端分析失敗或解析錯誤，將在 {delay:.2f} 秒後進行第 {attempt + 1} 次重試...")
            time.sleep(delay)

        logger.error(f"[AgentNewsAnalyzer] 已達到最大重試次數 ({max_retries})，放棄雲端分析。")
        return None

    def _infer_market(self, source: str, content: str) -> str:
        text = (source + content).lower()
        if any(k in text for k in ["nasdaq", "nyse", "fed", "美股"]): return "US"
        return "TW"
