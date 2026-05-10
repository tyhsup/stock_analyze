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

try:
    from groq import Groq
except ImportError:
    raise ImportError("無法載入 groq 庫。請執行: pip install groq")

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Stage 1：Chinese-Optimized Sentiment Scorer (本地 GPU)
# ══════════════════════════════════════════════════════════════════
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
# Stage 2：Llama 3.3 70B 定性解釋器 (Groq API)
# ══════════════════════════════════════════════════════════════════
LLAMA_SYSTEM_PROMPT = """你是專業金融分析師。本地模型已提供情緒初步評分，請你補充定性分析並輸出 JSON：
{
  "market": "TW 或 US",
  "impact_scope": "短期 或 中期 或 長期",
  "reasoning_summary": "50字以內的深度判斷理由"
}
只輸出 JSON。"""


class AgentNewsAnalyzer:
    """
    金融新聞混合分析器。
    Stage 1: 本地 Roberta 模型 (快速評分)
    Stage 2: Groq Llama 70B (深度解釋)
    """

    def __init__(self):
        # 讀取 API Key (優先從環境變數，次之從 .env)
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            from dotenv import load_dotenv
            dotenv_path = os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", ".env")
            load_dotenv(dotenv_path)
            self.api_key = os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("找不到 GROQ_API_KEY。")

        self.llm_model = "llama-3.3-70b-versatile"
        self.groq_client = Groq(api_key=self.api_key)

        logger.info("[AgentNewsAnalyzer] 載入本地 Scorer...")
        self.scorer = FinBertScorer()

    def analyze_news(self, news_data: dict, force_llm: bool = False) -> dict:
        title   = news_data.get("title",   news_data.get("標題", ""))
        date    = news_data.get("date",    news_data.get("發布時間", ""))
        content = news_data.get("content", news_data.get("內文", ""))
        link    = news_data.get("link",    news_data.get("連結", ""))
        source  = news_data.get("source",  news_data.get("來源", ""))

        # Stage 1: 本地快速評分
        score_res = self.scorer.analyze(title, content)
        
        # 升級判斷
        llama_res = {}
        should_upgrade = force_llm or len(content) > 200 or score_res["confidence"] < 0.75
        
        if should_upgrade:
            llama_res = self._call_llama(title, content, score_res) or {}

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
            "market": llama_res.get("market", self._infer_market(source, content)),
            "impact_scope": llama_res.get("impact_scope", "短期"),
            "reasoning_summary": llama_res.get("reasoning_summary", f"本地評分完成 (信心度 {score_res['confidence']:.0%})")
        }

    def _call_llama(self, title: str, content: str, score_res: dict) -> Optional[dict]:
        prompt = f"新聞：{title}\n內容摘要：{content[:300]}\n本地評分：{score_res}\n請提供市場與定性分析。"
        try:
            completion = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": LLAMA_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            logger.warning(f"Llama 呼叫失敗: {e}")
            return None

    def _infer_market(self, source: str, content: str) -> str:
        text = (source + content).lower()
        if any(k in text for k in ["nasdaq", "nyse", "fed", "美股"]): return "US"
        return "TW"
