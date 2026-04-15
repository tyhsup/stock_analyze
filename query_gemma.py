import requests

prompt = """
我們已經完成了 v1.0 (基礎架構解耦)。接下來請提供具體的 Python 實作指引與程式碼，幫助我進入「v2.0 語義特徵強化」。
目前在 `dataset_builders.py` 中，有一個 `SentimentProbabilityModel` 類別，它呼叫 `StockUtils.Sentiment_indicators` 取得新聞情緒，回傳 `Pos_Ratio` 與 `Neg_Ratio` 兩行數據。

需求：
1. 繼續使用 `StockUtils.Sentiment_indicators` 取得歷史新聞（假設回傳的 DataFrame 內包含新聞標題 'Title' 與內容 'Content' 欄位）。
2. 在 `SentimentProbabilityModel` 內引入 HuggingFace `transformers` 與 `ProsusAI/finbert`。
3. 捨棄原本的 `Pos_Ratio`, `Neg_Ratio`，將每一則新聞利用 FinBERT 取出 `[CLS]` 提取 768 維度的隱藏層特徵。如果同一天有多則新聞，請取當日新聞特徵向量的平均。
4. 返回以日期為 index 的 DataFrame，各欄位名稱可為 'finbert_emb_0' 到 'finbert_emb_767'。
5. 考慮推論速度，請告訴我使用 PyTorch 抓取最終層 (last hidden state) 的最佳寫法（不需要分類層 Outputs，只需要 backbone embedding）。

請給出具備完整註解的改寫後 `SentimentProbabilityModel` 程式碼，並指出是否有其他需安裝的相依套件。
"""

OLLAMA_URL = "http://localhost:11434/api/generate"
payload = {
    "model": "gemma4:26b",
    "prompt": prompt,
    "stream": False,
    "options": {"temperature": 0.3}
}

try:
    response = requests.post(OLLAMA_URL, json=payload, timeout=600)
    with open('gemma_finbert_result.md', 'w', encoding='utf-8') as f:
        f.write(response.json().get("response", "No response from Gemma"))
    print("Done")
except Exception as e:
    print(f"Error: {e}")
