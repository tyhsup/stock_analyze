import requests

prompt = """
我們正在進行 v3.0：結構化關聯建模 (Relational Modeling)。
目前的架構是 TensorFlow/Keras 基礎的多模態 LSTM 模型，包含：
1. 時間序列分支 (LSTM)
2. 外部特徵分支 (Dense, 包含 v2.0 的 FinBERT 768維 Embedding)
3. 目前僅支援單一股票預測。

目標：
引入 GNN (Graph Neural Network) 或 Node Transformer 概念，讓模型能學習股票間的連動性。

請針對以下具體問題提供技術建議與虛擬代碼：
1. **鄰接矩陣構建**：在 MySQL 資料庫環境下，如何最有效率地在訓練前構建「產業分類 (Sector)」或「歷史相關性 (Correlation)」鄰接矩陣？
2. **Keras GNN 層實現**：請提供一個簡單的 Graph Convolutional Layer (GCN) 或 Multi-head Attention based Node Transformer 類別（繼承 tf.keras.layers.Layer），並說明如何將其整合進目前的 `build_multi_input_model` 中。
3. **多股票批次處理 (DataLoader)**：若要實現跨節點傳播，原本的單股票 `IntegratedStockPredModel` 應如何調整以同時處理相關聯的股票節點數據？
4. **效能考量**：若圖節點過多（例如全台股），如何平衡運算複雜度與預測精度？

請提供詳細的分析報告與 Python 代碼範例。
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
    with open('gemma_v3_gnn_plan.md', 'w', encoding='utf-8') as f:
        f.write(response.json().get("response", "No response from Gemma"))
    print("Done")
except Exception as e:
    print(f"Error: {e}")
