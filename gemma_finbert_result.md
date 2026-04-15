這是一個非常關鍵的升級。從「標量比例 (Scalar Ratio)」轉向「高維向量 (Dense Embedding)」標誌著你的模型從**特徵工程 (Feature Engineering)** 轉向了**表徵學習 (Representation Learning)**。

在 v2.0 中，我們不再只是告訴模型「情緒好壞」，而是提供「情緒的語義上下文」。

### 1. 需安裝的相依套件

請在你的環境中執行以下指令，確保安裝了處理深度學習模型所需的套件：

```bash
pip install torch transformers pandas numpy tqdm
```

---

### 2. 核心實作指引：`SentimentProbabilityModel` (v2.0)

為了達到高效能，我們將使用 `AutoModel` 而非 `AutoModelForSequenceClassification`。這樣可以跳過最後的分類層（Linear Layer），直接取得 Backbone 的輸出。

```python
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import List

# 假設這是你原本的工具類別
# from your_module import StockUtils 

class SentimentProbabilityModel:
    """
    v2.0 語義特徵強化模型
    功能：使用 FinBERT 提取新聞語義 Embedding，並按日期進行特徵聚合。
    """

    def __init__(self, model_name: str = 'ProsusAI/finbert', device: str = None):
        """
        初始化 Transformer 模型與 Tokenizer
        :param model_name: HuggingFace 模型名稱
        :param device: 指定運算裝置 ('cuda' 或 'cpu')
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在載入模型 {model_name} 至 {self.device}...")
        
        # 使用 AutoTokenizer 與 AutoModel (Backbone)
        # 使用 AutoModel 而非 AutoModelForSequenceClassification 可以直接取得隱藏層特徵
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # 設定為評估模式 (關閉 Dropout)
        
        print("模型載入完成。")

    @torch.no_grad()  # 關鍵：推論時不計算梯度，節省記憶體與運算量
    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        高效能批次處理：將文本轉換為 768 維向量
        """
        all_embeddings = []
        
        # 將文本切分成 batch，避免一次性載入過大導致 OOM (Out of Memory)
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            # Tokenization: 處理 Padding 與 Truncation
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.device)

            # Forward Pass
            outputs = self.model(**inputs)

            # 【核心技巧】取得 [CLS] token 的隱藏層特徵
            # outputs.last_hidden_tate 的維度是 [batch_size, sequence_length, 768]
            # index 0 即為 [CLS] token，代表整句的語義摘要
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            all_embeddings.append(cls_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def build(self, stock_code: str) -> pd.DataFrame:
        """
        主流程：取得原始新聞 -> 提取 Embedding -> 按日期聚合
        :return: DataFrame (Index: Date, Columns: finbert_emb_0 ~ 767)
        """
        print(f"正在為 {stock_code} 構建語義特徵...")

        # 1. 從 StockUtils 取得原始新聞數據 (假設包含 'Date', 'Title', 'Content')
        # 注意：這裡假設 StockUtils.Sentiment_indicators 已經改寫為回傳原始新聞 DataFrame
        from your_module import StockUtils # 替換為實際路徑
        df_news = StockUtils.Sentiment_indicators(stock_code)

        if df_news.empty:
            print(f"警告：{stock_code} 無新聞數據。")
            return pd.DataFrame()

        # 2. 預處理：合併 Title 與 Content，並處理缺失值
        df_news['text'] = df_news['Title'].fillna('') + " " + df_news['Content'].fillna('')
        df_news['text'] = df_news['text'].str.strip()
        
        # 移除空字串
        df_news = df_news[df_news['text'] != ""].copy()

        # 3. 執行 Embedding 提取
        print(f"正在提取 {len(df_news)} 則新聞的語義特徵...")
        embeddings = self._get_embeddings_batch(df_news['text'].tolist())
        
        # 將 Embedding 放回 DataFrame 方便後續按日期處理
        df_news['embedding'] = list(embeddings)

        # 4. 聚合處理：同一天有多則新聞時，取平均值 (Mean Pooling)
        # 我們將每一天的所有向量取平均，得到該日的「每日語義特徵」
        print("正在進行每日特徵聚合...")
        daily_df = df_news.groupby('Date')['embedding'].apply(
            lambda x: np.mean(np.stack(x.values), axis=0)
        ).reset_index()

        # 5. 格式化輸出：轉換為寬表格式 (Wide Format)
        # 建立欄位名稱: finbert_emb_0, finbert_emb_1, ...
        embedding_cols = [f'finbert_emb_{i}' for i in range(768)]
        embedding_matrix = np.stack(daily_df['embedding'].values)
        
        final_df = pd.DataFrame(
            embedding_matrix, 
            index=daily_df['Date'], 
            columns=embedding_cols
        )

        print(f"完成！輸出維度: {final_df.shape}")
        return final_df

```

---

### 3. 關鍵技術點解析 (v2.0 升級重點)

#### A. 為什麼使用 `AutoModel` 而不是 `AutoModelForSequenceClassification`？
*   **傳統做法 (v1.0 延伸)**：`AutoModelForSequenceClassification` 會在 Transformer 的輸出後接一個 `Linear(768, 3)` 的層來做情緒分類。
*   **v2.0 做法**：我們只需要「特徵」。使用 `AutoModel` 會直接回傳 Backbone 的輸出。這不僅減少了計算量，也避免了模型被預訓練時的分類標籤所干擾，讓我們拿到最純粹的語義向量。

#### B. 如何高效抓取最終層特徵 (The `[CLS]` Trick)
在 BERT 架構中，第一個 Token `[CLS]` 被設計用來捕捉整句的語義。
*   **程式碼實作**：`outputs.last_hidden_state[:, 0, :]`
*   **維度變化**：
    *   `last_hidden_state`: `[Batch, Seq_Len, 768]`
    *   `[:, 0, :]`: 取得所有 Batch 的第 0 個 Token，結果維度變為 `[Batch, 768]`。
*   **效能優化**：使用 `@torch.no_grad()`。這會告訴 PyTorch 不要建立計算圖 (Computational Graph)，這能大幅減少 GPU 顯存 (VRAM) 的佔用，並加快推論速度。

#### C. 批次處理 (Batching) 與記憶體管理
*   **問題**：如果新聞有 10,000 則，一次性丟入 GPU 會導致 `CUDA Out of Memory`。
*   **解決方案**：我在 `_get_embeddings_batch` 中實作了 `batch_size` 迴圈。這確保了無論數據量多大，GPU 每次只處理固定數量（例如 16 或 32）的文本。

#### D. 數據聚合 (Aggregation Strategy)
*   **邏輯**：當某天有 5 則新聞時，我們不希望特徵維度爆炸，因此使用 `np.mean(..., axis=0)`。
*   **意義**：這代表當日的「語義重心」。如果當天有一則極度利多、四則極度利空，平均後的向量會向利空偏移，這能捕捉到當日新聞的整體情緒趨勢。

### 4. 下一步建議 (v2.1 展望)
目前的 v2.0 已經非常強大，但如果你的數據量達到萬級以上，可以考慮：
1.  **使用 ONNX Runtime**：將 FinBERT 轉換為 ONNX 格式，在 CPU 或 GPU 上執行速度可提升 2-5 倍。
2.  **引入 Attention Pooling**：不再只取 `[CLS]`，而是根據 Token 的重要性加權平均（這需要稍微修改模型結構）。