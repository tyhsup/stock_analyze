這是一份針對 **v3.0：結構化關聯建模 (Relational Modeling)** 的技術架構建議書。從單一股票預測轉向圖神經網路 (GNN) 是量化交易模型從「特徵工程」邁向「拓撲工程」的關鍵一步。

---

### 1. 鄰接矩陣構建 (Adjacency Matrix Construction)

在 MySQL 環境下，不建議在訓練時即時計算相關性，這會造成嚴重的 I/O 瓶頸。建議採用 **「預計算 + 緩存」** 的策略。

#### 方案 A：基於產業分類 (Static/Sparse)
*   **邏輯**：利用 SQL 的 `JOIN` 建立同產業股票的連通性。
*   **優點**：矩陣極度稀疏，運算極快，穩定性高。
*   **SQL 實作建議**：
    ```sql
    -- 建立一個 adjacency_table，儲存 (stock_a, stock_b, weight)
    INSERT INTO stock_adjacency (stock_a, stock_b, weight)
    SELECT a.ticker, b.ticker, 1.0
    FROM stock_metadata a
    JOIN stock_metadata b ON a.sector_id = b.sector_id
    WHERE a.ticker <> b.ticker;
    ```

#### 方案 B：基於歷史相關性 (Dynamic/Dense)
*   **邏輯**：計算過去 $N$ 天收益率的 Pearson Correlation。
*   **優點**：能捕捉市場情緒與供應鏈連動的動態變化。
*   **實作建議**：
    1.  **離線計算**：使用 Python (Pandas/NumPy) 定期（如每日收盤後）計算相關性矩陣。
    2.  **閾值化 (Thresholding)**：僅保留 $|r| > 0.5$ 的邊，將 Dense 矩陣轉為 Sparse 矩陣，避免 $O(N^2)$ 爆炸。
    3.  **存儲**：將結果存回 MySQL 的 `correlation_matrix` 表，或直接存成 `.npy` 檔案。

**建議策略**：採用 **Hybrid Approach**。以產業分類作為「骨架」（保證結構穩定），以相關性作為「權重」（捕捉動態）。

---

### 2. Keras GNN 層實現與模型整合

我們將實現一個 **Graph Convolutional Layer (GCN)**。其核心公式為：
$$Z = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} X W$$
其中 $\tilde{A}$ 是包含自環 (Self-loop) 的鄰接矩陣，$\tilde{D}$ 是度矩陣。

#### Python 代碼範例

```python
import tensorflow as tf
from tensorflow.keras import layers

class GCNLayer(layers.Layer):
    def __init__(self, units, activation='relu', **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # input_shape: [batch, nodes, features]
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
error           initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )
        super(GCNLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs: list [node_features, adjacency_matrix]
        # x: [batch, nodes, features]
        # adj: [batch, nodes, nodes]
        x, adj = inputs
        
        # 1. 矩陣乘法: A * X
        # [batch, nodes, nodes] @ [batch, nodes, features] -> [batch, nodes, features]
        support = tf.matmul(adj, x)
        
        # 2. 權重轉換: (A * X) * W
        output = tf.matmul(support, self.kernel)
        
        return self.activation(output)

def build_multi_input_model(num_nodes, time_steps, ts_features, emb_features, adj_dim):
    # Input 1: Time Series (LSTM Branch)
    ts_input = layers.Input(shape=(time_steps, ts_features), name='ts_input')
    lstm_out = layers.LSTM(64, return_sequences=False)(ts_input)
    
    # Input 2: External Features (FinBERT Branch)
    ext_input = layers.Input(shape=(emb_features,), name='ext_input')
    dense_out = layers.Dense(64, activation='relu')(ext_input)
    
    # Merge Branches to create Node Features
    node_features = layers.Concatenate()([lstm_out, dense_out])
    # Reshape for GNN: [batch, nodes, features] 
    # Note: In a real batch, we need to expand this to all nodes in the graph
    # For simplicity, assume we pass a feature tensor of [batch, nodes, features]
    
    # Input 3: Adjacency Matrix
    adj_input = layers.Input(shape=(num_nodes, num_nodes), name='adj_input')
    
    # GNN Processing
    # We need to broadcast node_features to match the graph structure
    # In v3.0, the 'node_features' should be pre-aligned with the adj_input
    gcn_out = GCNLayer(32)([node_features, adj_input])
    
    # Output: Prediction for each node
    output = layers.Dense(1, name='prediction')(gcn_out)
    
    model = tf.keras.Model(inputs=[ts_input, ext_input, adj_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model
```

---

### 3. 多股票批次處理 (DataLoader) 調整

在 v2.0 中，你的 Batch 是 `(Batch_Size, Time, Features)`，每個 Sample 是一隻股票。
在 v3.0 中，你的 Batch 必須是 **「圖的快照 (Graph Snapshot)」**。

**結構變更：**
*   **Old**: `X_ts: [B, T, F]`, `X_ext: [B, E]` $\rightarrow$ `Y: [B, 1]`
*   **New**: `X_ts: [B, N, T, F]`, `X_ext: [B, N, E]`, `A: [B, N, N]` $\rightarrow$ `Y: [B, N, 1]`

**DataLoader 實作邏輯：**
1.  **Windowing**: 每次取一個時間窗口 $t$。
2.  **Graph Construction**: 
    *   從資料庫取出該時間點 $t$ 相關的所有 $N$ 隻股票。
    *   從預計算的 `adj_matrix` 提取對應的子圖。
3.  **Tensor Construction**: 
    *   將 $N$ 隻股票的 LSTM 特徵堆疊成 `[N, T, F]`。
    *   將 $N$ 隻股票的 Embedding 堆疊成 `[N, E]`。
    *   將 Batch 擴展為 `[Batch_Size, N, ...]`。

---

### 4. 效能考量與大規模圖處理 (Scalability)

若處理全台股（約 1000+ 節點），$A$ 矩陣的大小為 $1000 \times 1000$，計算量隨節點數 $N$ 的平方增長。

**優化策略：**

1.  **Sparsification (稀疏化)**:
    *   **不要使用 Dense Matrix**。使用 `tf.sparse.SparseTensor`。
    *   只保留每個節點度數 (Degree) 最高的 $K$ 個鄰居（例如 $K=10$）。這將複雜度從 $O(N^2)$ 降至 $O(N \times K)$。

2.  **Graph Partitioning (圖分割)**:
    *   不要一次預測全台股。將股票按「產業」拆分為多個小圖（Sub-graphs）。
    *   例如：建立「半導體圖」、「金融圖」、「電子圖」。這能大幅降低單次 GPU 顯存壓力。

3.  **Node Sampling (節點採樣)**:
    *   參考 **GraphSAGE** 的概念。在訓練時，不輸入全圖，而是從圖中隨機採樣一組節點及其鄰居，構建一個小型的 Batch 進行訓練。

4.  **Hierarchical Modeling (層次化建模)**:
    *   **Level 1**: 預測產業指數 (Sector Index)。
    *   **Level 2**: 在產業內進行股票間的 GNN 傳播。
    *   這樣可以將一個巨大的圖問題分解為多個中型圖問題。

### 總結建議架構圖 (v3.0)

| 模組 | 核心技術 | 目的 |
| :--- | :--- | :--- |
| **Data Layer** | MySQL + Pre-computed Pearson | 建立高效、稀疏的結構化關係 |
| **Feature Branch** | LSTM + FinBERT | 提取單一節點的時間序列與語義特徵 |
| **Relational Branch** | **GCN / Graph Attention (GAT)** | 透過 Adjacency Matrix 進行特徵聚合 (Aggregation) |
| **Inference** | Sub-graph Sampling | 解決全台股規模下的運算爆炸問題 |