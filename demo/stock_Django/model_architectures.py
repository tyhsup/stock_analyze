import tensorflow as tf
from keras.layers import Dense, LSTM, Dropout, Input, Concatenate, TimeDistributed, Reshape
from keras.models import Model
import keras

class GCNLayer(keras.layers.Layer):
    """
    簡單的圖卷積層 (Graph Convolutional Layer)
    公式: Z = A * X * W
    """
    def __init__(self, units, activation='relu', **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        # input_shape[0] 是節點特徵 [batch, nodes, features]
        # input_shape[1] 是鄰接矩陣 [batch, nodes, nodes]
        feat_dim = input_shape[0][-1]
        self.kernel = self.add_weight(
            shape=(feat_dim, self.units),
            initializer='glorot_uniform',
            name='kernel'
        )
        super(GCNLayer, self).build(input_shape)

    def call(self, inputs):
        features, adj = inputs
        # 1. 聚合鄰居特徵: A * X
        # [batch, nodes, nodes] @ [batch, nodes, features] -> [batch, nodes, features]
        support = tf.matmul(adj, features)
        # 2. 線性變換: (A * X) * W
        output = tf.matmul(support, self.kernel)
        return self.activation(output)

class CrossModalAttention(keras.layers.Layer):
    """
    跨模態注意力融合層 (Cross-Modal Attention Fusion)
    將技術面、輿情面、財務面特徵進行動態權重匯聚
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1, **kwargs):
        super(CrossModalAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.mha = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.layernorm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, inputs):
        # inputs: [ts_feat, senti_feat, fin_feat]
        # 每個輸入形狀皆為 [batch, nodes, d_model]
        
        # 1. 堆疊模態: [batch, nodes, 3, d_model]
        stacked = tf.stack(inputs, axis=2)
        
        # 2. 自注意力機制 (在模態維度進行 MHA)
        # 我們將 (nodes * batch) 視為序列背景，對 3 個模態進行交互
        batch_size = tf.shape(stacked)[0]
        num_nodes = tf.shape(stacked)[1]
        
        # 重塑為 [Batch*Nodes, 3, d_model] 進行標準 Transformer-like MHA
        reshaped = tf.reshape(stacked, [-1, 3, self.d_model])
        attn_output = self.mha(reshaped, reshaped)
        attn_output = self.dropout(attn_output)
        
        # 3. 殘差連接與 LayerNorm
        out = self.layernorm(reshaped + attn_output)
        
        # 4. 匯聚模態 (使用 Global Average Pooling Over Modalities 或 Flatten)
        # 這裡選擇取平均，代表融合後的節點特徵
        fused = tf.reduce_mean(out, axis=1) # [Batch*Nodes, d_model]
        
        # 5. 回原形狀: [Batch, Nodes, d_model]
        return tf.reshape(fused, [batch_size, num_nodes, self.d_model])

class StockModelArchitectures:
    @staticmethod
    def build_multi_input_model(lstm_feat_dim, senti_feat_dim, fin_feat_dim, config):
        """建構 v4.0 Cross-Modal Attention + GNN 模型"""
        time_steps = config.get('time_steps', 20)
        predict_steps = config.get('predict_steps', 5)
        num_nodes = config.get('neighbor_count', 5) + 1
        d_model = config.get('attention_dim', 128)
        
        # --- 輸入端定義 ---
        input_ts = Input(shape=(num_nodes, time_steps, lstm_feat_dim), name="time_series_input")
        input_senti = Input(shape=(num_nodes, senti_feat_dim), name="sentiment_input")
        input_fin = Input(shape=(num_nodes, fin_feat_dim), name="financial_input")
        input_adj = Input(shape=(num_nodes, num_nodes), name="adj_input")
        
        # --- 1. 技術面分支 (Temporal) ---
        x = TimeDistributed(LSTM(config.get('lstm_units_1', 128), return_sequences=True))(input_ts)
        x = TimeDistributed(Dropout(config.get('dropout_rate', 0.2)))(x)
        x = TimeDistributed(LSTM(config.get('lstm_units_2', d_model)))(x) # Output: (Batch, Nodes, d_model)
        
        # --- 2. 輿情面分支 (Sentiment) ---
        y = TimeDistributed(Dense(config.get('dense_units_senti', 256), activation='relu'))(input_senti)
        y = TimeDistributed(Dense(d_model, activation='relu'))(y) # Output: (Batch, Nodes, d_model)
        
        # --- 3. 財務籌碼面分支 (Financials) ---
        z = TimeDistributed(Dense(config.get('dense_units_fin', 64), activation='relu'))(input_fin)
        z = TimeDistributed(Dense(d_model, activation='relu'))(z) # Output: (Batch, Nodes, d_model)
        
        # --- 4. Cross-Modal Attention 融合 ---
        fused_features = CrossModalAttention(d_model=d_model, num_heads=config.get('num_heads', 4))([x, y, z])
        
        # --- 5. GNN 關聯建模 ---
        gnn_out = GCNLayer(config.get('gnn_units', 64), activation='relu')([fused_features, input_adj])
        gnn_out = Dropout(config.get('dropout_rate', 0.2))(gnn_out)
        
        # 提取目標節點
        target_features = keras.layers.Lambda(lambda t: t[:, 0, :], name="target_selector")(gnn_out)
        
        # 最後預測
        dense_final = Dense(config.get('dense_units_combined', 128), activation='relu')(target_features)
        out_price = Dense(predict_steps, name='price_prediction')(dense_final)
        
        model = Model(inputs=[input_ts, input_senti, input_fin, input_adj], outputs=out_price)
        
        # 針對 GPU 優化編譯
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate', 0.0005)), 
            loss=tf.keras.losses.Huber(delta=1.0), 
            metrics=['mae']
        )
        return model
