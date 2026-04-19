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

class StockModelArchitectures:
    @staticmethod
    def build_multi_input_model(lstm_feat_dim, ext_feat_dim, config):
        """建構 v3.0 GNN 增強型模型"""
        time_steps = config.get('time_steps', 20)
        predict_steps = config.get('predict_steps', 5)
        num_nodes = config.get('neighbor_count', 5) + 1 # 1 target + K neighbors
        
        # 4D Input for Time Series: (Batch, Nodes, Time, Feat)
        input_ts = Input(shape=(num_nodes, time_steps, lstm_feat_dim), name="time_series_input")
        # 3D Input for External: (Batch, Nodes, Feat)
        input_ext = Input(shape=(num_nodes, ext_feat_dim), name="external_features_input")
        # 3D Input for Adjacency Matrix: (Batch, Nodes, Nodes)
        input_adj = Input(shape=(num_nodes, num_nodes), name="adj_input")
        
        # --- 節點特徵提取分支 ---
        # 使用 TimeDistributed 將同一套 LSTM 套用在圖中的每個節點
        x = TimeDistributed(LSTM(config.get('lstm_units_1', 128), return_sequences=True))(input_ts)
        x = TimeDistributed(Dropout(config.get('dropout_rate', 0.2)))(x)
        x = TimeDistributed(LSTM(config.get('lstm_units_2', 64)))(x) # Output: (Batch, Nodes, 64)
        
        # 外部特徵提取 (Dense)
        y = TimeDistributed(Dense(config.get('dense_units_ext', 256), activation='relu'))(input_ext)
        y = TimeDistributed(Dropout(config.get('dropout_rate', 0.2)))(y) # Output: (Batch, Nodes, 256)
        
        # 拼接跨模態特徵
        node_features = Concatenate(axis=-1)([x, y]) # Output: (Batch, Nodes, 320)
        
        # --- GNN 關聯建模 ---
        # 透過 GCNLayer 讓節點間交換資訊
        gnn_out = GCNLayer(config.get('gnn_units', 32), activation='relu')([node_features, input_adj])
        gnn_out = Dropout(config.get('dropout_rate', 0.2))(gnn_out)
        
        # 提取目標節點 (index 0) 的預測結果
        # 我們只關心目標股票的預測，或者也可以輸出全圖的預測
        # 這裡採取 Reshape + Lambda 取出 Node 0
        def select_target_node(tensor):
            return tensor[:, 0, :]
            
        target_features = keras.layers.Lambda(select_target_node, name="target_selector")(gnn_out)
        
        # 最後的預測層
        z = Dense(config.get('dense_units_combined', 128), activation='relu')(target_features)
        out_price = Dense(predict_steps, name='price_prediction')(z)
        
        model = Model(inputs=[input_ts, input_ext, input_adj], outputs=out_price)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate', 0.001)), 
            loss=tf.keras.losses.Huber(delta=config.get('huber_delta', 1.0)), 
            metrics=['mae']
        )
        return model
