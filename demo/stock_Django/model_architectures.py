import tensorflow as tf
from keras.layers import Dense, LSTM, Dropout, Input, Concatenate
from keras.models import Model

class StockModelArchitectures:
    @staticmethod
    def build_multi_input_model(lstm_feat_dim, ext_feat_dim, config):
        """建構 Multi-Input DNN 模型 (LSTM + Dense)"""
        time_steps = config.get('time_steps', 20)
        predict_steps = config.get('predict_steps', 5)
        
        input_ts = Input(shape=(time_steps, lstm_feat_dim), name="time_series_input")
        
        # LSTM 分支
        x = LSTM(config.get('lstm_units_1', 128), return_sequences=True)(input_ts)
        x = Dropout(config.get('dropout_rate', 0.2))(x)
        x = LSTM(config.get('lstm_units_2', 64))(x)
        
        # 外部特徵分支 (情緒、籌碼、基本面等)
        input_ext = Input(shape=(ext_feat_dim,), name="external_features_input")
        y = Dense(config.get('dense_units_ext', 64), activation='relu')(input_ext)
        y = Dropout(config.get('dropout_rate', 0.2))(y)
        
        # 原有 Concatenate 做法 (預留後續升級 Cross-Modal 的結構點)
        combined = Concatenate()([x, y])
        z = Dense(config.get('dense_units_combined', 64), activation='relu')(combined)
        z = Dropout(config.get('dropout_rate', 0.2))(z)
        
        out_price = Dense(predict_steps, name='price_prediction')(z)
        
        model = Model(inputs=[input_ts, input_ext], outputs=out_price)
        
        # 使用 Huber Loss 取代 MSE，降低異常跳空值的影響
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate', 0.001)), 
            loss=tf.keras.losses.Huber(delta=config.get('huber_delta', 1.0)), 
            metrics=['mae']
        )
        return model
