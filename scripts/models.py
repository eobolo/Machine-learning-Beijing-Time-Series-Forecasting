import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, BatchNormalization, Bidirectional

def build_lstm_model(n_features, seq_length=12, units=25, learning_rate=0.0001, activation='sigmoid', batch_norm=False):
    """
    Build a standard LSTM model for PM2.5 prediction.
    Used for Models 1–4.
    """
    input_layer = Input(shape=(seq_length, n_features))
    lstm_out = LSTM(units, return_sequences=True, activation='tanh')(input_layer)
    lstm_out = LSTM(units, return_sequences=True, activation='tanh')(lstm_out)
    if batch_norm:
        lstm_out = BatchNormalization()(lstm_out)
    output_last = lstm_out[:, -1, :]
    final_output = Dense(1, activation=activation)(output_last)

    model = Model(inputs=input_layer, outputs=final_output)
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate), loss='mse')
    return model

def build_bilstm_model(n_features, seq_length=48, units=50, learning_rate=0.00001, activation='sigmoid', batch_norm=True):
    """
    Build a Bidirectional LSTM model for PM2.5 prediction.
    Used for Models 5–7.
    """
    input_layer = Input(shape=(seq_length, n_features))
    lstm_out = Bidirectional(LSTM(units, return_sequences=True, activation='tanh', kernel_initializer=tf.keras.initializers.LecunNormal(seed=21)))(input_layer)
    lstm_out = Bidirectional(LSTM(units, return_sequences=True, activation='tanh'))(lstm_out)
    if batch_norm:
        lstm_out = BatchNormalization()(lstm_out)
    output_last = lstm_out[:, -1, :]
    final_output = Dense(1, activation=activation)(output_last)

    model = Model(inputs=input_layer, outputs=final_output)
    if activation == 'sigmoid':
        model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate), loss='mse')
    else:
        # For linear output (Model 7), add RMSE metric
        def rmse_metric(y_true, y_pred):
            return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
        model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate), loss='mse', metrics=[rmse_metric])
    return model