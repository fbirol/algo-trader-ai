# strategies/lstm_strategy.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

def prepare_lstm_data(data, feature_col='close', window_size=60):
    """
    LSTM için veri hazırla: sliding window kullanarak X ve y oluştur.
    """
    # Sadece kapanış fiyatı ile çalış
    dataset = data[feature_col].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    # LSTM için 3D formata dönüştür: (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y, scaler

def build_lstm_model(input_shape):
    """
    LSTM modelini oluştur.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))  # Tahmin: bir sonraki adım

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(data, window_size=60, epochs=50, batch_size=32, save_model_path=None):
    """
    LSTM modelini eğit.
    """
    X, y, scaler = prepare_lstm_data(data, window_size=window_size)

    # Eğitim/test ayırımı
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm_model((X_train.shape[1], 1))
    
    print("LSTM modeli eğitiliyor...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
              validation_data=(X_test, y_test), verbose=1)

    # Model ve scaler kaydet
    if save_model_path:
        model.save(save_model_path)
        scaler_path = save_model_path.replace(".h5", "_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"LSTM modeli kaydedildi: {save_model_path}")
        print(f"Scaler kaydedildi: {scaler_path}")

    return model, scaler, X_test, y_test

def predict_next_price(model, scaler, latest_data, window_size=60):
    """
    En son veriyle bir sonraki fiyatı tahmin et.
    """
    # Son window_size kadar veriyi al
    last_window = latest_data[-window_size:]
    last_window_scaled = scaler.transform(last_window.reshape(-1, 1))

    X_input = last_window_scaled.reshape(1, window_size, 1)

    pred_scaled = model.predict(X_input)
    pred_price = scaler.inverse_transform(pred_scaled)

    return pred_price[0][0]