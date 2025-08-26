# strategies/ensemble_strategy.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def generate_ensemble_signal(data, ma_weight=1.0, xgb_weight=1.0, lstm_weight=1.0):
    """
    Üç modelin sinyalini birleştir: MA, XGBoost, LSTM
    Ağırlıklı oylama ile nihai sinyal üret.
    """
    data = data.copy()
    signals = pd.DataFrame(index=data.index)

    # ----------------------------
    # 1. MA Crossover Sinyali
    # ----------------------------
    data['MA50'] = data['close'].rolling(50).mean()
    data['MA200'] = data['close'].rolling(200).mean()
    signals['MA'] = (data['MA50'] > data['MA200']).astype(int)

    # ----------------------------
    # 2. XGBoost Sinyali (önceden eğitilmiş modelle)
    # ----------------------------
    try:
        from strategies.xgboost_strategy import add_features
        import joblib

        # Özellik ekle
        data_xgb = add_features(data)
        features = ['MA5', 'MA10', 'MA20', 'MA50', 'RSI', 'MACD', 'MACD_signal',
                   'Volatility', 'Price_Change', 'Volume_Change', 'BB_width']

        # Model yükle
        model = joblib.load("models/xgboost_model.pkl")
        X = data_xgb[features].dropna()
        valid_idx = X.index

        preds = model.predict(X)
        xgb_signal = pd.Series(preds, index=valid_idx)
        signals['XGB'] = 0
        signals.loc[valid_idx, 'XGB'] = xgb_signal
    except Exception as e:
        print(f"XGBoost hatası: {e}")
        signals['XGB'] = 0  # Hata olursa pasif

    # ----------------------------
    # 3. LSTM Sinyali
    # ----------------------------
    try:
        from tensorflow.keras.models import load_model
        import joblib

        lstm_model = load_model("models/lstm_model.h5")
        scaler = joblib.load("models/lstm_model_scaler.pkl")
        window_size = 60

        dataset = data['close'].values.reshape(-1, 1)
        scaled_data = scaler.transform(dataset)

        lstm_preds = []
        for i in range(window_size, len(scaled_data)):
            X = scaled_data[i - window_size:i].reshape(1, window_size, 1)
            pred = lstm_model.predict(X, verbose=0)
            lstm_preds.append(pred[0, 0])

        # Tahmin > mevcut fiyat → AL (1), değilse SAT (0)
        lstm_signals = []
        for i in range(window_size, len(data)):
            pred_price = scaler.inverse_transform([[lstm_preds[i - window_size]]])[0, 0]
            current_price = data['close'].iloc[i]
            lstm_signals.append(1 if pred_price > current_price else 0)

        signals['LSTM'] = 0
        signals.loc[data.index[window_size]:, 'LSTM'] = lstm_signals
    except Exception as e:
        print(f"LSTM hatası: {e}")
        signals['LSTM'] = 0  # Hata olursa pasif

    # ----------------------------
    # 4. Ağırlıklı Oylama
    # ----------------------------
    weighted_sum = (
        signals['MA'] * ma_weight +
        signals['XGB'] * xgb_weight +
        signals['LSTM'] * lstm_weight
    )
    total_weight = ma_weight + xgb_weight + lstm_weight
    avg_score = weighted_sum / total_weight

    # Sinyal: 0.5'ten büyükse AL (1), değilse SAT (0)
    final_signal = (avg_score >= 0.5).astype(int)

    # Pozisyon değişimi (AL/SAT için)
    position = final_signal.diff()

    return final_signal, position, signals