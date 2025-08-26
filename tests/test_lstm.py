import pandas as pd
import yfinance as yf
from strategies.lstm_strategy import train_lstm_model, predict_next_price
import numpy as np

# Veri indir
data = yf.download("SPY", start="2010-01-01", end="2024-01-01")
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data.rename(columns={'Close': 'close'}, inplace=True)

# Model eğit
model, scaler, X_test, y_test = train_lstm_model(
    data,
    window_size=60,
    epochs=5,  # Hızlı test için az
    save_model_path = os.path.join("models", "lstm_model.h5")
)

# Son fiyatı tahmin et
window_size = 60
last_60_days = data['close'].values[-window_size:]
pred_price = predict_next_price(model, scaler, last_60_days, window_size)

current_price = data['close'].iloc[-1]
print(f"Geçerli Fiyat: ${current_price:.2f}")
print(f"Tahmini Sonraki Fiyat: ${pred_price:.2f}")
print(f"Tahmini Değişim: %{(pred_price - current_price) / current_price * 100:.2f}")