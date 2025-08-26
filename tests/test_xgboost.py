import pandas as pd
import yfinance as yf
from strategies.xgboost_strategy import add_features, train_xgboost_model

# Veri indir
data = yf.download("SPY", start="2010-01-01", end="2024-01-01")
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'}, inplace=True)

# Özellik ekle
data = add_features(data)

# Özellik listesi
features = [
    'MA5', 'MA10', 'MA20', 'MA50',
    'RSI', 'MACD', 'MACD_signal',
    'Volatility', 'Price_Change', 'Volume_Change',
    'BB_width'
]

# Model eğit
model, acc, clean_data = train_xgboost_model(data, features, save_model_path="xgboost_model.pkl")