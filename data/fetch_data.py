# data/fetch_data.py
import yfinance as yf
import pandas as pd

def fetch_yfinance_data(ticker, start, end):
    """
    Yahoo Finance'ten veri indir ve temizle.
    """
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            return None

        # MultiIndex düzleştirme
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Standart sütun isimleri
        data.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }, inplace=True)

        # NaN kontrolü
        data.dropna(inplace=True)
        return data

    except Exception as e:
        print(f"Veri indirme hatası: {e}")
        return None