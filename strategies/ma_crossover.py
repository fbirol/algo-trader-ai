# strategies/ma_crossover.py
import pandas as pd

def generate_signals(data, short_window=50, long_window=200):
    """
    MA kesişimi ile AL/SAT sinyali üret.
    """
    data = data.copy()
    data['MA_short'] = data['close'].rolling(short_window).mean()
    data['MA_long'] = data['close'].rolling(long_window).mean()

    # Sinyal: MA_short > MA_long ise 1 (AL), değilse 0
    data['signal'] = (data['MA_short'] > data['MA_long']).astype(int)

    # Pozisyon: sinyal değişimi = işlem fırsatı
    data['position'] = data['signal'].diff()

    return data