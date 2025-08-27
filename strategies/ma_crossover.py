# strategies/ma_crossover.py
import pandas as pd

class MACrossoverStrategy:
    def __init__(self, short_window=50, long_window=200):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        data = data.copy()
        data['MA_short'] = data['close'].rolling(self.short_window).mean()
        data['MA_long'] = data['close'].rolling(self.long_window).mean()
        signal = (data['MA_short'] > data['MA_long']).astype(int)
        return signal