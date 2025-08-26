# strategy_backtest.py
def run_ma_strategy(data, short_window=50, long_window=200):
    # MA hesapla
    data['MA_short'] = data['close'].rolling(short_window).mean()
    data['MA_long'] = data['close'].rolling(long_window).mean()

    # Sinyal üret
    data['Signal'] = (data['MA_short'] > data['MA_long']).astype(int)
    data['Position'] = data['Signal'].diff()

    # Performans simülasyonu
    capital = 10000
    shares = 0
    position = 0
    for i in range(long_window, len(data)):
        price = data['close'].iloc[i]
        sig = data['Position'].iloc[i]
        if sig == 1 and position == 0:
            shares = capital / price
            capital = 0
            position = 1
        elif sig == -1 and position == 1:
            capital = shares * price
            shares = 0
            position = 0
    if position == 1:
        capital = shares * data['close'].iloc[-1]
    return capital