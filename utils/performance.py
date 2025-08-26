# utils/performance.py
import matplotlib.pyplot as plt
import numpy as np

def backtest_strategy(data, initial_capital=10000):
    """
    Basit bir backtest simülasyonu.
    """
    capital = initial_capital
    shares = 0
    position = 0  # 0: nakit, 1: uzun
    trades = []

    for i in range(len(data)):
        row = data.iloc[i]
        close_price = row['close']
        signal = row['position']

        if signal == 1 and position == 0:  # AL
            shares = capital / close_price
            capital = 0
            position = 1
            trades.append({
                'type': 'AL',
                'price': close_price,
                'date': row.name,
                'capital': shares * close_price
            })

        elif signal == -1 and position == 1:  # SAT
            capital = shares * close_price
            shares = 0
            position = 0
            trades.append({
                'type': 'SAT',
                'price': close_price,
                'date': row.name,
                'capital': capital
            })

    # Son pozisyon açık kaldıysa kapat
    if position == 1:
        capital = shares * data['close'].iloc[-1]

    return {
        "final_capital": capital,
        "trades": trades,
        "shares": shares,
        "final_position": position
    }

def calculate_performance(data, trades):
    """
    Temel performans metrikleri.
    """
    if len(trades) == 0:
        return {"total_return": 0, "sharpe_ratio": 0}

    returns = []
    in_position = False
    entry_price = 0

    for trade in trades:
        if trade['type'] == 'AL':
            entry_price = trade['price']
            in_position = True
        elif trade['type'] == 'SAT' and in_position:
            ret = (trade['price'] - entry_price) / entry_price
            returns.append(ret)
            in_position = False

    total_return = (sum(returns) * 100) if returns else 0
    sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns) if len(returns) > 1 else 0

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "win_rate": sum(1 for r in returns if r > 0) / len(returns) * 100 if returns else 0
    }

def plot_strategy(data, ticker, short_window, long_window):
    """
    Stratejiyi görselleştir.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['close'], label=f'{ticker} Fiyat', alpha=0.7)
    ax.plot(data['MA_short'], label=f'{short_window}-gün MA', alpha=0.7)
    ax.plot(data['MA_long'], label=f'{long_window}-gün MA', alpha=0.7)

    # AL sinyalleri
    buys = data[data['position'] == 1]
    ax.plot(buys.index, buys['MA_short'], '^', markersize=10, color='g', label='AL')

    # SAT sinyalleri
    sells = data[data['position'] == -1]
    ax.plot(sells.index, sells['MA_long'], 'v', markersize=10, color='r', label='SAT')

    ax.set_title(f'{ticker} - MA Kesişim Stratejisi')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig