# utils/performance.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Sharpe oranı hesapla (yıllık)
    """
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - risk_free_rate / 252  # Günlük risk-free
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_drawdown(equity_curve):
    """
    Maksimum drawdown hesapla
    """
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def backtest_strategy_with_equity(data, signal_column='signal', initial_capital=10000):
    """
    Sinyal sütununa göre backtest yap ve equity curve oluştur.
    """
    capital = initial_capital
    shares = 0
    position = 0
    equity_curve = []
    trades = []

    for i in range(len(data)):
        row = data.iloc[i]
        close_price = row['close']
        signal = row[signal_column] if signal_column in row else 0

        # Mevcut equity
        current_equity = capital + (shares * close_price)
        equity_curve.append(current_equity)

        # Sinyal değişimi: AL/SAT
        if signal == 1 and position == 0:
            shares = capital / close_price
            capital = 0
            position = 1
            trades.append(('AL', close_price, i))
        elif signal == 0 and position == 1:
            capital = shares * close_price
            shares = 0
            position = 0
            trades.append(('SAT', close_price, i))

    # Son pozisyon açık kaldıysa kapat
    if position == 1:
        capital = shares * data['close'].iloc[-1]

    # Getiri serisi
    equity_curve = np.array(equity_curve)
    returns = pd.Series(equity_curve).pct_change().dropna()

    final_capital = capital
    total_return = (final_capital - initial_capital) / initial_capital * 100
    sharpe = calculate_sharpe_ratio(returns)
    max_dd = calculate_drawdown(equity_curve)
    win_rate = 0  # Basit versiyon için geçici

    if len(trades) > 1 and trades[0][0] == 'AL':
        wins = 0
        for i in range(1, len(trades)):
            if trades[i][0] == 'SAT':
                buy_price = trades[i-1][1]
                sell_price = trades[i][1]
                if sell_price > buy_price:
                    wins += 1
        win_rate = (wins / (len(trades) // 2)) * 100 if trades[-1][0] == 'SAT' else 0

    return {
        "final_capital": final_capital,
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd * 100,
        "win_rate": win_rate,
        "trade_count": len([t for t in trades if t[0] == 'SAT']),
        "equity_curve": equity_curve
    }
