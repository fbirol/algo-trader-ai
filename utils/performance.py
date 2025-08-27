# utils/performance.py
import numpy as np
import pandas as pd

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / returns.std()

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    downside = returns[returns < 0].std()
    if downside == 0:
        return np.inf
    return np.sqrt(252) * excess_returns.mean() / downside

def calculate_drawdown(equity_curve):
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def calculate_profit_factor(trades):
    wins = [t for t in trades if t['type'].startswith('SAT') and 'value' in t]
    if len(wins) == 0:
        return 0.0
    gross_profit = sum(t['value'] for t in wins if t['value'] > 0)
    gross_loss = abs(sum(t['value'] for t in wins if t['value'] < 0))
    return gross_profit / gross_loss if gross_loss != 0 else np.inf

def calculate_win_rate(trades):
    if len(trades) == 0:
        return 0.0
    profits = []
    for i in range(1, len(trades)):
        if trades[i]['type'].startswith('SAT') and trades[i-1]['type'] == 'AL':
            profit = trades[i]['value'] - trades[i-1]['value']
            profits.append(profit)
    if len(profits) == 0:
        return 0.0
    win_count = sum(1 for p in profits if p > 0)
    return (win_count / len(profits)) * 100

def calculate_all_metrics(equity_curve, trades, total_bars):
    returns = pd.Series(equity_curve).pct_change().dropna()
    total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
    cagr = ((equity_curve[-1] / equity_curve[0]) ** (252 / total_bars) - 1) * 100 if total_bars > 252 else 0

    return {
        "total_return": round(total_return, 2),
        "cagr": round(cagr, 2),
        "sharpe_ratio": round(calculate_sharpe_ratio(returns), 3),
        "sortino_ratio": round(calculate_sortino_ratio(returns), 3),
        "max_drawdown": round(calculate_drawdown(equity_curve) * 100, 2),
        "win_rate": round(calculate_win_rate(trades), 1),
        "profit_factor": round(calculate_profit_factor(trades), 2),
        "trade_count": len([t for t in trades if t['type'].startswith('SAT')]),
        "final_equity": round(equity_curve[-1], 2)
    }