# test_backtester.py
from backtester.core import Backtester, BacktestConfig
from strategies.ma_crossover import MACrossoverStrategy
from data.fetch_data import fetch_yfinance_data

data = fetch_yfinance_data("SPY", "2020-01-01", "2023-01-01")
strategy = MACrossoverStrategy(short_window=50, long_window=200)
config = BacktestConfig(initial_capital=1000, mode="long_only")
backtester = Backtester(config)
results = backtester.run(data, strategy)

print("Son Sermaye:", results["performance"]["final_equity"])
print("Toplam Getiri:", results["performance"]["total_return"], "%")