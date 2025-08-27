# test_optimize.py
from optimizer.grid_search import optimize
from strategies.ma_crossover import MACrossoverStrategy
from data.fetch_data import fetch_yfinance_data
from backtester.core import BacktestConfig

# Veri
data = fetch_yfinance_data("SPY", "2020-01-01", "2023-01-01")

# Konfigürasyon
config = BacktestConfig(initial_capital=1000, commission=0.001)

# Parametre aralığı
param_ranges = {
    "short_window": [10, 20, 50],
    "long_window": [50, 100, 200]
}

# Optimize et
results = optimize(
    strategy_class=MACrossoverStrategy,
    data=data,
    param_ranges=param_ranges,
    config=config,
    mode="long_short",
    metric="sharpe_ratio",
    n_jobs=4
)

# Kaydet
results.to_csv("optimization_results.csv", index=False)
print(results.head())