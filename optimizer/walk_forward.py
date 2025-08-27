# optimizer/walk_forward.py
import pandas as pd
import numpy as np
from typing import List, Dict
from backtester.core import Backtester, BacktestConfig
from utils.performance import calculate_all_metrics

def walk_forward_analysis(
    strategy_class,
    data: pd.DataFrame,
    warmup_period: str = "2 years",
    window_size: str = "1 year",
    step_size: str = "3 months",
    metric: str = "sharpe_ratio",
    config_kwargs: dict = None
):
    """
    Walk-Forward Analiz yap.
    """
    config_kwargs = config_kwargs or {}
    results = []

    # Tarih bazlı ilerleme için indeksi datetime yap
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Veri indeksi Datetime olmalı.")

    data = data.sort_index()
    dates = data.index

    # Başlangıç ve pencere hesaplamaları
    start_date = dates[0]
    warmup_date = start_date + pd.Timedelta(warmup_period)
    window_delta = pd.Timedelta(window_size)
    step_delta = pd.Timedelta(step_size)

    current_date = warmup_date

    while current_date + window_delta <= dates[-1]:
        train_end = current_date
        test_start = current_date
        test_end = current_date + step_delta

        train_data = data[start_date:train_end]
        test_data = data[test_start:test_end]

        if len(train_data) == 0 or len(test_data) == 0:
            current_date += step_delta
            continue

        # En iyi parametreyi bul (basit grid search)
        best_params = _optimize_for_period(strategy_class, train_data, config_kwargs)
        strategy = strategy_class(**best_params)

        # Test et
        backtester = Backtester(BacktestConfig(**config_kwargs))
        test_results = backtester.run(test_data, strategy)
        perf = test_results['performance']

        results.append({
            "test_period": f"{test_start.date()} - {test_end.date()}",
            "test_return": perf["total_return"],
            "test_sharpe": perf["sharpe_ratio"],
            "test_max_dd": perf["max_drawdown"],
            "params": best_params
        })

        current_date += step_delta

    return pd.DataFrame(results)

def _optimize_for_period(strategy_class, train_data, config_kwargs):
    """
    Eğitim verisinde en iyi parametreyi bul.
    Basit örnek: MA crossover için en iyi window.
    """
    best_score = -np.inf
    best_params = {"short_window": 50, "long_window": 200}

    for short in [20, 50]:
        for long in [100, 200]:
            if short >= long:
                continue
            strategy = strategy_class(short_window=short, long_window=long)
            backtester = Backtester(BacktestConfig(**config_kwargs))
            try:
                result = backtester.run(train_data, strategy)
                score = result['performance']['sharpe_ratio']
                if score > best_score:
                    best_score = score
                    best_params = {"short_window": short, "long_window": long}
            except:
                continue

    return best_params