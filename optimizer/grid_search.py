# optimizer/grid_search.py
import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, List
import joblib
from backtester.core import Backtester, BacktestConfig
import concurrent.futures

def run_single_backtest(config, data, strategy_class, params, mode, currency):
    """
    Tek bir parametre seti için backtest çalıştır.
    """
    try:
        strategy = strategy_class(**params)
        backtester = Backtester(BacktestConfig(
            initial_capital=config.initial_capital,
            commission=config.commission,
            slippage=config.slippage,
            mode=mode,
            currency=currency
        ))
        results = backtester.run(data, strategy)
        perf = results['performance']
        return {**params, **perf}
    except Exception as e:
        print(f"Hata ({params}): {e}")
        return {**params, "error": str(e)}

def optimize(
    strategy_class,
    data,
    param_ranges: Dict[str, List],
    config,
    mode="long_only",
    currency="TL",
    metric="sharpe_ratio",
    n_jobs=4
):
    """
    Grid search ile parametre optimizasyonu.
    """
    keys = param_ranges.keys()
    values = param_ranges.values()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = []
        for v in product(*values):
            params = dict(zip(keys, v))
            futures.append(
                executor.submit(
                    run_single_backtest,
                    config, data, strategy_class, params, mode, currency
                )
            )

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)

    df = pd.DataFrame(results)
    df.sort_values(by=metric, ascending=False, inplace=True)
    return df