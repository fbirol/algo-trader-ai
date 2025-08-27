# backtester/core.py
import pandas as pd
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class BacktestConfig:
    initial_capital: float = 1000.0
    commission: float = 0.001  # %0.1
    slippage: float = 0.0005   # %0.05
    mode: str = "long_only"    # "long_only" veya "long_short"
    currency: str = "TL"       # "TL", "USD"


class Backtester:
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.results = {}

    def run(self, data: pd.DataFrame, strategy) -> Dict[str, Any]:
        """
        Ortak backtest döngüsü.
        :param data: OHLCV verisi (sütunlar: open, high, low, close, volume)
        :param strategy: Strateji nesnesi (generate_signals metoduna sahip)
        :return: Backtest sonuçları
        """
        # Sinyal üret
        signals = strategy.generate_signals(data)
        if signals is None or len(signals) != len(data):
            raise ValueError("Strateji geçerli sinyal üretmedi.")

        # Backtest simülasyonu
        equity_curve, trades = self._simulate(data, signals)

        # Performans hesapla
        performance = self._calculate_performance(data, equity_curve, trades)

        # Sonuçları birleştir
        self.results = {
            "equity_curve": equity_curve,
            "trades": trades,
            "performance": performance,
            "config": self.config
        }

        return self.results

    def _simulate(self, data: pd.DataFrame, signals: pd.Series):
        """
        Long-Short destekli emir simülasyonu.
        """
        capital = self.config.initial_capital
        shares = 0.0
        position = 0  # -1: short, 0: nakit, 1: long
        equity_curve = []
        trades = []

        for i in range(len(data)):
            row = data.iloc[i]
            close_price = row['close']
            signal = signals.iloc[i]

            # Mevcut portföy değeri (long/short farkı: short pozisyonda borç var)
            if position == 1:
                current_equity = capital + (shares * close_price)
            elif position == -1:
                # Kısa pozisyon: borç = shares * close_price, sermaye = başlangıç + kar/zarar
                current_equity = capital
            else:
                current_equity = capital

            equity_curve.append(current_equity)

            # Long-Short modu
            if self.config.mode == "long_short":
                # LONG: 1 → long pozisyon aç
                if signal == 1 and position <= 0:
                    if position == -1:  # Önce short varsa kapat
                        capital += shares * close_price  # kısa pozisyon kapat
                        capital -= capital * self.config.commission
                        trades.append({
                            "type": "COVER (short kapat)",
                            "price": close_price,
                            "value": capital,
                            "date": row.name,
                            "commission": capital * self.config.commission
                        })
                    # Şimdi long aç
                    price_with_slippage = close_price * (1 + self.config.slippage)
                    shares = capital / price_with_slippage
                    capital = 0
                    position = 1
                    trades.append({
                        "type": "BUY (long)",
                        "price": close_price,
                        "value": shares * close_price,
                        "date": row.name,
                        "commission": shares * price_with_slippage * self.config.commission
                    })

                # SHORT: 0 → short pozisyon aç (tahmin: fiyat düşecek)
                elif signal == 0 and position >= 0:
                    if position == 1:  # Önce long varsa kapat
                        capital = shares * close_price
                        capital -= capital * self.config.commission
                        shares = 0
                        trades.append({
                            "type": "SELL (long kapat)",
                            "price": close_price,
                            "value": capital,
                            "date": row.name,
                            "commission": capital * self.config.commission
                        })
                    # Şimdi short aç
                    # varsayım: 1x kaldıraç, spot değil futures/CFD
                    short_value = capital  # kısa pozisyon değeri
                    shares = short_value / close_price  # borç alındı
                    position = -1
                    trades.append({
                        "type": "SHORT",
                        "price": close_price,
                        "value": short_value,
                        "date": row.name,
                        "commission": short_value * self.config.commission
                    })

            # Long-Only (mevcut mantık)
            elif self.config.mode == "long_only":
                if signal == 1 and position == 0:
                    price_with_slippage = close_price * (1 + self.config.slippage)
                    shares = capital / price_with_slippage
                    capital = 0
                    position = 1
                    trades.append({
                        "type": "AL",
                        "price": close_price,
                        "value": shares * close_price,
                        "date": row.name,
                        "commission": shares * price_with_slippage * self.config.commission
                    })
                elif signal == 0 and position == 1:
                    capital = shares * close_price
                    capital -= capital * self.config.commission
                    shares = 0
                    position = 0
                    trades.append({
                        "type": "SAT",
                        "price": close_price,
                        "value": capital,
                        "date": row.name,
                        "commission": capital * self.config.commission
                    })

        # Son pozisyonu kapat
        if position == 1 and shares > 0:
            last_price = data['close'].iloc[-1]
            capital = shares * last_price
            capital -= capital * self.config.commission
            trades.append({
                "type": "SAT (son)",
                "price": last_price,
                "value": capital,
                "date": data.index[-1],
                "commission": capital * self.config.commission
            })
        elif position == -1 and shares > 0:
            last_price = data['close'].iloc[-1]
            capital += shares * last_price  # kısa pozisyon kapat
            capital -= capital * self.config.commission
            trades.append({
                "type": "COVER (son)",
                "price": last_price,
                "value": capital,
                "date": data.index[-1],
                "commission": capital * self.config.commission
            })

        return np.array(equity_curve), trades

    def _calculate_performance(self, data: pd.DataFrame, equity_curve: np.ndarray, trades: list):
        """
        Tüm performans metriklerini hesapla.
        """
        from utils.performance import calculate_all_metrics
        return calculate_all_metrics(equity_curve, trades, len(data))