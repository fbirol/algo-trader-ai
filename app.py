# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from data.fetch_data import fetch_yfinance_data
from strategies.ma_crossover import generate_signals
from utils.performance import backtest_strategy, calculate_performance
from utils.performance import plot_strategy

st.title("📊 Algoritmik Ticaret Simülatörü")
st.write("Hareketli Ortalama Kesişimi (Golden Cross) Stratejisi")

# Sidebar
ticker = st.sidebar.text_input("Hisse Sembolü", "SPY")
start_date = st.sidebar.date_input("Başlangıç Tarihi", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("Bitiş Tarihi", pd.to_datetime("2024-01-01"))

short_window = st.sidebar.slider("Kısa MA (gün)", 10, 100, 50)
long_window = st.sidebar.slider("Uzun MA (gün)", 100, 200, 200)

if st.button("Stratejiyi Çalıştır"):
    with st.spinner("Veri indiriliyor ve analiz yapılıyor..."):
        # 1. Veri al
        data = fetch_yfinance_data(ticker, start_date, end_date)
        if data is None or data.empty:
            st.error("Veri alınamadı. Sembolü kontrol edin.")
        else:
            # 2. Sinyal üret
            data = generate_signals(data, short_window, long_window)

            # 3. Backtest
            results = backtest_strategy(data, initial_capital=10000)
            final_capital = results["final_capital"]
            trades = results["trades"]

            # 4. Performans metrikleri
            perf_metrics = calculate_performance(data, trades)

            # 5. Görselleştir
            fig = plot_strategy(data, ticker, short_window, long_window)
            st.pyplot(fig)

            # 6. Sonuçları göster
            st.success(f"Son Sermaye: ${final_capital:,.2f}")
            st.info(f"Toplam Getiri: %{perf_metrics['total_return']:.2f}")
            st.write(f"Sharpe Oranı: {perf_metrics['sharpe_ratio']:.2f}")
            st.write(f"Toplam İşlem Sayısı: {len(trades)}")