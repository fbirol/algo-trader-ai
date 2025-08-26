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

# app.py içine yeni bir sekme ekle
with st.sidebar:
    st.markdown("---")
    ai_enabled = st.checkbox("🤖 AI Modunu Etkinleştir")

if ai_enabled:
    st.subheader("🤖 XGBoost ile AI Sinyal Tahmini")

    if st.button("AI Modeli Eğit ve Tahmin Et"):
        with st.spinner("AI modeli eğitiliyor..."):
            from strategies.xgboost_strategy import add_features, train_xgboost_model, predict_signal

            # 1. Veri al
            data_raw = fetch_yfinance_data(ticker, start_date, end_date)
            if data_raw is None or data_raw.empty:
                st.error("Veri alınamadı.")
            else:
                # 2. Özellik ekle
                data_featured = add_features(data_raw)

                # 3. Özellik listesi
                features = [
                    'MA5', 'MA10', 'MA20', 'MA50',
                    'RSI', 'MACD', 'MACD_signal',
                    'Volatility', 'Price_Change', 'Volume_Change',
                    'BB_width'
                ]

                # 4. Model eğit
                try:
                    model, acc, clean_data = train_xgboost_model(data_featured, features)

                    # 5. Son sinyali tahmin et
                    latest_row = clean_data[features].iloc[-1:].values.reshape(1, -1)
                    signal, confidence = predict_signal(model, latest_row)

                    # 6. Göster
                    st.success(f"🤖 Tahmin: **{signal}**")
                    st.info(f"Güven: %{confidence*100:.1f}")
                    st.write(f"Model Doğruluğu: %{acc*100:.1f}")

                except Exception as e:
                    st.error(f"Model hatası: {e}")

if st.checkbox("🧠 LSTM ile Fiyat Tahmini"):
    st.subheader("🧠 LSTM Zaman Serisi Tahmini")

    if st.button("LSTM Modeli Eğit ve Tahmin Et"):
        with st.spinner("LSTM modeli eğitiliyor..."):
            from strategies.lstm_strategy import train_lstm_model, predict_next_price

            # Veri al
            data_raw = fetch_yfinance_data(ticker, start_date, end_date)
            if data_raw is None or data_raw.empty:
                st.error("Veri alınamadı.")
            else:
                try:
                    # Model eğit
                    model, scaler, X_test, y_test = train_lstm_model(
                        data_raw,
                        window_size=60,
                        epochs=10,  # Daha hızlı test
                        save_model_path="lstm_model.h5"
                    )

                    # Tahmin yap
                    window_size = 60
                    last_60 = data_raw['close'].values[-window_size:]
                    pred = predict_next_price(model, scaler, last_60, window_size)
                    current = data_raw['close'].iloc[-1]
                    change_pct = (pred - current) / current * 100

                    # Göster
                    st.success(f"Tahmini Fiyat: ${pred:.2f}")
                    st.info(f"Geçerli Fiyat: ${current:.2f}")
                    st.write(f"Tahmini Getiri: %{change_pct:.2f}")

                    if change_pct > 0:
                        st.write("🤖 **Sinyal: AL** (Yukarı yönlü hareket bekleniyor)")
                    else:
                        st.write("🤖 **Sinyal: SAT/Bekle** (Aşağı yönlü hareket bekleniyor)")

                except Exception as e:
                    st.error(f"LSTM hatası: {e}")
