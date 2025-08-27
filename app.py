# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Modüller
from data.fetch_data import fetch_yfinance_data
from data.load_custom_txt import load_custom_txt
from data.load_csv import load_csv_data
from strategies.ma_crossover import MACrossoverStrategy
from strategies.xgboost_strategy import add_features as xgb_add_features
from strategies.xgboost_strategy import train_xgboost_model, predict_signal
from strategies.lstm_strategy import train_lstm_model, predict_next_price
from strategies.ensemble_strategy import generate_ensemble_signal
from backtester.core import Backtester, BacktestConfig
from optimizer.grid_search import optimize
from optimizer.walk_forward import walk_forward_analysis
import joblib
from tensorflow.keras.models import load_model
import os

# ---------------------------------------------------------------------
# Sayfa Yapılandırması
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="AlgoTrader AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 AlgoTrader AI")
st.markdown("### **Yapay Zekâ Destekli Algoritmik Ticaret Simülatörü**")
st.markdown("---")

# ---------------------------------------------------------------------
# Yardımcı Fonksiyon: Veri Yükleme
# ---------------------------------------------------------------------
@st.cache_data
def get_data(source_type, filepath_or_ticker, start_date=None, end_date=None):
    """
    Ortak veri yükleme fonksiyonu.
    """
    if source_type == "yfinance":
        data = fetch_yfinance_data(filepath_or_ticker, start_date, end_date)
    elif source_type == "csv":
        data = load_csv_data(filepath_or_ticker)
    elif source_type == "txt":
        data = load_custom_txt(filepath_or_ticker)
    else:
        raise ValueError("Geçersiz veri kaynağı")

    if data is None or data.empty:
        return None

    # Tarih filtresi
    if start_date is not None:
        data = data[data.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        data = data[data.index <= pd.to_datetime(end_date)]

    return data if not data.empty else None

# ---------------------------------------------------------------------
# Sidebar: Genel Ayarlar
# ---------------------------------------------------------------------
st.sidebar.header("⚙️ Genel Ayarlar")
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Veri Kaynağı")

data_source = st.sidebar.radio("Veri Kaynağı", ["Yahoo Finance", "CSV Yükle", "Özel TXT Yükle"])

data = None
source_type = None
filepath_or_ticker = None

# Veri yükleme ve tarih aralığı
if data_source == "Yahoo Finance":
    ticker = st.sidebar.text_input("Hisse Sembolü", "SPY")
    start_date = st.sidebar.date_input("Başlangıç", pd.to_datetime("2020-01-01"), key="start_yf")
    end_date = st.sidebar.date_input("Bitiş", pd.to_datetime("2023-01-01"), key="end_yf")
    source_type = "yfinance"
    filepath_or_ticker = ticker

elif data_source == "CSV Yükle":
    uploaded_file = st.sidebar.file_uploader("CSV Yükle", type="csv", key="csv_uploader")
    if uploaded_file is not None:
        filepath = "temp_uploaded_data.csv"
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        filepath_or_ticker = filepath
        source_type = "csv"
        data = get_data(source_type, filepath_or_ticker, None, None)
        if data is not None:
            start_date = st.sidebar.date_input("Başlangıç (isteğe bağlı)", value=None, key="start_csv")
            end_date = st.sidebar.date_input("Bitiş (isteğe bağlı)", value=None, key="end_csv")
        else:
            st.error("CSV verisi yüklenemedi.")

else:  # Özel TXT Yükle
    uploaded_file = st.sidebar.file_uploader("TXT Yükle", type="txt", key="txt_uploader")
    if uploaded_file is not None:
        filepath = "temp_uploaded_data.txt"
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        filepath_or_ticker = filepath
        source_type = "txt"
        data = get_data(source_type, filepath_or_ticker, None, None)
        if data is not None:
            start_date = st.sidebar.date_input("Başlangıç (isteğe bağlı)", value=None, key="start_txt")
            end_date = st.sidebar.date_input("Bitiş (isteğe bağlı)", value=None, key="end_txt")
        else:
            st.error("TXT verisi yüklenemedi.")

# Veri aralığı bilgisi
if data is not None and not data.empty:
    st.sidebar.info(f"Veri aralığı: {data.index.min():%Y-%m-%d} → {data.index.max():%Y-%m-%d}")

# Veriyi filtrele (tarih bazlı)
if filepath_or_ticker and source_type:
    data = get_data(source_type, filepath_or_ticker, start_date, end_date)
    if data is not None and not data.empty:
        st.success(f"✅ Veri yüklendi. {len(data)} satır.")
    else:
        st.error("❌ Veri yüklenemedi veya boş.")

# Genel ayarlar
st.sidebar.markdown("---")
currency = st.sidebar.selectbox("Para Birimi", ["USD", "TL"], index=0)
mode = st.sidebar.selectbox("Ticaret Modu", ["long_only", "long_short"], index=0)
commission = st.sidebar.slider("Komisyon (%)", 0.0, 1.0, 0.1) / 100
slippage = st.sidebar.slider("Slippage (%)", 0.0, 1.0, 0.05) / 100
initial_capital = st.sidebar.number_input("Başlangıç Sermayesi", 1000, 100000, 10000)

# ---------------------------------------------------------------------
# Sekmeler
# ---------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Temel Strateji",
    "🤖 XGBoost AI",
    "🧠 LSTM Tahmini",
    "⚖️ Model Karşılaştırması",
    "🎛️ Ensemble & Optimizasyon",
    "🔍 Walk-Forward Analiz"
])

# ---------------------------------------------------------------------
# TAB 1: Temel MA Stratejisi
# ---------------------------------------------------------------------
with tab1:
    st.subheader("📊 Hareketli Ortalama (Golden Cross) Stratejisi")
    short_win = st.slider("Kısa MA (gün)", 5, 100, 50, key="ma_short_tab1")
    long_win = st.slider("Uzun MA (gün)", 50, 200, 200, key="ma_long_tab1")

    if st.button("📌 Stratejiyi Çalıştır", key="run_ma"):
        if data is None or data.empty:
            st.error("❌ Veri alınamadı.")
        else:
            strategy = MACrossoverStrategy(short_window=short_win, long_window=long_win)
            config = BacktestConfig(initial_capital=initial_capital, commission=commission, slippage=slippage, mode=mode, currency=currency)
            backtester = Backtester(config)
            results = backtester.run(data, strategy)
            perf = results['performance']

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Son Sermaye", f"${perf['final_equity']:,.2f}")
            col2.metric("Toplam Getiri", f"%{perf['total_return']:.2f}")
            col3.metric("Sharpe Oranı", f"{perf['sharpe_ratio']:.3f}")
            col4.metric("Max Drawdown", f"%{perf['max_drawdown']:.2f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=results['equity_curve'], mode='lines', name='Portföy Değeri'))
            fig.update_layout(title="Portföy Büyümesi (Equity Curve)", xaxis_title="Tarih", yaxis_title="Değer", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 2: XGBoost AI
# ---------------------------------------------------------------------
with tab2:
    st.subheader("🤖 XGBoost ile AI Sinyal Tahmini")
    if st.button("🧠 Model Eğit ve Tahmin Et"):
        if data is None or data.empty:
            st.error("❌ Veri alınamadı.")
        else:
            try:
                data_feat = xgb_add_features(data)
                features = ['MA5', 'MA10', 'MA20', 'MA50', 'RSI', 'MACD', 'MACD_signal', 'Volatility', 'Price_Change', 'Volume_Change', 'BB_width']
                model, acc, clean_data = train_xgboost_model(data_feat, features)
                latest = clean_data[features].iloc[-1:].values.reshape(1, -1)
                signal, confidence = predict_signal(model, latest)

                st.success(f"**Tahmin: {signal}**")
                st.info(f"**Güven: %{confidence * 100:.1f}**")
                st.write(f"Model Doğruluğu: %{acc * 100:.1f}")
            except Exception as e:
                st.error(f"❌ XGBoost hatası: {e}")

# ---------------------------------------------------------------------
# TAB 3: LSTM Tahmini
# ---------------------------------------------------------------------
with tab3:
    st.subheader("🧠 LSTM ile Fiyat Tahmini")
    if st.button("🔮 LSTM Modeli Eğit ve Tahmin Et"):
        if data is None or data.empty:
            st.error("❌ Veri alınamadı.")
        else:
            try:
                model, scaler, _, _ = train_lstm_model(data, window_size=60, epochs=5, save_model_path="models/lstm_model.h5")
                last_60 = data['close'].values[-60:]
                pred = predict_next_price(model, scaler, last_60, 60)
                current = data['close'].iloc[-1]
                change_pct = (pred - current) / current * 100

                col1, col2 = st.columns(2)
                col1.metric("Geçerli Fiyat", f"${current:.2f}")
                col2.metric("Tahmini Fiyat", f"${pred:.2f}")
                st.write(f"**Tahmini Getiri: %{change_pct:.2f}**")
                if change_pct > 0:
                    st.success("✅ **Sinyal: AL**")
                else:
                    st.warning("⚠️ **Sinyal: SAT**")
            except Exception as e:
                st.error(f"❌ LSTM hatası: {e}")

# ---------------------------------------------------------------------
# TAB 4: Model Karşılaştırması
# ---------------------------------------------------------------------
with tab4:
    st.subheader("⚖️ Üç Modelin Karşılaştırması")

    if st.button("📊 Karşılaştır"):
        if data is None or data.empty:
            st.error("❌ Veri alınamadı.")
        else:
            results = []

            # ----------------------------
            # 1. MA Crossover Stratejisi
            # ----------------------------
            try:
                strategy_ma = MACrossoverStrategy(short_window=50, long_window=200)
                config = BacktestConfig(initial_capital=initial_capital, commission=commission, mode=mode)
                backtester = Backtester(config)
                data_with_signal = data.copy()
                data_with_signal['signal'] = strategy_ma.generate_signals(data)
                result_ma = backtester.run(data_with_signal, strategy_ma)
                perf_ma = result_ma['performance']
                results.append({"Model": "MA Crossover", **perf_ma})
            except Exception as e:
                st.warning(f"MA hatası: {e}")

            # ----------------------------
            # 2. XGBoost Stratejisi
            # ----------------------------
            try:
                data_xgb = xgb_add_features(data.copy())
                features = ['MA5', 'MA10', 'MA20', 'MA50', 'RSI', 'MACD', 'MACD_signal',
                           'Volatility', 'Price_Change', 'Volume_Change', 'BB_width']
                model, acc, clean_data = train_xgboost_model(data_xgb, features)
                preds = model.predict(clean_data[features])
                signal_series = pd.Series(preds, index=clean_data.index)

                strategy_dummy = type('DummyStrategy', (), {
                    'generate_signals': lambda d: signal_series.reindex(d.index, fill_value=0)
                })()

                backtester = Backtester(BacktestConfig(initial_capital=initial_capital, commission=commission, mode=mode))
                result_xgb = backtester.run(data_xgb, strategy_dummy)
                perf_xgb = result_xgb['performance']
                results.append({"Model": "XGBoost", **perf_xgb})
            except Exception as e:
                st.warning(f"XGBoost hatası: {e}")

            # ----------------------------
            # 3. LSTM Stratejisi
            # ----------------------------
            try:
                from tensorflow.keras.models import load_model
                lstm_model = load_model("models/lstm_model.h5")
                scaler = joblib.load("models/lstm_model_scaler.pkl")
                window_size = 60

                # Tahmin üret
                dataset = data['close'].values.reshape(-1, 1)
                scaled_data = scaler.transform(dataset)
                signals = []
                for i in range(window_size, len(scaled_data)):
                    X = scaled_data[i-window_size:i, 0].reshape(1, window_size, 1)
                    pred = lstm_model.predict(X, verbose=0)[0, 0]
                    current = scaled_data[i, 0]
                    signals.append(1 if pred > current else 0)

                signal_series = pd.Series(signals, index=data.index[window_size:])

                strategy_dummy = type('DummyStrategy', (), {
                    'generate_signals': lambda d: signal_series.reindex(d.index, fill_value=0)
                })()

                backtester = Backtester(BacktestConfig(initial_capital=initial_capital, commission=commission, mode=mode))
                result_lstm = backtester.run(data.iloc[window_size:], strategy_dummy)
                perf_lstm = result_lstm['performance']
                results.append({"Model": "LSTM", **perf_lstm})
            except Exception as e:
                st.warning(f"LSTM hatası: {e}")

            # ----------------------------
            # Sonuçları Göster
            # ----------------------------
            if results:
                df_results = pd.DataFrame(results)[[
                    "Model", "total_return", "sharpe_ratio", "max_drawdown",
                    "win_rate", "trade_count"
                ]]
                df_results["total_return"] = df_results["total_return"].round(2)
                df_results["sharpe_ratio"] = df_results["sharpe_ratio"].round(3)
                df_results["max_drawdown"] = df_results["max_drawdown"].round(2)
                df_results["win_rate"] = df_results["win_rate"].round(1)

                st.dataframe(df_results.style.format({
                    "total_return": "{}%",
                    "max_drawdown": "{}%",
                    "win_rate": "{}%",
                }).background_gradient(subset=["total_return"], cmap="RdYlGn"))

                # Equity Curve
                fig = go.Figure()
                for r in results:
                    fig.add_trace(go.Scatter(y=r['equity_curve'], name=r['Model']))
                fig.update_layout(title="Equity Curve Karşılaştırması", xaxis_title="Zaman", yaxis_title="Portföy Değeri")
                st.plotly_chart(fig)

# ---------------------------------------------------------------------
# TAB 5: Ensemble & Optimizasyon
# ---------------------------------------------------------------------
with tab5:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Ensemble Strateji")
        if st.button("Ensemble Hesapla"):
            try:
                final_signal, _, _ = generate_ensemble_signal(data)
                # Sinyali DataFrame'e ekle
                data_with_signal = data.copy()
                data_with_signal['signal'] = final_signal
                # Backtest
                config = BacktestConfig(initial_capital=initial_capital, commission=commission, mode=mode)
                backtester = Backtester(config)
                result = backtester.run(data_with_signal, lambda d: d['signal'])  # Basit strateji
                perf = result['performance']

                st.metric("Son Sermaye", f"${perf['final_equity']:,.2f}")
                st.metric("Getiri", f"%{perf['total_return']:.2f}")
                st.metric("Sharpe", f"{perf['sharpe_ratio']:.3f}")
            except Exception as e:
                st.error(f"Ensemble hatası: {e}")
                
    with col2:
        st.subheader("🎛️ Parametre Optimizasyonu")
        if st.button("Optimize Et"):
            try:
                param_ranges = {"short_window": [10, 20, 50], "long_window": [50, 100, 200]}
                config = BacktestConfig(initial_capital=initial_capital, commission=commission)
                results_df = optimize(MACrossoverStrategy, data, param_ranges, config, mode=mode)
                st.dataframe(results_df.head(10))
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 CSV İndir", csv, "optimization.csv", "text/csv")
            except Exception as e:
                st.error(f"Optimizasyon hatası: {e}")

# ---------------------------------------------------------------------
# TAB 6: Walk-Forward Analiz
# ---------------------------------------------------------------------
with tab6:
    st.subheader("🔍 Walk-Forward Analiz")
    if st.button("WFA Çalıştır"):
        if data is None or data.empty:
            st.error("❌ Veri alınamadı.")
        else:
            try:
                results_df = walk_forward_analysis(
                    MACrossoverStrategy,
                    data,
                    warmup_period="365 days",
                    window_size="365 days",
                    step_size="90 days",
                    config_kwargs={
                        "initial_capital": initial_capital,
                        "commission": commission,
                        "mode": mode,
                        "currency": currency
                    }
                )
                st.dataframe(results_df)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=results_df['test_period'], y=results_df['test_return'], mode='lines+markers'))
                fig.update_layout(title="Test Dönemleri (Getiri %)", xaxis_title="Test Dönemi", yaxis_title="Getiri (%)")
                st.plotly_chart(fig)
                st.info(f"Ortalama Getiri: %{results_df['test_return'].mean():.2f}")
            except Exception as e:
                st.error(f"WFA hatası: {e}")