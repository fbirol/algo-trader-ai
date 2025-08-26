import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib  # Modeli kaydetmek için

def add_features(data):
    """
    Özellik mühendisliği: Teknik göstergeler ekle.
    """
    data = data.copy()

    # 1. Hareketli Ortalamalar
    data['MA5'] = data['close'].rolling(5).mean()
    data['MA10'] = data['close'].rolling(10).mean()
    data['MA20'] = data['close'].rolling(20).mean()
    data['MA50'] = data['close'].rolling(50).mean()

    # 2. RSI (Relative Strength Index)
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # 3. MACD
    exp1 = data['close'].ewm(span=12).mean()
    exp2 = data['close'].ewm(span=26).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_signal'] = data['MACD'].ewm(span=9).mean()

    # 4. Volatilite (20 günlük standart sapma)
    data['Volatility'] = data['close'].pct_change().rolling(20).std()

    # 5. Fiyat değişimleri
    data['Price_Change'] = data['close'].pct_change()

    # 6. Hacim değişimi
    data['Volume_Change'] = data['volume'].pct_change()

    # 7. Bollinger Band genişliği
    data['BB_middle'] = data['close'].rolling(20).mean()
    data['BB_std'] = data['close'].rolling(20).std()
    data['BB_upper'] = data['BB_middle'] + (data['BB_std'] * 2)
    data['BB_lower'] = data['BB_middle'] - (data['BB_std'] * 2)
    data['BB_width'] = data['BB_upper'] - data['BB_lower']

    return data

def add_target(data, horizon=1):
    """
    Hedef değişken oluştur: 1 gün sonra kapanış fiyatı artacak mı?
    """
    data['target'] = (data['close'].shift(-horizon) > data['close']).astype(int)
    return data

def prepare_data_for_training(data, features, target='target'):
    """
    Eğitim için NaN temizliği ve veri bölme.
    """
    # Gerekli sütunlar eksikse ekle
    if target not in data.columns:
        data = add_target(data)

    # NaN olan satırları temizle
    clean_data = data[features + [target]].dropna()

    X = clean_data[features]
    y = clean_data[target]

    return X, y, clean_data

def train_xgboost_model(data, features, save_model_path=None):
    """
    XGBoost modelini eğit ve değerlendir.
    """
    X, y, clean_data = prepare_data_for_training(data, features)

    # Eğitim ve test seti ayır (zamansal sıralı)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # XGBoost modeli
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Tahmin ve değerlendirme
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"XGBoost Modeli Doğruluk (Accuracy): {acc:.3f}")
    print("Sınıflandırma Raporu:")
    print(report)

    # Modeli kaydet (isteğe bağlı)
    if save_model_path:
        model_path = os.path.join("models", "xgboost_model.pkl")
        joblib.dump(model, model_path)
        print(f"Model kaydedildi: {save_model_path}")

    return model, acc, clean_data

def predict_signal(model, latest_features):
    """
    En güncel veriyle sinyal tahmini yap.
    """
    pred = model.predict(latest_features)[0]
    proba = model.predict_proba(latest_features)[0]
    confidence = max(proba)  # En yüksek olasılık
    return "AL" if pred == 1 else "SAT", confidence