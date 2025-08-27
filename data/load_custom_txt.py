# data/load_custom_txt.py
import pandas as pd
from typing import Optional
import re

def load_custom_txt(filepath: str) -> Optional[pd.DataFrame]:
    """
    Özel formatlı TXT dosyasını yükle ve OHLCV formatına çevir.
    Örnek satır:
        VIP'VIP-X030,200706111315,548.250,548.500,548.000,548.500,371,5465499000,B
    """
    try:
        # Sütun isimleri
        column_names = [
            "symbol_raw",     # VIP'VIP-X030
            "timestamp",      # 200706111315
            "open",
            "high",
            "low",
            "close",
            "volume",
            "extra",          # 5465499000
            "flag"            # B
        ]

        # CSV oku (header yok, virgülle ayrılmış)
        data = pd.read_csv(
            filepath,
            header=None,
            names=column_names,
            delimiter=",",
            skipinitialspace=True  # Boşlukları temizle
        )

        # Boşluk ve tırnak temizliği
        data["symbol_raw"] = data["symbol_raw"].str.strip().str.replace("'", "", regex=False)
        data["flag"] = data["flag"].str.strip()

        # Zaman damgasını çevir: YYYYMMDDHHmm → datetime
        data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y%m%d%H%M", errors="coerce")
        
        # Geçersiz tarihleri çıkar
        data.dropna(subset=["timestamp"], inplace=True)

        # Sayısal sütunları dönüştür
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        # Eksik verileri temizle
        data.dropna(subset=numeric_cols, inplace=True)

        # Sırala
        data.sort_values("timestamp", inplace=True)

        # Gerekli sütunları seç ve isimlendir
        data.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        }, inplace=True)

        # timestamp'i indeks yap
        data.set_index("timestamp", inplace=True)

        # Artık gereksiz sütunları at
        data.drop(columns=["symbol_raw", "extra", "flag"], errors="ignore", inplace=True)

        return data

    except Exception as e:
        print(f"Custom TXT yükleme hatası: {e}")
        return None