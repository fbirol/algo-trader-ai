# data/load_csv.py
import pandas as pd
from typing import Optional

def load_csv_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    CSV dosyasından OHLCV verisi yükle ve temizle.
    """
    try:
        data = pd.read_csv(filepath)
        
        # Gerekli sütunlar var mı kontrol et
        required_columns = {'open', 'high', 'low', 'close', 'volume', 'timestamp'}
        if not required_columns.issubset(data.columns.str.lower()):
            raise ValueError(f"Eksik sütunlar. Gerekli: {required_columns}")

        # Sütun isimlerini standardize et
        column_mapping = {col.lower(): col.lower() for col in data.columns}
        data.rename(columns=column_mapping, inplace=True)

        # Tarih sütununu indeks yap
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)

        # NaN temizliği
        data.dropna(inplace=True)

        # Sıralı indeks
        data.sort_index(inplace=True)

        return data

    except Exception as e:
        print(f"CSV yükleme hatası: {e}")
        return None