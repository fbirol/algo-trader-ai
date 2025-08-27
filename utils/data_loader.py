# utils/data_loader.py
def get_data(source_type, filepath_or_ticker, start_date=None, end_date=None):
    if source_type == "yfinance":
        from data.fetch_data import fetch_yfinance_data
        return fetch_yfinance_data(filepath_or_ticker, start_date, end_date)
    elif source_type == "csv":
        from data.load_csv import load_csv_data
        data = load_csv_data(filepath_or_ticker)
    elif source_type == "txt":
        from data.load_custom_txt import load_custom_txt
        data = load_custom_txt(filepath_or_ticker)
    else:
        raise ValueError("Geçersiz veri kaynağı")

    if data is None or data.empty:
        return None

    # Otomatik tarih setle
    if start_date is not None:
        data = data[data.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        data = data[data.index <= pd.to_datetime(end_date)]

    return data if not data.empty else None