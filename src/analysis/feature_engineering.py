import pandas as pd
import os

MASTER_CSV = "data/raw/weather_jawa_master.csv"

def load_data():
    if os.path.exists(MASTER_CSV):
        return pd.read_csv(MASTER_CSV)
    else:
        raise FileNotFoundError(f"{MASTER_CSV} tidak ditemukan. Jalankan fetch_weather.py terlebih dahulu.")

def create_features(df):
    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
    
    df["rain_anomaly"] = df.groupby("location")["precipitation"].transform(lambda x: x - x.mean())
    df["temp_anomaly"] = df.groupby("location")["temp_avg"].transform(lambda x: x - x.mean())
    
    return df

def save_processed(df):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/climate_features.csv", index=False)
    print("fitur cuaca diproses dan disimpan.")

if __name__ == "__main__":
    df = load_data()
    df = create_features(df)
    save_processed(df)