import pandas as pd
import os

def load_features():
    return pd.read_csv("data/processed/climate_features.csv")

def calculate_drought_index(df):

    # Normalisasi hujan: 0mm = 1 (Kering), >=20mm = 0 (Basah)
    rain_score = 1 - (df["precipitation"] / 20).clip(upper=1)
    
    # Normalisasi suhu: <25C = 0, >=35C = 1
    temp_score = ((df["temp_max"] - 25) / 10).clip(lower=0, upper=1)
    
    df["drought_score"] = (0.7 * rain_score) + (0.3 * temp_score)
    return df

def classify_risk(df):
    conditions = []
    for score in df["drought_score"]:
        if score < 0.5:
            conditions.append("Low")
        elif score < 0.75:
            conditions.append("Moderate")
        else:
            conditions.append("High")
    df["risk_level"] = conditions
    return df

def save_results(df):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/drought_risk.csv", index=False)
    print("Indeks kekeringan diperbaiki dan disimpan.")

if __name__ == "__main__":
    df = load_features()
    df = calculate_drought_index(df)
    df = classify_risk(df)
    save_results(df)