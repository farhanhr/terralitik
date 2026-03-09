import pandas as pd
import os

def load_data():
    
    files = os.listdir("data/raw")
    latest = sorted(files)[-1]
    path = f"data/raw/{latest}"
    df = pd.read_csv(path)

    return df

def create_features(df):

    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
    df["rain_anomaly"] = df["precipitation"] - df["precipitation"].mean()
    df["temp_anomaly"] = df["temp_avg"] - df["temp_avg"].mean()

    return df

def save_processed(df):

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/climate_features.csv", index=False)
    print("Processed data saved")

if __name__ == "__main__":
    df = load_data()
    df = create_features(df)
    save_processed(df)
    print(df.head())