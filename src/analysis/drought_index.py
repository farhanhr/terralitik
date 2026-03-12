import pandas as pd
import os

def load_features():

    df = pd.read_csv("data/processed/climate_features.csv")

    return df


def calculate_drought_index(df):

    rain_norm = (df["rain_anomaly"] - df["rain_anomaly"].min()) / (
        df["rain_anomaly"].max() - df["rain_anomaly"].min()
    )

    temp_norm = (df["temp_anomaly"] - df["temp_anomaly"].min()) / (
        df["temp_anomaly"].max() - df["temp_anomaly"].min()
    )

    df["drought_score"] = (
        0.6 * (1 - rain_norm) +   
        0.4 * temp_norm           
    )

    return df


def classify_risk(df):

    conditions = []

    for score in df["drought_score"]:

        if score < 0.4:
            conditions.append("Low")

        elif score < 0.7:
            conditions.append("Moderate")

        else:
            conditions.append("High")

    df["risk_level"] = conditions

    return df


def save_results(df):

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv("data/processed/drought_risk.csv", index=False)

    print("Saved drought risk dataset")


if __name__ == "__main__":

    df = load_features()

    df = calculate_drought_index(df)

    df = classify_risk(df)

    save_results(df)

    print(df.head())