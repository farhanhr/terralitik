import pandas as pd
import os


LOCATION_COORDS = {
    "Karawang": (-6.3227, 107.3376),
    "Indramayu": (-6.3275, 108.3200),
    "Klaten": (-7.7050, 110.6062),
    "Ngawi": (-7.4039, 111.4461),
    "Banyuwangi": (-8.2192, 114.3691),
}

def load_features():

    df = pd.read_csv("data/processed/climate_features.csv")

    return df

def add_coordinates(df):

    latitudes = []
    longitudes = []

    for loc in df["location"]:

        lat, lon = LOCATION_COORDS[loc]

        latitudes.append(lat)
        longitudes.append(lon)

    df["lat"] = latitudes
    df["lon"] = longitudes

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

    df = add_coordinates(df)

    save_results(df)

    print(df.head())