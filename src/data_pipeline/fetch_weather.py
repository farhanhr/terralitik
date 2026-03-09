import requests
import pandas as pd
from datetime import datetime
import os

API_URL = "https://api.open-meteo.com/v1/forecast"

LOCATIONS = [
    {"name": "Karawang", "lat": -6.3227, "lon": 107.3376},
    {"name": "Indramayu", "lat": -6.3275, "lon": 108.3200},
    {"name": "Klaten", "lat": -7.7050, "lon": 110.6062},
    {"name": "Ngawi", "lat": -7.4039, "lon": 111.4461},
    {"name": "Banyuwangi", "lat": -8.2192, "lon": 114.3691}
]


def fetch_weather_for_location(location):

    params = {
        "latitude": location["lat"],
        "longitude": location["lon"],
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum"
        ],
        "timezone": "auto"
    }

    response = requests.get(API_URL, params=params)

    data = response.json()

    daily = data["daily"]

    df = pd.DataFrame({
        "date": daily["time"],
        "location": location["name"],
        "temp_max": daily["temperature_2m_max"],
        "temp_min": daily["temperature_2m_min"],
        "precipitation": daily["precipitation_sum"]
    })

    return df


def fetch_all_locations():

    all_data = []

    for location in LOCATIONS:

        df = fetch_weather_for_location(location)

        all_data.append(df)

    combined = pd.concat(all_data)

    return combined


def save_data(df):

    os.makedirs("data/raw", exist_ok=True)

    filename = f"data/raw/weather_{datetime.now().date()}.csv"

    df.to_csv(filename, index=False)

    print("Saved:", filename)


if __name__ == "__main__":

    df = fetch_all_locations()

    print(df.head())

    save_data(df)