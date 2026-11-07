import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURES = ["precipitation", "temp_avg", "rain_anomaly", "temp_anomaly"]


def make_dataset(n_dates=60, locations=("Alpha", "Beta", "Gamma")):
    rows = []
    dates = pd.date_range("2025-01-01", periods=n_dates, freq="D")
    for location in locations:
        for index, date in enumerate(dates):
            precipitation = (index * 7 + len(location)) % 21
            temp_max = 27 + (index % 5)
            temp_avg = temp_max - 2
            rain_score = 1 - min(precipitation / 20, 1)
            temp_score = min(max((temp_max - 25) / 10, 0), 1)
            rows.append(
                {
                    "date": date,
                    "location": location,
                    "lat": -7.0,
                    "lon": 110.0,
                    "temp_max": temp_max,
                    "temp_min": temp_max - 4,
                    "precipitation": precipitation,
                    "temp_avg": temp_avg,
                    "rain_anomaly": precipitation - 10,
                    "temp_anomaly": temp_avg - 28,
                    "drought_score": 0.7 * rain_score + 0.3 * temp_score,
                    "risk_level": "Low",
                }
            )
    return pd.DataFrame(rows)


def make_future_df(days=3, location="Alpha", start=None):
    start = start or (pd.Timestamp.now().normalize() + pd.Timedelta(days=2))
    rows = []
    for offset in range(days):
        rows.append(
            {
                "date": start + pd.Timedelta(days=offset),
                "location": location,
                "lat": -7.0,
                "lon": 110.0,
                "temp_max": 30.0,
                "temp_min": 26.0,
                "precipitation": 2.0,
                "temp_avg": 28.0,
                "rain_anomaly": -8.0,
                "temp_anomaly": 0.0,
                "drought_score": 0.8,
                "risk_level": "High",
            }
        )
    return pd.DataFrame(rows)
