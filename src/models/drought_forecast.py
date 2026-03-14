import pandas as pd
from xgboost import XGBRegressor

FEATURES = ["precipitation", "temp_avg", "rain_anomaly", "temp_anomaly"]

def train_model(df):
    X = df[FEATURES]
    y = df["drought_score"]
    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    return model

def forecast_next_days(model, df, location, days=30):
    loc_df = df[df["location"] == location].copy()
    if loc_df.empty:
        return [0.0] * days

    latest = loc_df.iloc[-1]
    precipitation = latest["precipitation"]
    temp_avg = latest["temp_avg"]
    mean_rain = df["precipitation"].mean()
    mean_temp = df["temp_avg"].mean()
    
    forecasts = []

    for day in range(days):
        # Transisi logis ke nilai rata-rata historis (Mean Reversion)
        precipitation = (precipitation * 0.7) + (mean_rain * 0.3)
        temp_avg = (temp_avg * 0.8) + (mean_temp * 0.2)
        
        rain_anomaly = precipitation - mean_rain
        temp_anomaly = temp_avg - mean_temp
        
        X_future = pd.DataFrame([{
            "precipitation": precipitation,
            "temp_avg": temp_avg,
            "rain_anomaly": rain_anomaly,
            "temp_anomaly": temp_anomaly
        }])
        
        pred = model.predict(X_future)[0]
        pred_score = max(0.0, min(1.0, float(pred)))
        forecasts.append(round(pred_score, 3))
        
    return forecasts

def crop_failure_risk(score):
    if score < 0.4: return "Low"
    elif score < 0.7: return "Moderate"
    else: return "High"