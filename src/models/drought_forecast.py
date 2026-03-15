import pandas as pd
from xgboost import XGBRegressor
from datetime import datetime

FEATURES = ["precipitation", "temp_avg", "rain_anomaly", "temp_anomaly"]

def train_model(df):
    X = df[FEATURES]
    y = df["drought_score"]
    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    return model

def forecast_next_days(model, df, location):
    loc_df = df[df["location"] == location].copy()
    loc_df['date'] = pd.to_datetime(loc_df['date'])
    
    tomorrow = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    
    future_df = loc_df[loc_df['date'] >= tomorrow].copy()
    
    if future_df.empty:
        return [], []

    X_future = future_df[FEATURES]
    predictions = model.predict(X_future)
    
    forecast_scores = [max(0.0, min(1.0, float(pred))) for pred in predictions]
    forecast_dates = future_df['date'].dt.strftime('%Y-%m-%d').tolist()
    
    return forecast_dates, forecast_scores

def crop_failure_risk(score):
    if score < 0.5: return "Low"
    elif score < 0.75: return "Moderate"
    else: return "High"