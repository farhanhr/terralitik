import requests
import pandas as pd
import geopandas as gpd
import time
import os

API_URL = "https://api.open-meteo.com/v1/forecast"
GEOJSON_PATH = "data/geospatial/jawa_kabupaten.geojson"
MASTER_CSV = "data/raw/weather_jawa_master.csv"

def get_all_locations():
    gdf = gpd.read_file(GEOJSON_PATH)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf['centroid'] = gdf.geometry.centroid
        
    locations = []
    for _, row in gdf.iterrows():
        loc_name = row['regency_city']
        lat, lon = row['centroid'].y, row['centroid'].x
        if pd.isna(lat) or pd.isna(lon) or not loc_name:
            continue
        locations.append({"name": loc_name, "lat": lat, "lon": lon})
    return locations

def fetch_weather_for_location(location):
    params = {
        "latitude": location["lat"],
        "longitude": location["lon"],
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "auto",
        "past_days": 14,      
        "forecast_days": 16  
    }
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        daily = response.json()["daily"]
        
        return pd.DataFrame({
            "date": daily["time"],
            "location": location["name"],
            "lat": location["lat"],
            "lon": location["lon"],
            "temp_max": daily["temperature_2m_max"],
            "temp_min": daily["temperature_2m_min"],
            "precipitation": daily["precipitation_sum"]
        })
    except Exception as e:
        print(f"Gagal di {location['name']}: {e}")
        return pd.DataFrame()

def fetch_all_locations():
    locations = get_all_locations()
    new_data_list = []
    
    print(f"🔄 Menarik data riil & forecast satelit untuk {len(locations)} wilayah...")
    for i, loc in enumerate(locations):
        df_new = fetch_weather_for_location(loc)
        if not df_new.empty:
            new_data_list.append(df_new)
        if (i + 1) % 40 == 0:
            time.sleep(5) 

    if new_data_list:
        new_df = pd.concat(new_data_list, ignore_index=True)
        os.makedirs("data/raw", exist_ok=True)
        
        if os.path.exists(MASTER_CSV):
            old_df = pd.read_csv(MASTER_CSV)
            combined_df = pd.concat([old_df, new_df], ignore_index=True)
            final_df = combined_df.drop_duplicates(subset=["date", "location"], keep="last")
        else:
            final_df = new_df
            
        final_df = final_df.sort_values(by=["location", "date"]).reset_index(drop=True)
        final_df.to_csv(MASTER_CSV, index=False)
        print("✅ Data tersimpan secara dinamis!")
        return final_df
        
    return pd.DataFrame()

if __name__ == "__main__":
    fetch_all_locations()