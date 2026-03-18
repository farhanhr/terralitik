import requests
import pandas as pd
import geopandas as gpd
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

def fetch_weather_batch(locations_batch):
    lats = ",".join([str(loc["lat"]) for loc in locations_batch])
    lons = ",".join([str(loc["lon"]) for loc in locations_batch])
    
    params = {
        "latitude": lats,
        "longitude": lons,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "auto",
        "past_days": 3,
        "forecast_days": 14
    }
    
    try:
        response = requests.get(API_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if isinstance(data, dict) and "daily" in data:
            data = [data]
            
        batch_df_list = []
        for i, loc_data in enumerate(data):
            loc_info = locations_batch[i]
            daily = loc_data.get("daily", {})
            
            if not daily:
                continue
                
            df = pd.DataFrame({
                "date": daily.get("time", []),
                "location": loc_info["name"],
                "lat": loc_info["lat"],
                "lon": loc_info["lon"],
                "temp_max": daily.get("temperature_2m_max", []),
                "temp_min": daily.get("temperature_2m_min", []),
                "precipitation": daily.get("precipitation_sum", [])
            })
            batch_df_list.append(df)
            
        return pd.concat(batch_df_list, ignore_index=True) if batch_df_list else pd.DataFrame()
        
    except Exception as e:
        print(f"Gagal mengambil batch API: {e}")
        return pd.DataFrame()

def fetch_all_locations():
    locations = get_all_locations()
    all_data_list = []
    
    chunk_size = 50 
    print(f"Mulai penarikan data satelit secara Batch untuk {len(locations)} wilayah...")
    
    for i in range(0, len(locations), chunk_size):
        batch = locations[i:i + chunk_size]
        print(f"Memproses batch {i+1} hingga {min(i+chunk_size, len(locations))}...")
        
        df_batch = fetch_weather_batch(batch)
        if not df_batch.empty:
            all_data_list.append(df_batch)

    if all_data_list:
        new_df = pd.concat(all_data_list, ignore_index=True)
        os.makedirs("data/raw", exist_ok=True)
        
        if os.path.exists(MASTER_CSV):
            old_df = pd.read_csv(MASTER_CSV)
            combined_df = pd.concat([old_df, new_df], ignore_index=True)
            final_df = combined_df.drop_duplicates(subset=["date", "location"], keep="last")
        else:
            final_df = new_df
            
        final_df = final_df.sort_values(by=["location", "date"]).reset_index(drop=True)
        final_df.to_csv(MASTER_CSV, index=False)
        print(f"Finish! total updated data: {len(final_df)}")
        return final_df
        
    return pd.DataFrame()

if __name__ == "__main__":
    fetch_all_locations()