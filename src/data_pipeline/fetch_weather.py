import requests
import pandas as pd
import geopandas as gpd
import time
import os

API_URL = "https://api.open-meteo.com/v1/forecast"
GEOJSON_PATH = "data/geospatial/jawa_kabupaten.geojson"

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
            
        locations.append({
            "name": loc_name, 
            "lat": lat,
            "lon": lon
        })
    return locations

def fetch_weather_for_location(location):
    params = {
        "latitude": location["lat"],
        "longitude": location["lon"],
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "auto"
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
    all_data = []
    
    print(f"🔄 Menarik data cuaca untuk {len(locations)} wilayah di Jawa...")
    for i, loc in enumerate(locations):
        df = fetch_weather_for_location(loc)
        if not df.empty:
            all_data.append(df)
        if (i + 1) % 40 == 0:
            time.sleep(5)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        os.makedirs("data/raw", exist_ok=True)
        path = f"data/raw/weather_jawa_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
        final_df.to_csv(path, index=False)
        print(f"✅ Data tersimpan: {path}")
        return final_df
    return pd.DataFrame()

if __name__ == "__main__":
    fetch_all_locations()