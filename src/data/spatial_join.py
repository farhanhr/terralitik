import geopandas as gpd

def clean_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower()
    remove_words = ["kabupaten", "kab.", "kota", "city", "regency"]
    for w in remove_words:
        name = name.replace(w, "")
    return name.strip()

def load_regency_geojson(path):
    gdf = gpd.read_file(path)
    # Membaca format properties yang baru
    gdf = gdf[["province", "regency_city", "geometry"]]
    gdf["regency_clean"] = gdf["regency_city"].apply(clean_name)
    return gdf

def attach_regency(df, geojson_path):
    gdf = load_regency_geojson(geojson_path)
    df["location_clean"] = df["location"].apply(clean_name)
    
    merged = df.merge(
        gdf,
        left_on="location_clean",
        right_on="regency_clean",
        how="left"
    )
    return merged