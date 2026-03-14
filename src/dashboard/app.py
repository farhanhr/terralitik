import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data.spatial_join import attach_regency
from models.drought_forecast import train_model, forecast_next_days, crop_failure_risk

st.set_page_config(page_title="Terralitik", layout="wide")

st.title("🌱 Terralitik — AI Drought Early Warning System")
st.subheader("Monitoring Risiko Kekeringan Pertanian Pulau Jawa")

GEOJSON_PATH = "data/geospatial/jawa_kabupaten.geojson"

@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("data/processed/drought_risk.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = attach_regency(df, GEOJSON_PATH)
    return df

df = load_and_prepare_data()

with open(GEOJSON_PATH, encoding="utf-8") as f:
    geojson = json.load(f)

@st.cache_resource
def get_model(data):
    return train_model(data)

model = get_model(df)

st.header("🗺 Peta Persebaran Risiko Kekeringan Terkini")

latest_date = df['date'].max()
df_latest = df[df['date'] == latest_date]

fig_map = px.choropleth_mapbox(
    df_latest,
    geojson=geojson,
    locations="location", 
    featureidkey="properties.regency_city",
    color="drought_score",
    color_continuous_scale="RdYlGn_r", 
    range_color=[0, 1],
    mapbox_style="carto-positron",
    zoom=5.5,
    center={"lat": -7.25, "lon": 110.0},
    opacity=0.7,
    hover_name="location",
    hover_data={"drought_score": True, "risk_level": True, "location": False},
    labels={'drought_score': 'Indeks Risiko'}
)

fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig_map, use_container_width=True)

st.divider()
st.header("📊 Analisis & Prediksi Wilayah")

locations_list = sorted(df["location"].unique())
selected_location = st.selectbox("Pilih Wilayah untuk Analisis Mendalam:", locations_list)

loc_df = df[df["location"] == selected_location].sort_values("date")

tab1, tab2 = st.tabs(["Histori Cuaca & Anomali", "Prediksi Risiko AI (30 Hari)"])

with tab1:
    st.markdown(f"**Tren Cuaca Terkini di {selected_location}**")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        fig_rain = px.bar(
            loc_df, x="date", y="precipitation", 
            title="Curah Hujan (mm)", 
            labels={"precipitation": "Curah Hujan", "date": "Tanggal"},
            color_discrete_sequence=["#1f77b4"]
        )
        st.plotly_chart(fig_rain, use_container_width=True)
        
    with col_chart2:
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=loc_df["date"], y=loc_df["temp_max"], mode='lines+markers', name='Suhu Max', line=dict(color='red')))
        fig_temp.add_trace(go.Scatter(x=loc_df["date"], y=loc_df["temp_min"], mode='lines+markers', name='Suhu Min', line=dict(color='blue')))
        fig_temp.update_layout(title="Pergerakan Suhu (°C)", xaxis_title="Tanggal", yaxis_title="Suhu")
        st.plotly_chart(fig_temp, use_container_width=True)

with tab2:
    forecast = forecast_next_days(model, df, selected_location, days=30)
    
    forecast_df = pd.DataFrame({
        "Hari Ke-": list(range(1, 31)),
        "Prediksi Skor Kekeringan": forecast
    })

    fig_forecast = px.line(
        forecast_df, x="Hari Ke-", y="Prediksi Skor Kekeringan",
        title=f"Proyeksi Model XGBoost: Risiko Kekeringan {selected_location}",
        markers=True,
        color_discrete_sequence=["#d62728"]
    )
    fig_forecast.add_hline(y=0.75, line_dash="dash", line_color="red", annotation_text="Batas Risiko Tinggi")
    fig_forecast.add_hline(y=0.50, line_dash="dash", line_color="orange", annotation_text="Batas Risiko Sedang")
    st.plotly_chart(fig_forecast, use_container_width=True)

st.divider()
col1, col2 = st.columns(2)

risk_future = crop_failure_risk(forecast[-1])
latest_risk = loc_df.iloc[-1]["risk_level"]

with col1:
    st.subheader("🌾 Indikator Gagal Panen (Proyeksi Akhir Bulan)")
    if risk_future == "High":
        st.error("⚠ RISIKO TINGGI: Model AI mendeteksi probabilitas gagal panen serius.")
    elif risk_future == "Moderate":
        st.warning("⚠ RISIKO SEDANG: Kurangnya curah hujan stabil, perlu intervensi irigasi.")
    else:
        st.success("✅ RISIKO RENDAH: Kelembapan tanah diproyeksikan aman untuk panen.")

with col2:
    st.subheader("⚠ Status Saat Ini & Rekomendasi")
    if latest_risk == "High":
        st.error(f"Peringatan untuk {selected_location}: Siapkan pompa irigasi darurat dan asuransi pertanian.")
    elif latest_risk == "Moderate":
        st.warning(f"Informasi {selected_location}: Pantau kapasitas waduk lokal.")
    else:
        st.success(f"Status {selected_location}: Curah hujan saat ini mencukupi.")