import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import urllib.parse

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
    forecast_dates, forecast_scores = forecast_next_days(model, df, selected_location)
    
    if forecast_dates:
        forecast_df = pd.DataFrame({
            "Tanggal": forecast_dates,
            "Prediksi Skor Kekeringan": forecast_scores
        })

        fig_forecast = px.line(
            forecast_df, x="Tanggal", y="Prediksi Skor Kekeringan",
            title=f"Proyeksi Model XGBoost (Berbasis Data Satelit): {selected_location}",
            markers=True,
            color_discrete_sequence=["#d62728"]
        )
        fig_forecast.add_hline(y=0.75, line_dash="dash", line_color="red", annotation_text="Batas Risiko Tinggi")
        fig_forecast.add_hline(y=0.50, line_dash="dash", line_color="orange", annotation_text="Batas Risiko Sedang")
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.warning("Data satelit masa depan belum tersedia. Silakan perbarui data harian.")

st.divider()
col1, col2 = st.columns(2)

future_risk_score = forecast_scores[-1] if forecast_scores else loc_df.iloc[-1]["drought_score"]
risk_future = crop_failure_risk(future_risk_score)
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

st.divider()
st.header("💼 Analisis Risiko Ekonomi & Tata Kelola")


BASELINE_YIELD_VALUE_PER_HA = 25000000

if forecast_scores:
    avg_forecast_score = sum(forecast_scores) / len(forecast_scores)
else:
    avg_forecast_score = loc_df.iloc[-1]["drought_score"]

if avg_forecast_score < 0.5:
    potential_loss_percentage = 0.0
elif avg_forecast_score < 0.75:
    potential_loss_percentage = ((avg_forecast_score - 0.5) / 0.25) * 0.30
else:
    potential_loss_percentage = 0.30 + ((avg_forecast_score - 0.75) / 0.25) * 0.70

potential_loss_percentage = min(potential_loss_percentage, 1.0)

col_eco1, col_eco2, col_eco3 = st.columns(3)

with col_eco1:
    st.metric(
        label="Rata-Rata Indeks Kerentanan (30 Hari)",
        value=f"{avg_forecast_score * 100:.1f}%",
        delta="Kritis" if avg_forecast_score >= 0.75 else "Aman",
        delta_color="inverse" if avg_forecast_score >= 0.75 else "normal"
    )

with col_eco2:
    st.metric(
        label="Estimasi Penurunan Hasil Panen",
        value=f"{potential_loss_percentage * 100:.1f}%",
        delta="- Tonase" if potential_loss_percentage > 0 else "Normal",
        delta_color="red" if potential_loss_percentage > 0 else "normal"
    )

with col_eco3:
    est_loss_value = BASELINE_YIELD_VALUE_PER_HA * potential_loss_percentage
    st.metric(
        label="Potensi Kerugian per Hektar",
        value=f"Rp {est_loss_value:,.0f}".replace(',', '.'),
        delta="Risiko Finansial" if est_loss_value > 0 else "Aman",
        delta_color="inverse" if est_loss_value > 0 else "normal"
    )

st.subheader("🔍 Analisis Faktor Risiko")

latest_loc_data = loc_df.iloc[-1]

fig_xai = go.Figure(go.Waterfall(
    name="20", orientation="h",
    measure=["relative", "relative", "total"],
    y=["Anomali Curah Hujan", "Anomali Suhu", "Total Skor Kekeringan"],
    
    x=[
        -latest_loc_data['rain_anomaly'] * 0.1, 
        latest_loc_data['temp_anomaly'] * 0.2,  
        latest_loc_data['drought_score']
    ],
    connector={"line":{"color":"rgb(63, 63, 63)"}},
))
fig_xai.update_layout(
    title=f"Kontribusi Variabel Iklim terhadap Risiko Saat Ini di {selected_location}",
    showlegend=False,
    height=350,
    margin=dict(l=0, r=0, t=40, b=0)
)
st.plotly_chart(fig_xai, use_container_width=True)

st.divider()
st.header("🚨 Distribusi Peringatan Dini")
st.markdown("Kirimkan peringatan terenkripsi dan instruksi mitigasi langsung ke grup koordinator lapangan atau petani.")

wa_message = f"""
*⚠️ PERINGATAN DINI KEKERINGAN TERRALITIK*
Wilayah: *{selected_location}*
Tanggal Update: {latest_date.strftime('%d %b %Y')}

*Status Analisis AI:*
Tingkat Risiko: *{latest_risk.upper()}*
Skor Kekeringan: {avg_forecast_score:.2f} (0=Aman, 1=Kritis)

*Rekomendasi Tindakan Segera:*
"""

if latest_risk == "High":
    wa_message += "- Aktifkan pompa air cadangan dari embung terdekat.\n- Tunda penanaman bibit baru hingga kelembapan tanah stabil.\n- Persiapkan asuransi klaim gagal panen (AUTP)."
elif latest_risk == "Moderate":
    wa_message += "- Kurangi volume penyiraman, terapkan sistem irigasi bergilir.\n- Pantau kapasitas sumber air permukaan secara harian."
else:
    wa_message += "- Kondisi normal. Lakukan pemeliharaan irigasi rutin."

wa_message += "\n\n_Dihasilkan secara otomatis oleh sistem AI Terralitik._"

encoded_message = urllib.parse.quote(wa_message)
whatsapp_url = f"https://wa.me/?text={encoded_message}"

if latest_risk in ["High", "Moderate"]:
    st.warning(f"Sistem mendeteksi anomali di {selected_location}. Segera distribusikan peringatan ke aparat desa!")
    st.link_button(
        "📱 Bagikan Peringatan via WhatsApp", 
        whatsapp_url, 
        type="primary"
    )
else:
    st.success("Kondisi iklim saat ini terpantau aman. Tidak ada tindakan darurat yang diperlukan.")