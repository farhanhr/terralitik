import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import urllib.parse
from models.drought_forecast import train_model, forecast_next_days, crop_failure_risk

try:
    from models.ai_assistant import get_ai_recommendation
except ImportError:
    pass 

st.markdown(
    '<meta name="dicoding:email" content="farhanhr00@gmail.com">',
    unsafe_allow_html=True
)

st.set_page_config(page_title="Terralitik | EWS", page_icon="🌱", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0;}
    .sub-header { font-size: 1.2rem; color: #64748B; margin-bottom: 2rem;}
    </style>
    <div class="main-header">🌱 Terralitik</div>
    <div class="sub-header">Java Drought Risk Early Warning System</div>
""", unsafe_allow_html=True)

GEOJSON_PATH = "data/geospatial/jawa_kabupaten.geojson"

@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("data/processed/drought_risk.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_and_prepare_data()

with open(GEOJSON_PATH, encoding="utf-8") as f:
    geojson = json.load(f)

@st.cache_resource
def get_model(data):
    return train_model(data)

model = get_model(df)

df_map = df.dropna(subset=['drought_score']).sort_values('date').groupby('location').last().reset_index()


for feature in geojson["features"]:
    props = feature["properties"]
    feature["id"] = props.get("regency_city", "")

st.markdown("### 🗺 Persebaran Risiko Kekeringan Terkini")

fig_map = px.choropleth_map(
    df_map,
    geojson=geojson,
    locations="location",
    featureidkey="id",
    color="drought_score",
    color_continuous_scale="RdYlGn_r", 
    range_color=[0, 1],
    map_style="carto-positron", 
    zoom=6,
    center={"lat": -7.25, "lon": 110.0},
    opacity=0.8,
    hover_name="location",
    hover_data={"drought_score": True, "risk_level": True},
    labels={'drought_score': 'Indeks Risiko (0-1)'}
)
fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=500)
st.plotly_chart(fig_map, width='stretch')

st.divider()

st.markdown("### 📊 Analisis & Prediksi Spesifik Wilayah")
locations_list = sorted(df["location"].unique())
selected_location = st.selectbox("Pilih Kabupaten/Kota untuk dianalisis:", locations_list)

loc_df = df[df["location"] == selected_location].sort_values("date")
valid_loc_df = loc_df.dropna(subset=['drought_score', 'risk_level'])

forecast_dates, forecast_scores = forecast_next_days(model, df, selected_location)

if forecast_scores:
    future_risk_score = forecast_scores[-1]
    avg_forecast_score = sum(forecast_scores) / len(forecast_scores)
elif not valid_loc_df.empty:
    future_risk_score = valid_loc_df.iloc[-1]["drought_score"]
    avg_forecast_score = valid_loc_df.iloc[-1]["drought_score"]
else:
    future_risk_score = 0.0
    avg_forecast_score = 0.0

risk_future = crop_failure_risk(future_risk_score)

if not valid_loc_df.empty:
    latest_risk = valid_loc_df.iloc[-1]["risk_level"]
    latest_date_valid = valid_loc_df.iloc[-1]["date"]
else:
    latest_risk = "Low"
    latest_date_valid = pd.Timestamp.now()

tab1, tab2, tab3 = st.tabs(["📈 Proyeksi AI", "🤖 Asisten Mitigasi", "🔍 Data History"])

with tab1:
    if forecast_dates:
        forecast_df = pd.DataFrame({
            "Tanggal": forecast_dates,
            "Prediksi Skor Kekeringan": forecast_scores
        })
        fig_forecast = px.line(
            forecast_df, x="Tanggal", y="Prediksi Skor Kekeringan",
            markers=True, color_discrete_sequence=["#d62728"]
        )
        fig_forecast.add_hline(y=0.75, line_dash="dash", line_color="red", annotation_text="Krisis (Gagal Panen)")
        fig_forecast.add_hline(y=0.50, line_dash="dash", line_color="orange", annotation_text="Waspada")
        st.plotly_chart(fig_forecast, width='stretch')
    else:
        st.warning("Data satelit masa depan belum tersedia untuk wilayah ini.")

BASELINE_YIELD_VALUE_PER_HA = 25000000

if avg_forecast_score < 0.5:
    potential_loss_percentage = 0.0
elif avg_forecast_score < 0.75:
    potential_loss_percentage = ((avg_forecast_score - 0.5) / 0.25) * 0.30
else:
    potential_loss_percentage = 0.30 + ((avg_forecast_score - 0.75) / 0.25) * 0.70

potential_loss_percentage = min(potential_loss_percentage, 1.0)
est_loss_value = BASELINE_YIELD_VALUE_PER_HA * potential_loss_percentage

with tab2:
    st.markdown("#### 💬 Asisten Terralitik")
    st.info("Asisten akan menerjemahkan data teknis menjadi instruksi mitigasi.")
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    
    if st.button("Generate Rekomendasi Mitigasi dengan AI", type="primary"):
        with st.spinner("AI sedang menganalisis data satelit dan merumuskan kebijakan..."):
            if "get_ai_recommendation" in locals():
                ai_response = get_ai_recommendation(
                    selected_location, latest_risk, avg_forecast_score, est_loss_value, api_key
                )
                st.write(ai_response)
            else:
                st.error("Modul AI belum terhubung dengan benar.")

with tab3:
    col_temp, col_rain = st.columns(2)
    with col_temp:
        max_temp = max(loc_df["temp_max"].max(), loc_df["temp_min"].max())
        min_temp = min(0, loc_df["temp_min"].min())
        upper_temp = max_temp * 1.3
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=loc_df["date"], y=loc_df["temp_max"], mode='lines+markers', name='Suhu Max', line=dict(color='red')))
        fig_temp.add_trace(go.Scatter(x=loc_df["date"], y=loc_df["temp_min"], mode='lines+markers', name='Suhu Min', line=dict(color='blue')))
        fig_temp.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            height=400,
            title="Pergerakan Suhu (°C)",
            xaxis_title="Tanggal",
            yaxis_title="Suhu",
            yaxis=dict(range=[min_temp, upper_temp])
        )
        st.plotly_chart(fig_temp, width='stretch')
            
    with col_rain:
        max_rain = loc_df["precipitation"].max()
        upper_rain = max_rain * 1.3
        fig_rain = px.bar(loc_df, x="date", y="precipitation", title="Histori Hujan Terakhir (mm)")
        fig_rain.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            height=400,
            yaxis=dict(range=[0, upper_rain])
        )
        st.plotly_chart(fig_rain, width='stretch')

    if not valid_loc_df.empty:
        latest_loc_data = valid_loc_df.iloc[-1]
        fig_xai = go.Figure(go.Waterfall(
            name="20", orientation="h", measure=["relative", "relative", "total"],
            y=["Anomali Hujan", "Anomali Suhu", "Total Skor"],
            x=[-latest_loc_data['rain_anomaly'] * 0.1, latest_loc_data['temp_anomaly'] * 0.2, latest_loc_data['drought_score']],
            connector={"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig_xai.update_layout(title="Faktor Penyumbang Risiko Saat Ini", showlegend=False, margin=dict(l=0, r=0, t=40, b=0), height=300)
        st.plotly_chart(fig_xai, width='stretch')
    else:
        st.warning("Data histori tidak tersedia.")
st.divider()
st.markdown("### 💼 Analisis Risiko Ekonomi & Tata Kelola")

col_eco1, col_eco2, col_eco3 = st.columns(3)
with col_eco1:
    st.metric(label="Rata-Rata Kerentanan", value=f"{avg_forecast_score * 100:.1f}%", delta="Kritis" if avg_forecast_score >= 0.75 else "Aman", delta_color="inverse" if avg_forecast_score >= 0.75 else "normal")
with col_eco2:
    st.metric(label="Estimasi Penurunan Panen", value=f"{potential_loss_percentage * 100:.1f}%", delta="- Tonase" if potential_loss_percentage > 0 else "Normal", delta_color="inverse" if potential_loss_percentage > 0 else "normal")
with col_eco3:
    st.metric(label="Potensi Kerugian per Hektar", value=f"Rp {est_loss_value:,.0f}".replace(',', '.'), delta="Risiko Finansial" if est_loss_value > 0 else "Aman", delta_color="inverse" if est_loss_value > 0 else "normal")

if forecast_dates:
    report_df = pd.DataFrame({
        "Tanggal": forecast_dates,
        "Lokasi": [selected_location] * len(forecast_dates),
        "Prediksi_Skor_Kekeringan": forecast_scores,
        "Estimasi_Kerugian_Rp_Per_Ha": [est_loss_value] * len(forecast_dates)
    })
    csv = report_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Laporan Prediksi (CSV)",
        data=csv,
        file_name=f"Laporan_Terralitik_{selected_location}.csv",
        mime="text/csv",
    )

st.markdown("### 🚨 Distribusi Peringatan Dini")
if not valid_loc_df.empty:
    wa_message = f"""*⚠️ PERINGATAN DINI KEKERINGAN TERRALITIK*
Wilayah: *{selected_location}* | Update: {latest_date_valid.strftime('%d %b %Y')}

*Status Analisis:*
Tingkat Risiko: *{latest_risk.upper()}*
Skor Kekeringan: {avg_forecast_score:.2f} (Batas Kritis > 0.75)

*Rekomendasi Tindakan Segera:*
"""
    if latest_risk == "High" or avg_forecast_score >= 0.75:
        wa_message += "- Aktifkan pompa air cadangan dari embung terdekat.\n- Tunda penanaman bibit baru hingga kelembapan tanah stabil.\n- Persiapkan asuransi klaim gagal panen (AUTP)."
    elif latest_risk == "Moderate" or avg_forecast_score >= 0.5:
        wa_message += "- Kurangi volume penyiraman, terapkan sistem irigasi bergilir.\n- Pantau kapasitas sumber air permukaan secara harian."
    else:
        wa_message += "- Kondisi terpantau aman. Lakukan pemeliharaan irigasi rutin."

    wa_message += "\n\n_Dihasilkan oleh sistem AI Terralitik._"
    encoded_message = urllib.parse.quote(wa_message)
    whatsapp_url = f"https://wa.me/?text={encoded_message}"

    if latest_risk in ["High", "Moderate"] or avg_forecast_score >= 0.5:
        st.warning(f"Sistem mendeteksi peningkatan kerentanan di {selected_location}. Distribusikan mitigasi!")
        st.link_button("Bagikan Peringatan via WhatsApp", whatsapp_url, type="primary")
    else:
        st.success("Kondisi iklim saat ini terpantau aman. Tidak ada tindakan darurat yang diperlukan.")