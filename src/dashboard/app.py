import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Terralitik", layout="centered")

st.title("🌱 Terralitik — AI Drought Early Warning System")
st.subheader("Monitoring Risiko Kekeringan untuk Wilayah Pertanian Indonesia")

df = pd.read_csv("data/processed/drought_risk.csv")

if "lat" not in df.columns:
    st.error("Dataset belum memiliki koordinat")
    st.stop()

st.header("🗺️ Peta Risiko Kekeringan")

fig = px.scatter_mapbox(
    df,
    lat="lat",
    lon="lon",
    color="risk_level",
    hover_name="location",
    hover_data=["drought_score"],
    zoom=4,
    height=500,
    color_discrete_map={
        "Low": "green",
        "Moderate": "orange",
        "High": "red"
    }
)

fig.update_layout(mapbox_style="open-street-map")

st.plotly_chart(fig, use_container_width=True)

st.header("📍 Analisis Wilayah")

locations = df["location"].unique()

selected_location = st.selectbox(
    "Pilih Wilayah Pertanian",
    locations
)

loc_df = df[df["location"] == selected_location]

st.subheader("🌧 Curah Hujan")

rain_chart = px.line(
    loc_df,
    x="date",
    y="precipitation",
    title="Curah Hujan Harian"
)

st.plotly_chart(rain_chart, use_container_width=True)

st.subheader("🌡 Temperatur")

temp_chart = px.line(
    loc_df,
    x="date",
    y="temp_avg",
    title="Temperatur Rata-rata"
)

st.plotly_chart(temp_chart, use_container_width=True)


st.header("⚠ Early Warning System")

latest = loc_df.iloc[-1]

risk = latest["risk_level"]

if risk == "High":

    st.error(f"""
    ⚠ RISIKO KEKERINGAN TINGGI

    Wilayah: {selected_location}

    Rekomendasi:
    - Tingkatkan irigasi
    - Gunakan varietas tahan kering
    - Monitor kondisi tanah
    """)

elif risk == "Moderate":

    st.warning(f"""
    ⚠ WASPADA KEKERINGAN

    Wilayah: {selected_location}

    Curah hujan mulai menurun.
    Perhatikan pola irigasi.
    """)

else:

    st.success(f"""
    ✅ KONDISI AMAN

    Wilayah: {selected_location}

    Risiko kekeringan rendah.
    """)


