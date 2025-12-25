import os
import time
import json
import base64

import streamlit as st
import pandas as pd
import folium
import numpy as np
from groq import Groq
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from streamlit_folium import st_folium
from sklearn.cluster import KMeans

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="PrimeCore | Global AI Trip Planner",
    layout="wide",
    page_icon="🚀",
)

if "trip_plan" not in st.session_state:
    st.session_state.trip_plan = None
if "trip_df" not in st.session_state:
    st.session_state.trip_df = None
if "distance" not in st.session_state:
    st.session_state.distance = 0

# =========================================================
# CSS
# =========================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700;800&display=swap');
.stApp {
    background: linear-gradient(rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0.75)),
                url(https://images.unsplash.com/photo-1507525428034-b723cf961d3e?fit=crop&w=1950&q=80);
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'Inter', sans-serif;
}
.header-container {
    display: flex;
    align-items: center;
    gap: 18px;
    margin-bottom: 5px;
}
.brand-title {
    font-size: 30px !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #00d4ff, #00ff88);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    margin: 0 !important;
    line-height: 1;
}
.brand-tagline {
    font-size: 11px !important;
    color: #00d4ff !important;
    text-transform: uppercase;
    letter-spacing: 4px;
    font-weight: 400;
    margin: 0 !important;
    opacity: 0.8;
}
.architect-pill {
    display: inline-block;
    padding: 10px 30px;
    margin-top: 15px !important;
    background: rgba(0, 212, 255, 0.12);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-left: 6px solid #00d4ff;
    border-radius: 6px;
    font-size: 32px !important;
    font-weight: 300 !important;
    letter-spacing: 6px !important;
    color: #ffffff !important;
    text-transform: uppercase;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
}
.stat-box, .place-card, .restaurant-card, .itinerary-box {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(15px);
    padding: 20px;
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 15px;
}
.visit-link {
    color: #00d4ff !important;
    text-decoration: none;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# HEADER
# =========================================================
def get_base64_image(path: str):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

logo_b64 = get_base64_image("logo.png")

if logo_b64:
    header_html = f"""
    <div class="header-container">
        <img src="data:image/png;base64,{logo_b64}" width="65">
        <div>
            <p class="brand-title">PRIMECORE</p>
            <p class="brand-tagline">Technology Solutions</p>
        </div>
    </div>
    <div class="architect-pill">JOURNEY ARCHITECT</div>
    """
else:
    header_html = """
    <div class="header-container">
        <div style="width:50px; height:50px; background:#00d4ff; border-radius:10px;"></div>
        <div>
            <p class="brand-title">PRIMECORE</p>
            <p class="brand-tagline">Technology Solutions</p>
        </div>
    </div>
    <div class="architect-pill">JOURNEY ARCHITECT</div>
    """

st.markdown(header_html, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# GROQ CLIENT (API KEY FROM ENV/SECRETS)
# =========================================================
# Put your key in environment or .streamlit/secrets.toml as GROQ_API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))

client = Groq(api_key=GROQ_API_KEY)

# =========================================================
# BACKEND HELPERS
# =========================================================
def get_itinerary_ai(origin: str, dest: str, days: int, members: int) -> dict:
    prompt = f"""
Comprehensive Travel Analytics for {origin} to {dest}. {days} days, {members} travelers.
1. Calculate total trip budget in $.
2. Summary itinerary (max 2 lines per day).
3. Detailed info (5 lines) for 3 primary landmarks.
4. Provide 3 specific Hotels and 3 specific Restaurants.
5. CRITICAL: Provide verified URLs for every Hotel and Restaurant.
Return ONLY a JSON object:
{{
  "totalbudget": "XXXXX",
  "travelmode": "Best transport",
  "itinerary": {{"Day 1": "...", "Day 2": "..."}},
  "places": [{{"name": "Name", "info": "5 lines info", "time": "Best time"}}],
  "restaurants": [{{"name": "Name", "specialty": "Dish", "link": "URL"}}],
  "hotels": [{{"name": "Name", "tier": "Budget/Luxury", "price": "$X", "link": "URL"}}],
  "mapcoords": ["Spot 1", "Spot 2", "Spot 3"]
}}
"""

    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Groq text model.[web:23]
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(chat.choices[0].message.content)

def get_mapping(origin, dest, landmarks):
    geolocator = Nominatim(user_agent="primecoretrending2025")
    points = []
    route = [origin] + landmarks + [dest]

    for q in route:
        try:
            time.sleep(1.2)
            loc = geolocator.geocode(q)
            if loc:
                points.append({"name": q, "lat": loc.latitude, "lon": loc.longitude})
        except Exception:
            continue

    return pd.DataFrame(points)

def cluster_attractions(df, n_clusters):
    if df is None or df.empty or len(df) < 2:
        return df
    coords = df[["lat", "lon"]].values
    kmeans = KMeans(
        n_clusters=min(n_clusters, len(df)),
        random_state=42,
        n_init="auto",
    )
    df["cluster"] = kmeans.fit_predict(coords)
    return df.sort_values("cluster")

# =========================================================
# INPUT UI
# =========================================================
st.markdown("<br>", unsafe_allow_html=True)

with st.container():
    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        origin = st.text_input("Origin City", "Mumbai, India")
    with c2:
        dest = st.text_input("Destination", "Zurich, Switzerland")
    with c3:
        days = st.number_input("Days", 1, 30, 4)

    col_a, col_b = st.columns(2)
    with col_a:
        members = st.number_input("Persons", 1, 20, 2)
    with col_b:
        st.sidebar.write("Groq API key is configured on the server.")

# =========================================================
# RUN BUTTON
# =========================================================
if st.button("🚀 EXECUTE TRIP ANALYTICS"):
    if not GROQ_API_KEY:
        st.error("Groq API key is not set on the server.")
    else:
        try:
            with st.spinner("PrimeCore Engine Processing Global Logistics..."):
                plan = get_itinerary_ai(origin, dest, days, members)
                st.session_state.trip_plan = plan

                st.session_state.trip_df = get_mapping(
                    origin, dest, plan.get("mapcoords", [])
                )
                df = cluster_attractions(st.session_state.trip_df, days)
                st.session_state.trip_df = df

                if df is None or df.empty or "name" not in df.columns:
                    st.session_state.distance = 0
                else:
                    start = df[df["name"].str.lower().str.contains(origin.lower())]
                    end = df[df["name"].str.lower().str.contains(dest.lower())]
                    if not start.empty and not end.empty:
                        st.session_state.distance = int(
                            geodesic(
                                (start.lat.values[0], start.lon.values[0]),
                                (end.lat.values[0], end.lon.values[0]),
                            ).km
                        )
                    else:
                        st.session_state.distance = 0
        except Exception as e:
            st.error(f"Trip generation failed: {e}")

# =========================================================
# OUTPUT
# =========================================================
if st.session_state.trip_plan:
    p = st.session_state.trip_plan
    df = st.session_state.trip_df

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(
            f"""
        <div class="stat-box">
            <b>📏 Distance</b><br>
            {st.session_state.distance} KM
        </div>
        """,
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f"""
        <div class="stat-box">
            <b>💰 AI Budget</b><br>
            ${p.get("totalbudget", "N/A")}
        </div>
        """,
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f"""
        <div class="stat-box">
            <b>🚗 Transport</b><br>
            {p.get("travelmode", "N/A")}
        </div>
        """,
            unsafe_allow_html=True,
        )
    with s4:
        st.markdown(
            f"""
        <div class="stat-box">
            <b>👥 Travelers</b><br>
            {members} Pax
        </div>
        """,
            unsafe_allow_html=True,
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 PLANS", "🗺️ PLACE EXPLORER", "🍽️ DINING & HOTELS", "🗺️ ML ROUTE MAP"]
    )

    with tab1:
        st.subheader("Precise Daily Schedule")
        for day_label, activity in p.get("itinerary", {}).items():
            st.markdown(
                f"""
            <div class="itinerary-box">
                <b>{day_label}</b><br>{activity}
            </div>
            """,
                unsafe_allow_html=True,
            )

    with tab2:
        st.subheader("Tourist Insights")
        for place in p.get("places", []):
            st.markdown(
                f"""
            <div class="place-card">
                <h3 style="color:#00d4ff; margin:0">{place['name']}</h3>
                <p style="font-size:18px">{place['info']}</p>
                <p><b>📅 Best Visit:</b> {place['time']}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with tab3:
        col_food, col_stay = st.columns(2)
        with col_food:
            st.subheader("Top Eateries")
            for res in p.get("restaurants", []):
                st.markdown(
                    f"""
                <div class="restaurant-card">
                    <h4 style="color:#00d4ff; margin:0">{res['name']}</h4>
                    <p style="margin:0">Cuisine: {res['specialty']}</p>
                    <a href="{res['link']}" target="_blank" class="visit-link">🌐 Website</a>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        with col_stay:
            st.subheader("Top Hotel Picks")
            for hot in p.get("hotels", []):
                st.markdown(
                    f"""
                <div class="restaurant-card">
                    <h4 style="color:#00ff88; margin:0">{hot['name']}</h4>
                    <p style="margin:0">{hot['tier']} • {hot['price']}</p>
                    <a href="{hot['link']}" target="_blank" class="visit-link" style="color:#00ff88 !important">🌐 Book Now</a>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    with tab4:
        st.subheader("🧠 Geographic Intelligent Path (KMeans Clustering)")
        if df is not None and not df.empty:
            m = folium.Map(location=[df.lat.mean(), df.lon.mean()], zoom_start=4)
            path = list(zip(df.lat, df.lon))
            folium.PolyLine(path, color="#00d4ff", weight=5, opacity=0.8).add_to(m)

            cluster_colors = [
                "red", "blue", "green", "orange",
                "purple", "pink", "darkred", "lightred",
            ]
            if "cluster" in df.columns:
                for _, row in df.iterrows():
                    color = cluster_colors[int(row["cluster"]) % len(cluster_colors)]
                    folium.Marker(
                        [row.lat, row.lon],
                        popup=f"{row.name}<br>Cluster {row.cluster}",
                        icon=folium.Icon(color=color, icon="info-sign"),
                    ).add_to(m)
            else:
                for _, row in df.iterrows():
                    folium.Marker(
                        [row.lat, row.lon],
                        popup=row.name,
                        icon=folium.Icon(color="blue", icon="info-sign"),
                    ).add_to(m)

            st_folium(m, width=1100, height=500, key="primeroutemap")

# =========================================================
# RESET
# =========================================================
st.sidebar.button("🔄 Reset Application", on_click=lambda: st.session_state.clear())
