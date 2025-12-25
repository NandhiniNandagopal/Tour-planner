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
# UTILS
# =========================================================
def safe_get(d, key, default="N/A"):
    return d[key] if isinstance(d, dict) and key in d else default

def total_route_distance(df):
    if df is None or len(df) < 2:
        return 0
    dist = 0
    for i in range(len(df) - 1):
        dist += geodesic(
            (df.lat.iloc[i], df.lon.iloc[i]),
            (df.lat.iloc[i + 1], df.lon.iloc[i + 1])
        ).km
    return int(dist)

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700;800&display=swap');
.stApp {
    background: linear-gradient(rgba(0,0,0,.75), rgba(0,0,0,.75)),
    url(https://images.unsplash.com/photo-1507525428034-b723cf961d3e);
    background-size: cover;
    background-attachment: fixed;
    font-family: 'Inter', sans-serif;
}
.stat-box,.place-card,.restaurant-card,.itinerary-box{
    background: rgba(255,255,255,.06);
    backdrop-filter: blur(14px);
    padding: 18px;
    border-radius: 18px;
    margin-bottom: 14px;
}
.visit-link{color:#00d4ff;text-decoration:none;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
def get_base64_image(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

logo = get_base64_image("logo.png")

st.markdown(f"""
<div style="display:flex;align-items:center;gap:15px">
    {"<img src='data:image/png;base64,"+logo+"' width='60'>" if logo else ""}
    <div>
        <h2 style="margin:0;color:#00d4ff">PRIMECORE</h2>
        <small style="letter-spacing:3px">JOURNEY ARCHITECT</small>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# GROQ CLIENT
# =========================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
client = Groq(api_key=GROQ_API_KEY)

# =========================================================
# AI ENGINE
# =========================================================
def get_itinerary_ai(origin, dest, days, members, style, interests, budget):
    prompt = f"""
You are an expert AI travel planner.

Trip:
- From: {origin}
- To: {dest}
- Days: {days}
- Travelers: {members}

Preferences:
- Travel style: {style}
- Interests: {", ".join(interests)}
- Budget level: {budget}

Return ONLY valid JSON.
"""
    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(chat.choices[0].message.content)

# =========================================================
# GEO + ML
# =========================================================
def get_mapping(origin, dest, landmarks):
    geo = Nominatim(user_agent="primecore")
    points = []
    for q in [origin] + landmarks + [dest]:
        try:
            time.sleep(1)
            loc = geo.geocode(q)
            if loc:
                points.append({
                    "name": q,
                    "lat": loc.latitude,
                    "lon": loc.longitude
                })
        except:
            pass
    return pd.DataFrame(points)

def cluster_attractions(df, days):
    if df is None or df.empty:
        return df

    coords = df[["lat", "lon"]].values

    # soften start & end points
    coords[0] *= 0.7
    coords[-1] *= 0.7

    k = min(days, len(df))
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    df["day"] = model.fit_predict(coords) + 1
    return df.sort_values("day")

# =========================================================
# INPUT UI
# =========================================================
if "last_trip" in st.session_state:
    st.info(
        f"Last trip: {st.session_state.last_trip['origin']} → "
        f"{st.session_state.last_trip['dest']}"
    )

c1, c2, c3 = st.columns(3)
origin = c1.text_input("Origin", "Mumbai, India")
dest = c2.text_input("Destination", "Zurich, Switzerland")
days = c3.number_input("Days", 1, 30, 4)

members = st.number_input("Persons", 1, 20, 2)

travel_style = st.selectbox(
    "Travel Style",
    ["Relaxed", "Adventure", "Luxury", "Budget", "Cultural"]
)

interests = st.multiselect(
    "Interests",
    ["Nature", "Food", "History", "Shopping", "Nightlife", "Photography"],
    default=["Food", "Nature"]
)

budget_level = st.select_slider(
    "Budget Preference", ["Low", "Medium", "High"], value="Medium"
)

# =========================================================
# RUN
# =========================================================
if st.button("🚀 EXECUTE TRIP ANALYTICS"):
    if not GROQ_API_KEY:
        st.error("Groq API key missing")
    else:
        with st.status("Planning journey...", expanded=True) as status:
            st.write("🧠 Generating itinerary")
            plan = get_itinerary_ai(
                origin, dest, days, members,
                travel_style, interests, budget_level
            )
            st.session_state.trip_plan = plan

            st.write("🗺 Mapping locations")
            df = get_mapping(origin, dest, safe_get(plan, "mapcoords", []))

            st.write("🧠 Optimizing route")
            df = cluster_attractions(df, days)
            st.session_state.trip_df = df

            st.session_state.distance = total_route_distance(df)

            st.session_state.last_trip = {
                "origin": origin,
                "dest": dest,
                "style": travel_style
            }

            status.update(label="Journey ready", state="complete")

# =========================================================
# OUTPUT
# =========================================================
if st.session_state.trip_plan:
    p = st.session_state.trip_plan
    df = st.session_state.trip_df

    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(f"<div class='stat-box'>📏 {st.session_state.distance} km</div>", unsafe_allow_html=True)
    s2.markdown(f"<div class='stat-box'>💰 {safe_get(p,'totalbudget')}</div>", unsafe_allow_html=True)
    s3.markdown(f"<div class='stat-box'>🚗 {safe_get(p,'travelmode')}</div>", unsafe_allow_html=True)
    s4.markdown(f"<div class='stat-box'>👥 {members} pax</div>", unsafe_allow_html=True)

    with st.expander("💸 Budget Breakdown"):
        for k, v in safe_get(p, "budgetbreakdown", {}).items():
            st.write(f"**{k.title()}**: {v}")

    tabs = st.tabs(["📋 Plan", "🗺 Places", "🍽 Hotels & Food", "🧠 ML Route"])

    with tabs[0]:
        for d, a in safe_get(p, "itinerary", {}).items():
            st.markdown(
                f"<div class='itinerary-box'><b>{d}</b><br>{a}</div>",
                unsafe_allow_html=True
            )

    with tabs[1]:
        for pl in safe_get(p, "places", []):
            st.markdown(
                f"<div class='place-card'><b>{pl['name']}</b><br>{pl['info']}</div>",
                unsafe_allow_html=True
            )

    with tabs[2]:
        for r in safe_get(p, "restaurants", []):
            st.markdown(
                f"<div class='restaurant-card'>{r['name']} – {r['specialty']}<br>"
                f"<a href='{r['link']}' class='visit-link'>Visit</a></div>",
                unsafe_allow_html=True
            )
        for h in safe_get(p, "hotels", []):
            st.markdown(
                f"<div class='restaurant-card'>{h['name']} – {h['tier']}<br>"
                f"<a href='{h['link']}' class='visit-link'>Book</a></div>",
                unsafe_allow_html=True
            )

    with tabs[3]:
        st.caption("Route optimized using K-Means clustering to minimize daily travel distance.")
        if df is not None and not df.empty:
            m = folium.Map(
                location=[df.lat.mean(), df.lon.mean()],
                zoom_start=4
            )
            for _, r in df.iterrows():
                folium.Marker(
                    [r.lat, r.lon],
                    popup=f"{r.name} (Day {r.day})"
                ).add_to(m)
            folium.PolyLine(list(zip(df.lat, df.lon))).add_to(m)
            st_folium(m, width=1100, height=500)

    st.download_button(
        "📥 Download Trip (JSON)",
        data=json.dumps(p, indent=2),
        file_name="trip_plan.json"
    )

st.sidebar.button("🔄 Reset", on_click=lambda: st.session_state.clear())
