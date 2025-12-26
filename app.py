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

for key in ["trip_plan", "trip_df", "distance"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(0,0,0,.78), rgba(0,0,0,.78)),
    url(https://images.unsplash.com/photo-1507525428034-b723cf961d3e);
    background-size: cover;
    background-attachment: fixed;
}
.stat-box, .card {
    background: rgba(255,255,255,.06);
    backdrop-filter: blur(15px);
    padding: 18px;
    border-radius: 18px;
    margin-bottom: 14px;
}
a { color:#00d4ff !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<h1 style="color:#00d4ff;">PRIMECORE</h1>
<p style="letter-spacing:4px;color:#aaa;">JOURNEY ARCHITECT</p>
""", unsafe_allow_html=True)

# =========================================================
# GROQ CLIENT
# =========================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
client = Groq(api_key=GROQ_API_KEY)

# =========================================================
# AI CORE
# =========================================================
def get_itinerary_ai(origin, dest, days, members, theme):
    prompt = f"""
You are an elite travel AI.

Trip:
Origin: {origin}
Destination: {dest}
Days: {days}
People: {members}
Theme: {theme}

Return ONLY valid JSON:

{{
 "totalbudget": "number",
 "travelmode": "string",
 "weather": "2 line summary",
 "itinerary": {{"Day 1":"...", "Day 2":"..."}},
 "places":[{{"name":"", "info":"5 lines", "time":""}}],
 "restaurants":[{{"name":"","specialty":"","link":""}}],
 "hotels":[{{"name":"","tier":"","price":"","link":""}}],
 "mapcoords":["place1","place2","place3"]
}}
"""
    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(chat.choices[0].message.content)

# =========================================================
# GEO + ML
# =========================================================
def geocode_places(places):
    geo = Nominatim(user_agent="primecore2025")
    data = []
    for p in places:
        try:
            time.sleep(1)
            loc = geo.geocode(p)
            if loc:
                data.append({"name": p, "lat": loc.latitude, "lon": loc.longitude})
        except:
            pass
    return pd.DataFrame(data)

def cluster_route(df, days):
    if df is None or df.empty:
        return df
    k = min(days, len(df))
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    df["cluster"] = km.fit_predict(df[["lat","lon"]])
    return df.sort_values("cluster")

# =========================================================
# INPUT UI
# =========================================================
c1,c2,c3,c4 = st.columns(4)
with c1:
    origin = st.text_input("Origin", "Mumbai, India")
with c2:
    dest = st.text_input("Destination", "Zurich, Switzerland")
with c3:
    days = st.number_input("Days",1,30,4)
with c4:
    members = st.number_input("People",1,20,2)

theme = st.selectbox(
    "Trip Style",
    ["Luxury", "Adventure", "Cultural", "Budget", "Romantic"]
)

# =========================================================
# EXECUTE
# =========================================================
if st.button("🚀 BUILD MY TRIP"):
    if not GROQ_API_KEY:
        st.error("Groq API Key missing.")
    else:
        with st.spinner("Architecting your journey..."):
            plan = get_itinerary_ai(origin,dest,days,members,theme)
            st.session_state.trip_plan = plan

            df = geocode_places([origin]+plan["mapcoords"]+[dest])
            df = cluster_route(df, days)
            st.session_state.trip_df = df

            if len(df) >= 2:
                st.session_state.distance = int(
                    geodesic(
                        (df.lat.iloc[0], df.lon.iloc[0]),
                        (df.lat.iloc[-1], df.lon.iloc[-1])
                    ).km
                )

# =========================================================
# OUTPUT
# =========================================================
if st.session_state.trip_plan:
    p = st.session_state.trip_plan
    df = st.session_state.trip_df

    st.markdown("## 📊 Trip Analytics")
    a,b,c,d = st.columns(4)
    a.metric("Distance (km)", st.session_state.distance)
    b.metric("Budget ($)", p["totalbudget"])
    c.metric("Mode", p["travelmode"])
    d.metric("Per Day ($)", round(float(p["totalbudget"])/days,2))

    st.info(f"🌦️ Weather Insight: {p.get('weather','N/A')}")

    tabs = st.tabs(["📅 Plan","📍 Places","🍽️ Stay & Food","🧠 Smart Map"])

    with tabs[0]:
        for d,txt in p["itinerary"].items():
            st.markdown(f"<div class='card'><b>{d}</b><br>{txt}</div>", unsafe_allow_html=True)

    with tabs[1]:
        for pl in p["places"]:
            st.markdown(f"<div class='card'><h4>{pl['name']}</h4>{pl['info']}<br><b>Best:</b> {pl['time']}</div>", unsafe_allow_html=True)

    with tabs[2]:
        for r in p["restaurants"]:
            st.markdown(f"<div class='card'><b>{r['name']}</b><br>{r['specialty']}<br><a href='{r['link']}' target='_blank'>Visit</a></div>", unsafe_allow_html=True)
        for h in p["hotels"]:
            st.markdown(f"<div class='card'><b>{h['name']}</b><br>{h['tier']} • {h['price']}<br><a href='{h['link']}' target='_blank'>Book</a></div>", unsafe_allow_html=True)

    with tabs[3]:
        if df is not None and not df.empty:
            m = folium.Map(location=[df.lat.mean(), df.lon.mean()], zoom_start=4)
            folium.PolyLine(list(zip(df.lat, df.lon))).add_to(m)
            for _,r in df.iterrows():
                folium.Marker([r.lat,r.lon], popup=r.name).add_to(m)
            st_folium(m, height=500, width=1100)

    st.download_button(
        "📥 Download Trip JSON",
        json.dumps(p, indent=2),
        file_name="primecore_trip.json"
    )

# =========================================================
# RESET
# =========================================================
st.sidebar.button("🔄 Reset", on_click=lambda: st.session_state.clear())
