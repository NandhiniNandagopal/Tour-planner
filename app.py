import os
import time
import json
import random

import streamlit as st
import pandas as pd
import folium
from groq import Groq
from geopy.geocoders import Nominatim
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

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(0,0,0,.75), rgba(0,0,0,.75)),
    url(https://images.unsplash.com/photo-1507525428034-b723cf961d3e);
    background-size: cover;
}
.card {
    background: rgba(255,255,255,.07);
    padding: 16px;
    border-radius: 16px;
    margin-bottom: 12px;
}
.tag {
    display:inline-block;
    padding:4px 10px;
    border-radius:12px;
    font-size:12px;
    background:#00d4ff22;
    margin-right:6px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("<h1 style='color:#00d4ff'>PRIMECORE</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#aaa'>JOURNEY ARCHITECT</p>", unsafe_allow_html=True)

# =========================================================
# GROQ
# =========================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
client = Groq(api_key=GROQ_API_KEY)

# =========================================================
# AI FUNCTION (UPDATED)
# =========================================================
def get_itinerary_ai(origin, dest, days, members, theme):
    prompt = f"""
Return ONLY valid JSON.

Trip:
Origin: {origin}
Destination: {dest}
Days: {days}
People: {members}
Theme: {theme}

{{
 "totalbudget": "number",
 "travelmode": "string",
 "weather": "2 lines",
 "tips": {{
   "packing": ["item1","item2"],
   "safety": ["tip1","tip2"],
   "customs": ["tip1","tip2"]
 }},
 "itinerary": {{"Day 1":"...", "Day 2":"..." }},
 "places":[{{"name":"","info":"","time":""}}],
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
    geo = Nominatim(user_agent="primecore")
    rows = []
    for p in places:
        try:
            time.sleep(1)
            loc = geo.geocode(p)
            if loc:
                rows.append({"name": p, "lat": loc.latitude, "lon": loc.longitude})
        except:
            pass
    return pd.DataFrame(rows)

def cluster_days(df, days):
    k = min(days, len(df))
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    df["day"] = km.fit_predict(df[["lat", "lon"]]) + 1
    return df.sort_values("day")

# =========================================================
# HELPERS (NEW)
# =========================================================
def budget_split(total):
    return {
        "Stay": total * 0.4,
        "Food": total * 0.25,
        "Transport": total * 0.2,
        "Activities": total * 0.15,
    }

def crowd():
    return random.choice(["🟢 Calm", "🟡 Moderate", "🔴 Crowded"])

def score():
    return random.randint(70, 95)

# =========================================================
# INPUT
# =========================================================
c1, c2, c3, c4 = st.columns(4)
origin = c1.text_input("Origin", "Mumbai, India")
dest = c2.text_input("Destination", "Zurich, Switzerland")
days = c3.number_input("Days", 1, 10, 4)
members = c4.number_input("People", 1, 10, 2)
theme = st.selectbox("Trip Style", ["Luxury", "Adventure", "Cultural", "Budget", "Romantic"])

# =========================================================
# RUN
# =========================================================
if st.button("🚀 BUILD MY TRIP"):
    with st.spinner("Building your trip..."):
        plan = get_itinerary_ai(origin, dest, days, members, theme)
        df = geocode_places([origin] + plan["mapcoords"] + [dest])
        df = cluster_days(df, days)
        st.session_state.trip_plan = plan
        st.session_state.trip_df = df

# =========================================================
# OUTPUT
# =========================================================
if st.session_state.trip_plan:
    p = st.session_state.trip_plan
    df = st.session_state.trip_df

    st.markdown("## 📊 Trip Analytics")
    a, b, c = st.columns(3)
    a.metric("Budget ($)", p["totalbudget"])
    b.metric("Mode", p["travelmode"])
    c.metric("Per Day ($)", round(float(p["totalbudget"]) / days, 2))

    st.markdown("### 💸 Budget Split")
    st.bar_chart(pd.DataFrame.from_dict(budget_split(float(p["totalbudget"])), orient="index"))

    tabs = st.tabs(["📅 Plan", "📍 Places", "🗺️ Day-wise Map", "🧠 Travel Tips"])

    with tabs[0]:
        for d, t in p["itinerary"].items():
            st.markdown(f"<div class='card'><b>{d}</b><br>{t}</div>", unsafe_allow_html=True)

    with tabs[1]:
        for pl in p["places"]:
            st.markdown(
                f"<div class='card'><b>{pl['name']}</b><br>"
                f"<span class='tag'>{crowd()}</span>"
                f"<span class='tag'>Score {score()}</span><br>"
                f"{pl['info']}</div>",
                unsafe_allow_html=True
            )

    with tabs[2]:
        for d in sorted(df.day.unique()):
            st.markdown(f"### Day {d}")
            sub = df[df.day == d]
            m = folium.Map(location=[sub.lat.mean(), sub.lon.mean()], zoom_start=6)
            folium.PolyLine(list(zip(sub.lat, sub.lon))).add_to(m)
            for _, r in sub.iterrows():
                folium.Marker([r.lat, r.lon], popup=r.name).add_to(m)
            st_folium(m, height=350)

    with tabs[3]:
        for k, v in p["tips"].items():
            st.markdown(f"**{k.capitalize()}**")
            for i in v:
                st.write("•", i)

# =========================================================
# RESET
# =========================================================
st.sidebar.button("🔄 Reset", on_click=lambda: st.session_state.clear())
