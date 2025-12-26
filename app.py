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
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="PrimeCore | Global AI Trip Planner",
    layout="wide",
    page_icon="🚀",
)

# =========================================================
# SESSION STATE
# =========================================================
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
    background: rgba(255,255,255,.08);
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
st.markdown("<p style='color:#aaa'>Journey Architect</p>", unsafe_allow_html=True)

# =========================================================
# GROQ CONFIG
# =========================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
client = Groq(api_key=GROQ_API_KEY)

# =========================================================
# AI ITINERARY FUNCTION
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
# GEO FUNCTIONS
# =========================================================
@st.cache_data(show_spinner=False)
def geocode_places(places):
    geo = Nominatim(user_agent="primecore")
    rows = []

    for place in places:
        try:
            time.sleep(0.6)
            loc = geo.geocode(place)
            if loc:
                rows.append({
                    "name": place,
                    "lat": loc.latitude,
                    "lon": loc.longitude
                })
        except:
            continue

    return pd.DataFrame(rows)

def cluster_days(df, days):
    if df is None or df.empty or len(df) < 2:
        return df

    k = min(days, len(df))
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")

    df = df.copy()
    df["day"] = km.fit_predict(df[["lat", "lon"]]) + 1
    return df.sort_values("day")

# =========================================================
# HELPERS
# =========================================================
def budget_split(total):
    return {
        "Stay": total * 0.40,
        "Food": total * 0.25,
        "Transport": total * 0.20,
        "Activities": total * 0.15,
    }

def crowd_level():
    return random.choice(["🟢 Calm", "🟡 Moderate", "🔴 Crowded"])

def place_score():
    return random.randint(72, 96)

# =========================================================
# INPUT SECTION
# =========================================================
c1, c2, c3, c4 = st.columns(4)

origin = c1.text_input("Origin", "Mumbai, India")
dest = c2.text_input("Destination", "Zurich, Switzerland")
days = c3.number_input("Days", 1, 14, 4)
members = c4.number_input("People", 1, 10, 2)

theme = st.selectbox(
    "Trip Style",
    ["Luxury", "Adventure", "Cultural", "Budget", "Romantic"]
)

# =========================================================
# BUILD TRIP
# =========================================================
if st.button("🚀 BUILD MY TRIP"):
    with st.spinner("Designing your journey..."):
        plan = get_itinerary_ai(origin, dest, days, members, theme)

        locations = [origin] + plan.get("mapcoords", []) + [dest]
        df = geocode_places(locations)

        if df is not None and not df.empty and len(df) >= 2:
            df = cluster_days(df, days)

        st.session_state.trip_plan = plan
        st.session_state.trip_df = df

# =========================================================
# OUTPUT
# =========================================================
if st.session_state.trip_plan:
    plan = st.session_state.trip_plan
    df = st.session_state.trip_df

    st.markdown("## 📊 Trip Overview")
    a, b, c = st.columns(3)

    a.metric("Total Budget ($)", plan["totalbudget"])
    b.metric("Travel Mode", plan["travelmode"])
    c.metric("Per Day ($)", round(float(plan["totalbudget"]) / days, 2))

    st.markdown("### 💸 Budget Breakdown")
    st.bar_chart(
        pd.DataFrame.from_dict(
            budget_split(float(plan["totalbudget"])),
            orient="index",
            columns=["Amount"]
        )
    )

    tabs = st.tabs(["📅 Plan", "📍 Places", "🗺️ Day-wise Map", "🧠 Travel Tips"])

    # ---------------- PLAN TAB ----------------
    with tabs[0]:
        for day, text in plan["itinerary"].items():
            st.markdown(
                f"<div class='card'><b>{day}</b><br>{text}</div>",
                unsafe_allow_html=True
            )

    # ---------------- PLACES TAB ----------------
    with tabs[1]:
        for place in plan["places"]:
            st.markdown(
                f"""
                <div class='card'>
                    <b>{place['name']}</b><br>
                    <span class='tag'>{crowd_level()}</span>
                    <span class='tag'>Score {place_score()}</span><br>
                    {place['info']}
                </div>
                """,
                unsafe_allow_html=True
            )

    # ---------------- MAP TAB ----------------
    with tabs[2]:
        if df is not None and not df.empty and "day" in df.columns:
            for d in sorted(df["day"].unique()):
                st.markdown(f"### Day {d}")
                sub = df[df["day"] == d]

                m = folium.Map(
                    location=[sub.lat.mean(), sub.lon.mean()],
                    zoom_start=6
                )

                folium.PolyLine(
                    list(zip(sub.lat, sub.lon))
                ).add_to(m)

                for _, r in sub.iterrows():
                    folium.Marker(
                        [r.lat, r.lon],
                        popup=r.name
                    ).add_to(m)

                st_folium(m, height=350)
        else:
            st.info("🗺️ Not enough location data to generate maps.")

    # ---------------- TIPS TAB ----------------
    with tabs[3]:
        for section, items in plan["tips"].items():
            st.markdown(f"**{section.capitalize()}**")
            for i in items:
                st.write("•", i)

# =========================================================
# RESET
# =========================================================
st.sidebar.button(
    "🔄 Reset App",
    on_click=lambda: st.session_state.clear()
)
