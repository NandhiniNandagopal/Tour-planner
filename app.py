import os
import time
import json
import random

import streamlit as st
import pandas as pd
import folium
import numpy as np
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

SESSION_KEYS = [
    "trip_plan", "trip_df", "theme", "origin", "dest", "days", "members"
]
for k in SESSION_KEYS:
    st.session_state.setdefault(k, None)

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
.card {
    background: rgba(255,255,255,.07);
    backdrop-filter: blur(14px);
    padding: 18px;
    border-radius: 18px;
    margin-bottom: 14px;
}
.tag {
    display:inline-block;
    padding:4px 10px;
    border-radius:12px;
    font-size:12px;
    margin-right:6px;
    background:#00d4ff22;
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
def get_itinerary_ai(origin, dest, days, members, theme, variant="default"):
    variation_note = "" if variant == "default" else "Generate an alternative itinerary with different places."

    prompt = f"""
You are an elite travel AI.

{variation_note}

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
 "tips": {{
    "packing": ["item1","item2"],
    "safety": ["tip1","tip2"],
    "customs": ["tip1","tip2"]
 }},
 "itinerary": {{"Day 1":"...", "Day 2":"..." }},
 "places":[{{"name":"", "info":"5 lines", "time":""}}],
 "restaurants":[{{"name":"","specialty":"","link":""}}],
 "hotels":[{{"name":"","tier":"","price":"","link":""}}],
 "mapcoords":["place1","place2","place3"]
}}
"""
    try:
        chat = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(chat.choices[0].message.content)
    except:
        return None

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
    df["day"] = km.fit_predict(df[["lat", "lon"]]) + 1
    return df.sort_values("day")

# =========================================================
# HELPERS
# =========================================================
def budget_split(total):
    return {
        "Stay": round(total * 0.4, 2),
        "Food": round(total * 0.25, 2),
        "Transport": round(total * 0.2, 2),
        "Activities": round(total * 0.15, 2),
    }

def crowd_level():
    return random.choice(["🟢 Calm", "🟡 Moderate", "🔴 Crowded"])

def place_score():
    return random.randint(70, 95)

# =========================================================
# INPUT UI
# =========================================================
c1, c2, c3, c4 = st.columns(4)
with c1:
    origin = st.text_input("Origin", st.session_state.origin or "Mumbai, India")
with c2:
    dest = st.text_input("Destination", st.session_state.dest or "Zurich, Switzerland")
with c3:
    days = st.number_input("Days", 1, 30, st.session_state.days or 4)
with c4:
    members = st.number_input("People", 1, 20, st.session_state.members or 2)

theme = st.selectbox(
    "Trip Style",
    ["Luxury", "Adventure", "Cultural", "Budget", "Romantic"],
    index=["Luxury", "Adventure", "Cultural", "Budget", "Romantic"].index(
        st.session_state.theme or "Luxury"
    ),
)

# =========================================================
# EXECUTION
# =========================================================
if st.button("🚀 BUILD MY TRIP"):
    if not GROQ_API_KEY:
        st.error("Groq API Key missing.")
    else:
        st.session_state.update({
            "origin": origin, "dest": dest,
            "days": days, "members": members, "theme": theme
        })

        with st.status("Planning your journey...", expanded=True) as status:
            status.update(label="Generating itinerary...")
            plan = get_itinerary_ai(origin, dest, days, members, theme)

            if not plan:
                st.error("AI service failed. Try again.")
                st.stop()

            status.update(label="Mapping locations...")
            df = geocode_places([origin] + plan["mapcoords"] + [dest])
            df = cluster_route(df, days)

            status.update(label="Optimizing experience...")
            st.session_state.trip_plan = plan
            st.session_state.trip_df = df

            status.update(label="Trip ready ✨", state="complete")

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

    st.markdown("### 💸 Smart Budget Split")
    split = budget_split(float(p["totalbudget"]))
    st.bar_chart(pd.DataFrame.from_dict(split, orient="index"))

    st.info(f"🌦️ Weather Insight: {p.get('weather','N/A')}")

    tabs = st.tabs(["📅 Plan", "📍 Places", "🍽️ Stay & Food", "🗺️ Day-wise Maps", "🧠 Travel Tips"])

    with tabs[0]:
        for d, txt in p["itinerary"].items():
            st.markdown(f"<div class='card'><b>{d}</b><br>{txt}</div>", unsafe_allow_html=True)

    with tabs[1]:
        for pl in p["places"]:
            st.markdown(
                f"<div class='card'><h4>{pl['name']}</h4>"
                f"<span class='tag'>{crowd_level()}</span>"
                f"<span class='tag'>Score: {place_score()}</span><br>"
                f"{pl['info']}<br><b>Best:</b> {pl['time']}</div>",
                unsafe_allow_html=True
            )

    with tabs[2]:
        for r in p["restaurants"]:
            st.markdown(f"<div class='card'><b>{r['name']}</b><br>{r['specialty']}<br>"
                        f"<a href='{r['link']}' target='_blank'>Visit</a></div>",
                        unsafe_allow_html=True)
        for h in p["hotels"]:
            st.markdown(f"<div class='card'><b>{h['name']}</b><br>{h['tier']} • {h['price']}<br>"
                        f"<a href='{h['link']}' target='_blank'>Book</a></div>",
                        unsafe_allow_html=True)

    with tabs[3]:
        for day in sorted(df.day.unique()):
            ddf = df[df.day == day]
            st.markdown(f"### Day {day}")
            m = folium.Map(location=[ddf.lat.mean(), ddf.lon.mean()], zoom_start=6)
            folium.PolyLine(list(zip(ddf.lat, ddf.lon))).add_to(m)
            for _, r in ddf.iterrows():
                folium.Marker([r.lat, r.lon], popup=r.name).add_to(m)
            st_folium(m, height=350, width=1000)

    with tabs[4]:
        for k, v in p["tips"].items():
            st.markdown(f"**{k.capitalize()}**")
            for i in v:
                st.write("•", i)

    if st.button("🔁 Show Alternate Plan"):
        alt = get_itinerary_ai(origin, dest, days, members, theme, variant="alt")
        if alt:
            st.session_state.trip_plan = alt

    st.download_button(
        "📥 Download Trip JSON",
        json.dumps(p, indent=2),
        file_name="primecore_trip.json"
    )

# =========================================================
# RESET
# =========================================================
st.sidebar.button("🔄 Reset", on_click=lambda: st.session_state.clear())
