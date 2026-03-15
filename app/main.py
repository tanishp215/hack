import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st

# set_page_config is handled by each page module so it can set its own
# title and icon. main.py only sets it as a fallback when run directly
# without navigating to a sub-page first.
try:
    st.set_page_config(
        page_title="PlasticFlow",
        page_icon="🌊",
        layout="wide",
    )
except st.errors.StreamlitAPIException:
    pass  # already set by the page module

page = st.sidebar.radio(
    "Navigate",
    ["🗺️ Overview (Map)", "📊 Statistical Insights"],
)

if page == "📊 Statistical Insights":
    import app.page_statistics as page_statistics
    page_statistics.render()
elif page == "🗺️ Overview (Map)":
    st.title("Overview Map")
    st.info("Coming soon — Lagrangian particle drift simulation.")

st.sidebar.markdown("---")
st.sidebar.caption("Data source: NOAA NCEI Marine Microplastics Database")
