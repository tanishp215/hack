"""Main Streamlit entrypoint for the PlasticFlow app."""

from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

APP_TITLE = "PlasticFlow"
PAGE_OPTIONS = (
    "Global Observations Map",
    "Ocean Currents",
    "Particle Drift",
    "Statistical Insights",
)


def render_navigation() -> str:
    """Render the main app navigation and return the selected page label."""

    return st.sidebar.radio("Navigate", PAGE_OPTIONS)


def render_selected_page(selected_page: str) -> None:
    """Render the page selected in the app sidebar."""

    if selected_page == "Ocean Currents":
        from app import page_currents

        page_currents.render()
        return

    if selected_page == "Particle Drift":
        from app import page_drift

        page_drift.render()
        return

    if selected_page == "Statistical Insights":
        from app import page_statistics

        page_statistics.render()
        return

    from app import page_observations

    page_observations.render()


def render_footer() -> None:
    """Render shared sidebar footer content."""

    st.sidebar.markdown("---")
    st.sidebar.caption("Data: NOAA NCEI Marine Microplastics & NASA OSCAR Surface Currents")


def main() -> None:
    """Render the PlasticFlow Streamlit app."""

    try:
        st.set_page_config(
            page_title=APP_TITLE,
            page_icon="🌊",
            layout="wide",
        )
    except st.errors.StreamlitAPIException:
        pass

    selected_page = render_navigation()
    render_selected_page(selected_page)
    render_footer()


if __name__ == "__main__":
    main()
