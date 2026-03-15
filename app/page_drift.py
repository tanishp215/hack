"""Particle Drift page — animated cumulative density heatmap of microplastic drift."""

from __future__ import annotations

import json
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAJECTORIES_PATH = REPO_ROOT / "results" / "all_trajectories.parquet"
_HOTSPOT_PATH = REPO_ROOT / "results" / "hotspot_boundaries.json"

# Heatmap colorscale: transparent → deep blue → teal → amber → red
_DRIFT_COLORSCALE = [
    [0.00, "rgba(0, 0, 0, 0)"],
    [0.10, "rgba(8, 48, 107, 0.4)"],
    [0.25, "rgba(33, 113, 181, 0.5)"],
    [0.40, "rgba(0, 200, 150, 0.55)"],
    [0.55, "rgba(160, 220, 50, 0.6)"],
    [0.70, "rgba(255, 200, 0, 0.7)"],
    [0.85, "rgba(255, 100, 0, 0.8)"],
    [1.00, "rgba(220, 30, 30, 0.9)"],
]

_HOTSPOT_LINE_COLOR = "rgba(255, 255, 255, 0.35)"
_HOTSPOT_FILL_COLOR = "rgba(255, 255, 255, 0.06)"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def _load_trajectories() -> pd.DataFrame:
    df = pd.read_parquet(_TRAJECTORIES_PATH)
    mask = df["lon"] > 180
    df.loc[mask, "lon"] = df.loc[mask, "lon"] - 360.0
    return df


@st.cache_data
def _load_hotspot_polygons() -> list[dict]:
    with open(_HOTSPOT_PATH) as f:
        payload = json.load(f)
    return payload.get("polygons", payload if isinstance(payload, list) else [])


@st.cache_data
def _precompute_density_frames(
    _traj_hash: str,
    traj_lats: np.ndarray,
    traj_lons: np.ndarray,
    traj_days: np.ndarray,
    frame_days: list[int],
) -> tuple[dict, float]:
    """Precompute cumulative 1° density grid for each animation frame.

    Returns (frames_dict, zmax) where frames_dict maps day → (lats, lons, counts).
    """
    lat_bins = np.arange(-90, 91, 1.0)
    lon_bins = np.arange(-180, 181, 1.0)
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2

    cumulative = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1), dtype=np.float32)
    frames: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for day in frame_days:
        day_mask = traj_days == day
        if day_mask.any():
            h, _, _ = np.histogram2d(
                traj_lats[day_mask], traj_lons[day_mask],
                bins=[lat_bins, lon_bins],
            )
            cumulative += h.astype(np.float32)

        nz = cumulative > 0
        if nz.any():
            lat_grid, lon_grid = np.meshgrid(lat_centers, lon_centers, indexing="ij")
            frames[day] = (
                lat_grid[nz].astype(np.float32),
                lon_grid[nz].astype(np.float32),
                cumulative[nz].copy(),
            )
        else:
            frames[day] = (np.array([]), np.array([]), np.array([]))

    # Consistent zmax across all frames (95th percentile of final)
    final_counts = frames[frame_days[-1]][2]
    zmax = float(np.percentile(final_counts, 95)) if len(final_counts) > 0 else 1.0

    return frames, zmax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _day_label(day: int) -> str:
    if day == 0:
        return "Day 0"
    years = day // 365
    months = (day % 365) // 30
    if years >= 1 and months > 0:
        return f"Year {years}, Month {months}"
    if years >= 1:
        return f"Year {years}"
    return f"Month {max(1, day // 30)}"


def _build_frame(
    day: int,
    density_frames: dict,
    zmax: float,
    current_df: pd.DataFrame,
    polygons: list[dict],
    show_hotspots: bool,
    heatmap_radius: int,
) -> go.Figure:
    """Build a single animation frame."""
    fig = go.Figure()

    # Layer 1: Cumulative density heatmap
    if day in density_frames:
        lats, lons, counts = density_frames[day]
        if len(lats) > 0:
            fig.add_trace(go.Densitymapbox(
                lat=lats.tolist(),
                lon=lons.tolist(),
                z=counts.tolist(),
                radius=heatmap_radius,
                colorscale=_DRIFT_COLORSCALE,
                zmin=0,
                zmax=zmax,
                opacity=0.85,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Accumulation", font=dict(color="white", size=12)),
                    tickfont=dict(color="white"),
                    x=0.99,
                    len=0.5,
                ),
                hoverinfo="skip",
            ))

    # Layer 2: Hotspot overlay
    if show_hotspots and polygons:
        for poly in polygons:
            coords = poly.get("polygon", [])
            if len(coords) < 3:
                continue
            plons = [p["longitude"] for p in coords] + [coords[0]["longitude"]]
            plats = [p["latitude"] for p in coords] + [coords[0]["latitude"]]
            fig.add_trace(go.Scattermapbox(
                lon=plons, lat=plats,
                mode="lines",
                fill="toself",
                fillcolor=_HOTSPOT_FILL_COLOR,
                line=dict(color=_HOTSPOT_LINE_COLOR, width=1),
                hoverinfo="skip",
                showlegend=False,
            ))

    # Layer 3: Current particle positions
    if not current_df.empty:
        fig.add_trace(go.Scattermapbox(
            lat=current_df["lat"],
            lon=current_df["lon"],
            mode="markers",
            marker=dict(size=4, color="white", opacity=0.7),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=15, lon=0),
            zoom=1.3,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=650,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Particle Drift animation page."""
    st.markdown(
        "Watch microplastics drift across the world's oceans over 5 years. "
        "The heatmap builds as particles accumulate — compare with real "
        "observation hotspots to see where gyres trap debris."
    )

    # Load data
    df = _load_trajectories()
    polygons = _load_hotspot_polygons()
    all_cities = sorted(df["city"].unique())
    all_days = sorted(df["day"].unique())

    # -- Sidebar controls --
    st.sidebar.markdown("### Drift Controls")

    selected_cities = st.sidebar.multiselect(
        "Cities",
        all_cities,
        default=all_cities,
    )

    speed = st.sidebar.slider(
        "Animation Speed",
        min_value=0.05,
        max_value=0.5,
        value=0.15,
        step=0.05,
        help="Seconds per frame",
    )

    frame_skip = st.sidebar.selectbox(
        "Frame Skip",
        [1, 2, 4, 8],
        index=2,
        help="Use every Nth time step (higher = faster, fewer frames)",
    )

    heatmap_radius = st.sidebar.slider(
        "Heatmap Radius",
        min_value=5,
        max_value=25,
        value=12,
        step=1,
    )

    show_hotspots = st.sidebar.checkbox("Show Hotspot Overlay", value=False)

    # -- Filter data by selected cities --
    if selected_cities:
        filtered = df[df["city"].isin(selected_cities)]
    else:
        filtered = df

    # -- Determine animation frames --
    frame_days = all_days[::frame_skip]
    n_frames = len(frame_days)

    # -- Precompute density --
    city_key = ",".join(sorted(selected_cities)) if selected_cities else "all"
    density_frames, zmax = _precompute_density_frames(
        city_key,
        filtered["lat"].values,
        filtered["lon"].values,
        filtered["day"].values,
        frame_days,
    )

    # -- Session state for animation --
    if "drift_playing" not in st.session_state:
        st.session_state.drift_playing = False
        st.session_state.drift_frame = 0

    # -- Play / Pause / Reset buttons --
    bcol1, bcol2, bcol3 = st.sidebar.columns(3)
    if bcol1.button("Play", use_container_width=True):
        st.session_state.drift_playing = True
    if bcol2.button("Pause", use_container_width=True):
        st.session_state.drift_playing = False
    if bcol3.button("Reset", use_container_width=True):
        st.session_state.drift_playing = False
        st.session_state.drift_frame = 0

    # -- Placeholders for dynamic content --
    metrics_ph = st.empty()
    map_ph = st.empty()
    caption_ph = st.empty()

    def _show_frame(idx: int) -> None:
        day = frame_days[min(idx, n_frames - 1)]
        current_df = filtered[filtered["day"] == day]

        with metrics_ph.container():
            c1, c2, c3 = st.columns(3)
            c1.metric("Time", _day_label(day))
            c2.metric("Day", f"{day} / {all_days[-1]}")
            c3.metric("Particles", len(current_df))

        fig = _build_frame(
            day, density_frames, zmax,
            current_df, polygons, show_hotspots, heatmap_radius,
        )
        map_ph.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
        caption_ph.caption(
            f"Frame {idx + 1}/{n_frames} — **{_day_label(day)}** "
            f"({len(selected_cities)} cities, {len(current_df)} particles)"
        )

    if st.session_state.drift_playing:
        start = st.session_state.drift_frame
        for i in range(start, n_frames):
            st.session_state.drift_frame = i
            _show_frame(i)
            _time.sleep(speed)
        # Animation finished
        st.session_state.drift_playing = False
        st.session_state.drift_frame = n_frames - 1
    else:
        _show_frame(st.session_state.drift_frame)
