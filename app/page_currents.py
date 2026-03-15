"""Ocean Currents page — vector field visualization of OSCAR surface currents."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
_VECTORS_PATH = REPO_ROOT / "results" / "current_vectors.npz"

# Sequential colorscale for speed bins (dark blue → cyan → yellow → orange → red)
_SPEED_COLORS = [
    "#08306b", "#08519c", "#2171b5", "#4292c6",
    "#6baed6", "#41ab5d", "#addd8e", "#fee08b",
    "#fdae61", "#f46d43", "#d73027", "#a50026",
]


@st.cache_data
def _load_vectors() -> dict:
    data = np.load(_VECTORS_PATH)
    return {k: data[k] for k in data.files}


def _build_arrow_traces(
    lat: np.ndarray,
    lon: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    speed: np.ndarray,
    scale: float,
    min_speed: float,
) -> list[go.Scattermapbox]:
    """Build colored line-segment traces representing current arrows."""
    # Create meshgrid: u/v shape is (n_lon, n_lat)
    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing="ij")

    # Flatten
    flat_lat = lat_grid.ravel()
    flat_lon = lon_grid.ravel()
    flat_u = u.ravel()
    flat_v = v.ravel()
    flat_speed = speed.ravel()

    # Filter: ocean only + above minimum speed
    mask = flat_speed > min_speed
    flat_lat = flat_lat[mask]
    flat_lon = flat_lon[mask]
    flat_u = flat_u[mask]
    flat_v = flat_v[mask]
    flat_speed = flat_speed[mask]

    if len(flat_speed) == 0:
        return []

    # Compute arrow endpoints
    # Scale factor: convert m/s to visual degrees (tuned for aesthetics)
    cos_lat = np.cos(np.radians(flat_lat))
    cos_lat = np.clip(cos_lat, 0.1, None)
    end_lat = flat_lat + flat_v * scale
    end_lon = flat_lon + flat_u * scale / cos_lat

    # Split into speed bins for coloring
    n_bins = len(_SPEED_COLORS)
    speed_min = flat_speed.min()
    speed_max = flat_speed.max()
    bin_edges = np.linspace(speed_min, speed_max, n_bins + 1)

    traces = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            bin_mask = (flat_speed >= lo) & (flat_speed <= hi)
        else:
            bin_mask = (flat_speed >= lo) & (flat_speed < hi)

        if not bin_mask.any():
            continue

        # Build disconnected line segments with None separators
        n_pts = bin_mask.sum()
        seg_lats = np.empty(n_pts * 3)
        seg_lons = np.empty(n_pts * 3)

        idx = np.where(bin_mask)[0]
        seg_lats[0::3] = flat_lat[idx]
        seg_lats[1::3] = end_lat[idx]
        seg_lats[2::3] = np.nan
        seg_lons[0::3] = flat_lon[idx]
        seg_lons[1::3] = end_lon[idx]
        seg_lons[2::3] = np.nan

        traces.append(go.Scattermapbox(
            lat=seg_lats.tolist(),
            lon=seg_lons.tolist(),
            mode="lines",
            line=dict(color=_SPEED_COLORS[i], width=1.2),
            hoverinfo="skip",
            showlegend=False,
        ))

    return traces


def _build_figure(
    vectors: dict,
    scale: float,
    min_speed: float,
) -> go.Figure:
    fig = go.Figure()

    lat = vectors["lat"]
    lon = vectors["lon"]
    u = vectors["u"]
    v = vectors["v"]
    speed = vectors["speed"]

    # Speed heatmap underglow
    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing="ij")
    flat_speed = speed.ravel()
    ocean_mask = flat_speed > min_speed
    if ocean_mask.any():
        fig.add_trace(go.Densitymapbox(
            lat=lat_grid.ravel()[ocean_mask].tolist(),
            lon=lon_grid.ravel()[ocean_mask].tolist(),
            z=flat_speed[ocean_mask].tolist(),
            radius=12,
            colorscale=[
                [0.0, "rgba(0,0,0,0)"],
                [0.2, "rgba(8,48,107,0.3)"],
                [0.4, "rgba(33,113,181,0.4)"],
                [0.6, "rgba(65,171,93,0.4)"],
                [0.8, "rgba(253,174,97,0.5)"],
                [1.0, "rgba(215,48,39,0.6)"],
            ],
            zmin=0,
            zmax=float(np.percentile(flat_speed[ocean_mask], 95)),
            opacity=0.6,
            showscale=True,
            colorbar=dict(
                title=dict(text="Speed (m/s)", font=dict(color="white")),
                tickfont=dict(color="white"),
                x=0.99,
            ),
            hoverinfo="skip",
        ))

    # Arrow traces
    arrow_traces = _build_arrow_traces(lat, lon, u, v, speed, scale, min_speed)
    for t in arrow_traces:
        fig.add_trace(t)

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=10, lon=0),
            zoom=1.3,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=700,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render() -> None:
    """Render the Ocean Currents page."""
    st.markdown(
        "Mean ocean surface current vectors from NASA's OSCAR dataset. "
        "Arrow direction shows current flow; color intensity shows speed."
    )

    vectors = _load_vectors()
    speed = vectors["speed"]
    ocean_speed = speed[speed > 0]

    # Sidebar controls
    st.sidebar.markdown("### Current Controls")
    min_speed = st.sidebar.slider(
        "Min Speed (m/s)",
        min_value=0.0,
        max_value=0.3,
        value=0.02,
        step=0.01,
        help="Hide weak currents below this threshold",
    )
    scale = st.sidebar.slider(
        "Arrow Scale",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="Visual length of arrows",
    )

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Speed", f"{ocean_speed.mean():.3f} m/s")
    col2.metric("Max Speed", f"{ocean_speed.max():.3f} m/s")
    visible = np.count_nonzero(speed > min_speed)
    col3.metric("Visible Arrows", f"{visible:,}")

    # Map
    fig = _build_figure(vectors, scale, min_speed)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
