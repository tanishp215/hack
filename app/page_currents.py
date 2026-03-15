"""Ocean Currents page — streamline visualization of OSCAR surface currents."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
_STREAMLINES_PATH = REPO_ROOT / "results" / "streamlines.npz"

# Vibrant speed colorscale: cool blue ocean → hot yellow/red for fast currents
_N_SPEED_BINS = 8
_SPEED_PALETTE = [
    "#2166ac",  # deep blue (slowest)
    "#4393c3",  # medium blue
    "#67a9cf",  # sky blue
    "#41ae76",  # teal-green
    "#f7e53b",  # bright yellow
    "#fdae61",  # orange
    "#f46d43",  # red-orange
    "#d73027",  # red (fastest)
]


@st.cache_data
def _load_streamlines() -> dict:
    data = np.load(_STREAMLINES_PATH)
    return {k: data[k] for k in data.files}


def _split_streamlines_by_speed(
    lats: np.ndarray,
    lons: np.ndarray,
    speeds: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, str, float]]:
    """Split the NaN-separated streamline arrays into speed-binned groups.

    Each streamline segment between NaNs gets assigned to a speed bin
    based on the per-streamline mean speed stored in `speeds`.
    """
    # Find NaN positions (streamline separators)
    nan_mask = np.isnan(lats)
    nan_idx = np.where(nan_mask)[0]

    # Assign each streamline to a speed bin
    bin_edges = np.linspace(speeds.min(), speeds.max() + 1e-6, _N_SPEED_BINS + 1)
    bin_assignments = np.digitize(speeds, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, _N_SPEED_BINS - 1)

    # Group streamline segments by bin
    bins: dict[int, tuple[list, list]] = {i: ([], []) for i in range(_N_SPEED_BINS)}

    start = 0
    stream_idx = 0
    for ni in nan_idx:
        seg_lats = lats[start:ni]
        seg_lons = lons[start:ni]
        if len(seg_lats) >= 2:
            b = bin_assignments[stream_idx]
            bins[b][0].extend(seg_lats.tolist() + [None])
            bins[b][1].extend(seg_lons.tolist() + [None])
        stream_idx += 1
        start = ni + 1

    # Post-process: break segments that cross the antimeridian (lon jump > 90°)
    for i in range(_N_SPEED_BINS):
        old_lats, old_lons = bins[i]
        if not old_lats:
            continue
        new_lats: list = []
        new_lons: list = []
        for j in range(len(old_lats)):
            if j > 0 and old_lons[j] is not None and old_lons[j - 1] is not None:
                if abs(old_lons[j] - old_lons[j - 1]) > 90:
                    new_lats.append(None)
                    new_lons.append(None)
            new_lats.append(old_lats[j])
            new_lons.append(old_lons[j])
        bins[i] = (new_lats, new_lons)

    result = []
    for i in range(_N_SPEED_BINS):
        if bins[i][0]:
            width = 1.0 + 2.5 * (i / max(_N_SPEED_BINS - 1, 1))
            result.append((
                np.array(bins[i][0], dtype=object),
                np.array(bins[i][1], dtype=object),
                _SPEED_PALETTE[i],
                width,
            ))
    return result


def _build_figure(
    data: dict,
    min_speed: float = 0.0,
    line_scale: float = 1.0,
) -> go.Figure:
    fig = go.Figure()

    lats = data["stream_lats"]
    lons = data["stream_lons"]
    speeds = data["stream_speeds"]
    grid_lat = data["grid_lat"]
    grid_lon = data["grid_lon"]
    grid_speed = data["grid_speed"]  # (lat, lon) shape

    # Filter streamlines by minimum speed
    if min_speed > 0:
        keep = speeds >= min_speed
        # Rebuild lats/lons keeping only segments for kept streamlines
        nan_mask = np.isnan(lats)
        nan_idx = np.where(nan_mask)[0]
        new_lats: list[float] = []
        new_lons: list[float] = []
        new_speeds: list[float] = []
        start = 0
        stream_idx = 0
        for ni in nan_idx:
            if stream_idx < len(keep) and keep[stream_idx]:
                new_lats.extend(lats[start:ni].tolist())
                new_lats.append(np.nan)
                new_lons.extend(lons[start:ni].tolist())
                new_lons.append(np.nan)
                new_speeds.append(speeds[stream_idx])
            stream_idx += 1
            start = ni + 1
        lats = np.array(new_lats)
        lons = np.array(new_lons)
        speeds = np.array(new_speeds)

    # Layer 1: Ocean speed heatmap — gives the ocean a blue tint distinct from land
    lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)
    flat_speed = grid_speed.ravel()
    ocean_mask = flat_speed > 0.001
    if ocean_mask.any():
        fig.add_trace(go.Densitymapbox(
            lat=lat_grid.ravel()[ocean_mask].tolist(),
            lon=lon_grid.ravel()[ocean_mask].tolist(),
            z=flat_speed[ocean_mask].tolist(),
            radius=18,
            colorscale=[
                [0.0, "rgba(8, 30, 60, 0.4)"],
                [0.3, "rgba(10, 50, 100, 0.5)"],
                [0.6, "rgba(20, 80, 140, 0.45)"],
                [0.8, "rgba(40, 120, 180, 0.35)"],
                [1.0, "rgba(60, 160, 200, 0.3)"],
            ],
            zmin=0,
            zmax=float(np.percentile(flat_speed[ocean_mask], 95)),
            opacity=0.7,
            showscale=False,
            hoverinfo="skip",
        ))

    if len(speeds) == 0:
        return fig

    # Layer 2: Streamlines — the main vector field
    binned = _split_streamlines_by_speed(lats, lons, speeds)
    for seg_lats, seg_lons, color, width in binned:
        width *= line_scale
        fig.add_trace(go.Scattermapbox(
            lat=seg_lats.tolist(),
            lon=seg_lons.tolist(),
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Colorbar legend (invisible scatter with color mapping)
    fig.add_trace(go.Scattermapbox(
        lat=[None],
        lon=[None],
        mode="markers",
        marker=dict(
            size=0,
            color=[0],
            colorscale=[
                [0.0, _SPEED_PALETTE[0]],
                [0.5, _SPEED_PALETTE[4]],
                [1.0, _SPEED_PALETTE[-1]],
            ],
            cmin=float(speeds.min()),
            cmax=float(speeds.max()),
            showscale=True,
            colorbar=dict(
                title=dict(text="Current Speed (m/s)", font=dict(color="white")),
                tickfont=dict(color="white"),
                x=0.99,
                len=0.6,
            ),
        ),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.update_layout(
        mapbox=dict(
            style="white-bg",
            layers=[{
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "CARTO",
                "source": [
                    "https://basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}@2x.png"
                ],
            }],
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
        "Streamlines show the mean ocean surface current flow from NASA's "
        "OSCAR dataset (1/3° grid, 315 pentad composites averaged). "
        "Color and thickness indicate current speed — "
        "note the major gyres, equatorial jets, and western boundary currents."
    )

    data = _load_streamlines()
    speeds = data["stream_speeds"]

    # Sidebar controls
    st.sidebar.markdown("### Current Display")
    min_speed = st.sidebar.slider(
        "Min Speed Filter (m/s)",
        min_value=0.0,
        max_value=float(round(speeds.max(), 2)),
        value=0.0,
        step=0.01,
        help="Hide streamlines slower than this threshold",
    )
    line_scale = st.sidebar.slider(
        "Line Thickness",
        min_value=0.3,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Visual thickness multiplier for streamlines",
    )

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Speed", f"{speeds.mean():.3f} m/s")
    col2.metric("Max Speed", f"{speeds.max():.3f} m/s")
    col3.metric("Streamlines", f"{len(speeds):,}")

    # Map
    fig = _build_figure(data, min_speed=min_speed, line_scale=line_scale)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
