"""Particle Drift page — animated microplastic drift with cumulative density heatmap.

Uses st.rerun() for flicker-free animation: the plotly chart stays in the same
DOM position across reruns so Plotly.react() does an efficient in-place update
instead of destroying and recreating the element.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAJECTORIES_PATH = REPO_ROOT / "results" / "all_trajectories.parquet"
_NOAA_CSV_PATH = REPO_ROOT / "data" / "NOAA.csv"
_STREAMLINES_PATH = REPO_ROOT / "results" / "streamlines.npz"

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

# Density heatmap: transparent → warm amber → hot red (more vivid)
_DRIFT_COLORSCALE = [
    [0.00, "rgba(0, 0, 0, 0)"],
    [0.05, "rgba(20, 20, 80, 0.3)"],
    [0.15, "rgba(255, 220, 50, 0.45)"],
    [0.35, "rgba(255, 140, 20, 0.65)"],
    [0.55, "rgba(255, 60, 10, 0.8)"],
    [0.80, "rgba(200, 20, 20, 0.9)"],
    [1.00, "rgba(140, 0, 40, 0.95)"],
]

# NOAA observation density overlay — blue/purple tones to contrast with drift heatmap
_NOAA_COLORSCALE = [
    [0.0, "rgba(0, 0, 0, 0)"],
    [0.2, "rgba(100, 140, 255, 0.25)"],
    [0.5, "rgba(130, 100, 255, 0.4)"],
    [0.8, "rgba(180, 80, 255, 0.55)"],
    [1.0, "rgba(220, 60, 255, 0.7)"],
]

# Streamline speed palette — bolder than before so currents are clearly visible
_N_SPEED_BINS = 6
_STREAM_PALETTE = [
    "rgba(30, 80, 160, 0.50)",   # deep blue (slowest)
    "rgba(50, 120, 190, 0.55)",  # medium blue
    "rgba(70, 160, 210, 0.58)",  # sky blue
    "rgba(50, 170, 120, 0.55)",  # teal-green
    "rgba(200, 190, 50, 0.55)",  # yellow
    "rgba(210, 100, 50, 0.60)",  # orange (fastest)
]

# Trail / particle colors
_TRAIL_COLOR = "rgba(0, 255, 200, 0.25)"
_PARTICLE_COLOR = "#00ffd0"


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
def _load_noaa_density() -> tuple[list, list, list]:
    """Load NOAA observation positions and bin into a 2° density grid.

    Returns (lats, lons, counts) for non-zero cells — ready for Densitymapbox.
    """
    raw = pd.read_csv(_NOAA_CSV_PATH, usecols=["Latitude (degree)", "Longitude (degree)"])
    obs_lat = raw["Latitude (degree)"].values
    obs_lon = raw["Longitude (degree)"].values

    lat_bins = np.arange(-90, 91, 2.0)
    lon_bins = np.arange(-180, 181, 2.0)
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2

    h, _, _ = np.histogram2d(obs_lat, obs_lon, bins=[lat_bins, lon_bins])
    nz = h > 0
    lat_grid, lon_grid = np.meshgrid(lat_centers, lon_centers, indexing="ij")
    return (
        lat_grid[nz].tolist(),
        lon_grid[nz].tolist(),
        h[nz].tolist(),
    )


@st.cache_data
def _load_streamlines() -> dict:
    data = np.load(_STREAMLINES_PATH)
    return {k: data[k] for k in data.files}


@st.cache_data
def _split_streamlines_by_speed(
    _cache_key: str,
    lats: np.ndarray,
    lons: np.ndarray,
    speeds: np.ndarray,
) -> list[tuple[list, list, str, float]]:
    """Split NaN-separated streamlines into speed-binned groups for coloring."""
    nan_mask = np.isnan(lats)
    nan_idx = np.where(nan_mask)[0]

    bin_edges = np.linspace(speeds.min(), speeds.max() + 1e-6, _N_SPEED_BINS + 1)
    bin_assignments = np.clip(np.digitize(speeds, bin_edges) - 1, 0, _N_SPEED_BINS - 1)

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

    # Post-process: break any segment that crosses the antimeridian (lon jump > 90°)
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
            width = 1.0 + 2.0 * (i / max(_N_SPEED_BINS - 1, 1))
            result.append((bins[i][0], bins[i][1], _STREAM_PALETTE[i], width))
    return result


@st.cache_data
def _precompute_density_frames(
    _traj_hash: str,
    traj_lats: np.ndarray,
    traj_lons: np.ndarray,
    traj_days: np.ndarray,
    frame_days: list[int],
) -> tuple[dict, float]:
    """Precompute cumulative 2-degree density grid for each time step."""
    lat_bins = np.arange(-90, 91, 2.0)
    lon_bins = np.arange(-180, 181, 2.0)
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

    final_counts = frames[frame_days[-1]][2]
    zmax = float(np.percentile(final_counts, 95)) if len(final_counts) > 0 else 1.0
    return frames, zmax


@st.cache_data
def _precompute_trails(
    _traj_hash: str,
    traj_lats: np.ndarray,
    traj_lons: np.ndarray,
    traj_days: np.ndarray,
    traj_keys: np.ndarray,
    frame_days: list[int],
) -> dict[int, tuple[list, list]]:
    """For each frame day, build NaN-separated trail lines for particles.

    Inserts breaks when longitude jumps >90° (antimeridian crossing) to
    prevent horizontal lines spanning the entire map.
    """
    unique_keys = np.unique(traj_keys)
    key_indices = {k: np.where(traj_keys == k)[0] for k in unique_keys}

    trails: dict[int, tuple[list, list]] = {}
    for day in frame_days:
        all_lats: list = []
        all_lons: list = []
        for k in unique_keys:
            idx = key_indices[k]
            mask = traj_days[idx] <= day
            if mask.sum() < 2:
                continue
            trail_idx = idx[mask]
            plats = traj_lats[trail_idx]
            plons = traj_lons[trail_idx]
            # Break trail at antimeridian crossings (lon jump > 90°)
            for i in range(len(plats)):
                if i > 0 and abs(plons[i] - plons[i - 1]) > 90:
                    all_lats.append(None)
                    all_lons.append(None)
                all_lats.append(float(plats[i]))
                all_lons.append(float(plons[i]))
            all_lats.append(None)
            all_lons.append(None)
        trails[day] = (all_lats, all_lons)
    return trails


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _day_label(day: int) -> str:
    if day == 0:
        return "Release Day"
    years = day // 365
    months = (day % 365) // 30
    if years >= 1 and months > 0:
        return f"Year {years}, Month {months}"
    if years >= 1:
        return f"Year {years}"
    return f"Month {max(1, day // 30)}"


def _build_figure(
    day: int,
    stream_bins: list[tuple[list, list, str, float]],
    density_frames: dict,
    zmax: float,
    trail_data: tuple[list, list] | None,
    current_df: pd.DataFrame,
    noaa_density: tuple[list, list, list] | None,
) -> go.Figure:
    """Build the map figure for a given animation frame."""
    fig = go.Figure()

    # Layer 1: Speed-colored ocean current streamlines (visible background)
    for seg_lats, seg_lons, color, width in stream_bins:
        fig.add_trace(go.Scattermapbox(
            lat=seg_lats,
            lon=seg_lons,
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Layer 2: Cumulative density heatmap (where particles have accumulated)
    if day in density_frames:
        lats, lons, counts = density_frames[day]
        if len(lats) > 0:
            fig.add_trace(go.Densitymapbox(
                lat=lats.tolist(),
                lon=lons.tolist(),
                z=counts.tolist(),
                radius=25,
                colorscale=_DRIFT_COLORSCALE,
                zmin=0,
                zmax=zmax,
                opacity=0.9,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Plastic Accumulation",
                        font=dict(color="white", size=12),
                    ),
                    tickfont=dict(color="white"),
                    x=0.99,
                    len=0.5,
                ),
                hoverinfo="skip",
            ))

    # Layer 3: NOAA observation density (smooth blobs replacing angular polygons)
    if noaa_density is not None:
        nlats, nlons, ncounts = noaa_density
        fig.add_trace(go.Densitymapbox(
            lat=nlats,
            lon=nlons,
            z=ncounts,
            radius=30,
            colorscale=_NOAA_COLORSCALE,
            zmin=0,
            zmax=float(np.percentile(ncounts, 90)),
            opacity=0.6,
            showscale=False,
            hoverinfo="skip",
        ))

    # Layer 4: Particle trails (paths traveled so far)
    if trail_data is not None:
        t_lats, t_lons = trail_data
        if len(t_lats) > 0:
            fig.add_trace(go.Scattermapbox(
                lat=t_lats,
                lon=t_lons,
                mode="lines",
                line=dict(color=_TRAIL_COLOR, width=1.5),
                hoverinfo="skip",
                showlegend=False,
            ))

    # Layer 5: Current particle positions (bright dots)
    if not current_df.empty:
        fig.add_trace(go.Scattermapbox(
            lat=current_df["lat"],
            lon=current_df["lon"],
            mode="markers",
            marker=dict(size=5, color=_PARTICLE_COLOR, opacity=0.9),
            text=current_df["city"],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
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
    """Render the Particle Drift page with flicker-free animated playback."""

    st.markdown(
        "We simulate releasing microplastic particles from each of **18 major "
        "coastal cities** and track where ocean currents carry them over **5 years**. "
        "Press **Play** to watch plastics spread and accumulate in ocean gyres."
    )

    # -- Load data --
    df = _load_trajectories()
    stream_data = _load_streamlines()

    stream_bins = _split_streamlines_by_speed(
        "drift",
        stream_data["stream_lats"],
        stream_data["stream_lons"],
        stream_data["stream_speeds"],
    )

    all_cities = sorted(df["city"].unique())
    all_days = sorted(df["day"].unique())

    # -- Sidebar controls --
    st.sidebar.markdown("### Drift Settings")

    selected_cities = st.sidebar.multiselect(
        "Source Cities",
        all_cities,
        default=all_cities,
        help="Which coastal cities to release particles from",
    )

    anim_speed = st.sidebar.slider(
        "Animation Speed (s/frame)",
        min_value=0.05,
        max_value=0.5,
        value=0.15,
        step=0.05,
        help="Seconds between animation frames",
    )

    show_hotspots = st.sidebar.checkbox(
        "Compare with NOAA Observations",
        value=False,
        help="Overlay real NOAA microplastic observation density (purple glow)",
    )

    # -- Filter data --
    filtered = df[df["city"].isin(selected_cities)] if selected_cities else df

    # Filter out frozen particles (those that never move from start position)
    filtered = filtered.copy()
    filtered["_key"] = filtered["city"] + "_" + filtered["particle_id"].astype(str)
    start_pos = filtered[filtered["day"] == 0].set_index("_key")[["lat", "lon"]]
    end_pos = filtered[filtered["day"] == all_days[-1]].set_index("_key")[["lat", "lon"]]
    if not start_pos.empty and not end_pos.empty:
        joined = start_pos.join(end_pos, lsuffix="_s", rsuffix="_e", how="inner")
        displacement = np.sqrt(
            (joined["lat_e"] - joined["lat_s"]) ** 2
            + (joined["lon_e"] - joined["lon_s"]) ** 2
        )
        moving_keys = set(displacement[displacement > 0.5].index)
        if moving_keys:
            filtered = filtered[filtered["_key"].isin(moving_keys)]

    # -- Animation frames: every 8th time step → ~33 frames --
    frame_days = all_days[::8]

    # -- Precompute density + trails --
    city_key = ",".join(sorted(selected_cities)) if selected_cities else "all"
    density_frames, zmax = _precompute_density_frames(
        city_key,
        filtered["lat"].values,
        filtered["lon"].values,
        filtered["day"].values,
        frame_days,
    )

    # Trails: subsample to 150 particles for performance
    all_keys = filtered["_key"].unique()
    if len(all_keys) > 150:
        rng = np.random.RandomState(42)
        keep_keys = rng.choice(all_keys, 150, replace=False)
        trail_df = filtered[filtered["_key"].isin(keep_keys)]
    else:
        trail_df = filtered

    trail_frames = _precompute_trails(
        city_key + "_trails",
        trail_df["lat"].values,
        trail_df["lon"].values,
        trail_df["day"].values,
        trail_df["_key"].values,
        frame_days,
    )

    # NOAA density overlay (loaded once, cached)
    noaa_density = _load_noaa_density() if show_hotspots else None

    # -- Session state for animation --
    if "drift_frame_idx" not in st.session_state:
        st.session_state.drift_frame_idx = 0
    if "drift_playing" not in st.session_state:
        st.session_state.drift_playing = False

    n_frames = len(frame_days)
    st.session_state.drift_frame_idx = min(
        st.session_state.drift_frame_idx, n_frames - 1
    )

    # -- Playback controls --
    btn_cols = st.columns([1, 1, 1, 4])
    with btn_cols[0]:
        if st.button("Play", use_container_width=True):
            st.session_state.drift_playing = True
            if st.session_state.drift_frame_idx >= n_frames - 1:
                st.session_state.drift_frame_idx = 0
    with btn_cols[1]:
        if st.button("Pause", use_container_width=True):
            st.session_state.drift_playing = False
    with btn_cols[2]:
        if st.button("Reset", use_container_width=True):
            st.session_state.drift_playing = False
            st.session_state.drift_frame_idx = 0

    # -- Current frame data --
    idx = st.session_state.drift_frame_idx
    day = frame_days[idx]
    current_df = filtered[filtered["day"] == day]
    n_particles = len(current_df)
    n_cities = len(selected_cities) if selected_cities else len(all_cities)

    # -- Metrics row --
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Time", _day_label(day))
    c2.metric("Day", f"{day:,} / {all_days[-1]:,}")
    c3.metric("Active Particles", f"{n_particles:,}")
    c4.metric("Source Cities", n_cities)

    # -- Build and display map (single call, stable DOM position = no flicker) --
    trail = trail_frames.get(day)
    fig = _build_figure(
        day, stream_bins, density_frames, zmax,
        trail, current_df, noaa_density,
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    # -- Contextual caption --
    if day == 0:
        st.caption(
            f"**{_day_label(day)}** — {n_particles} particles positioned "
            f"at {n_cities} coastal cities, ready to drift with ocean currents."
        )
    elif day < 365:
        st.caption(
            f"**{_day_label(day)}** — Particles begin dispersing along "
            f"major current systems. Watch them follow the streamlines."
        )
    elif day < 1095:
        st.caption(
            f"**{_day_label(day)}** — Particles are carried by ocean "
            f"gyres. Notice accumulation zones forming (orange/red regions)."
        )
    else:
        st.caption(
            f"**{_day_label(day)}** — After {day // 365}+ years, "
            f"particles concentrate in subtropical gyres — matching real-world "
            f"NOAA observations. Toggle 'Compare with NOAA Observations' to verify."
        )

    # -- Auto-advance if playing (flicker-free: st.rerun keeps chart in place) --
    if st.session_state.drift_playing:
        if st.session_state.drift_frame_idx < n_frames - 1:
            time.sleep(anim_speed)
            st.session_state.drift_frame_idx += 1
            st.rerun()
        else:
            st.session_state.drift_playing = False
