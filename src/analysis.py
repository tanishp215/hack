"""Statistical analysis and visualization functions for the microplastics dashboard.

All functions accept a cleaned pandas DataFrame as their first argument and
return either a summary DataFrame or a Plotly figure.  No function renders
anything itself — all rendering is delegated to the Streamlit page layer.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0088
DEFAULT_HOTSPOT_POINTS_BASENAME = "hotspot_points"
DEFAULT_HOTSPOT_POLYGONS_FILENAME = "hotspot_boundaries.json"
REQUIRED_CLUSTERING_COLUMNS = ("latitude", "longitude", "density")
SUPPORTED_POINTS_FORMATS = frozenset({"parquet", "csv"})

# ---------------------------------------------------------------------------
# Color palette — reused across all charts
# ---------------------------------------------------------------------------

PRIMARY_COLOR: str = "#4C72B0"
SECONDARY_COLOR: str = "#55A868"
ACCENT_COLOR: str = "#C44E52"
PALETTE: list[str] = [
    "#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD",
]

# Neutral / semantic colors reused across chart builders
NEUTRAL_GREY: str = "#999999"    # error bars, weak-correlation annotations
MUTED_GREY: str = "#888888"      # reference lines, secondary annotation text
MUTED_FILL: str = "#E8E8E8"      # shaded background regions (e.g. vrect)
NOISE_COLOR: str = "#CCCCCC"     # DBSCAN noise points
DARK_TEXT: str = "#1A1A1A"       # map text labels

# Correlation effect-size annotation colors
CORR_STRONG: str = "#2ca02c"     # green  — |ρ| > 0.3
CORR_MODERATE: str = "#ff7f0e"   # orange — 0.1 ≤ |ρ| ≤ 0.3
# CORR_WEAK reuses NEUTRAL_GREY

# Keep the private alias so CHART_THEME.colorway still works
_COLOR_PALETTE = PALETTE

CHART_THEME: dict = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FAFAFA",
    font=dict(family="Inter", color="#1A1A1A"),
    title=dict(
        font=dict(size=16, color="#1A1A1A", family="Inter"),
        x=0,
        xanchor="left",
    ),
    xaxis=dict(
        title_font=dict(size=12, color="#666666"),
        showline=False,
        showgrid=False,
        tickfont=dict(color="#666666"),
    ),
    yaxis=dict(
        title_font=dict(size=12, color="#666666"),
        showline=False,
        gridcolor="#E8E8E8",
        gridwidth=1,
        tickfont=dict(color="#666666"),
    ),
    legend=dict(
        x=1, y=1, xanchor="right", yanchor="top",
        bgcolor="rgba(0,0,0,0)", borderwidth=0,
    ),
    colorway=_COLOR_PALETTE,
    margin=dict(l=60, r=30, t=50, b=50),
)


# ---------------------------------------------------------------------------
# Defensive validation
# ---------------------------------------------------------------------------

def validate_dataframe(df: pd.DataFrame, required_cols: list[str]) -> None:
    """Raise a descriptive ValueError if any required column is missing.

    Args:
        df: DataFrame to validate.
        required_cols: Column names that must be present.

    Raises:
        ValueError: Lists every missing column in the message.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required column(s): {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )


# ---------------------------------------------------------------------------
# Analysis functions — return DataFrames
# ---------------------------------------------------------------------------

def compute_basin_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics of microplastics density by ocean basin.

    Only rows with valid ``ocean`` and ``measurement`` values are included;
    NaNs in other columns are left untouched.

    Args:
        df: Cleaned microplastics DataFrame containing at least the columns
            ``ocean`` (str) and ``measurement`` (float).

    Returns:
        A DataFrame indexed by ocean basin with columns:
        ``mean``, ``median``, ``std``, ``count``, ``p25``, ``p75``.
        Sorted descending by mean density.
    """
    validate_dataframe(df, ["ocean", "measurement"])
    subset = df.dropna(subset=["ocean", "measurement"])

    stats_df = subset.groupby("ocean")["measurement"].agg(
        mean="mean",
        median="median",
        std="std",
        count="count",
        p25=lambda s: s.quantile(0.25),
        p75=lambda s: s.quantile(0.75),
    )

    stats_df.index.name = "ocean_basin"
    stats_df = stats_df.sort_values("mean", ascending=False)
    return stats_df


def compute_temporal_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate observation count and mean/median density by calendar year.

    Only rows with valid ``year`` and ``measurement`` values contribute to
    the aggregation.  Years with fewer than 3 observations are dropped as
    too noisy to plot reliably.  Results are sorted ascending by year.

    Args:
        df: Cleaned microplastics DataFrame containing at least the columns
            ``year`` (int-like) and ``measurement`` (float).

    Returns:
        A DataFrame with ``year`` as a regular column and columns:
        ``observation_count`` (int), ``mean_density`` (float),
        ``median_density`` (float).  Sorted ascending by year.
    """
    validate_dataframe(df, ["year", "measurement"])
    subset = df.dropna(subset=["year", "measurement"]).copy()
    subset["year"] = subset["year"].astype(int)

    grouped = subset.groupby("year")["measurement"].agg(
        observation_count="count",
        mean_density="mean",
        median_density="median",
    ).reset_index()

    grouped = grouped[grouped["observation_count"] >= 3]
    grouped = grouped.sort_values("year").reset_index(drop=True)
    return grouped


def get_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise pre-computed DBSCAN clusters from the DataFrame.

    This function does *not* run clustering itself — it expects the DataFrame
    to already contain a ``cluster`` column produced by a prior Stage 1A step.
    Noise points (``cluster == -1``) are excluded from the summary.

    Args:
        df: Cleaned microplastics DataFrame containing at least the columns
            ``cluster`` (int), ``latitude`` (float), ``longitude`` (float),
            and ``measurement`` (float).

    Returns:
        A DataFrame with one row per cluster and columns:
        ``cluster_id``, ``label``, ``center_lat``, ``center_lon``,
        ``observation_count``, ``mean_density``, ``max_density``.
        Sorted descending by ``observation_count``.
    """
    validate_dataframe(df, ["cluster", "latitude", "longitude", "measurement"])
    clusters = df[df["cluster"] != -1].copy()

    summary = (
        clusters.groupby("cluster")
        .agg(
            center_lat=("latitude", "mean"),
            center_lon=("longitude", "mean"),
            observation_count=("latitude", "count"),
            mean_density=("measurement", "mean"),
            max_density=("measurement", "max"),
        )
        .reset_index()
        .rename(columns={"cluster": "cluster_id"})
    )

    summary["center_lat"] = summary["center_lat"].round(2)
    summary["center_lon"] = summary["center_lon"].round(2)
    summary = summary.sort_values("observation_count", ascending=False).reset_index(drop=True)

    # Renumber from 1 regardless of DBSCAN's 0-based labels
    summary.insert(1, "label", [f"Cluster {i + 1}" for i in range(len(summary))])

    return summary


def compute_correlations(df: pd.DataFrame, feature_cols: list[str]) -> list[dict]:
    """Compute Spearman rank correlations between each feature and density.

    Only columns that actually exist in ``df`` are processed; missing columns
    emit a warning and are skipped.  NaNs are dropped per pair so each
    correlation uses the maximum available sample.

    Args:
        df: Cleaned microplastics DataFrame containing ``measurement`` and
            (some of) the columns named in ``feature_cols``.
        feature_cols: Column names to correlate against ``measurement``.

    Returns:
        A list of dicts sorted by ``abs(rho)`` descending, each with keys:
        ``feature`` (str), ``rho`` (float), ``p_value`` (float), ``n`` (int).
    """
    validate_dataframe(df, ["measurement"])
    results: list[dict] = []

    for col in feature_cols:
        if col not in df.columns:
            logger.warning("compute_correlations: column %r not found — skipping.", col)
            continue

        subset = df[[col, "measurement"]].dropna()
        n = len(subset)
        if n < 3:
            logger.warning("compute_correlations: column %r has only %d valid rows — skipping.", col, n)
            continue

        rho, p_value = stats.spearmanr(
            subset[col].astype(float), subset["measurement"].astype(float)
        )
        results.append({"feature": col, "rho": float(rho), "p_value": float(p_value), "n": int(n)})

    results.sort(key=lambda d: abs(d["rho"]), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Shared layout helper
# ---------------------------------------------------------------------------

def apply_standard_layout(fig: go.Figure, title: str) -> go.Figure:
    """Apply the PlasticFlow standard visual theme to a Plotly figure.

    Sets font, background, gridlines, margins, and title consistently
    across every chart in the dashboard.

    Args:
        fig: The Plotly Figure to update (mutated in-place).
        title: Chart title string.

    Returns:
        The same figure with the standard layout applied.
    """
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color="#1A1A1A", family="Inter, Arial, sans-serif"),
            x=0,
            xanchor="left",
        ),
        font=dict(family="Inter, Arial, sans-serif", size=13, color="#1A1A1A"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        margin=dict(l=60, r=30, t=60, b=60),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#f0f0f0",
        gridwidth=1,
        showline=False,
        tickfont=dict(size=12, color="#666666"),
        title_font=dict(size=12, color="#666666"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#f0f0f0",
        gridwidth=1,
        showline=False,
        tickfont=dict(size=12, color="#666666"),
        title_font=dict(size=12, color="#666666"),
    )
    return fig


# ---------------------------------------------------------------------------
# Chart-builder functions — return Plotly figures
# ---------------------------------------------------------------------------

def build_basin_chart(basin_stats: pd.DataFrame) -> go.Figure:
    """Build a bar chart of mean microplastics density by ocean basin.

    Each bar represents one ocean basin's mean density with ±1 SD error bars.
    A horizontal dashed line marks the global mean across all basins.
    Hover tooltips show basin name, mean, median, std, and count.
    The ``CHART_THEME`` is applied to the returned figure.

    Args:
        basin_stats: Output of :func:`compute_basin_statistics` — a DataFrame
            indexed by ``ocean_basin`` with columns ``mean``, ``median``,
            ``std``, ``count``, ``p25``, ``p75``.

    Returns:
        A Plotly Figure ready for ``st.plotly_chart``.
    """
    basins = basin_stats.index.tolist()
    n = len(basins)
    bar_colors = [PALETTE[i % len(PALETTE)] for i in range(n)]

    global_mean = basin_stats["mean"].mean()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=basins,
        y=basin_stats["mean"],
        error_y=dict(type="data", array=basin_stats["std"].tolist(), visible=True,
                     color=NEUTRAL_GREY, thickness=1.5, width=6),
        marker=dict(color=bar_colors, opacity=0.85, line=dict(width=0)),
        customdata=basin_stats[["median", "std", "count"]].values,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Mean: %{y:.4f} pieces/m³<br>"
            "Median: %{customdata[0]:.4f} pieces/m³<br>"
            "Std dev: %{customdata[1]:.4f}<br>"
            "Observations: %{customdata[2]:,}<extra></extra>"
        ),
        name="Mean density",
    ))

    # Global mean reference line
    fig.add_hline(
        y=global_mean,
        line=dict(color=MUTED_GREY, width=1.5, dash="dash"),
        annotation_text="Global Mean",
        annotation_position="top right",
        annotation_font=dict(size=11, color=MUTED_GREY, family="Inter"),
    )

    apply_standard_layout(fig, "Mean Microplastic Density by Ocean Basin")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text="Ocean Basin", showgrid=False)
    fig.update_yaxes(
        title_text="Mean Density (pieces/m³)",
        rangemode="tozero",
        tickformat=",.2f",
    )

    return fig


def build_temporal_chart(temporal_df: pd.DataFrame) -> go.Figure:
    """Build a dual-axis chart: observation count (bars) and mean density (line).

    Uses ``make_subplots`` with ``secondary_y=True``.  Observation count bars
    sit on the left y-axis; mean density line sits on the right y-axis so
    density — the more analytically interesting series — is visually prominent.
    A shaded region flags the expanded research period from 2010 onward.
    The ``CHART_THEME`` is applied to the returned figure.

    Args:
        temporal_df: Output of :func:`compute_temporal_trends` — a DataFrame
            with columns ``year``, ``observation_count``, ``mean_density``,
            and ``median_density``.

    Returns:
        A Plotly Figure with two y-axes, ready for ``st.plotly_chart``.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    years = temporal_df["year"].tolist()
    last_year = max(years)

    # --- Bars: observation count (left / primary y-axis) ---
    fig.add_trace(
        go.Bar(
            x=years,
            y=temporal_df["observation_count"],
            name="Observations",
            marker=dict(color=PALETTE[4], opacity=0.6, line=dict(width=0)),
            customdata=temporal_df[["mean_density", "observation_count"]].values,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Observations: %{y:,}<br>"
                "Mean density: %{customdata[0]:.4f} pieces/m³<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    # --- Line: mean density (right / secondary y-axis) ---
    fig.add_trace(
        go.Scatter(
            x=years,
            y=temporal_df["mean_density"],
            name="Mean Density",
            mode="lines+markers",
            line=dict(color=PRIMARY_COLOR, width=2),
            marker=dict(size=5, color=PRIMARY_COLOR),
            customdata=temporal_df[["mean_density", "observation_count"]].values,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Mean density: %{y:.4f} pieces/m³<br>"
                "Observations: %{customdata[1]:,}<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    # --- Shaded region: expanded research period (2010 → last year) ---
    fig.add_vrect(
        x0=2010, x1=last_year,
        fillcolor=MUTED_FILL, opacity=0.35,
        layer="below", line_width=0,
        annotation_text="Expanded research period",
        annotation_position="top left",
        annotation_font=dict(size=11, color=MUTED_GREY, family="Inter"),
    )

    # --- Layout ---
    apply_standard_layout(fig, "Temporal Trends: Research Activity vs. Measured Density")
    fig.update_layout(
        legend=dict(
            x=0.01, y=0.99, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.85)", borderwidth=0,
        ),
        bargap=0.2,
    )
    fig.update_xaxes(title_text="Year", showgrid=False)
    fig.update_yaxes(
        title_text="Number of Observations",
        gridcolor="#f0f0f0", gridwidth=1, showline=False,
        rangemode="tozero", tickformat=",d",
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Mean Density (pieces/m³)",
        showgrid=False, showline=False,
        rangemode="tozero", tickformat=",.4f",
        secondary_y=True,
    )

    return fig


def build_cluster_map(df: pd.DataFrame, cluster_summary: pd.DataFrame) -> go.Figure:
    """Build a Plotly geo scatter map coloured by DBSCAN cluster membership.

    # Expected real-world cluster locations (if data is accurate):
    #   - North Pacific Gyre  (~30°N, 140°W) — Great Pacific Garbage Patch
    #   - North Atlantic Gyre (~30°N,  40°W) — North Atlantic Garbage Patch
    #   - South Pacific Gyre  (~40°S, 120°W)
    #   - South Atlantic Gyre (~35°S,  15°W)
    #   - Indian Ocean Gyre   (~30°S,  80°E)

    Uses ``go.Scattergeo`` with Plotly's Natural Earth projection so country
    names are always rendered in English (no tile API key required).  All raw
    data points are plotted, noise in light grey, clusters in ``PALETTE``
    colors.  Cluster centers are overlaid as larger bordered markers.

    Args:
        df: Cleaned microplastics DataFrame with ``latitude``, ``longitude``,
            ``cluster`` (int), and ``measurement`` columns.
        cluster_summary: Output of :func:`get_cluster_summary` with columns
            ``cluster_id``, ``label``, ``center_lat``, ``center_lon``,
            ``observation_count``, ``mean_density``, ``max_density``.

    Returns:
        A Plotly Figure using go.Scattergeo, ready for ``st.plotly_chart``.
    """
    # Build cluster_id → color lookup; noise gets fixed grey
    id_to_color: dict[int, str] = {
        row.cluster_id: PALETTE[i % len(PALETTE)]
        for i, row in enumerate(cluster_summary.itertuples())
    }

    # --- Log-scale density for continuous coloring ---
    log_density = np.log10(df["measurement"].clip(lower=1e-4) + 1e-4)
    cmin = float(log_density.quantile(0.01))
    cmax = float(log_density.quantile(0.99))

    # Build colorbar tick labels at clean powers of 10
    _tick_raw = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1_000, 10_000, 100_000, 800_000]
    _tick_vals = [float(np.log10(v + 1e-4)) for v in _tick_raw]
    _tick_text = ["<0.001", "0.001", "0.01", "0.1", "1", "10", "100", "1k", "10k", "100k", "800k"]
    # Keep only ticks within the data range
    _ticks = [(v, t) for v, t in zip(_tick_vals, _tick_text) if cmin <= v <= cmax]
    tick_vals = [v for v, _ in _ticks]
    tick_text = [t for _, t in _ticks]

    fig = go.Figure()

    # --- All observation points colored by log-density (single trace, shared colorbar) ---
    fig.add_trace(go.Scattergeo(
        lat=df["latitude"],
        lon=df["longitude"],
        mode="markers",
        name="Observations",
        marker=dict(
            size=4,
            color=log_density,
            colorscale="YlOrRd",
            cmin=cmin,
            cmax=cmax,
            opacity=0.65,
            showscale=True,
            colorbar=dict(
                title=dict(text="Density<br>(pieces/m³)", font=dict(size=11)),
                tickvals=tick_vals,
                ticktext=tick_text,
                thickness=14,
                len=0.75,
                x=1.01,
            ),
        ),
        customdata=np.stack([df["measurement"], df["cluster"]], axis=1),
        hovertemplate=(
            "Lat: %{lat:.3f}, Lon: %{lon:.3f}<br>"
            "Density: %{customdata[0]:.4f} pieces/m³<extra></extra>"
        ),
        showlegend=False,
    ))

    # --- Cluster centers — numbered markers (number shown on marker) ---
    cluster_numbers = [str(i + 1) for i in range(len(cluster_summary))]
    fig.add_trace(go.Scattergeo(
        lat=cluster_summary["center_lat"],
        lon=cluster_summary["center_lon"],
        mode="markers+text",
        name="Cluster Centers",
        text=cluster_numbers,
        textposition="middle center",
        textfont=dict(size=11, color="white", family="Inter, Arial, sans-serif"),
        marker=dict(
            size=22,
            color=[id_to_color[cid] for cid in cluster_summary["cluster_id"]],
            opacity=1.0,
            line=dict(width=2, color="white"),
            showscale=False,
        ),
        customdata=cluster_summary[["label", "observation_count", "mean_density", "max_density", "center_lat", "center_lon"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Center: %{customdata[4]:.2f}°, %{customdata[5]:.2f}°<br>"
            "Observations: %{customdata[1]:,}<br>"
            "Mean density: %{customdata[2]:.4f} pieces/m³<br>"
            "Max density: %{customdata[3]:.4f} pieces/m³<extra></extra>"
        ),
        showlegend=True,
    ))

    apply_standard_layout(fig, "DBSCAN Cluster Map — Microplastic Hotspots")
    fig.update_layout(
        geo=dict(
            projection_type="natural earth",
            showland=True, landcolor="#f5f5f5",
            showocean=True, oceancolor="#d8eaf8",
            showcoastlines=True, coastlinecolor="#aaaaaa", coastlinewidth=0.5,
            showlakes=True, lakecolor="#d8eaf8",
            showcountries=True, countrycolor="#cccccc", countrywidth=0.5,
            showframe=False,
            bgcolor="white",
        ),
        legend=dict(
            x=0.01, y=0.99, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.85)", borderwidth=0,
            font=dict(size=11, family="Inter, Arial, sans-serif"),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        height=550,
    )

    return fig


def build_correlation_charts(
    df: pd.DataFrame,
    correlations: list[dict],
    feature_cols: list[str],
) -> list[go.Figure]:
    """Build one scatter-with-trendline figure per feature–density pair.

    Each figure includes raw scatter points (small, semi-transparent), a
    numpy-polyfit OLS trendline, and a top-left annotation with the Spearman
    ρ, p-value, and sample size.  Annotation color reflects effect size:
    green (|ρ| > 0.3), orange (0.1–0.3), grey (< 0.1).

    Columns in ``feature_cols`` that are absent from ``df`` are skipped with
    a warning.  ``CHART_THEME`` is applied to every figure.

    Args:
        df: Cleaned microplastics DataFrame containing ``measurement`` and
            (some of) the columns named in ``feature_cols``.
        correlations: Output of :func:`compute_correlations` — list of dicts
            with keys ``feature``, ``rho``, ``p_value``, ``n``.
        feature_cols: Feature columns to chart (same candidates passed to
            :func:`compute_correlations`).

    Returns:
        A list of Plotly Figures, one per valid feature, ready for
        ``st.plotly_chart``.
    """
    validate_dataframe(df, ["measurement"])

    # Build lookup from correlations list for quick access
    corr_lookup: dict[str, dict] = {c["feature"]: c for c in correlations}

    # Friendly axis labels (unit hints where known)
    axis_labels: dict[str, str] = {
        "latitude": "Latitude (°)",
        "longitude": "Longitude (°)",
        "year": "Year",
        "depth": "Depth (m)",
        "distance_to_coast": "Distance to Coast (km)",
    }

    figs: list[go.Figure] = []

    for col in feature_cols:
        if col not in df.columns:
            logger.warning("build_correlation_charts: column %r not found — skipping.", col)
            continue

        subset = df[[col, "measurement"]].dropna()
        x = subset[col].astype(float).values
        y = subset["measurement"].astype(float).values

        # Trendline fitted in log-y space (density is log-distributed)
        y_pos = y.clip(min=1e-4)
        x_sorted = np.sort(x)
        coeffs = np.polyfit(x, np.log10(y_pos), 1)
        y_trend = 10 ** np.polyval(coeffs, x_sorted)

        # Annotation color by effect size
        corr = corr_lookup.get(col, {})
        rho = corr.get("rho", float("nan"))
        p_val = corr.get("p_value", float("nan"))
        n = corr.get("n", len(subset))

        abs_rho = abs(rho)
        if abs_rho > 0.3:
            annot_color = CORR_STRONG
        elif abs_rho >= 0.1:
            annot_color = CORR_MODERATE
        else:
            annot_color = NEUTRAL_GREY

        x_label = axis_labels.get(col, col.replace("_", " ").title())

        fig = go.Figure()

        # Scatter points
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Observations",
            marker=dict(size=5, color=PALETTE[0], opacity=0.4, line=dict(width=0)),
            hovertemplate=(
                f"{x_label}: %{{x}}<br>"
                "Density: %{y:.4f} pieces/m³<extra></extra>"
            ),
        ))

        # Trendline
        fig.add_trace(go.Scatter(
            x=x_sorted,
            y=y_trend,
            mode="lines",
            name="OLS trend",
            line=dict(color=ACCENT_COLOR, width=2, dash="dash"),
            hoverinfo="skip",
        ))

        apply_standard_layout(fig, f"Density vs. {x_label}")
        fig.update_xaxes(title_text=x_label)
        fig.update_yaxes(title_text="Density (pieces/m³)", type="log")

        fig.add_annotation(
            text=f"Spearman ρ = {rho:.3f}, p = {p_val:.4f}, n = {n:,}",
            xref="paper", yref="paper",
            x=0.02, y=0.97,
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=12, color=annot_color, family="Inter, Arial, sans-serif"),
            bgcolor="rgba(255,255,255,0.85)",
            borderpad=4,
        )

        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# Hotspot clustering helpers
# ---------------------------------------------------------------------------

def validate_hotspot_input_dataframe(
    df: pd.DataFrame,
    *,
    required_columns: tuple[str, ...] = REQUIRED_CLUSTERING_COLUMNS,
) -> pd.DataFrame:
    """Return a validated observations DataFrame for hotspot clustering."""

    if df is None:
        raise ValueError("Observations DataFrame is required for hotspot clustering.")

    _validate_required_columns(
        df,
        required_columns,
        error_prefix="Observations DataFrame is missing required columns for clustering",
    )

    validated_df = df.copy()
    for column in required_columns:
        validated_df[column] = pd.to_numeric(validated_df[column], errors="coerce")

    validated_df = validated_df.dropna(subset=list(required_columns)).reset_index(drop=True)
    if validated_df.empty:
        required_list = ", ".join(required_columns)
        raise ValueError(
            "Observations DataFrame does not contain any valid rows with numeric "
            f"{required_list} values."
        )

    return validated_df


def validate_hotspot_polygons_payload(payload: object) -> list[dict[str, Any]]:
    """Validate and return hotspot polygon records from a JSON payload."""

    if not isinstance(payload, dict):
        raise ValueError("Hotspot boundaries JSON must contain an object payload.")

    polygons = payload.get("polygons", [])
    if not isinstance(polygons, list):
        raise ValueError("Hotspot boundaries file must contain a 'polygons' list.")

    for polygon in polygons:
        if not isinstance(polygon, dict):
            raise ValueError("Each hotspot polygon payload must be a JSON object.")
        if "polygon" not in polygon:
            raise ValueError("Each hotspot polygon payload must include a 'polygon' field.")
        if not isinstance(polygon["polygon"], list):
            raise ValueError("Each hotspot polygon field must be a list of coordinates.")

    return polygons


def filter_high_density_observations(
    df: pd.DataFrame,
    density_percentile: float = 75.0,
) -> pd.DataFrame:
    """Return hotspot candidate observations above a density percentile threshold."""

    _validate_cluster_parameters(density_percentile=density_percentile)

    validated_df = validate_hotspot_input_dataframe(df, required_columns=("density",))
    density_threshold = validated_df["density"].quantile(density_percentile / 100.0)
    if pd.isna(density_threshold):
        raise ValueError(
            "Unable to compute a density percentile threshold from the observations DataFrame."
        )

    high_density_df = validated_df.loc[validated_df["density"] >= density_threshold].copy()
    high_density_df["density_percentile_threshold"] = density_threshold
    return high_density_df.reset_index(drop=True)


def cluster_hotspot_observations(
    df: pd.DataFrame,
    density_percentile: float = 75.0,
    eps_km: float = 250.0,
    min_samples: int = 3,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Cluster high-density observations and return hotspot points and polygons."""

    _validate_cluster_parameters(
        density_percentile=density_percentile,
        eps_km=eps_km,
        min_samples=min_samples,
    )

    validated_df = validate_hotspot_input_dataframe(df)
    high_density_df = filter_high_density_observations(
        validated_df,
        density_percentile=density_percentile,
    )
    if high_density_df.empty:
        return high_density_df.assign(cluster_label=pd.Series(dtype="int64")), []

    cluster_labels = _fit_hotspot_dbscan(
        high_density_df,
        eps_km=eps_km,
        min_samples=min_samples,
    )

    labeled_hotspots = high_density_df.copy()
    labeled_hotspots["cluster_label"] = cluster_labels.astype(int)
    clustered_hotspots = labeled_hotspots.loc[
        labeled_hotspots["cluster_label"] != -1
    ].copy()

    boundary_polygons = build_cluster_boundary_polygons(clustered_hotspots)
    return clustered_hotspots.reset_index(drop=True), boundary_polygons


def build_cluster_boundary_polygons(
    hotspot_points: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build serializable boundary polygons for each hotspot cluster."""

    if hotspot_points.empty:
        return []

    _validate_required_columns(
        hotspot_points,
        ("cluster_label", "latitude", "longitude"),
        error_prefix="Hotspot points are missing required columns for polygon creation",
    )

    polygons: list[dict[str, Any]] = []
    grouped_points = hotspot_points.groupby("cluster_label", sort=True)

    for cluster_label, cluster_df in grouped_points:
        if cluster_label == -1:
            continue

        cluster_points = _extract_cluster_coordinates(cluster_df)
        if cluster_points.size == 0:
            continue

        ring = _build_cluster_ring(cluster_points)
        polygons.append(_build_polygon_record(cluster_label, cluster_df, ring))

    return polygons


def precompute_and_save_hotspot_clusters(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    density_percentile: float = 75.0,
    eps_km: float = 250.0,
    min_samples: int = 3,
    points_format: str = "parquet",
) -> dict[str, Path]:
    """Compute hotspot clusters and save reusable point and polygon artifacts."""

    hotspot_points, boundary_polygons = cluster_hotspot_observations(
        df,
        density_percentile=density_percentile,
        eps_km=eps_km,
        min_samples=min_samples,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points_path = _build_hotspot_points_path(output_dir, points_format)
    polygons_path = output_dir / DEFAULT_HOTSPOT_POLYGONS_FILENAME

    _save_hotspot_points(hotspot_points, points_path, points_format)
    _save_hotspot_polygons(
        boundary_polygons,
        polygons_path,
        density_percentile=density_percentile,
        eps_km=eps_km,
        min_samples=min_samples,
    )

    return {
        "points_path": points_path,
        "polygons_path": polygons_path,
    }


def load_precomputed_hotspot_clusters(
    output_dir: str | Path,
    *,
    points_format: str = "parquet",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Load precomputed hotspot points and boundary polygons from disk."""

    output_dir = Path(output_dir)
    points_path = _build_hotspot_points_path(output_dir, points_format)
    polygons_path = output_dir / DEFAULT_HOTSPOT_POLYGONS_FILENAME

    _require_existing_file(points_path, "Hotspot points file")
    _require_existing_file(polygons_path, "Hotspot polygon file")

    hotspot_points = _load_hotspot_points(points_path, points_format)
    polygons_payload = _load_json_payload(polygons_path)
    return hotspot_points, validate_hotspot_polygons_payload(polygons_payload)


def run_hotspot_precompute(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    density_percentile: float = 75.0,
    eps_km: float = 250.0,
    min_samples: int = 3,
    points_format: str = "parquet",
) -> dict[str, Path]:
    """Load cleaned observations data and save hotspot artifacts."""

    observations_df = _read_observations_frame(input_path)
    return precompute_and_save_hotspot_clusters(
        observations_df,
        output_dir,
        density_percentile=density_percentile,
        eps_km=eps_km,
        min_samples=min_samples,
        points_format=points_format,
    )


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: tuple[str, ...] | list[str],
    *,
    error_prefix: str,
) -> None:
    """Raise a ``ValueError`` when a DataFrame is missing required columns."""

    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"{error_prefix}: {', '.join(missing_columns)}.")


def _validate_cluster_parameters(
    *,
    density_percentile: float,
    eps_km: float | None = None,
    min_samples: int | None = None,
) -> None:
    """Validate user-facing hotspot clustering parameters."""

    if not 0 <= density_percentile <= 100:
        raise ValueError("density_percentile must be between 0 and 100.")
    if eps_km is not None and eps_km <= 0:
        raise ValueError("eps_km must be greater than zero.")
    if min_samples is not None and min_samples < 1:
        raise ValueError("min_samples must be at least 1.")


def _fit_hotspot_dbscan(
    hotspot_df: pd.DataFrame,
    *,
    eps_km: float,
    min_samples: int,
) -> np.ndarray:
    """Fit haversine DBSCAN to hotspot candidate coordinates."""

    coordinates_radians = np.radians(
        hotspot_df.loc[:, ["latitude", "longitude"]].to_numpy(dtype=float)
    )
    eps_radians = eps_km / EARTH_RADIUS_KM

    dbscan = DBSCAN(
        eps=eps_radians,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    )
    return dbscan.fit_predict(coordinates_radians)


def _extract_cluster_coordinates(cluster_df: pd.DataFrame) -> np.ndarray:
    """Return cluster coordinates as ``[longitude, latitude]`` pairs."""

    return cluster_df.loc[:, ["longitude", "latitude"]].to_numpy(dtype=float)


def _build_polygon_record(
    cluster_label: int | np.integer[Any],
    cluster_df: pd.DataFrame,
    ring: np.ndarray,
) -> dict[str, Any]:
    """Return one serializable polygon payload for a hotspot cluster."""

    return {
        "cluster_label": int(cluster_label),
        "point_count": int(len(cluster_df)),
        "centroid": {
            "latitude": float(cluster_df["latitude"].mean()),
            "longitude": float(cluster_df["longitude"].mean()),
        },
        "polygon": [
            {"longitude": float(longitude), "latitude": float(latitude)}
            for longitude, latitude in ring
        ],
    }


def _build_cluster_ring(cluster_points: np.ndarray) -> np.ndarray:
    """Return a closed polygon ring for a cluster of ``[lon, lat]`` points."""

    if len(cluster_points) == 0:
        raise ValueError("Cluster polygon generation requires at least one coordinate pair.")
    if len(cluster_points) < 3:
        return _bounding_box_ring(cluster_points)

    try:
        hull = ConvexHull(cluster_points)
        hull_points = cluster_points[hull.vertices]
    except QhullError:
        return _bounding_box_ring(cluster_points)

    return _close_ring(hull_points)


def _bounding_box_ring(cluster_points: np.ndarray) -> np.ndarray:
    """Return a rectangular fallback ring that encloses a cluster."""

    min_longitude, min_latitude = cluster_points.min(axis=0)
    max_longitude, max_latitude = cluster_points.max(axis=0)

    longitude_padding = max((max_longitude - min_longitude) * 0.05, 1e-6)
    latitude_padding = max((max_latitude - min_latitude) * 0.05, 1e-6)

    ring = np.array(
        [
            [min_longitude - longitude_padding, min_latitude - latitude_padding],
            [max_longitude + longitude_padding, min_latitude - latitude_padding],
            [max_longitude + longitude_padding, max_latitude + latitude_padding],
            [min_longitude - longitude_padding, max_latitude + latitude_padding],
        ]
    )
    return _close_ring(ring)


def _close_ring(points: np.ndarray) -> np.ndarray:
    """Return polygon coordinates with the first point repeated at the end."""

    if np.array_equal(points[0], points[-1]):
        return points
    return np.vstack([points, points[0]])


def _build_hotspot_points_path(output_dir: Path, points_format: str) -> Path:
    """Return the hotspot points artifact path for the requested format."""

    normalized_format = _normalize_points_format(points_format)
    return output_dir / f"{DEFAULT_HOTSPOT_POINTS_BASENAME}.{normalized_format}"


def _normalize_points_format(points_format: str) -> str:
    """Return a normalized hotspot points file format."""

    normalized_format = points_format.lower()
    if normalized_format not in SUPPORTED_POINTS_FORMATS:
        raise ValueError("points_format must be either 'parquet' or 'csv'.")
    return normalized_format


def _save_hotspot_points(
    hotspot_points: pd.DataFrame,
    points_path: Path,
    points_format: str,
) -> None:
    """Write hotspot point artifacts to disk."""

    normalized_format = _normalize_points_format(points_format)
    if normalized_format == "parquet":
        hotspot_points.to_parquet(points_path, index=False)
        return
    hotspot_points.to_csv(points_path, index=False)


def _load_hotspot_points(points_path: Path, points_format: str) -> pd.DataFrame:
    """Read hotspot point artifacts from disk."""

    normalized_format = _normalize_points_format(points_format)
    if normalized_format == "parquet":
        return pd.read_parquet(points_path)
    return pd.read_csv(points_path)


def _save_hotspot_polygons(
    boundary_polygons: list[dict[str, Any]],
    polygons_path: Path,
    *,
    density_percentile: float,
    eps_km: float,
    min_samples: int,
) -> None:
    """Write hotspot polygon metadata to disk as JSON."""

    payload = {
        "version": 1,
        "density_percentile": density_percentile,
        "eps_km": eps_km,
        "min_samples": min_samples,
        "cluster_count": len(boundary_polygons),
        "polygons": boundary_polygons,
    }
    with polygons_path.open("w", encoding="utf-8") as polygons_file:
        json.dump(payload, polygons_file, indent=2)


def _require_existing_file(path: Path, description: str) -> None:
    """Raise ``FileNotFoundError`` when an expected artifact file is missing."""

    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def _load_json_payload(path: Path) -> dict[str, Any]:
    """Read and return a JSON payload from disk."""

    with path.open("r", encoding="utf-8") as json_file:
        payload = json.load(json_file)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _read_observations_frame(input_path: str | Path) -> pd.DataFrame:
    """Read a cleaned observations DataFrame from CSV or parquet."""

    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    if suffix == ".csv":
        return pd.read_csv(input_path)

    raise ValueError("input_path must point to a .csv or .parquet file.")


def _build_hotspot_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for offline hotspot precomputation."""

    parser = argparse.ArgumentParser(
        description="Precompute hotspot clustering artifacts for the Streamlit app."
    )
    parser.add_argument("input_path", help="Path to a cleaned observations CSV or parquet file.")
    parser.add_argument("output_dir", help="Directory where hotspot artifacts should be saved.")
    parser.add_argument(
        "--density-percentile",
        type=float,
        default=75.0,
        help="Density percentile cutoff for hotspot candidate selection.",
    )
    parser.add_argument(
        "--eps-km",
        type=float,
        default=250.0,
        help="DBSCAN neighborhood radius in kilometers.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum number of hotspot points required for a cluster.",
    )
    parser.add_argument(
        "--points-format",
        choices=tuple(sorted(SUPPORTED_POINTS_FORMATS)),
        default="parquet",
        help="Output format for the hotspot points artifact.",
    )
    return parser


def _run_quick_smoke_test() -> None:
    """Run the historical zero-argument smoke test for the analysis module."""

    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.data_loader import load_microplastics

    df = load_microplastics()

    basin_stats = compute_basin_statistics(df)
    print(basin_stats.to_string())

    fig = build_basin_chart(basin_stats)
    print(f"\nChart title: {fig.layout.title.text}")
    print(f"Traces: {len(fig.data)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        parser = _build_hotspot_argument_parser()
        arguments = parser.parse_args()
        saved_paths = run_hotspot_precompute(
            arguments.input_path,
            arguments.output_dir,
            density_percentile=arguments.density_percentile,
            eps_km=arguments.eps_km,
            min_samples=arguments.min_samples,
            points_format=arguments.points_format,
        )
        print(f"Saved hotspot points to {saved_paths['points_path']}")
        print(f"Saved hotspot polygons to {saved_paths['polygons_path']}")
    else:
        _run_quick_smoke_test()
