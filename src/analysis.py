"""Statistical analysis and visualization functions for the microplastics dashboard.

All functions accept a cleaned pandas DataFrame as their first argument and
return either a summary DataFrame or a Plotly figure.  No function renders
anything itself — all rendering is delegated to the Streamlit page layer.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette — reused across all charts
# ---------------------------------------------------------------------------

PRIMARY_COLOR: str = "#4C72B0"
SECONDARY_COLOR: str = "#55A868"
ACCENT_COLOR: str = "#C44E52"
PALETTE: list[str] = [
    "#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD",
]

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
                     color="#999999", thickness=1.5, width=6),
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
        line=dict(color="#888888", width=1.5, dash="dash"),
        annotation_text="Global Mean",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#888888", family="Inter"),
    )

    apply_standard_layout(fig, "Mean Microplastic Density by Ocean Basin")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text="Ocean Basin", showgrid=False)
    fig.update_yaxes(title_text="Mean Density (pieces/m³)")

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
        fillcolor="#E8E8E8", opacity=0.35,
        layer="below", line_width=0,
        annotation_text="Expanded research period",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#888888", family="Inter"),
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
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Mean Density (pieces/m³)",
        showgrid=False, showline=False,
        secondary_y=True,
    )

    return fig


def build_cluster_map(df: pd.DataFrame, cluster_summary: pd.DataFrame) -> go.Figure:
    """Build a Plotly mapbox scatter map coloured by DBSCAN cluster membership.

    # Expected real-world cluster locations (if data is accurate):
    #   - North Pacific Gyre  (~30°N, 140°W) — Great Pacific Garbage Patch
    #   - North Atlantic Gyre (~30°N,  40°W) — North Atlantic Garbage Patch
    #   - South Pacific Gyre  (~40°S, 120°W)
    #   - South Atlantic Gyre (~35°S,  15°W)
    #   - Indian Ocean Gyre   (~30°S,  80°E)

    Uses ``mapbox_style="carto-positron"`` (no API key required).  All raw
    data points are plotted, noise in light grey, clusters in ``PALETTE``
    colors.  Cluster centers are overlaid as larger bordered markers.

    Args:
        df: Cleaned microplastics DataFrame with ``latitude``, ``longitude``,
            ``cluster`` (int), and ``measurement`` columns.
        cluster_summary: Output of :func:`get_cluster_summary` with columns
            ``cluster_id``, ``label``, ``center_lat``, ``center_lon``,
            ``observation_count``, ``mean_density``, ``max_density``.

    Returns:
        A Plotly Figure using go.Scattermapbox, ready for ``st.plotly_chart``.
    """
    # Build cluster_id → color lookup; noise gets fixed grey
    id_to_color: dict[int, str] = {
        row.cluster_id: PALETTE[i % len(PALETTE)]
        for i, row in enumerate(cluster_summary.itertuples())
    }
    id_to_label: dict[int, str] = {
        row.cluster_id: row.label
        for row in cluster_summary.itertuples()
    }

    fig = go.Figure()

    # --- Noise points (cluster == -1) ---
    noise = df[df["cluster"] == -1]
    if len(noise):
        fig.add_trace(go.Scattermapbox(
            lat=noise["latitude"],
            lon=noise["longitude"],
            mode="markers",
            name="Noise",
            marker=dict(size=4, color="#CCCCCC", opacity=0.3),
            hovertemplate=(
                "Lat: %{lat:.3f}, Lon: %{lon:.3f}<br>"
                "Density: %{customdata:.4f} pieces/m³<br>"
                "Cluster: Noise<extra></extra>"
            ),
            customdata=noise["measurement"],
        ))

    # --- Cluster data points — one trace per cluster for legend + color ---
    for _, row in cluster_summary.iterrows():
        cid = row["cluster_id"]
        label = row["label"]
        color = id_to_color[cid]
        pts = df[df["cluster"] == cid]

        fig.add_trace(go.Scattermapbox(
            lat=pts["latitude"],
            lon=pts["longitude"],
            mode="markers",
            name=label,
            marker=dict(size=5, color=color, opacity=0.5),
            customdata=pts["measurement"],
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Lat: %{lat:.3f}, Lon: %{lon:.3f}<br>"
                "Density: %{customdata:.4f} pieces/m³<extra></extra>"
            ),
            legendgroup=label,
        ))

    # --- Cluster centers (larger bordered markers) ---
    fig.add_trace(go.Scattermapbox(
        lat=cluster_summary["center_lat"],
        lon=cluster_summary["center_lon"],
        mode="markers+text",
        name="Cluster Centers",
        text=cluster_summary["label"],
        textposition="top right",
        textfont=dict(size=10, color="#1A1A1A", family="Inter"),
        marker=dict(
            size=14,
            color=[id_to_color[cid] for cid in cluster_summary["cluster_id"]],
            opacity=1.0,
            symbol="circle",
        ),
        customdata=cluster_summary[["label", "observation_count", "mean_density", "max_density"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Center: %{lat:.2f}°, %{lon:.2f}°<br>"
            "Observations: %{customdata[1]:,}<br>"
            "Mean density: %{customdata[2]:.4f} pieces/m³<br>"
            "Max density: %{customdata[3]:.4f} pieces/m³<extra></extra>"
        ),
        showlegend=True,
    ))

    # --- Map center and zoom from data extent ---
    center_lat = float(df["latitude"].mean())
    center_lon = float(df["longitude"].mean())

    apply_standard_layout(fig, "DBSCAN Cluster Map — Microplastic Hotspots")
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=1.2,
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

        # OLS trendline via numpy polyfit
        coeffs = np.polyfit(x, y, 1)
        x_sorted = np.sort(x)
        y_trend = np.polyval(coeffs, x_sorted)

        # Annotation color by effect size
        corr = corr_lookup.get(col, {})
        rho = corr.get("rho", float("nan"))
        p_val = corr.get("p_value", float("nan"))
        n = corr.get("n", len(subset))

        abs_rho = abs(rho)
        if abs_rho > 0.3:
            annot_color = "#2ca02c"   # green
        elif abs_rho >= 0.1:
            annot_color = "#ff7f0e"   # orange
        else:
            annot_color = "#999999"   # grey

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
        fig.update_yaxes(title_text="Density (pieces/m³)")

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
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.data_loader import load_microplastics

    df = load_microplastics()

    basin_stats = compute_basin_statistics(df)
    print(basin_stats.to_string())

    fig = build_basin_chart(basin_stats)
    print(f"\nChart title: {fig.layout.title.text}")
    print(f"Traces: {len(fig.data)}")
