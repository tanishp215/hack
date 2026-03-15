"""Stage 1B — Statistical Insights page for the PlasticFlow Streamlit app."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import streamlit as st

# set_page_config must be the first Streamlit call in the file
st.set_page_config(
    page_title="Statistical Insights — Microplastics",
    page_icon="📊",
    layout="wide",
)

# Global CSS: hide Streamlit chrome, set font, add tab padding
st.markdown(
    """
    <style>
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    html, body, [class*="css"] {
        font-family: "Inter", "Arial", sans-serif;
    }
    div[data-testid="stTabContent"] {
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

from src.analysis import (
    build_basin_chart,
    build_cluster_map,
    build_correlation_charts,
    build_temporal_chart,
    compute_basin_statistics,
    compute_correlations,
    compute_temporal_trends,
    get_cluster_summary,
)
from src.data_loader import load_microplastics


def _fmt_density(v: float) -> str:
    """Format a density value with appropriate precision and units."""
    if v >= 1000:
        return f"{v / 1000:.1f}K pieces/m³"
    if v >= 1:
        return f"{v:.1f} pieces/m³"
    if v >= 0.001:
        return f"{v:.3f} pieces/m³"
    return f"{v:.2e} pieces/m³"


# ---------------------------------------------------------------------------
# Data loading — parquet preferred, CSV fallback
# ---------------------------------------------------------------------------

_PARQUET_PATH = "data/microplastics.parquet"
_CSV_PATH = "data/microplastics.csv"

_FEATURE_COLS = ["latitude", "depth", "distance_to_coast", "year", "longitude"]


@st.cache_data
def _load_data() -> pd.DataFrame:
    """Load cleaned microplastics data; parquet if available, else CSV."""
    if os.path.exists(_PARQUET_PATH):
        return pd.read_parquet(_PARQUET_PATH)
    if not os.path.exists(_CSV_PATH):
        raise FileNotFoundError(_CSV_PATH)
    return load_microplastics(_CSV_PATH)


@st.cache_data(ttl=3600)
def _cached_basin_stats(df_hash: str, df: pd.DataFrame) -> pd.DataFrame:
    """Cached wrapper for compute_basin_statistics."""
    return compute_basin_statistics(df)


@st.cache_data(ttl=3600)
def _cached_temporal_trends(df_hash: str, df: pd.DataFrame) -> pd.DataFrame:
    """Cached wrapper for compute_temporal_trends."""
    return compute_temporal_trends(df)


@st.cache_data(ttl=3600)
def _cached_correlations(df_hash: str, df: pd.DataFrame, feature_cols: tuple) -> list:
    """Cached wrapper for compute_correlations."""
    return compute_correlations(df, list(feature_cols))


@st.cache_data(ttl=3600)
def _cached_cluster_data(df_hash: str, df: pd.DataFrame) -> pd.DataFrame:
    """Run DBSCAN and return a clustered copy of df; cached to avoid re-running on every interaction."""
    from sklearn.cluster import DBSCAN

    coords = df[["latitude", "longitude"]].values
    cluster_labels = DBSCAN(eps=3.0, min_samples=30).fit_predict(coords)
    result = df.copy()
    result["cluster"] = cluster_labels
    return result


def _df_hash(df: pd.DataFrame) -> str:
    """Stable cache key from DataFrame shape and column names."""
    return f"{df.shape}_{df.columns.tolist()}"


# ---------------------------------------------------------------------------
# Module-scope filtered_df — populated by render(), used by all tabs
# ---------------------------------------------------------------------------

filtered_df: pd.DataFrame = pd.DataFrame()


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Statistical Insights page."""
    global filtered_df

    try:
        df = _load_data()
    except FileNotFoundError:
        st.error(
            "Cleaned data not found. Please run the data pipeline from "
            "the Home page first."
        )
        st.stop()
        return

    # ------------------------------------------------------------------ #
    # Header
    # ------------------------------------------------------------------ #
    st.title("Statistical Insights")
    st.markdown(
        """
        This page explores the NOAA NCEI Marine Microplastics dataset through four
        complementary lenses: how concentration varies across **ocean basins**, how
        **sampling activity and measured density have changed over time**, where
        geographic **hotspot clusters** emerge using DBSCAN, and what **correlations**
        exist between density and spatial or temporal features.  Use the sidebar
        filters to narrow the analysis to specific basins or time windows — all charts
        update together.
        """
    )

    # ------------------------------------------------------------------ #
    # Sidebar filters
    # ------------------------------------------------------------------ #
    st.sidebar.header("Data Filters")

    all_oceans: list[str] = sorted(df["ocean"].dropna().unique().tolist())
    selected_oceans: list[str] = st.sidebar.multiselect(
        "Ocean basins",
        options=all_oceans,
        default=all_oceans,
    )

    year_min = int(df["year"].dropna().min())
    year_max = int(df["year"].dropna().max())
    year_range: tuple[int, int] = st.sidebar.slider(
        "Year range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
    )

    st.sidebar.caption("Filters apply to all charts on this page.")

    # ------------------------------------------------------------------ #
    # Apply filters — assign to module-scope filtered_df
    # ------------------------------------------------------------------ #
    filtered_df = df[
        df["ocean"].isin(selected_oceans)
        & (df["year"] >= year_range[0])
        & (df["year"] <= year_range[1])
    ].copy()

    st.sidebar.caption(
        f"Showing **{len(filtered_df):,}** of **{len(df):,}** observations"
    )

    if filtered_df.empty:
        st.warning("No data matches the current filters. Adjust the sidebar selections.")
        return

    # ------------------------------------------------------------------ #
    # Tabs
    # ------------------------------------------------------------------ #
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌊 Ocean Basins",
        "📅 Temporal Trends",
        "🗺️ Hotspot Clusters",
        "🔗 Correlations",
    ])

    # ---- Tab 1: Ocean Basins ----------------------------------------- #
    with tab1:
        with st.spinner("Crunching basin stats..."):
            basin_stats = _cached_basin_stats(_df_hash(filtered_df), filtered_df)
            basin_stats = basin_stats[basin_stats["count"] >= 10]

        if basin_stats.empty:
            st.warning("No basins with ≥10 observations match the current filters.")
        else:
            # ── FIX 5: Dynamic key-finding callout ──────────────────────
            worst_basin = basin_stats["mean"].idxmax()
            best_basin  = basin_stats["mean"].idxmin()
            worst_mean  = float(basin_stats.loc[worst_basin, "mean"])
            best_mean   = float(basin_stats.loc[best_basin,  "mean"])
            ratio       = worst_mean / best_mean if best_mean > 0 else float("nan")

            st.info(
                f"**Key finding:** The **{worst_basin}** has the highest mean microplastic "
                f"density at {_fmt_density(worst_mean)} — approximately **{ratio:.0f}×** higher "
                f"than the {best_basin} ({_fmt_density(best_mean)}). Values are shown on "
                f"a log scale because density spans several orders of magnitude across basins."
            )
            st.divider()

            # ── FIX 4: Chart + basin snapshot panel side by side ────────
            fig = build_basin_chart(basin_stats)
            col_chart, col_snapshot = st.columns([3, 2])

            with col_chart:
                st.plotly_chart(fig, use_container_width=True)

            with col_snapshot:
                st.subheader("Basin Snapshot")
                sorted_basins = basin_stats.sort_values("mean", ascending=False)

                for rank, basin in enumerate(sorted_basins.index, start=1):
                    row = sorted_basins.loc[basin]
                    basin_mean = float(row["mean"])
                    n_nonzero  = int(row["n_nonzero"]) if "n_nonzero" in row.index else int(row["count"])
                    total      = int(row["count"])
                    st.metric(
                        label=f"#{rank} {basin}",
                        value=_fmt_density(basin_mean),
                        delta=f"{n_nonzero:,} detections / {total:,} total",
                        delta_color="off",
                    )

                st.caption(
                    "Values shown are geometric mean density (non-zero observations only)."
                )

            # ── Raw stats expander ───────────────────────────────────────
            with st.expander("View raw basin statistics table"):
                display = basin_stats[["mean", "median", "p25", "p75", "count"]].copy().reset_index()
                display.insert(0, "Rank", range(1, len(display) + 1))
                display.columns = ["Rank", "Ocean Basin", "Mean", "Median", "Q25", "Q75", "Count"]
                st.dataframe(
                    display.style.format({
                        "Mean":   "{:.3g}",
                        "Median": "{:.3g}",
                        "Q25":    "{:.3g}",
                        "Q75":    "{:.3g}",
                        "Count":  "{:.0f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

    # ---- Tab 2: Temporal Trends -------------------------------------- #
    with tab2:
        st.warning(
            "⚠️ Interpretation note: The bar height shows how many samples were collected "
            "each year — not how much plastic exists. A spike in sample count reflects "
            "increased research activity. Compare the orange line (measured density) "
            "independently of the bars to assess real concentration trends."
        )
        st.divider()

        with st.spinner("Crunching temporal trends..."):
            temporal_df = _cached_temporal_trends(_df_hash(filtered_df), filtered_df)
            fig = build_temporal_chart(temporal_df)

        st.plotly_chart(fig, use_container_width=True)

        # Metrics
        peak_research_row = temporal_df.loc[temporal_df["observation_count"].idxmax()]
        peak_density_row = temporal_df.loc[temporal_df["mean_density"].idxmax()]

        peak_research_year = int(peak_research_row["year"])
        peak_density_year = int(peak_density_row["year"])

        # Delta = change from previous year in the sorted table
        temporal_sorted = temporal_df.sort_values("year").reset_index(drop=True)

        def _prev_year_delta(col: str, target_year: int) -> float | None:
            idx = temporal_sorted.index[temporal_sorted["year"] == target_year].tolist()
            if not idx or idx[0] == 0:
                return None
            return float(
                temporal_sorted.loc[idx[0], col] - temporal_sorted.loc[idx[0] - 1, col]
            )

        obs_delta = _prev_year_delta("observation_count", peak_research_year)
        density_delta = _prev_year_delta("mean_density", peak_density_year)

        col_left, col_right = st.columns(2)
        with col_left:
            st.metric(
                label="Peak Research Year",
                value=str(peak_research_year),
                delta=f"{int(obs_delta):+,} obs vs prior year" if obs_delta is not None else None,
            )
        with col_right:
            st.metric(
                label="Highest Mean Density Year",
                value=str(peak_density_year),
                delta=f"{density_delta:+.4f} pieces/m³ vs prior year" if density_delta is not None else None,
            )

    # ---- Tab 3: Hotspot Clusters ------------------------------------- #
    with tab3:
        st.info(
            "Clusters were identified using DBSCAN spatial clustering. Each numbered "
            "cluster corresponds to a region of elevated microplastic concentration. "
            "Grey points are ungrouped observations. The five major ocean gyres — "
            "natural circular current systems — are known accumulation zones."
        )
        st.divider()

        if "cluster" in filtered_df.columns:
            clustered_df = filtered_df
        else:
            with st.spinner("Running DBSCAN clustering..."):
                clustered_df = _cached_cluster_data(_df_hash(filtered_df), filtered_df)

        if "cluster" not in clustered_df.columns or clustered_df["cluster"].isna().all():
            st.warning(
                "Cluster data not available. Re-run the pipeline with "
                "clustering enabled."
            )
        else:
            cluster_summary = get_cluster_summary(clustered_df)
            fig = build_cluster_map(clustered_df, cluster_summary)

            col_map, col_table = st.columns([3, 2])

            with col_map:
                st.plotly_chart(fig, use_container_width=True)

            with col_table:
                st.subheader("Cluster Summary")
                loc_col = ["location"] if "location" in cluster_summary.columns else []
                display_cols = ["label"] + loc_col + ["center_lat", "center_lon", "observation_count", "mean_density"]
                summary_display = cluster_summary[display_cols].copy()
                summary_display.insert(0, "#", range(1, len(summary_display) + 1))
                rename_map = {
                    "label": "Cluster ID",
                    "location": "Nearest Landmark",
                    "center_lat": "Center Lat (°)",
                    "center_lon": "Center Lon (°)",
                    "observation_count": "Observations",
                    "mean_density": "Mean Density (pieces/m³)",
                }
                summary_display = summary_display.rename(columns=rename_map)

                top_obs_idx = summary_display["Observations"].idxmax()

                def _highlight_max(row):
                    return [
                        "background-color: #d0e8ff" if row.name == top_obs_idx else ""
                        for _ in row
                    ]

                st.dataframe(
                    summary_display.style
                        .apply(_highlight_max, axis=1)
                        .format({
                            "Center Lat (°)": "{:.2f}",
                            "Center Lon (°)": "{:.2f}",
                            "Mean Density (pieces/m³)": "{:.4f}",
                        }),
                    use_container_width=True,
                    hide_index=True,
                )

            cluster_labels_arr = clustered_df["cluster"].values
            n_clustered = int((cluster_labels_arr != -1).sum())
            n_noise = int((cluster_labels_arr == -1).sum())
            n_total = len(cluster_labels_arr)
            pct_clustered = 100 * n_clustered / n_total if n_total > 0 else 0.0

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Clustered Observations", f"{n_clustered:,}")
            with m2:
                st.metric("Noise (Ungrouped)", f"{n_noise:,}")
            with m3:
                st.metric("% Data Clustered", f"{pct_clustered:.1f}%")

    # ---- Tab 4: Correlations ----------------------------------------- #
    with tab4:
        st.info(
            "Spearman rank correlation measures the monotonic relationship between each "
            "feature and microplastic density. ρ near ±1 indicates strong correlation; "
            "near 0 indicates no relationship. P-values below 0.05 suggest the correlation "
            "is statistically significant."
        )
        st.divider()

        with st.spinner("Computing correlations..."):
            feature_cols = [c for c in _FEATURE_COLS if c in filtered_df.columns]
            correlations = _cached_correlations(
                _df_hash(filtered_df), filtered_df, tuple(feature_cols)
            )
            figs = build_correlation_charts(filtered_df, correlations, feature_cols)

        # Compact summary table — color-coded by effect size
        if correlations:
            summary_records = [
                {
                    "Feature": c["feature"],
                    "ρ (Spearman)": c["rho"],
                    "p-value": c["p_value"],
                    "n": c["n"],
                    "Significant?": "✅ Yes" if c["p_value"] < 0.05 else "❌ No",
                }
                for c in correlations
            ]
            corr_summary_df = pd.DataFrame(summary_records)

            def _style_correlation_table(df_in: pd.DataFrame):
                def _color_rho(val: float) -> str:
                    av = abs(val)
                    if av > 0.3:
                        return "color: #27ae60; font-weight: bold"
                    elif av > 0.1:
                        return "color: #e67e22; font-weight: bold"
                    return "color: #7f8c8d"

                return (
                    df_in.style
                    .applymap(_color_rho, subset=["ρ (Spearman)"])
                    .format({
                        "ρ (Spearman)": "{:+.4f}",
                        "p-value": lambda v: "< 0.0001" if v < 0.0001 else f"{v:.4f}",
                        "n": "{:,}",
                    })
                    .set_properties(**{"text-align": "left"})
                    .hide(axis="index")
                )

            st.dataframe(
                _style_correlation_table(corr_summary_df),
                use_container_width=True,
            )
        else:
            st.warning("No valid feature columns found for correlation analysis.")

        # Charts — smart layout based on count
        n_figs = len(figs)
        if n_figs == 0:
            pass
        elif n_figs == 1:
            st.plotly_chart(figs[0], use_container_width=True)
        elif n_figs == 2:
            c1, c2 = st.columns(2)
            c1.plotly_chart(figs[0], use_container_width=True)
            c2.plotly_chart(figs[1], use_container_width=True)
        elif n_figs == 3:
            c1, c2 = st.columns(2)
            c1.plotly_chart(figs[0], use_container_width=True)
            c2.plotly_chart(figs[1], use_container_width=True)
            _, c_center, _ = st.columns([1, 2, 1])
            c_center.plotly_chart(figs[2], use_container_width=True)
        else:
            for i in range(0, n_figs, 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < n_figs:
                        col.plotly_chart(figs[i + j], use_container_width=True)
