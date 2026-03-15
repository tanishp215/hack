"""Streamlit page for the global microplastics observations dashboard."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis import (
    DEFAULT_HOTSPOT_POLYGONS_FILENAME,
    validate_hotspot_polygons_payload,
)

PAGE_TITLE = "Global Observations Map"
REPO_ROOT = Path(__file__).resolve().parents[1]
OCEAN_BASINS = ("Atlantic", "Pacific", "Indian", "Arctic", "Southern")
REQUIRED_MAP_COLUMNS = (
    "latitude",
    "longitude",
    "density",
    "date",
    "ocean",
    "sampling_method",
)
OBSERVATION_COLUMN_ALIASES = {
    "latitude": ("latitude_(degree)", "lat"),
    "longitude": ("longitude_(degree)", "lon"),
    "density": ("microplastics_measurement", "measurement"),
    "date": ("sample_date", "observation_date"),
    "ocean": ("ocean_basin",),
    "sampling_method": ("marine_setting", "sampling"),
}
DEFAULT_DATASET_CANDIDATES = (
    REPO_ROOT / "data" / "NOAA.csv",
    REPO_ROOT / "data" / "NOAA.parquet",
    REPO_ROOT / "data" / "microplastics_cleaned.parquet",
    REPO_ROOT / "data" / "microplastics_cleaned.csv",
    REPO_ROOT / "data" / "microplastics.csv",
)
DEFAULT_HOTSPOT_BOUNDARY_CANDIDATES = (
    REPO_ROOT / "results" / DEFAULT_HOTSPOT_POLYGONS_FILENAME,
    REPO_ROOT / "data" / "hotspots" / DEFAULT_HOTSPOT_POLYGONS_FILENAME,
    REPO_ROOT / "data" / "processed" / DEFAULT_HOTSPOT_POLYGONS_FILENAME,
    REPO_ROOT / DEFAULT_HOTSPOT_POLYGONS_FILENAME,
)
DENSITY_COLOR_SCALE = (
    (0.0, "#27d3ff"),
    (0.25, "#3ba4ff"),
    (0.5, "#78e08f"),
    (0.75, "#ffd166"),
    (1.0, "#ff6b6b"),
)
MAP_BACKGROUND_COLOR = "#07131f"
TEXT_COLOR = "#e6eef8"
MUTED_TEXT_COLOR = "#91a7c0"
FONT_FAMILY = "'IBM Plex Sans', 'Segoe UI', sans-serif"
EMPTY_MAP_CENTER = {"lat": 15.0, "lon": 0.0}
EMPTY_MAP_ZOOM = 0.8
MAP_TILE_SOURCE = [
    "https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
]
MAP_LABEL_TILE_SOURCE = [
    "https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
]
MAP_TILE_ATTRIBUTION = "Source: Esri, HERE, Garmin, FAO, NOAA, USGS"
DENSITY_TICK_CANDIDATES = (
    0.0,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1_000.0,
    10_000.0,
    100_000.0,
    1_000_000.0,
)
PLOTLY_CHART_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


@dataclass(frozen=True)
class ObservationsFilterState:
    """Sidebar filter selections for the observations dashboard."""

    selected_oceans: tuple[str, ...]
    year_range: tuple[int, int]
    show_hotspot_clusters: bool


def resolve_dataset_path(
    candidate_paths: Sequence[str | Path] | None = None,
) -> Path:
    """Return the first existing cleaned observations dataset path."""

    candidates = candidate_paths or DEFAULT_DATASET_CANDIDATES
    return _resolve_first_existing_path(
        candidates,
        missing_message=(
            "No cleaned observations dataset was found. Expected a CSV or parquet file in "
            "one of the standard data directories."
        ),
    )


@st.cache_data(show_spinner=False)
def load_observations_data(dataset_path: str) -> pd.DataFrame:
    """Load the cleaned observations dataset from a CSV or parquet file."""

    return _read_tabular_dataset(Path(dataset_path))


def resolve_hotspot_boundaries_path(
    candidate_paths: Sequence[str | Path] | None = None,
) -> Path:
    """Return the first existing precomputed hotspot boundary file path."""

    candidates = candidate_paths or DEFAULT_HOTSPOT_BOUNDARY_CANDIDATES
    return _resolve_first_existing_path(
        candidates,
        missing_message=(
            "No precomputed hotspot boundary file was found. Run the offline hotspot "
            "precompute step before enabling the overlay."
        ),
    )


@st.cache_data(show_spinner=False)
def load_hotspot_boundary_polygons(boundaries_path: str) -> list[dict[str, Any]]:
    """Load precomputed hotspot boundary polygons from a JSON file."""

    boundaries_file = Path(boundaries_path)
    if not boundaries_file.exists():
        raise FileNotFoundError(f"Hotspot boundaries file not found: {boundaries_file}")

    with boundaries_file.open("r", encoding="utf-8") as boundaries_handle:
        payload = json.load(boundaries_handle)

    return validate_hotspot_polygons_payload(payload)


def validate_observations_dataframe(
    df: pd.DataFrame,
    *,
    required_columns: Iterable[str] = REQUIRED_MAP_COLUMNS,
) -> pd.DataFrame:
    """Return the observations DataFrame after required-column validation."""

    if df is None:
        raise ValueError("Observations data could not be loaded.")

    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(
            "The observations dataset is missing required columns for the map: "
            f"{missing_text}."
        )

    return df


def prepare_observations_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the cleaned observations DataFrame for filtering and mapping."""

    prepared_df = _normalize_input_columns(df)
    prepared_df = _coalesce_observation_columns(prepared_df)
    validate_observations_dataframe(prepared_df)

    prepared_df["latitude"] = pd.to_numeric(prepared_df["latitude"], errors="coerce")
    prepared_df["longitude"] = pd.to_numeric(prepared_df["longitude"], errors="coerce")
    prepared_df["density"] = pd.to_numeric(prepared_df["density"], errors="coerce")
    prepared_df["date"] = _parse_observation_dates(prepared_df["date"])
    prepared_df["ocean"] = _safe_text_series(prepared_df["ocean"], default="")
    prepared_df["sampling_method"] = _safe_text_series(
        prepared_df["sampling_method"],
        default="Unknown",
    )
    prepared_df["year"] = prepared_df["date"].dt.year.astype("Int64")
    prepared_df["ocean_basin"] = prepared_df["ocean"].map(_normalize_ocean_basin).astype("string")

    prepared_df = prepared_df.dropna(
        subset=["latitude", "longitude", "density", "date", "year"]
    ).copy()
    prepared_df = prepared_df.loc[prepared_df["ocean_basin"].notna()].reset_index(drop=True)

    if prepared_df.empty:
        raise ValueError(
            "The observations dataset does not contain any valid rows after coordinate, "
            "date, and ocean-basin validation."
        )

    return prepared_df


def build_sidebar_filters(df: pd.DataFrame) -> ObservationsFilterState:
    """Render sidebar controls and return the selected filter state."""

    available_basins = [basin for basin in OCEAN_BASINS if basin in set(df["ocean_basin"].dropna())]
    if not available_basins:
        available_basins = list(OCEAN_BASINS)

    year_min, year_max = _extract_year_bounds(df)

    with st.sidebar:
        st.header("Map Controls")
        st.caption("Refine the observations view for the demo map.")

        selected_oceans = st.multiselect(
            "Ocean basins",
            options=list(OCEAN_BASINS),
            default=available_basins,
            help="Choose the major ocean basins to display on the map.",
        )

        if year_min == year_max:
            st.caption(f"Observation year: {year_min}")
            selected_year_range = (year_min, year_max)
        else:
            selected_year_range = st.slider(
                "Observation year range",
                min_value=year_min,
                max_value=year_max,
                value=(year_min, year_max),
                help="Limit observations to a specific range of observation years.",
            )

        show_hotspot_clusters = st.checkbox(
            "Show hotspot clusters",
            value=False,
            help="Overlay precomputed hotspot boundaries on top of the observation map.",
        )

    return ObservationsFilterState(
        selected_oceans=tuple(selected_oceans),
        year_range=selected_year_range,
        show_hotspot_clusters=show_hotspot_clusters,
    )


def apply_filters(df: pd.DataFrame, filter_state: ObservationsFilterState) -> pd.DataFrame:
    """Apply the sidebar filters to the observations DataFrame."""

    filtered_df = _apply_ocean_basin_filter(df, filter_state.selected_oceans)
    filtered_df = _apply_year_range_filter(filtered_df, filter_state.year_range)
    return filtered_df.reset_index(drop=True)


def build_figure(
    observations_df: pd.DataFrame,
    *,
    hotspot_polygons: list[dict[str, Any]] | None = None,
) -> go.Figure:
    """Build the interactive observations map figure."""

    if observations_df.empty:
        return _build_empty_state_figure(
            "No observations match the current filters. Try widening the year range or "
            "selecting additional ocean basins."
        )

    display_df = _build_map_display_frame(observations_df)
    figure = go.Figure()

    if hotspot_polygons:
        add_hotspot_boundary_traces(figure, hotspot_polygons)

    _add_observation_trace(figure, display_df)
    _configure_map_layout(figure, display_df)

    return figure


def add_hotspot_boundary_traces(
    figure: go.Figure,
    hotspot_polygons: Sequence[dict[str, Any]],
) -> None:
    """Add precomputed hotspot boundary overlays to the map figure."""

    for polygon_record in hotspot_polygons:
        polygon_points = polygon_record.get("polygon", [])
        if len(polygon_points) < 3:
            continue

        longitudes = [point["longitude"] for point in polygon_points]
        latitudes = [point["latitude"] for point in polygon_points]
        cluster_label = polygon_record.get("cluster_label", "unknown")
        point_count = polygon_record.get("point_count", "unknown")

        figure.add_trace(
            go.Scattermapbox(
                lat=latitudes,
                lon=longitudes,
                mode="lines",
                fill="toself",
                fillcolor="rgba(39, 211, 255, 0.16)",
                line={"color": "rgba(80, 227, 255, 0.90)", "width": 2},
                hovertemplate=(
                    "<b>Hotspot Cluster %{customdata[0]}</b><br>"
                    "Clustered observations: %{customdata[1]}<extra></extra>"
                ),
                customdata=[[cluster_label, point_count] for _ in polygon_points],
                name=f"Hotspot {cluster_label}",
                showlegend=False,
            )
        )


def render_summary_section(
    filtered_df: pd.DataFrame,
    filter_state: ObservationsFilterState,
) -> None:
    """Render the summary metrics above the observations map."""

    st.markdown("### Filtered Summary")
    first_column, second_column, third_column = st.columns(3)

    _render_summary_card(
        first_column,
        title="Filtered observations",
        value=f"{len(filtered_df):,}",
        subtitle="Mapped NOAA microplastics records",
    )
    _render_summary_card(
        second_column,
        title="Year range",
        value=_format_year_range(filter_state.year_range),
        subtitle="Observation window currently displayed",
    )
    _render_summary_card(
        third_column,
        title="Ocean basins",
        value=_format_selected_basins(filter_state.selected_oceans),
        subtitle="Major basins included in the current view",
    )


def render_page_header() -> None:
    """Render the title and intro copy for the observations page."""

    st.title(PAGE_TITLE)
    st.caption("NOAA marine microplastics observations")
    st.write(
        "Each point represents a NOAA microplastics observation colored by measured "
        "density, making it easier to compare where concentrations appear strongest "
        "across the global ocean."
    )


def render_map_section(figure: go.Figure) -> None:
    """Render the observations map section."""

    st.markdown("### Observations Map")
    st.plotly_chart(
        figure,
        use_container_width=True,
        config=PLOTLY_CHART_CONFIG,
    )


def render_page() -> None:
    """Render the full observations dashboard page."""

    _inject_page_styles()
    render_page_header()

    try:
        prepared_observations = _load_prepared_observations_dataset()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Unable to load the observations dataset: {exc}")
        return

    filter_state = build_sidebar_filters(prepared_observations)
    filtered_observations = apply_filters(prepared_observations, filter_state)

    hotspot_polygons, hotspot_message = _load_optional_hotspot_polygons(
        filter_state.show_hotspot_clusters
    )
    if hotspot_message:
        st.info(hotspot_message)

    render_summary_section(filtered_observations, filter_state)

    if filtered_observations.empty:
        st.info(
            "No observations match the current filters. Adjust the year range or select "
            "additional ocean basins to repopulate the map."
        )

    figure = build_figure(filtered_observations, hotspot_polygons=hotspot_polygons)
    render_map_section(figure)


def render() -> None:
    """Render the observations page for the main app navigation."""

    render_page()


def _resolve_first_existing_path(
    candidate_paths: Sequence[str | Path],
    *,
    missing_message: str,
) -> Path:
    """Return the first existing path from a sequence of candidate paths."""

    resolved_candidates = [Path(candidate_path) for candidate_path in candidate_paths]
    for candidate_path in resolved_candidates:
        if candidate_path.exists():
            return candidate_path

    searched_paths = "\n".join(f"- {candidate_path}" for candidate_path in resolved_candidates)
    raise FileNotFoundError(f"{missing_message}\nSearched:\n{searched_paths}")


def _load_prepared_observations_dataset() -> pd.DataFrame:
    """Load and prepare the cleaned observations dataset for the dashboard."""

    dataset_path = resolve_dataset_path()
    raw_observations = load_observations_data(str(dataset_path))
    return prepare_observations_data(raw_observations)


def _load_optional_hotspot_polygons(
    show_hotspot_clusters: bool,
) -> tuple[list[dict[str, Any]], str | None]:
    """Load hotspot polygons when the overlay is enabled."""

    if not show_hotspot_clusters:
        return [], None

    try:
        hotspot_path = resolve_hotspot_boundaries_path()
        return load_hotspot_boundary_polygons(str(hotspot_path)), None
    except FileNotFoundError:
        return [], (
            "Hotspot overlay is unavailable because no precomputed hotspot boundary file "
            "was found."
        )
    except ValueError as exc:
        return [], f"Hotspot overlay was skipped because the boundary file is invalid: {exc}"
    except Exception as exc:  # pragma: no cover - defensive UI guard
        return [], f"Hotspot overlay was skipped because it could not be loaded: {exc}"


def _read_tabular_dataset(dataset_path: Path) -> pd.DataFrame:
    """Read a CSV or parquet dataset from disk."""

    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(dataset_path)
    if suffix == ".parquet":
        return pd.read_parquet(dataset_path)

    raise ValueError(f"Unsupported observations dataset format: {dataset_path.suffix}")


def _normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the input DataFrame with normalized column names."""

    normalized_df = df.copy()
    normalized_df.columns = [_normalize_column_name(column) for column in normalized_df.columns]
    return normalized_df


def _normalize_column_name(column: object) -> str:
    """Return a normalized snake-case representation of a column name."""

    return (
        str(column)
        .strip()
        .lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _coalesce_observation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill canonical observation columns from known source-column aliases."""

    prepared_df = df.copy()

    for canonical_column, aliases in OBSERVATION_COLUMN_ALIASES.items():
        candidate_columns = [
            column_name
            for column_name in (canonical_column, *aliases)
            if column_name in prepared_df.columns
        ]
        if not candidate_columns:
            continue

        coalesced_series: pd.Series | None = None
        for candidate_column in candidate_columns:
            candidate_series = _nullify_blank_strings(prepared_df[candidate_column])
            if coalesced_series is None:
                coalesced_series = candidate_series
            else:
                coalesced_series = coalesced_series.where(
                    coalesced_series.notna(),
                    candidate_series,
                )

        if coalesced_series is not None:
            prepared_df[canonical_column] = coalesced_series

    return prepared_df


def _nullify_blank_strings(series: pd.Series) -> pd.Series:
    """Return a series where empty strings are treated as missing values."""

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        text_series = series.astype("string").str.strip()
        return text_series.replace("", pd.NA)
    return series


def _parse_observation_dates(series: pd.Series) -> pd.Series:
    """Parse observation dates while handling NOAA-style timestamp strings."""

    parsed_dates = pd.to_datetime(
        series,
        errors="coerce",
        format="%m/%d/%Y %I:%M:%S %p",
    )
    remaining_mask = parsed_dates.isna() & series.notna()
    if remaining_mask.any():
        parsed_dates.loc[remaining_mask] = pd.to_datetime(
            series.loc[remaining_mask],
            errors="coerce",
        )

    return parsed_dates


def _safe_text_series(series: pd.Series, *, default: str) -> pd.Series:
    """Return a trimmed string series with empty values replaced by a default."""

    text_series = series.astype("string").str.strip()
    text_series = text_series.where(text_series.notna() & text_series.ne(""), default)
    return text_series


def _normalize_ocean_basin(value: object) -> str | None:
    """Normalize raw ocean labels to one of the supported major ocean basins."""

    if pd.isna(value):
        return None

    normalized_value = str(value).strip().lower()
    for basin in OCEAN_BASINS:
        if basin.lower() in normalized_value:
            return basin

    return None


def _extract_year_bounds(df: pd.DataFrame) -> tuple[int, int]:
    """Return the inclusive year bounds from a prepared observations DataFrame."""

    valid_years = df["year"].dropna().astype(int)
    if valid_years.empty:
        raise ValueError("The observations dataset does not contain any valid years.")

    return int(valid_years.min()), int(valid_years.max())


def _apply_ocean_basin_filter(
    df: pd.DataFrame,
    selected_oceans: Sequence[str],
) -> pd.DataFrame:
    """Return observations filtered to the selected ocean basins."""

    if not selected_oceans:
        return df.iloc[0:0].copy()

    return df.loc[df["ocean_basin"].isin(selected_oceans)].copy()


def _apply_year_range_filter(
    df: pd.DataFrame,
    year_range: tuple[int, int],
) -> pd.DataFrame:
    """Return observations filtered to the selected observation year range."""

    year_start, year_end = year_range
    year_series = df["year"].astype(int)
    return df.loc[year_series.between(year_start, year_end)].copy()


def _build_map_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a plotting-ready DataFrame with formatted hover values."""

    display_df = df.copy()
    display_df["density_display"] = display_df["density"].map(_format_density_value)
    display_df["date_display"] = display_df["date"].dt.strftime("%b %d, %Y").fillna("Unknown")
    display_df["ocean_display"] = _safe_text_series(display_df["ocean"], default="Unknown")
    display_df["sampling_method_display"] = _safe_text_series(
        display_df["sampling_method"],
        default="Unknown",
    )
    return display_df


def _add_observation_trace(figure: go.Figure, display_df: pd.DataFrame) -> None:
    """Add the main observations point layer to the map figure."""

    density_scale = _build_density_color_scale(display_df["density"])
    customdata = list(
        zip(
            display_df["density_display"],
            display_df["date_display"],
            display_df["ocean_display"],
            display_df["sampling_method_display"],
            strict=False,
        )
    )

    figure.add_trace(
        go.Scattermapbox(
            lat=display_df["latitude"],
            lon=display_df["longitude"],
            mode="markers",
            customdata=customdata,
            marker={
                "size": 8,
                "opacity": 0.72,
                "allowoverlap": True,
                "color": density_scale["color_values"],
                "colorscale": DENSITY_COLOR_SCALE,
                "cmin": density_scale["cmin"],
                "cmax": density_scale["cmax"],
                "colorbar": {
                    "title": {
                        "text": (
                            "Measured Density"
                            "<br><sup>pieces/m3, log scale, capped at 90th percentile</sup>"
                        )
                    },
                    "tickvals": density_scale["tickvals"],
                    "ticktext": density_scale["ticktext"],
                    "thickness": 16,
                    "len": 0.78,
                    "bgcolor": "rgba(7, 19, 31, 0.65)",
                    "outlinecolor": "rgba(145, 167, 192, 0.40)",
                },
            },
            hovertemplate=(
                "<b>Microplastics Observation</b><br>"
                "Density: %{customdata[0]}<br>"
                "Date: %{customdata[1]}<br>"
                "Ocean: %{customdata[2]}<br>"
                "Sampling method: %{customdata[3]}<extra></extra>"
            ),
            name="Observations",
            showlegend=False,
        )
    )


def _configure_map_layout(figure: go.Figure, display_df: pd.DataFrame) -> None:
    """Apply the shared dark dashboard styling to the map figure."""

    figure.update_layout(
        **_build_map_layout(
            center=_compute_map_center(display_df),
            zoom=_estimate_map_zoom(display_df),
        )
    )


def _build_empty_state_figure(message: str) -> go.Figure:
    """Return a dark themed empty-state figure for the observations map."""

    figure = go.Figure()
    figure.update_layout(
        **_build_map_layout(center=EMPTY_MAP_CENTER, zoom=EMPTY_MAP_ZOOM),
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 16, "color": TEXT_COLOR},
                "align": "center",
            }
        ],
    )
    return figure


def _build_map_layout(
    *,
    center: dict[str, float],
    zoom: float,
) -> dict[str, Any]:
    """Return the shared dark map layout configuration."""

    return {
        "mapbox": {
            "style": "white-bg",
            "center": center,
            "zoom": zoom,
            "layers": [
                {
                    "below": "traces",
                    "sourcetype": "raster",
                    "sourceattribution": MAP_TILE_ATTRIBUTION,
                    "source": MAP_TILE_SOURCE,
                },
                {
                    "below": "traces",
                    "sourcetype": "raster",
                    "sourceattribution": MAP_TILE_ATTRIBUTION,
                    "source": MAP_LABEL_TILE_SOURCE,
                }
            ],
        },
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        "height": 720,
        "paper_bgcolor": MAP_BACKGROUND_COLOR,
        "plot_bgcolor": MAP_BACKGROUND_COLOR,
        "font": {"color": TEXT_COLOR, "family": FONT_FAMILY},
        "hoverlabel": {
            "bgcolor": "rgba(7, 19, 31, 0.96)",
            "bordercolor": "rgba(145, 167, 192, 0.55)",
            "font": {"color": TEXT_COLOR, "size": 12},
            "align": "left",
        },
    }


def _compute_map_center(df: pd.DataFrame) -> dict[str, float]:
    """Return a sensible map center for the current filtered observations."""

    if df.empty:
        return EMPTY_MAP_CENTER

    return {
        "lat": float(df["latitude"].median()),
        "lon": float(df["longitude"].median()),
    }


def _estimate_map_zoom(df: pd.DataFrame) -> float:
    """Estimate a readable map zoom level from the filtered observations extent."""

    if df.empty:
        return EMPTY_MAP_ZOOM

    lat_span = float(df["latitude"].max() - df["latitude"].min())
    lon_span = float(df["longitude"].max() - df["longitude"].min())
    max_span = max(lat_span, lon_span)

    if max_span >= 120:
        return 0.8
    if max_span >= 75:
        return 1.2
    if max_span >= 40:
        return 1.8
    if max_span >= 20:
        return 2.4
    if max_span >= 10:
        return 3.2
    if max_span >= 5:
        return 4.1
    return 5.2


def _render_summary_card(
    column: Any,
    *,
    title: str,
    value: str,
    subtitle: str,
) -> None:
    """Render one summary card using a dark dashboard style."""

    column.markdown(
        f"""
        <div style="
            background: linear-gradient(160deg, rgba(13, 27, 42, 0.95), rgba(7, 19, 31, 0.98));
            border: 1px solid rgba(120, 224, 143, 0.18);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            min-height: 130px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.20);
        ">
            <div style="color: {MUTED_TEXT_COLOR}; font-size: 0.9rem; margin-bottom: 0.35rem;">
                {title}
            </div>
            <div style="color: {TEXT_COLOR}; font-size: 1.9rem; font-weight: 700; line-height: 1.1;">
                {value}
            </div>
            <div style="color: {MUTED_TEXT_COLOR}; font-size: 0.88rem; margin-top: 0.55rem;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_density_color_scale(density_series: pd.Series) -> dict[str, Any]:
    """Return clipped log-scaled colors and a readable density legend."""

    density_values = density_series.astype(float)
    positive_values = density_values.loc[density_values > 0]

    if positive_values.empty:
        floor_density = 1e-6
        upper_density = 1.0
    else:
        floor_density = max(float(positive_values.min()) / 2.0, 1e-6)
        upper_density = float(positive_values.quantile(0.90))
        upper_density = max(upper_density, float(positive_values.median()), floor_density * 10.0)

    clipped_density = density_values.clip(lower=0.0, upper=upper_density)
    color_values = np.log10(clipped_density + floor_density)
    cmin = float(np.log10(floor_density))
    cmax = float(np.log10(upper_density + floor_density))

    tick_values = [
        tick_value
        for tick_value in DENSITY_TICK_CANDIDATES
        if tick_value <= upper_density * 1.01
    ]
    if 0.0 not in tick_values:
        tick_values.insert(0, 0.0)
    if upper_density > tick_values[-1]:
        tick_values.append(upper_density)

    tickvals = [float(np.log10(tick_value + floor_density)) for tick_value in tick_values]
    ticktext = [_format_density_value(tick_value) for tick_value in tick_values]

    return {
        "color_values": color_values,
        "cmin": cmin,
        "cmax": cmax,
        "tickvals": tickvals,
        "ticktext": ticktext,
        "upper_density": upper_density,
    }


def _format_density_value(value: float) -> str:
    """Return a compact human-readable density label."""

    numeric_value = float(value)
    if numeric_value == 0:
        return "0"
    if abs(numeric_value) >= 1_000:
        return f"{numeric_value:,.0f}"
    if abs(numeric_value) >= 1:
        return f"{numeric_value:,.2f}"
    if abs(numeric_value) >= 0.01:
        return f"{numeric_value:,.3f}"
    return f"{numeric_value:.2e}"


def _format_year_range(year_range: tuple[int, int]) -> str:
    """Return a human-readable year-range label."""

    start_year, end_year = year_range
    if start_year == end_year:
        return str(start_year)
    return f"{start_year} - {end_year}"


def _format_selected_basins(selected_oceans: Sequence[str]) -> str:
    """Return a compact label for the selected ocean basin filters."""

    if not selected_oceans:
        return "None selected"
    if len(selected_oceans) == len(OCEAN_BASINS):
        return "All major basins"
    return ", ".join(selected_oceans)


def _inject_page_styles() -> None:
    """Apply lightweight dashboard styling for the observations page."""

    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #08121d 0%, #0d1b2a 100%);
        }
        div[data-testid="stSidebarUserContent"] h1,
        div[data-testid="stSidebarUserContent"] h2,
        div[data-testid="stSidebarUserContent"] label,
        div[data-testid="stSidebarUserContent"] p,
        div[data-testid="stSidebarUserContent"] span {
            color: #e6eef8;
        }
        div.block-container {
            padding-top: 2rem;
            padding-bottom: 2.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render_page()
