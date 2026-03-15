"""Hotspot clustering helpers for microplastics observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import DBSCAN

EARTH_RADIUS_KM = 6371.0088
DEFAULT_HOTSPOT_POINTS_BASENAME = "hotspot_points"
DEFAULT_HOTSPOT_POLYGONS_FILENAME = "hotspot_boundaries.json"


def filter_high_density_observations(
    df: pd.DataFrame,
    density_percentile: float = 75.0,
) -> pd.DataFrame:
    """Return the highest-density observations above a percentile threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned observations DataFrame that includes a numeric ``density``
        column.
    density_percentile : float, default 75.0
        Percentile cutoff used to define hotspot candidates. ``75.0`` keeps
        the top 25 percent of observations by density.

    Returns
    -------
    pd.DataFrame
        Copy of the filtered high-density observations.
    """

    if "density" not in df.columns:
        raise ValueError("Expected a 'density' column in the observations DataFrame.")
    if not 0 <= density_percentile <= 100:
        raise ValueError("density_percentile must be between 0 and 100.")

    density_threshold = df["density"].quantile(density_percentile / 100.0)
    high_density_df = df.loc[df["density"] >= density_threshold].copy()
    high_density_df["density_percentile_threshold"] = density_threshold

    return high_density_df.reset_index(drop=True)


def cluster_hotspot_observations(
    df: pd.DataFrame,
    density_percentile: float = 75.0,
    eps_km: float = 250.0,
    min_samples: int = 3,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Cluster high-density observations and build cluster boundary polygons.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned observations DataFrame with ``latitude``, ``longitude``, and
        ``density`` columns.
    density_percentile : float, default 75.0
        Percentile cutoff used to keep only high-density observations.
    eps_km : float, default 250.0
        DBSCAN neighborhood radius in kilometers.
    min_samples : int, default 3
        Minimum observations required for a dense cluster.

    Returns
    -------
    tuple[pd.DataFrame, list[dict[str, Any]]]
        Two values:
        - DataFrame of clustered hotspot points with a ``cluster_label`` column
        - Serializable list of boundary polygon dictionaries for each cluster
    """

    required_columns = {"latitude", "longitude", "density"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Observations DataFrame is missing required columns for clustering: "
            f"{', '.join(missing_columns)}."
        )
    if eps_km <= 0:
        raise ValueError("eps_km must be positive.")
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1.")

    high_density_df = filter_high_density_observations(
        df,
        density_percentile=density_percentile,
    )
    if high_density_df.empty:
        return high_density_df.assign(cluster_label=pd.Series(dtype="int64")), []

    coordinates = high_density_df[["latitude", "longitude"]].to_numpy(dtype=float)

    # Haversine distance expects coordinates in radians.
    clustering_model = DBSCAN(
        eps=eps_km / EARTH_RADIUS_KM,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    )
    cluster_labels = clustering_model.fit_predict(np.radians(coordinates))

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
    """Build serializable boundary polygons for each hotspot cluster.

    Parameters
    ----------
    hotspot_points : pd.DataFrame
        Clustered hotspot observations that include ``cluster_label``,
        ``latitude``, and ``longitude`` columns. Noise points should already be
        removed.

    Returns
    -------
    list[dict[str, Any]]
        One serializable dictionary per cluster containing polygon coordinates
        and light metadata for map overlays.
    """

    if hotspot_points.empty:
        return []

    required_columns = {"cluster_label", "latitude", "longitude"}
    missing_columns = sorted(required_columns - set(hotspot_points.columns))
    if missing_columns:
        raise ValueError(
            "Hotspot points are missing required columns for polygon creation: "
            f"{', '.join(missing_columns)}."
        )

    polygons: list[dict[str, Any]] = []

    for cluster_label, cluster_df in hotspot_points.groupby("cluster_label", sort=True):
        if cluster_label == -1:
            continue

        cluster_points = cluster_df[["longitude", "latitude"]].to_numpy(dtype=float)
        ring = _build_cluster_ring(cluster_points)

        polygons.append(
            {
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
        )

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
    """Compute hotspot clusters and save reusable artifacts to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned observations DataFrame used as input for hotspot clustering.
    output_dir : path-like
        Directory where the hotspot artifacts should be written.
    density_percentile : float, default 75.0
        Percentile cutoff used to keep the densest observations.
    eps_km : float, default 250.0
        DBSCAN neighborhood radius in kilometers.
    min_samples : int, default 3
        Minimum number of observations required to form a dense cluster.
    points_format : {"parquet", "csv"}, default "parquet"
        File format used for the hotspot points artifact. Parquet is the
        default because it preserves types cleanly for Streamlit reloads.

    Returns
    -------
    dict[str, Path]
        Paths to the saved hotspot points and polygon artifacts.
    """

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

    if points_format == "parquet":
        hotspot_points.to_parquet(points_path, index=False)
    elif points_format == "csv":
        hotspot_points.to_csv(points_path, index=False)
    else:
        raise ValueError("points_format must be either 'parquet' or 'csv'.")

    polygons_payload = {
        "version": 1,
        "density_percentile": density_percentile,
        "eps_km": eps_km,
        "min_samples": min_samples,
        "cluster_count": len(boundary_polygons),
        "polygons": boundary_polygons,
    }
    with polygons_path.open("w", encoding="utf-8") as polygons_file:
        json.dump(polygons_payload, polygons_file, indent=2)

    return {
        "points_path": points_path,
        "polygons_path": polygons_path,
    }


def load_precomputed_hotspot_clusters(
    output_dir: str | Path,
    *,
    points_format: str = "parquet",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Load precomputed hotspot points and boundary polygons from disk.

    Parameters
    ----------
    output_dir : path-like
        Directory containing the saved hotspot artifacts.
    points_format : {"parquet", "csv"}, default "parquet"
        File format used for the hotspot points artifact.

    Returns
    -------
    tuple[pd.DataFrame, list[dict[str, Any]]]
        The hotspot points DataFrame and the list of boundary polygons.
    """

    output_dir = Path(output_dir)
    points_path = _build_hotspot_points_path(output_dir, points_format)
    polygons_path = output_dir / DEFAULT_HOTSPOT_POLYGONS_FILENAME

    if points_format == "parquet":
        hotspot_points = pd.read_parquet(points_path)
    elif points_format == "csv":
        hotspot_points = pd.read_csv(points_path)
    else:
        raise ValueError("points_format must be either 'parquet' or 'csv'.")

    with polygons_path.open("r", encoding="utf-8") as polygons_file:
        polygons_payload = json.load(polygons_file)

    return hotspot_points, polygons_payload.get("polygons", [])


def _build_cluster_ring(cluster_points: np.ndarray) -> np.ndarray:
    """Return a closed polygon ring for a cluster of ``[lon, lat]`` points."""

    if len(cluster_points) < 3:
        return _bounding_box_ring(cluster_points)

    try:
        hull = ConvexHull(cluster_points)
        hull_points = cluster_points[hull.vertices]
    except QhullError:
        # Collinear or duplicate-heavy clusters still get a visible polygon.
        return _bounding_box_ring(cluster_points)

    return _close_ring(hull_points)


def _bounding_box_ring(cluster_points: np.ndarray) -> np.ndarray:
    """Return a closed rectangular ring that encloses the cluster points."""

    min_longitude, min_latitude = cluster_points.min(axis=0)
    max_longitude, max_latitude = cluster_points.max(axis=0)

    # A tiny padding avoids zero-area polygons for duplicated coordinates.
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
    """Return polygon coordinates with the starting point appended at the end."""

    if np.array_equal(points[0], points[-1]):
        return points

    return np.vstack([points, points[0]])


def _build_hotspot_points_path(output_dir: Path, points_format: str) -> Path:
    """Return the hotspot points artifact path for the requested format."""

    normalized_format = points_format.lower()
    if normalized_format not in {"parquet", "csv"}:
        raise ValueError("points_format must be either 'parquet' or 'csv'.")

    return output_dir / f"{DEFAULT_HOTSPOT_POINTS_BASENAME}.{normalized_format}"


def run_hotspot_precompute(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    density_percentile: float = 75.0,
    eps_km: float = 250.0,
    min_samples: int = 3,
    points_format: str = "parquet",
) -> dict[str, Path]:
    """Load cleaned observations from disk and save precomputed hotspot artifacts."""

    observations_df = _read_observations_frame(input_path)
    return precompute_and_save_hotspot_clusters(
        observations_df,
        output_dir,
        density_percentile=density_percentile,
        eps_km=eps_km,
        min_samples=min_samples,
        points_format=points_format,
    )


def _read_observations_frame(input_path: str | Path) -> pd.DataFrame:
    """Read a cleaned observations DataFrame from a CSV or parquet file."""

    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    if suffix == ".csv":
        return pd.read_csv(input_path)

    raise ValueError("input_path must point to a .csv or .parquet file.")


def _build_argument_parser() -> argparse.ArgumentParser:
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
        choices=("parquet", "csv"),
        default="parquet",
        help="Output format for the hotspot points artifact.",
    )

    return parser


__all__ = [
    "EARTH_RADIUS_KM",
    "build_cluster_boundary_polygons",
    "cluster_hotspot_observations",
    "filter_high_density_observations",
    "load_precomputed_hotspot_clusters",
    "precompute_and_save_hotspot_clusters",
    "run_hotspot_precompute",
]


if __name__ == "__main__":
    parser = _build_argument_parser()
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
