"""Loader and cleaner for the NOAA NCEI microplastics CSV.

Provides `load_noaa()` for downstream code and a CLI inspection mode.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Resolve the CSV path relative to the repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CSV_PATH = _REPO_ROOT / "data" / "NOAA.csv"

_RENAME_MAP = {
    "Latitude (degree)": "latitude",
    "Longitude (degree)": "longitude",
    "Ocean": "ocean",
    "Region": "region",
    "Marine Setting": "marine_setting",
    "Microplastics Measurement": "measurement",
    "Unit": "unit",
    "Concentration Class": "concentration_class",
    "Sample Date": "sample_date",
}


def load_noaa(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Return a cleaned DataFrame of NOAA microplastics observations.

    Parameters
    ----------
    csv_path : path, optional
        Override the default CSV location.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with snake_case columns, parsed dates,
        and rows with missing measurement or ocean removed.
    """
    csv_path = Path(csv_path) if csv_path is not None else _CSV_PATH
    df = pd.read_csv(csv_path)

    df = df.rename(columns=_RENAME_MAP)
    df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")
    df = df.dropna(subset=["measurement"])
    df = df[df["measurement"] >= 0].copy()
    df = df.dropna(subset=["ocean"])

    # Derived columns used by analysis and app layers
    df["year"] = df["sample_date"].dt.year.astype("Int64")
    df["month"] = df["sample_date"].dt.month.astype("Int64")

    # Latitude band classification
    abs_lat = df["latitude"].abs()
    df["lat_band"] = np.select(
        [abs_lat > 60, abs_lat > 45, abs_lat > 23, abs_lat <= 23],
        ["Polar", "Subpolar", "Temperate", "Subtropical/Tropical"],
        default="Subtropical/Tropical",
    )

    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load_noaa()
    print(f"Shape: {df.shape}")
    print(f"\n=== Date Range ===")
    print(f"{df['sample_date'].min()} → {df['sample_date'].max()}")
    print(f"\n=== Ocean Distribution ===")
    print(df["ocean"].value_counts().to_string())
    print(f"\n=== Unit Distribution ===")
    print(df["unit"].value_counts().to_string())
