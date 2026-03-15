"""Data loading and cleaning utilities for microplastics dataset."""

import numpy as np
import pandas as pd


def load_microplastics(path: str = "data/microplastics.csv") -> pd.DataFrame:
    """Load and clean the NOAA marine microplastics CSV.

    Renames columns to snake_case, parses dates, filters invalid measurements,
    and adds a latitude-band classification column.

    Args:
        path: Path to the raw microplastics CSV file.

    Returns:
        Cleaned DataFrame with snake_case columns, datetime parsing,
        year/month extraction, and lat_band classification.
    """
    column_map: dict[str, str] = {
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

    df = pd.read_csv(path)
    df = df.rename(columns=column_map)

    # Parse dates and extract year/month
    df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")
    df["year"] = df["sample_date"].dt.year.astype("Int64")
    df["month"] = df["sample_date"].dt.month.astype("Int64")

    # Drop rows with missing or negative measurements
    df = df.dropna(subset=["measurement"])
    df = df[df["measurement"] >= 0].copy()

    # Classify latitude bands
    abs_lat: pd.Series = df["latitude"].abs()
    conditions: list[pd.Series] = [
        abs_lat > 60,
        abs_lat > 45,
        abs_lat > 23,
        abs_lat <= 23,
    ]
    labels: list[str] = ["Polar", "Subpolar", "Temperate", "Subtropical/Tropical"]
    df["lat_band"] = np.select(conditions, labels, default="Subtropical/Tropical")

    return df
