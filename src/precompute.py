"""Precompute 5-year particle drift trajectories from 18 coastal cities.

Runs simulate_particles for each city and saves the concatenated results
to results/all_trajectories.parquet for use by the Streamlit app.
"""

import time
from pathlib import Path

import pandas as pd

# Imports use the same src/ package convention as other modules
from src.process_oscar import load_oscar
from src.simulate import simulate_particles

# Resolve output path relative to repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_PATH = _REPO_ROOT / "results" / "all_trajectories.parquet"

# 18 coastal release points — coordinates placed in ocean near the coast,
# not at city centers which may be inland/on land in the OSCAR grid
CITIES: dict[str, tuple[float, float]] = {
    "Shanghai":  (31.2, 122.0),
    "Jakarta":   (-6.1, 106.8),
    "Manila":    (14.5, 121.0),
    "Mumbai":    (19.0, 72.8),
    "Lagos":     (6.4, 3.4),
    "Santos":    (-23.9, -46.3),
    "New York":  (40.6, -73.8),
    "Los Angeles": (33.7, -118.3),
    "Istanbul":  (41.0, 29.0),
    "Bangkok":   (13.1, 100.5),
    "Tokyo":     (35.4, 139.8),
    "Cape Town": (-33.9, 18.4),
    "Lima":      (-12.0, -77.1),
    "Miami":     (25.76, -80.19),
    "London":    (51.5, 1.0),
    "Sydney":    (-33.8, 151.3),
    "Mombasa":   (-4.0, 39.7),
    "Dubai":     (25.2, 55.3),
}


if __name__ == "__main__":
    # Load the OSCAR dataset once — lazy via dask, so this is fast
    print("Loading OSCAR dataset...")
    ds = load_oscar()
    print(f"  {len(ds.time)} time steps, grid {ds.sizes['longitude']}x{ds.sizes['latitude']}")

    all_dfs: list[pd.DataFrame] = []
    total_start = time.time()

    # Run simulation for each city
    for city, (lat, lon) in CITIES.items():
        print(f"\n{city} (lat={lat}, lon={lon})...")
        t0 = time.time()

        df = simulate_particles(
            ds,
            start_lat=lat,
            start_lon=lon,
            n_particles=50,
            n_days=1825,     # 5 years
            dt_days=1.0,
            spread=0.5,
        )

        # Tag each row with the source city
        df["city"] = city

        elapsed = time.time() - t0
        print(f"  {len(df)} rows, {df['particle_id'].nunique()} particles, {elapsed:.1f}s")
        all_dfs.append(df)

    # Concatenate all city results into one DataFrame
    print("\nConcatenating...")
    result = pd.concat(all_dfs, ignore_index=True)

    # Save as parquet (compact, fast to reload)
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(_OUTPUT_PATH, index=False)

    total_elapsed = time.time() - total_start
    print(f"\nSaved {result.shape[0]} rows to {_OUTPUT_PATH}")
    print(f"Total elapsed: {total_elapsed:.1f}s")
