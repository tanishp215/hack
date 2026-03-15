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

# 18 coastal release points — pushed 2-3° offshore into open water where
# OSCAR has non-zero surface currents. Previous coords sat on land grid cells.
CITIES: dict[str, tuple[float, float]] = {
    "Shanghai":  (30.5, 124.0),   # East China Sea
    "Jakarta":   (-7.5, 106.0),   # Java Sea, south
    "Manila":    (14.0, 119.0),   # South China Sea, west
    "Mumbai":    (18.0, 71.0),    # Arabian Sea, west
    "Lagos":     (4.0, 2.5),      # Gulf of Guinea, south
    "Santos":    (-25.0, -43.5),  # South Atlantic, east
    "New York":  (40.0, -71.5),   # Atlantic shelf
    "Los Angeles": (33.0, -120.0),# Pacific, west
    "Istanbul":  (36.0, 28.0),    # Eastern Mediterranean
    "Bangkok":   (10.0, 100.0),   # Gulf of Thailand, south
    "Tokyo":     (34.5, 142.0),   # Kuroshio Current, east
    "Cape Town": (-35.5, 17.5),   # Benguela/Agulhas
    "Lima":      (-13.0, -79.0),  # Humboldt Current
    "Miami":     (27.0, -79.0),   # Gulf Stream
    "London":    (49.0, -5.0),    # Celtic Sea / Atlantic approach
    "Sydney":    (-35.0, 153.0),  # East Australian Current
    "Mombasa":   (-5.0, 41.5),    # Somali Current
    "Dubai":     (24.0, 58.0),    # Arabian Sea via Gulf of Oman
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
            n_particles=80,
            n_days=1825,     # 5 years
            dt_days=1.0,
            spread=1.5,
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
