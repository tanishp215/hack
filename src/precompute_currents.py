"""Precompute subsampled mean ocean current vectors from OSCAR data.

Produces results/current_vectors.npz for the Ocean Currents Streamlit page.
"""

import time
from pathlib import Path

import numpy as np

from src.process_oscar import load_oscar

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_PATH = _REPO_ROOT / "results" / "current_vectors.npz"

# Subsample factor: take every Nth grid point
_SUBSAMPLE = 6


if __name__ == "__main__":
    print("Loading OSCAR dataset...")
    ds = load_oscar()
    print(f"  {len(ds.time)} time steps, grid {ds.sizes['longitude']}x{ds.sizes['latitude']}")

    print("Computing temporal mean of u and v (this may take a few minutes)...")
    t0 = time.time()
    u_mean = ds["u"].mean(dim="time").values  # forces dask compute
    v_mean = ds["v"].mean(dim="time").values
    print(f"  Done in {time.time() - t0:.1f}s")

    # Coordinate arrays
    lon_full = ds["lon"].values  # shape (1440,), 0-360
    lat_full = ds["lat"].values  # shape (719,)

    # Subsample
    lon_sub = lon_full[::_SUBSAMPLE]
    lat_sub = lat_full[::_SUBSAMPLE]
    u_sub = u_mean[::_SUBSAMPLE, ::_SUBSAMPLE]  # (lon, lat) axis order
    v_sub = v_mean[::_SUBSAMPLE, ::_SUBSAMPLE]

    # Convert lon from 0-360 to -180..180
    lon_sub = np.where(lon_sub > 180, lon_sub - 360, lon_sub)
    sort_idx = np.argsort(lon_sub)
    lon_sub = lon_sub[sort_idx]
    u_sub = u_sub[sort_idx, :]
    v_sub = v_sub[sort_idx, :]

    # Replace NaN (land) with 0
    u_sub = np.nan_to_num(u_sub, nan=0.0)
    v_sub = np.nan_to_num(v_sub, nan=0.0)

    # Speed magnitude
    speed = np.sqrt(u_sub**2 + v_sub**2)

    # Save
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _OUTPUT_PATH,
        lat=lat_sub,
        lon=lon_sub,
        u=u_sub,
        v=v_sub,
        speed=speed,
    )

    print(f"\nSaved to {_OUTPUT_PATH}")
    print(f"  lat: {lat_sub.shape}, lon: {lon_sub.shape}")
    print(f"  u/v/speed: {u_sub.shape}")
    print(f"  Speed range: {speed[speed > 0].min():.4f} – {speed.max():.4f} m/s")
    print(f"  Non-zero points: {np.count_nonzero(speed)} / {speed.size}")
