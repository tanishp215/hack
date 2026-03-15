"""Lagrangian particle drift simulation using OSCAR surface currents.

Provides `simulate_particles()` which runs Euler advection and returns
a trajectory DataFrame suitable for visualization.

Performance: loads each time step's u/v grids into numpy once, then uses
scipy.ndimage.map_coordinates to interpolate all particles in a single
vectorized call per step (~1000x faster than per-particle xarray.interp).
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import map_coordinates


# Conversion constant: 1 degree of latitude ≈ 111,320 meters
_METERS_PER_DEG_LAT = 111_320.0

# Seconds in one day, used to convert m/s → m/day
_SECONDS_PER_DAY = 86_400.0


def _get_velocities_batch(
    ds: xr.Dataset,
    lats: np.ndarray,
    lons: np.ndarray,
    day_index: int,
    lon_arr: np.ndarray,
    lat_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Look up (u, v) for all particles at once using scipy map_coordinates.

    Parameters
    ----------
    ds : xr.Dataset
        OSCAR dataset.
    lats, lons : np.ndarray
        Particle positions (lons already in 0-360 range).
    day_index : int
        Time index into the dataset.
    lon_arr, lat_arr : np.ndarray
        1-D coordinate arrays from the dataset (cached by caller).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (u_vals, v_vals) arrays. NaN over land is replaced with 0.0.
    """
    # Load this time step's u and v grids into numpy arrays.
    # Shape: (1440, 719) — (longitude, latitude).
    # .values forces dask to compute; we do this once per step, not per particle.
    u_grid = ds["u"].isel(time=day_index).values
    v_grid = ds["v"].isel(time=day_index).values

    # Convert particle lat/lon to fractional grid indices.
    # np.interp maps geographic coords onto the array index space [0, N-1].
    lon_idx = np.interp(lons, lon_arr, np.arange(len(lon_arr)))
    lat_idx = np.interp(lats, lat_arr, np.arange(len(lat_arr)))

    # Build coordinate array for map_coordinates: shape (2, n_particles).
    # First row = longitude indices, second row = latitude indices,
    # matching the grid's (longitude, latitude) axis order.
    coords = np.array([lon_idx, lat_idx])

    # Bilinear interpolation of all particles in one vectorized call.
    # order=1 = bilinear; mode='nearest' clamps edges instead of extrapolating.
    u_vals = map_coordinates(u_grid, coords, order=1, mode="nearest")
    v_vals = map_coordinates(v_grid, coords, order=1, mode="nearest")

    # Replace NaN (land) with 0.0 so land particles freeze
    nan_mask = np.isnan(u_vals) | np.isnan(v_vals)
    u_vals[nan_mask] = 0.0
    v_vals[nan_mask] = 0.0

    return u_vals, v_vals


def simulate_particles(
    ds: xr.Dataset,
    start_lat: float,
    start_lon: float,
    n_particles: int = 50,
    n_days: int = 1825,
    dt_days: float = 1.0,
    spread: float = 0.5,
) -> pd.DataFrame:
    """Run Euler advection of particles through the OSCAR current field.

    Parameters
    ----------
    ds : xr.Dataset
        OSCAR dataset loaded via load_oscar().
    start_lat, start_lon : float
        Center of the initial particle release point.
    n_particles : int
        Number of particles to simulate.
    n_days : int
        Total simulation duration in days (default 1825 = 5 years).
    dt_days : float
        Time step in days for Euler integration.
    spread : float
        Initial particle positions are randomized within ±spread degrees.

    Returns
    -------
    pd.DataFrame
        Columns: particle_id, day, lat, lon. Positions recorded every 7 days.
    """
    # Total time steps available — we cycle with modulo for multi-year sims
    n_time_steps = len(ds.time)

    # Cache the 1-D coordinate arrays once (small, ~12KB total)
    lon_arr = ds["lon"].values   # shape (1440,), 0.0 to 359.75
    lat_arr = ds["lat"].values   # shape (719,)

    # Initialize particle positions: start point + random uniform offset
    rng = np.random.default_rng(42)
    lats = start_lat + rng.uniform(-spread, spread, size=n_particles)
    lons = start_lon + rng.uniform(-spread, spread, size=n_particles)

    # Track which particles are frozen (hit land). Once frozen, they stop.
    frozen = np.zeros(n_particles, dtype=bool)

    # Pre-allocate trajectory storage: record day 0 + every 7th step
    n_steps = int(n_days / dt_days)
    n_snapshots = 1 + n_steps // 7
    # Arrays: (n_snapshots, n_particles) for lat and lon
    all_lats = np.empty((n_snapshots, n_particles))
    all_lons = np.empty((n_snapshots, n_particles))
    all_days = np.empty(n_snapshots, dtype=int)

    # Record initial positions (day 0)
    all_lats[0] = lats
    all_lons[0] = lons
    all_days[0] = 0
    snap_idx = 1

    # Main simulation loop
    for step in range(1, n_steps + 1):
        current_day = step * dt_days

        # Cycle through available OSCAR time steps with modulo
        day_index = step % n_time_steps

        # Convert lons to 0-360 for OSCAR lookup
        lons_360 = lons % 360.0

        # Vectorized velocity lookup for ALL particles at once
        u_arr, v_arr = _get_velocities_batch(
            ds, lats, lons_360, day_index, lon_arr, lat_arr
        )

        # Zero out frozen particles (they don't move)
        u_arr[frozen] = 0.0
        v_arr[frozen] = 0.0

        # Freeze any non-frozen particle that got (0, 0) — it hit land
        just_landed = (~frozen) & (u_arr == 0.0) & (v_arr == 0.0)
        frozen |= just_landed

        # Convert velocity from m/s to degrees/day:
        #   dlat = v * seconds_per_day / meters_per_degree_lat
        #   dlon = u * seconds_per_day / (meters_per_degree_lat * cos(lat))
        # cos(lat) corrects for longitude lines converging toward poles
        cos_lat = np.cos(np.radians(lats))
        cos_lat = np.clip(cos_lat, 0.01, None)  # avoid div-by-zero at poles

        dlat = v_arr * _SECONDS_PER_DAY * dt_days / _METERS_PER_DEG_LAT
        dlon = u_arr * _SECONDS_PER_DAY * dt_days / (_METERS_PER_DEG_LAT * cos_lat)

        # Euler step: update positions
        lats += dlat
        lons += dlon

        # Clamp latitude to valid range
        lats = np.clip(lats, -90.0, 90.0)

        # Wrap longitude to -180..180
        lons = ((lons + 180.0) % 360.0) - 180.0

        # Record positions every 7 days
        if step % 7 == 0:
            all_lats[snap_idx] = lats
            all_lons[snap_idx] = lons
            all_days[snap_idx] = int(current_day)
            snap_idx += 1

    # Build DataFrame from pre-allocated arrays (no per-row dict overhead)
    particle_ids = np.tile(np.arange(n_particles), n_snapshots)
    days = np.repeat(all_days[:snap_idx], n_particles)
    flat_lats = all_lats[:snap_idx].ravel()
    flat_lons = all_lons[:snap_idx].ravel()

    return pd.DataFrame({
        "particle_id": particle_ids[:len(days)],
        "day": days,
        "lat": flat_lats,
        "lon": flat_lons,
    })


if __name__ == "__main__":
    import time
    from src.process_oscar import load_oscar

    # Load OSCAR dataset (lazy)
    ds = load_oscar()
    print(f"Dataset: {dict(ds.sizes)}")

    # Run 10 particles from offshore Miami for 30 days
    print("\nSimulating 10 particles, 30 days from offshore Miami...")
    t0 = time.time()
    df = simulate_particles(
        ds,
        start_lat=25.76,
        start_lon=-79.5,  # Slightly offshore so particles start in water
        n_particles=10,
        n_days=30,
        dt_days=1.0,
        spread=0.25,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")
    print(f"\nResult shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 10 rows:\n{df.head(10)}")
    print(f"\nLast 10 rows:\n{df.tail(10)}")
