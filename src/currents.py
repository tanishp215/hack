"""Interpolated ocean surface current lookup from OSCAR data.

Provides `get_velocity()` for the drift simulation and other analysis.
"""

import numpy as np
import xarray as xr


def get_velocity(
    ds: xr.Dataset, lat: float, lon: float, day_index: int
) -> tuple[float, float]:
    """Return interpolated (u, v) surface current velocity in m/s.

    Parameters
    ----------
    ds : xr.Dataset
        OSCAR dataset loaded via load_oscar() (lazily backed by dask).
    lat : float
        Latitude in degrees (-90 to 90).
    lon : float
        Longitude in degrees (-180 to 180 or 0 to 360).
    day_index : int
        Index into the time dimension (0 to len(ds.time)-1).

    Returns
    -------
    tuple[float, float]
        (u, v) velocity in m/s. Returns (0.0, 0.0) if the point is over land.
    """
    # Convert negative longitudes (e.g. -80 for Miami) to 0-360 range
    # because OSCAR uses 0-360 convention
    if lon < 0:
        lon += 360.0

    # Select the single time step we need — avoids loading all 315 days
    ds_t = ds.isel(time=day_index)

    # lon/lat are non-dimension coordinates, so .interp(lon=, lat=) won't work.
    # Instead, convert our target lat/lon to fractional grid indices and
    # interpolate along the actual dimensions (longitude, latitude).

    # Extract the 1-D coordinate arrays (these are small, already in memory)
    lon_arr = ds_t["lon"].values  # shape (1440,), 0.0 to 359.75
    lat_arr = ds_t["lat"].values  # shape (719,), e.g. -79.75 to 79.75

    # Find the fractional index into each dimension array.
    # np.interp maps our target value onto the index space [0, N-1].
    lon_idx = np.interp(lon, lon_arr, np.arange(len(lon_arr)))
    lat_idx = np.interp(lat, lat_arr, np.arange(len(lat_arr)))

    # Bilinearly interpolate u and v at the fractional grid position.
    # This uses the integer dimension names (longitude, latitude).
    point = ds_t.interp(
        longitude=lon_idx, latitude=lat_idx, method="linear"
    )

    # .values.item() forces dask to compute and extracts a Python float
    u = point["u"].values.item()
    v = point["v"].values.item()

    # NaN means the point is over land (no ocean current data).
    # Return zero velocity so the drift simulation treats land as stationary.
    if np.isnan(u) or np.isnan(v):
        return (0.0, 0.0)

    return (float(u), float(v))


if __name__ == "__main__":
    from process_oscar import load_oscar

    # Load the full OSCAR dataset (lazy — no data in RAM yet)
    ds = load_oscar()
    print(f"Dataset: {dict(ds.sizes)}")

    # Query Miami: lat=25.76, lon=-80.19 (negative, will be converted to 279.81)
    lat, lon, day = 25.76, -80.19, 0
    u, v = get_velocity(ds, lat, lon, day)
    print(f"\nMiami (lat={lat}, lon={lon}, day_index={day})")
    print(f"  u = {u:.4f} m/s (east-west)")
    print(f"  v = {v:.4f} m/s (north-south)")
    print(f"  speed = {np.sqrt(u**2 + v**2):.4f} m/s")
