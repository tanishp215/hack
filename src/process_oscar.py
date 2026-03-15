"""Loader for the merged OSCAR ocean surface current NetCDF file.

Provides `load_oscar()` for downstream code and a CLI inspection mode.
"""

import os
from pathlib import Path

import xarray as xr

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PATH = _REPO_ROOT / "data" / "currents.nc"


def load_oscar(path: os.PathLike | str | None = None) -> xr.Dataset:
    """Return a lazily-loaded xarray Dataset from the merged currents file.

    Parameters
    ----------
    path : path, optional
        Override the default currents.nc location.

    Returns
    -------
    xr.Dataset
        Lazily-loaded dataset (backed by dask arrays).
    """
    path = Path(path) if path is not None else _DEFAULT_PATH

    if not path.exists():
        raise FileNotFoundError(f"Currents file not found: {path}")

    return xr.open_dataset(path, chunks="auto")


if __name__ == "__main__":
    ds = load_oscar()
    print(f"Found {len(ds.time)} time steps")
    print(f"\n=== Dataset Info ===")
    print(f"Dimensions:  {dict(ds.sizes)}")
    print(f"Coordinates: {list(ds.coords)}")
    print(f"Variables:   {list(ds.data_vars)}")
    if "time" in ds.coords:
        print(f"Time range:  {ds.time.values[0]} → {ds.time.values[-1]}")
    print(f"\nFull summary:\n{ds}")
