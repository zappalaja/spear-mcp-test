"""
coord_utils.py — Shared coordinate conversion and spatial subsetting utilities.

Handles the critical mismatch between user-provided coordinates (typically
-180..180 longitude) and dataset conventions (SPEAR/GFDL uses 0..360).

All spatial subsetting across NetCDF and Zarr tools should use these functions
to ensure consistent, correct results regardless of how the user specifies
coordinates.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


# ── Longitude convention detection and conversion ────────────────────────────

def detect_lon_convention(lon_values: np.ndarray) -> str:
    """Detect whether a dataset uses 0..360 or -180..180 longitude convention.

    Returns:
        "0_360" if longitudes span roughly 0 to 360,
        "-180_180" if longitudes span roughly -180 to 180.
    """
    lon_min = float(lon_values.min())
    lon_max = float(lon_values.max())

    if lon_min >= -1.0 and lon_max > 180.0:
        return "0_360"
    return "-180_180"


def convert_lon(user_lon: float, convention: str) -> float:
    """Convert a single longitude value to match the dataset convention.

    Args:
        user_lon: Longitude as provided by the user (could be any convention).
        convention: Target convention, either "0_360" or "-180_180".

    Returns:
        Longitude value in the target convention.
    """
    if convention == "0_360":
        # Convert -180..180 → 0..360
        return user_lon % 360.0
    else:
        # Convert 0..360 → -180..180
        return ((user_lon + 180.0) % 360.0) - 180.0


def convert_lon_range(
    lon_range: List[float], ds_lon_values: np.ndarray
) -> Tuple[float, float]:
    """Convert a user-provided [lon_min, lon_max] to the dataset's convention.

    Args:
        lon_range: [min_lon, max_lon] as provided by the user.
        ds_lon_values: The actual longitude coordinate array from the dataset.

    Returns:
        (converted_min, converted_max) in the dataset's convention.
    """
    convention = detect_lon_convention(ds_lon_values)
    converted_min = convert_lon(lon_range[0], convention)
    converted_max = convert_lon(lon_range[1], convention)

    # Ensure min <= max after conversion (can flip when crossing 0/360 boundary)
    if converted_min > converted_max:
        converted_min, converted_max = converted_max, converted_min

    return converted_min, converted_max


def convert_lat_range(
    lat_range: List[float], ds_lat_values: np.ndarray
) -> Tuple[float, float]:
    """Normalize a user-provided [lat_min, lat_max], ensuring min <= max.

    Latitude doesn't typically need convention conversion, but we ensure
    proper ordering and detect if the dataset runs north-to-south.

    Args:
        lat_range: [min_lat, max_lat] as provided by the user.
        ds_lat_values: The actual latitude coordinate array from the dataset.

    Returns:
        (lat_min, lat_max) in the correct order for slicing.
    """
    lat_min, lat_max = lat_range[0], lat_range[1]

    # Ensure user values are ordered
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min

    # Check if dataset latitude is descending (north-to-south)
    if len(ds_lat_values) > 1 and ds_lat_values[0] > ds_lat_values[-1]:
        # For descending lat, slice needs max first, then min
        return lat_max, lat_min

    return lat_min, lat_max


# ── Unified spatial subsetting ───────────────────────────────────────────────

def subset_spatial(
    data_var: xr.DataArray,
    ds: xr.Dataset,
    lat_range: Optional[List[float]] = None,
    lon_range: Optional[List[float]] = None,
) -> Tuple[xr.DataArray, Dict[str, Any]]:
    """Apply spatial subsetting with automatic coordinate conversion, clamping,
    and nearest-neighbor snapping. Returns the subsetted DataArray and a dict
    of any coordinate adjustments made (for transparency in results).

    This is the single function that ALL tools should use for spatial selection.

    Args:
        data_var: The xarray DataArray to subset.
        ds: The parent Dataset (used to read coordinate arrays).
        lat_range: Optional [min_lat, max_lat] from the user.
        lon_range: Optional [min_lon, max_lon] from the user.

    Returns:
        (subsetted_data_var, coordinate_adjustments_dict)
    """
    coordinate_adjustments = {}

    # ── Latitude ──────────────────────────────────────────────────────────
    if lat_range and "lat" in ds.coords:
        lat_vals = ds["lat"].values
        lat_ds_min, lat_ds_max = float(lat_vals.min()), float(lat_vals.max())

        # Normalize user range (ensure ordering)
        req_lat_min, req_lat_max = lat_range[0], lat_range[1]
        if req_lat_min > req_lat_max:
            req_lat_min, req_lat_max = req_lat_max, req_lat_min

        # Clamp to dataset bounds
        clamped_lat_min = max(req_lat_min, lat_ds_min)
        clamped_lat_max = min(req_lat_max, lat_ds_max)

        # Snap to nearest grid points
        nearest_lat_min = float(
            ds["lat"].sel(lat=clamped_lat_min, method="nearest").values
        )
        nearest_lat_max = float(
            ds["lat"].sel(lat=clamped_lat_max, method="nearest").values
        )

        # Ensure min <= max after snapping
        if nearest_lat_min > nearest_lat_max:
            nearest_lat_min, nearest_lat_max = nearest_lat_max, nearest_lat_min

        # Record adjustment if values changed
        if (
            abs(nearest_lat_min - lat_range[0]) > 0.01
            or abs(nearest_lat_max - lat_range[1]) > 0.01
        ):
            coordinate_adjustments["latitude"] = {
                "requested": [lat_range[0], lat_range[1]],
                "actual": [nearest_lat_min, nearest_lat_max],
                "reason": "clamped to dataset bounds and snapped to nearest grid points",
            }

        # Handle descending latitude
        if len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]:
            data_var = data_var.sel(lat=slice(nearest_lat_max, nearest_lat_min))
        else:
            data_var = data_var.sel(lat=slice(nearest_lat_min, nearest_lat_max))

    # ── Longitude ─────────────────────────────────────────────────────────
    if lon_range and "lon" in ds.coords:
        lon_vals = ds["lon"].values
        lon_ds_min, lon_ds_max = float(lon_vals.min()), float(lon_vals.max())
        convention = detect_lon_convention(lon_vals)

        # Convert user longitudes to dataset convention
        req_lon_min = convert_lon(lon_range[0], convention)
        req_lon_max = convert_lon(lon_range[1], convention)

        # Ensure min <= max after conversion
        if req_lon_min > req_lon_max:
            req_lon_min, req_lon_max = req_lon_max, req_lon_min

        # Clamp to dataset bounds
        clamped_lon_min = max(req_lon_min, lon_ds_min)
        clamped_lon_max = min(req_lon_max, lon_ds_max)

        # Snap to nearest grid points
        nearest_lon_min = float(
            ds["lon"].sel(lon=clamped_lon_min, method="nearest").values
        )
        nearest_lon_max = float(
            ds["lon"].sel(lon=clamped_lon_max, method="nearest").values
        )

        # Ensure min <= max after snapping
        if nearest_lon_min > nearest_lon_max:
            nearest_lon_min, nearest_lon_max = nearest_lon_max, nearest_lon_min

        # Record adjustment — always show what conversion happened
        was_converted = (
            abs(lon_range[0] - req_lon_min) > 0.01
            or abs(lon_range[1] - req_lon_max) > 0.01
        )
        was_snapped = (
            abs(nearest_lon_min - req_lon_min) > 0.01
            or abs(nearest_lon_max - req_lon_max) > 0.01
        )

        if was_converted or was_snapped:
            reasons = []
            if was_converted:
                reasons.append(
                    f"converted from user convention to dataset convention ({convention})"
                )
            if was_snapped:
                reasons.append("snapped to nearest grid points")
            coordinate_adjustments["longitude"] = {
                "requested": [lon_range[0], lon_range[1]],
                "converted": [req_lon_min, req_lon_max],
                "actual": [nearest_lon_min, nearest_lon_max],
                "dataset_convention": convention,
                "reason": "; ".join(reasons),
            }

        data_var = data_var.sel(lon=slice(nearest_lon_min, nearest_lon_max))

    # ── Safety check: warn if selection resulted in empty data ────────────
    if data_var.size == 0:
        logger.warning(
            "Spatial subsetting resulted in empty selection. "
            f"lat_range={lat_range}, lon_range={lon_range}, "
            f"adjustments={coordinate_adjustments}"
        )

    return data_var, coordinate_adjustments
