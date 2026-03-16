"""MCP tool script for working with CMIP6 Zarr data on AWS S3."""

from typing import List, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import xarray as xr
import s3fs
import logging

from .coord_utils import subset_spatial

logger = logging.getLogger(__name__)

# Global variables to cache loaded dataset
_cached_dataset = None
_cached_zarr_path = None

# CMIP6 Configuration
CMIP6_BASE_BUCKET = "cmip6-pds"
CMIP6_BASE_PATH = "CMIP6/CMIP/NOAA-GFDL/GFDL-CM4/historical/r1i1p1f1/Amon/tas/gr1/v20180701"

def get_zarr_store_info(
    zarr_path: str = None,
    include_full_details: bool = False
) -> Dict[str, Any]:
    """
    Get metadata from a CMIP6 Zarr store without loading data.

    Args:
        zarr_path: S3 path to Zarr store (e.g., "s3://cmip6-pds/CMIP6/...")
                   If None, uses default CMIP6_BASE_PATH
        include_full_details: If True, includes detailed coordinate/variable info (slower)

    Returns:
        Dictionary containing Zarr store metadata
    """
    try:
        fs = s3fs.S3FileSystem(anon=True)

        # Use default path if none provided
        if zarr_path is None:
            zarr_path = f"s3://{CMIP6_BASE_BUCKET}/{CMIP6_BASE_PATH}"

        # Remove s3:// prefix for s3fs operations
        s3_path = zarr_path.replace("s3://", "")

        # Check if Zarr store exists
        if not fs.exists(s3_path):
            return {"error": f"Zarr store not found: {zarr_path}"}

        # Get basic store size
        try:
            store_size = 0
            for item in fs.walk(s3_path):
                if item[2]:  # files exist
                    for file in item[2]:
                        file_path = f"{item[0]}/{file}"
                        try:
                            store_size += fs.size(file_path)
                        except:
                            pass
            store_size_mb = round(store_size / (1024 * 1024), 2)
        except Exception as e:
            store_size_mb = "unknown"
            logger.warning(f"Could not calculate store size: {e}")

        metadata = {
            "zarr_store_path": zarr_path,
            "store_size_mb": store_size_mb,
            "data_format": "Zarr",
            "dataset": "CMIP6 GFDL-CM4"
        }

        # If full details requested, open the store
        if include_full_details:
            store = s3fs.S3Map(root=s3_path, s3=fs, check=False)
            ds = xr.open_zarr(store, consolidated=True)

            metadata["dimensions"] = dict(ds.dims)
            metadata["coordinates"] = {}
            metadata["variables"] = {}
            metadata["global_attributes"] = make_json_serializable(dict(ds.attrs))

            # Get coordinate information
            for coord_name, coord in ds.coords.items():
                coord_info = {
                    "dimensions": list(coord.dims),
                    "shape": list(coord.shape),
                    "dtype": str(coord.dtype),
                    "attributes": make_json_serializable(dict(coord.attrs))
                }

                # For small coordinate arrays, include values
                if coord.size <= 1000:
                    coord_info["values"] = make_json_serializable(coord.values.tolist())
                else:
                    min_val = coord.min().values
                    max_val = coord.max().values
                    coord_info["values_info"] = {
                        "size": int(coord.size),
                        "min": make_json_serializable(min_val),
                        "max": make_json_serializable(max_val),
                        "first_few": make_json_serializable(coord.values[:5].tolist()),
                        "last_few": make_json_serializable(coord.values[-5:].tolist())
                    }

                metadata["coordinates"][coord_name] = coord_info

            # Get variable information
            for var_name, var in ds.data_vars.items():
                metadata["variables"][var_name] = {
                    "dimensions": list(var.dims),
                    "shape": list(var.shape),
                    "dtype": str(var.dtype),
                    "size": int(var.size),
                    "long_name": var.attrs.get('long_name', 'N/A'),
                    "units": var.attrs.get('units', 'N/A'),
                    "standard_name": var.attrs.get('standard_name', 'N/A'),
                    "attributes": make_json_serializable(dict(var.attrs))
                }

        return metadata

    except Exception as e:
        return {"error": f"Failed to get Zarr store metadata: {str(e)}"}


def load_zarr_dataset(zarr_path: str = None):
    """
    Load Zarr dataset with lazy loading - data only fetched when accessed.

    Args:
        zarr_path: S3 path to Zarr store. If None, uses default CMIP6_BASE_PATH

    Returns:
        xarray Dataset with lazy-loaded Zarr data
    """
    global _cached_dataset, _cached_zarr_path

    # Use default path if none provided
    if zarr_path is None:
        zarr_path = f"s3://{CMIP6_BASE_BUCKET}/{CMIP6_BASE_PATH}"

    # Return cached dataset if same path
    if _cached_dataset is not None and _cached_zarr_path == zarr_path:
        return _cached_dataset

    try:
        fs = s3fs.S3FileSystem(anon=True)
        s3_path = zarr_path.replace("s3://", "")

        # Create S3Map for Zarr access
        store = s3fs.S3Map(root=s3_path, s3=fs, check=False)

        # Open Zarr store with lazy loading (consolidated metadata for faster access)
        _cached_dataset = xr.open_zarr(store, consolidated=True)
        _cached_zarr_path = zarr_path

        return _cached_dataset

    except Exception as e:
        raise ValueError(f"Failed to load Zarr dataset from {zarr_path}: {str(e)}")


def query_zarr_data(
    variable: str = "tas",
    start_date: str = None,
    end_date: str = None,
    lat_range: List[float] = None,
    lon_range: List[float] = None,
    zarr_path: str = None
) -> Dict[str, Any]:
    """
    Query Zarr data with spatial/temporal subsetting.

    Args:
        variable: Variable name (e.g., "tas" for temperature)
        start_date: Start date (e.g., "1850-01" or "1850-01-15")
        end_date: End date (e.g., "2014-12" or "2014-12-31")
        lat_range: [min_lat, max_lat] in degrees
        lon_range: [min_lon, max_lon] in degrees
        zarr_path: S3 path to Zarr store. If None, uses default

    Returns:
        Dictionary containing queried data and metadata
    """
    try:
        # Load dataset
        ds = load_zarr_dataset(zarr_path)

        # Check if variable exists
        if variable not in ds.data_vars:
            available_vars = list(ds.data_vars.keys())
            return {
                "error": f"Variable '{variable}' not found in dataset",
                "available_variables": available_vars
            }

        # Select variable
        data_var = ds[variable]

        # Spatial subsetting with automatic coordinate conversion
        data_var, coordinate_adjustments = subset_spatial(data_var, ds, lat_range, lon_range)

        # Apply temporal subsetting
        if start_date or end_date:
            time_slice = {}
            if start_date:
                time_slice['start'] = start_date
            if end_date:
                time_slice['stop'] = end_date
            data_var = data_var.sel(time=slice(time_slice.get('start'), time_slice.get('stop')))

        # Check data size
        data_size_mb = (data_var.size * 4) / (1024 * 1024)  # Assuming float32

        if data_size_mb > 50:
            return {
                "error": f"Requested data too large: {data_size_mb:.2f} MB",
                "message": "Please use smaller spatial/temporal ranges",
                "data_shape": list(data_var.shape),
                "suggestion": "Try reducing the lat/lon range or time period"
            }

        # Load the data (this triggers actual download from S3)
        data_values = data_var.values

        # Build result
        result = {
            "variable": variable,
            "zarr_store": zarr_path or f"s3://{CMIP6_BASE_BUCKET}/{CMIP6_BASE_PATH}",
            "query_parameters": {
                "start_date": start_date,
                "end_date": end_date,
                "lat_range": lat_range,
                "lon_range": lon_range
            },
            "data_info": {
                "shape": list(data_var.shape),
                "dimensions": list(data_var.dims),
                "dtype": str(data_var.dtype),
                "size_mb": round(data_size_mb, 2)
            },
            "coordinates": {},
            "data": make_json_serializable(data_values.tolist()),
            "attributes": make_json_serializable(dict(data_var.attrs))
        }

        # Add coordinate information
        for coord_name in data_var.coords:
            coord = data_var.coords[coord_name]
            result["coordinates"][coord_name] = {
                "values": make_json_serializable(coord.values.tolist()),
                "attributes": make_json_serializable(dict(coord.attrs))
            }

        # Add coordinate adjustment info if any
        if coordinate_adjustments:
            result["coordinate_adjustments"] = coordinate_adjustments
            result["note"] = "Coordinates were snapped to nearest grid points. See 'coordinate_adjustments' for details."

        return result

    except Exception as e:
        return {"error": f"Query failed: {str(e)}"}


def get_zarr_summary_statistics(
    variable: str = "tas",
    start_date: str = None,
    end_date: str = None,
    lat_range: List[float] = None,
    lon_range: List[float] = None,
    zarr_path: str = None
) -> Dict[str, Any]:
    """
    Get summary statistics for Zarr data without returning full arrays.

    Args:
        variable: Variable name
        start_date: Start date
        end_date: End date
        lat_range: Latitude range
        lon_range: Longitude range
        zarr_path: S3 path to Zarr store

    Returns:
        Dictionary with summary statistics
    """
    try:
        ds = load_zarr_dataset(zarr_path)

        if variable not in ds.data_vars:
            return {"error": f"Variable '{variable}' not found"}

        data_var = ds[variable]

        # Spatial subsetting with automatic coordinate conversion
        data_var, _ = subset_spatial(data_var, ds, lat_range, lon_range)

        if start_date or end_date:
            time_slice = {}
            if start_date:
                time_slice['start'] = start_date
            if end_date:
                time_slice['stop'] = end_date
            data_var = data_var.sel(time=slice(time_slice.get('start'), time_slice.get('stop')))

        # Compute statistics (lazy evaluation - only reads needed data)
        stats = {
            "variable": variable,
            "query_parameters": {
                "start_date": start_date,
                "end_date": end_date,
                "lat_range": lat_range,
                "lon_range": lon_range
            },
            "shape": list(data_var.shape),
            "data_size_mb": round((data_var.size * 4) / (1024 * 1024), 2),
            "statistics": {
                "min": float(data_var.min().values),
                "max": float(data_var.max().values),
                "mean": float(data_var.mean().values),
                "std": float(data_var.std().values)
            }
        }

        return stats

    except Exception as e:
        return {"error": f"Statistics calculation failed: {str(e)}"}


def test_cmip6_connection(zarr_path: str = None):
    """
    Test basic S3 connection to CMIP6 Zarr store.

    Args:
        zarr_path: S3 path to test. If None, uses default CMIP6_BASE_PATH

    Returns:
        Connection test results
    """
    try:
        if zarr_path is None:
            zarr_path = f"s3://{CMIP6_BASE_BUCKET}/{CMIP6_BASE_PATH}"

        fs = s3fs.S3FileSystem(anon=True)
        s3_path = zarr_path.replace("s3://", "")

        # Check if store exists
        if not fs.exists(s3_path):
            return {
                "status": "error",
                "error": f"Zarr store not found at {zarr_path}"
            }

        # Try to list contents
        contents = fs.ls(s3_path)[:10]

        # Try to open the store
        store = s3fs.S3Map(root=s3_path, s3=fs, check=False)
        ds = xr.open_zarr(store, consolidated=True)

        return {
            "status": "success",
            "message": "Successfully connected to CMIP6 Zarr store",
            "zarr_path": zarr_path,
            "sample_contents": [c.split('/')[-1] for c in contents],
            "dimensions": dict(ds.dims),
            "variables": list(ds.data_vars.keys()),
            "coordinates": list(ds.coords.keys())
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "zarr_path": zarr_path
        }


def make_json_serializable(obj):
    """Recursively convert objects to JSON-serializable format."""
    if isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        try:
            return [make_json_serializable(item) for item in obj]
        except:
            return str(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif not isinstance(obj, (str, int, float, bool, type(None))):
        return str(obj)
    return obj
