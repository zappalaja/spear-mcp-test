"""MCP tool script for working with the public SPEAR NetCDF output on AWS server."""
from typing import Annotated, List, Dict, Optional, Tuple, Union, Any
from urllib.parse import quote
import warnings
warnings.filterwarnings('ignore')

import aiohttp
import numpy as np
import pandas as pd
import xarray as xr
from async_lru import alru_cache
from loguru import logger
from pydantic import Field, HttpUrl
import cftime
import re
import logging
import tempfile
import os
import json
import s3fs
import datetime
from datetime import datetime as dt
import sys

from .coord_utils import subset_spatial, convert_lon, detect_lon_convention

# Global variables to cache loaded dataset (necessary).
_cached_dataset = None
_cached_file_path = None

# Cache for S3 directory listings (to avoid repeated slow S3 calls)
_dir_listing_cache = {}



def get_cached_file_list(dir_path: str) -> List[str]:
    """Get file list from cache or S3, caching the result."""
    global _dir_listing_cache

    if dir_path in _dir_listing_cache:
        return _dir_listing_cache[dir_path]

    fs = s3fs.S3FileSystem(anon=True)
    files_in_dir = fs.ls(dir_path, detail=False)
    nc_files = sorted([f for f in files_in_dir if f.endswith('.nc')])
    _dir_listing_cache[dir_path] = nc_files
    return nc_files


def parse_date_range_from_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract date range from NetCDF filename.

    Filenames typically have format: var_freq_model_scenario_ensemble_grid_STARTDATE-ENDDATE.nc
    Examples:
        - pr_6hr_GFDL-SPEAR-MED_scenarioSSP5-85_r1i1p1f1_gr3_20150101-20201231.nc
        - tas_Amon_GFDL-SPEAR-MED_historical_r1i1p1f1_gr3_192101-201412.nc

    Returns:
        Tuple of (start_date, end_date) as strings, or (None, None) if parsing fails
    """
    try:
        # Extract the date range part (before .nc)
        base = filename.replace('.nc', '')
        parts = base.split('_')

        # Date range is typically the last part
        date_part = parts[-1]

        if '-' in date_part:
            start, end = date_part.split('-')
            return (start, end)
    except Exception:
        pass

    return (None, None)


def find_file_for_date_range(
    nc_files: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Tuple[Optional[str], List[str]]:
    """
    Find the file(s) that contain the requested date range.

    Args:
        nc_files: List of S3 file paths
        start_date: Requested start date (YYYY-MM or YYYY-MM-DD format)
        end_date: Requested end date

    Returns:
        Tuple of (best_file, all_matching_files)
        - best_file: Single file that best matches the request (or first file if no dates specified)
        - all_matching_files: All files that overlap with the requested range
    """
    if not nc_files:
        return (None, [])

    # If no dates specified, return first file
    if not start_date and not end_date:
        return (nc_files[0], nc_files)

    # Parse requested dates to comparable format (YYYYMMDD)
    def normalize_date(date_str: str) -> str:
        """Convert date to YYYYMMDD format for comparison."""
        if not date_str:
            return ""
        # Remove dashes and handle different formats
        clean = date_str.replace('-', '')
        # Pad to 8 digits if needed (YYYYMM -> YYYYMM01)
        if len(clean) == 6:
            clean += "01"
        return clean

    req_start = normalize_date(start_date) if start_date else "00000000"
    req_end = normalize_date(end_date) if end_date else "99999999"

    matching_files = []
    best_file = None

    for f in nc_files:
        filename = f.split('/')[-1]
        file_start, file_end = parse_date_range_from_filename(filename)

        if file_start is None or file_end is None:
            # Can't parse date range, include it as potential match
            matching_files.append(f)
            continue

        # Normalize file dates
        f_start = normalize_date(file_start)
        f_end = normalize_date(file_end)

        # Check if file overlaps with requested range
        # Overlap exists if: file_start <= req_end AND file_end >= req_start
        if f_start <= req_end and f_end >= req_start:
            matching_files.append(f)

            # Best file is the one where requested start falls within its range
            if best_file is None and f_start <= req_start <= f_end:
                best_file = f

    # If no best file found but we have matches, use first match
    if best_file is None and matching_files:
        best_file = matching_files[0]

    # If still no matches, fall back to first file overall
    if best_file is None:
        best_file = nc_files[0]
        matching_files = [nc_files[0]]

    return (best_file, matching_files)

def get_s3_file_metadata_only(
    scenario: str = "scenarioSSP5-85",   # Default values, the LLM will replace the values when fuction calling.
    ensemble_member: str = "r15i1p1f1",
    frequency: str = "Amon",
    variable: str = "tas",
    grid: str = "gr3",
    version: str = "v20210201",
    filename: Optional[str] = None,  # Optional: exact filename if known (e.g., from browse_spear_directory)
    include_full_details: bool = False  # Set True to open file and get full coordinate info (slower)
) -> Dict[str, Any]:
    """
    Get metadata of a SPEAR NetCDF file. By default, extracts info from filename and S3
    (fast mode). Set include_full_details=True to open file for complete coordinate info (slower).

    Args:
        scenario: "historical" or "scenarioSSP5-85"
        ensemble_member: e.g., "r1i1p1f1", "r15i1p1f1"
        frequency: e.g., "Amon" (monthly), "day" (daily)
        variable: e.g., "tas", "pr", "ua"
        grid: e.g., "gr3"
        version: e.g., "v20210201"
        filename: Optional exact filename (e.g., "pr_day_GFDL-SPEAR-MED_historical_r4i1p1f1_gr3_19210101-19301231.nc")
                  If provided, uses this filename directly instead of constructing one.
        include_full_details: If True, opens the file to get full coordinate/dimension details (slower).

    Returns:
        Dictionary containing file metadata. Fast mode includes parsed filename info and file size.
        Full details mode adds dimensions, coordinates, and variable information from the file.
    """
    try:
        fs = s3fs.S3FileSystem(anon=True)

        # Build the directory path
        dir_path = f"noaa-gfdl-spear-large-ensembles-pds/SPEAR/GFDL-LARGE-ENSEMBLES/CMIP/NOAA-GFDL/GFDL-SPEAR-MED/{scenario}/{ensemble_member}/{frequency}/{variable}/{grid}/{version}"

        # If filename is provided, use it directly
        if filename:
            s3_file_path = f"{dir_path}/{filename}"
        else:
            # No filename provided - list files in directory and find the first matching .nc file
            try:
                nc_files = get_cached_file_list(dir_path)
                if not nc_files:
                    return {"error": f"No NetCDF files found in directory: {dir_path}"}
                # Use the first file found
                s3_file_path = nc_files[0]
                filename = s3_file_path.split('/')[-1]
            except Exception as e:
                return {"error": f"Failed to list files in directory {dir_path}: {str(e)}"}

        # Get file info from S3 (fast - doesn't open the file)
        try:
            file_info = fs.info(s3_file_path)
            file_size_bytes = file_info.get('size', 0)
            file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
        except Exception as e:
            return {"error": f"File not found or inaccessible: {s3_file_path}. Error: {str(e)}"}

        # Parse filename to extract metadata (e.g., pr_day_GFDL-SPEAR-MED_historical_r8i1p1f1_gr3_19210101-19301231.nc)
        parsed_info = {}
        if filename:
            parts = filename.replace('.nc', '').split('_')
            if len(parts) >= 6:
                parsed_info = {
                    "variable": parts[0],
                    "frequency": parts[1],
                    "model": parts[2],
                    "scenario": parts[3],
                    "ensemble_member": parts[4],
                    "grid": parts[5],
                }
                # Parse date range from last part (e.g., 19210101-19301231 or 192101-201412)
                if len(parts) >= 7:
                    date_range = parts[6]
                    if '-' in date_range:
                        start_date, end_date = date_range.split('-')
                        parsed_info["date_range"] = {
                            "start": start_date,
                            "end": end_date,
                            "format": "YYYYMMDD" if len(start_date) == 8 else "YYYYMM"
                        }

        # Build basic metadata (fast mode - no file opening)
        metadata = {
            "file_path": f"s3://{s3_file_path}",
            "filename": filename,
            "file_size_mb": file_size_mb,
            "file_size_bytes": file_size_bytes,
            "parsed_from_filename": parsed_info,
            "path_info": {
                "scenario": scenario,
                "ensemble_member": ensemble_member,
                "frequency": frequency,
                "variable": variable,
                "grid": grid,
                "version": version
            }
        }

        # If user wants full details, open the file (slower for large files)
        if include_full_details:
            with fs.open(s3_file_path, mode="rb") as f:
                ds = xr.open_dataset(f, engine="h5netcdf", decode_cf=True)

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
        return {"error": f"Failed to get file metadata: {str(e)}"}

def get_file_info_and_validation(
    scenario: str = "scenarioSSP5-85",  # Default values, the LLM will replace the values when fuction calling.
    ensemble_member: str = "r15i1p1f1",
    frequency: str = "Amon",
    variable: str = "tas",
    grid: str = "gr3",
    version: str = "v20210201",
    filename: Optional[str] = None  # Optional: exact filename if known (e.g., from browse_spear_directory)
) -> Dict[str, Any]:
    """
    Get comprehensive file information including data ranges for validation.

    Args:
        scenario: "historical" or "scenarioSSP5-85"
        ensemble_member: e.g., "r1i1p1f1", "r15i1p1f1"
        frequency: e.g., "Amon" (monthly), "day" (daily)
        variable: e.g., "tas", "pr", "ua"
        grid: e.g., "gr3"
        version: e.g., "v20210201"
        filename: Optional exact filename (e.g., "pr_day_GFDL-SPEAR-MED_historical_r4i1p1f1_gr3_19210101-19301231.nc")
                  If provided, uses this filename directly instead of constructing one.

    Returns:
        Dictionary containing file info plus data ranges for validation.
    """
    try:
        fs = s3fs.S3FileSystem(anon=True)

        # Build the directory path
        dir_path = f"noaa-gfdl-spear-large-ensembles-pds/SPEAR/GFDL-LARGE-ENSEMBLES/CMIP/NOAA-GFDL/GFDL-SPEAR-MED/{scenario}/{ensemble_member}/{frequency}/{variable}/{grid}/{version}"

        # List all files in directory to get complete time range across all chunks (cached)
        try:
            nc_files = get_cached_file_list(dir_path)
            if not nc_files:
                return {"error": f"No NetCDF files found in directory: {dir_path}"}
        except Exception as e:
            return {"error": f"Failed to list files in directory {dir_path}: {str(e)}"}

        # If filename is provided, use it directly; otherwise get aggregate info from all files
        if filename:
            s3_file_path = f"{dir_path}/{filename}"
            files_to_report = [s3_file_path]
        else:
            # Get time range from ALL files (for chunked data like 6hr)
            s3_file_path = nc_files[0]  # Use first file for metadata
            files_to_report = nc_files
            filename = s3_file_path.split('/')[-1]

        # Aggregate time range from all file names (fast - no file opening needed)
        all_starts = []
        all_ends = []
        for f in files_to_report:
            fname = f.split('/')[-1]
            file_start, file_end = parse_date_range_from_filename(fname)
            if file_start and file_end:
                all_starts.append(file_start)
                all_ends.append(file_end)

        # Check if the nc file exists.
        if not fs.exists(s3_file_path):
            return {"error": f"File not found: {s3_file_path}"}

        with fs.open(s3_file_path, mode="rb") as f:
            ds = xr.open_dataset(f, engine="h5netcdf", decode_cf=True)

            # Use aggregated time range from filenames if available (covers all chunks)
            if all_starts and all_ends:
                # Sort and get overall range
                all_starts.sort()
                all_ends.sort()
                time_start = all_starts[0]
                time_end = all_ends[-1]
                # Format nicely for validation
                # Convert YYYYMMDD to YYYY-MM-DD or YYYYMM to YYYY-MM
                if len(time_start) == 8:
                    time_start = f"{time_start[:4]}-{time_start[4:6]}-{time_start[6:8]}"
                elif len(time_start) == 6:
                    time_start = f"{time_start[:4]}-{time_start[4:6]}"
                if len(time_end) == 8:
                    time_end = f"{time_end[:4]}-{time_end[4:6]}-{time_end[6:8]}"
                elif len(time_end) == 6:
                    time_end = f"{time_end[:4]}-{time_end[4:6]}"
            else:
                # Fallback: use time from the opened file
                time_values = ds.time.values
                time_start = str(time_values[0])
                time_end = str(time_values[-1])

            # Extract spatial range.
            lat_range = None
            lon_range = None
            if 'lat' in ds.coords:
                lat_range = [float(ds.lat.values.min()), float(ds.lat.values.max())]
            if 'lon' in ds.coords:
                lon_range = [float(ds.lon.values.min()), float(ds.lon.values.max())]

            # Extract variable info.
            variables = {}
            for var_name, var in ds.data_vars.items():
                variables[var_name] = {
                    "dimensions": list(var.dims),
                    "shape": list(var.shape),
                    "dtype": str(var.dtype),
                    "long_name": var.attrs.get('long_name', 'N/A'),
                    "units": var.attrs.get('units', 'N/A'),
                    "description": var.attrs.get('standard_name', 'N/A')
                }

            # Comprehensive info to send back to the LLM!
            file_info = {
                "file_path": s3_file_path,
                "filename": filename,
                "scenario": scenario,
                "ensemble_member": ensemble_member,
                "frequency": frequency,
                "dimensions": dict(ds.dims),
                "time_range": {
                    "start": time_start,
                    "end": time_end,
                    "total_files": len(files_to_report)
                },
                "variables": variables,
                "global_attributes": make_json_serializable(dict(ds.attrs)),
                "note": "Time range aggregated across all available files" if len(files_to_report) > 1 else "Use this info to validate query parameters"
            }

            # Include list of available files if chunked
            if len(files_to_report) > 1:
                file_info["available_files"] = [f.split('/')[-1] for f in files_to_report]

            # Also includes the spatial info if it is available.
            if lat_range or lon_range:
                file_info["spatial_range"] = {}
                if lat_range:
                    file_info["spatial_range"]["latitude"] = lat_range
                    file_info["spatial_range"]["lat_points"] = len(ds.lat.values)
                if lon_range:
                    file_info["spatial_range"]["longitude"] = lon_range
                    file_info["spatial_range"]["lon_points"] = len(ds.lon.values)

            return file_info

    except Exception as e:
        return {"error": f"Failed to get file info: {str(e)}"}

def validate_query_parameters(
    start_date=None, 
    end_date=None, 
    lat_range=None, 
    lon_range=None, 
    variable=None,
    scenario: str = "scenarioSSP5-85",  # Default values, the LLM will replace the values when fuction calling.
    ensemble_member: str = "r15i1p1f1",
    frequency: str = "Amon",
    grid: str = "gr3",
    version: str = "v20210201"
) -> Dict[str, Any]:
    """
    Validate query parameters against available data ranges
    """
    try:
        # Extract file info for validation.
        file_info = get_file_info_and_validation(scenario, ensemble_member, frequency, variable, grid, version)
        if "error" in file_info:
            return file_info
        
        warnings = []
        errors = []
        
        # Validate variable.
        if variable and variable not in file_info["variables"]:
            errors.append(f"Variable '{variable}' not found. Available: {list(file_info['variables'].keys())}")
        
        # Validate time range.
        if start_date or end_date:
            file_start = file_info["time_range"]["start"]
            file_end = file_info["time_range"]["end"]
            
            if start_date:
                if start_date < file_start:
                    errors.append(f"Start date {start_date} is before file start {file_start}")
                if start_date > file_end:
                    errors.append(f"Start date {start_date} is after file end {file_end}")
            
            if end_date:
                if end_date > file_end:
                    errors.append(f"End date {end_date} is after file end {file_end}")
                if end_date < file_start:
                    errors.append(f"End date {end_date} is before file start {file_start}")
        
        # Validate spatial range (with coordinate convention conversion).
        if "spatial_range" in file_info:
            if "latitude" in file_info["spatial_range"] and lat_range:
                file_lat = file_info["spatial_range"]["latitude"]
                req_lat = sorted([lat_range[0], lat_range[1]])
                if req_lat[0] < file_lat[0] or req_lat[1] > file_lat[1]:
                    warnings.append(f"Requested lat range {lat_range} extends beyond file range {file_lat}")

            if "longitude" in file_info["spatial_range"] and lon_range:
                file_lon = file_info["spatial_range"]["longitude"]
                # Convert user longitudes to dataset convention before comparing
                if file_lon[1] > 180:
                    conv_lon = sorted([lon_range[0] % 360, lon_range[1] % 360])
                else:
                    conv_lon = sorted([lon_range[0], lon_range[1]])
                if conv_lon[0] < file_lon[0] or conv_lon[1] > file_lon[1]:
                    warnings.append(
                        f"Requested lon range {lon_range} (converted: {conv_lon}) "
                        f"extends beyond file range {file_lon}"
                    )
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "file_info": file_info
        }
        
    except Exception as e:
        return {"error": f"Validation failed: {str(e)}"}

def estimate_response_size(shape, dtype="float32", include_coords=True) -> int:
    """
    Estimate response size in bytes for given data shape
    """
    # Estimate bytes per value based on dtype and JSON overhead.
    bytes_per_value = {
        "float32": 12,  # JSON representation + overhead
        "float64": 15,
        "int32": 10,
        "int64": 12
    }
    
    data_bytes = np.prod(shape) * bytes_per_value.get(dtype, 12)
    
    # Add coordinate and metadata overhead.
    if include_coords:
        data_bytes *= 1.5
    
    return int(data_bytes)

def calculate_chunk_size(total_shape, max_response_bytes=800000) -> Dict[str, Any]:
    """
    Calculate optimal chunk size to stay under response limit (currently 1mb, but here the max_response_bytes is set to slightly less for the estimation).
    """
    total_size = estimate_response_size(total_shape)
    
    if total_size <= max_response_bytes:
        return {
            "needs_chunking": False,
            "total_chunks": 1,
            "chunk_shape": total_shape,
            "estimated_size": total_size
        }
    
    # Calculate chunks needed (we should always prioritize time dimension chunking).
    chunks_needed = int(np.ceil(total_size / max_response_bytes))
    
    # For 3D data (time, lat, lon), chunk along time dimension.
    if len(total_shape) == 3:
        time_chunk_size = max(1, total_shape[0] // chunks_needed)
        chunk_shape = (time_chunk_size, total_shape[1], total_shape[2])
        actual_chunks = int(np.ceil(total_shape[0] / time_chunk_size))
    else:
        # For other dimensions, chunk along first dimension.
        first_dim_chunk = max(1, total_shape[0] // chunks_needed)
        chunk_shape = (first_dim_chunk,) + total_shape[1:]
        actual_chunks = int(np.ceil(total_shape[0] / first_dim_chunk))
    
    return {
        "needs_chunking": True,
        "total_chunks": actual_chunks,
        "chunk_shape": chunk_shape,
        "estimated_size": estimate_response_size(chunk_shape)
    }

def get_s3_file_path(
    scenario: str = "scenarioSSP5-85",
    ensemble_member: str = "r15i1p1f1",
    frequency: str = "Amon",
    variable: str = "tas",
    grid: str = "gr3",
    version: str = "v20210201",
    filename: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> str:
    """Build the S3 file path for a SPEAR NetCDF file.

    For time-chunked data (like 6hr), finds the file containing the requested date range.
    """
    fs = s3fs.S3FileSystem(anon=True)
    dir_path = f"noaa-gfdl-spear-large-ensembles-pds/SPEAR/GFDL-LARGE-ENSEMBLES/CMIP/NOAA-GFDL/GFDL-SPEAR-MED/{scenario}/{ensemble_member}/{frequency}/{variable}/{grid}/{version}"

    if filename:
        return f"s3://{dir_path}/{filename}"
    else:
        nc_files = get_cached_file_list(dir_path)
        if not nc_files:
            raise ValueError(f"No NetCDF files found in directory: {dir_path}")

        # Find the file that contains the requested date range
        best_file, _ = find_file_for_date_range(nc_files, start_date, end_date)
        if best_file:
            return f"s3://{best_file}"
        return f"s3://{nc_files[0]}"


def load_dataset_if_needed(
    scenario: str = "scenarioSSP5-85",
    ensemble_member: str = "r15i1p1f1",
    frequency: str = "Amon",
    variable: str = "tas",
    grid: str = "gr3",
    version: str = "v20210201",
    filename: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Load dataset with LAZY loading - does NOT download entire file.
    Data is only fetched when actually accessed after subsetting.

    For time-chunked data (like 6hr), automatically finds the file containing the requested date range.
    """
    global _cached_dataset, _cached_file_path

    s3_path = get_s3_file_path(scenario, ensemble_member, frequency, variable, grid, version, filename, start_date, end_date)

    if _cached_dataset is None or _cached_file_path != s3_path:
        # Use xarray with fsspec for lazy S3 access - NO .load() call!
        # Data will only be downloaded when we actually access values after subsetting
        # IMPORTANT: Use storage_options for anonymous S3 access (public bucket)
        _cached_dataset = xr.open_dataset(
            s3_path,
            engine="h5netcdf",
            chunks={},  # Enable dask chunking for lazy loading
            storage_options={"anon": True}  # Anonymous access to public S3 bucket
        )
        _cached_file_path = s3_path

    return _cached_dataset

def query_netcdf_data(
    variable: str = "tas",   # Default values, the LLM will replace the values when fuction calling.
    start_date: str = None,
    end_date: str = None,
    lat_range: List[float] = None,
    lon_range: List[float] = None,
    chunk_index: int = 0,
    scenario: str = "scenarioSSP5-85",
    ensemble_member: str = "r15i1p1f1",
    frequency: str = "Amon",
    grid: str = "gr3",
    version: str = "v20210201"
) -> Dict[str, Any]:
    """
    Query NetCDF output with spatial/temporal subsetting and chunking
    
    Args:
        variable: Variable name (e.g., "tas")
        start_date: Start date in YYYY-MM format (e.g., "2020-01")
        end_date: End date in YYYY-MM format (e.g., "2021-12")
        lat_range: [min_lat, max_lat] in degrees
        lon_range: [min_lon, max_lon] in degrees
        chunk_index: Which chunk to return (0-based) if data needs chunking
        scenario: "historical" or "scenarioSSP5-85"
        ensemble_member: e.g., "r15i1p1f1"
        frequency: e.g., "Amon"
        grid: e.g., "gr3"
        version: e.g., "v20210201"
    """
    try:
        # Validate parameters first.
        validation = validate_query_parameters(start_date, end_date, lat_range, lon_range, variable,
                                             scenario, ensemble_member, frequency, grid, version)
        # Check if validation returned an error (e.g., file not found)
        if "error" in validation:
            return validation
        if not validation["valid"]:
            return {
                "error": "Invalid query parameters",
                "details": validation["errors"],
                "warnings": validation.get("warnings", [])
            }
        
        # Load dataset (with date range for chunked files like 6hr).
        ds = load_dataset_if_needed(scenario, ensemble_member, frequency, variable, grid, version,
                                     start_date=start_date, end_date=end_date)

        # Select variable.
        if variable not in ds.data_vars:
            return {"error": f"Variable '{variable}' not found in dataset"}

        data_var = ds[variable]

        # Spatial subsetting with automatic coordinate conversion
        data_var, coordinate_adjustments = subset_spatial(data_var, ds, lat_range, lon_range)

        # Extract temporal selection.
        if start_date or end_date:
            time_slice = {}
            if start_date:
                time_slice['start'] = start_date
            if end_date:
                time_slice['stop'] = end_date
            data_var = data_var.sel(time=slice(time_slice.get('start'), time_slice.get('stop')))
        
        # Check if chunking is needed for this data return.
        chunk_info = calculate_chunk_size(data_var.shape)
        
        if chunk_info["needs_chunking"]:
            if chunk_index >= chunk_info["total_chunks"]:
                return {"error": f"Chunk index {chunk_index} exceeds total chunks {chunk_info['total_chunks']}"}
            
            # Calculate chunk slice.
            if len(data_var.shape) == 3:  # (time, lat, lon)
                time_chunk_size = chunk_info["chunk_shape"][0]
                start_idx = chunk_index * time_chunk_size
                end_idx = min((chunk_index + 1) * time_chunk_size, data_var.shape[0])
                data_chunk = data_var.isel(time=slice(start_idx, end_idx))
            else:
                # Handle other dimensions.
                first_dim_chunk = chunk_info["chunk_shape"][0]
                start_idx = chunk_index * first_dim_chunk
                end_idx = min((chunk_index + 1) * first_dim_chunk, data_var.shape[0])
                data_chunk = data_var.isel({data_var.dims[0]: slice(start_idx, end_idx)})
        else:
            data_chunk = data_var
        
        # Convert to JSON-serializable format, which is necesary for proper communication of data to the LLM.
        result = {
            "variable": variable,
            "file_info": {
                "scenario": scenario,
                "ensemble_member": ensemble_member,
                "frequency": frequency,
                "grid": grid,
                "version": version
            },
            "query_parameters": {
                "start_date": start_date,
                "end_date": end_date,
                "lat_range": lat_range,
                "lon_range": lon_range
            },
            "data_info": {
                "shape": list(data_chunk.shape),
                "dimensions": list(data_chunk.dims),
                "dtype": str(data_chunk.dtype)
            },
            "chunking_info": {
                "is_chunked": chunk_info["needs_chunking"],
                "current_chunk": chunk_index,
                "total_chunks": chunk_info["total_chunks"],
                "estimated_response_size_bytes": chunk_info["estimated_size"]
            },
            "coordinates": {},
            "data": make_json_serializable(data_chunk.values.tolist()),
            "attributes": make_json_serializable(dict(data_chunk.attrs))
        }
        
        # Add coordinate information.
        for coord_name in data_chunk.coords:
            coord = data_chunk.coords[coord_name]
            result["coordinates"][coord_name] = {
                "values": make_json_serializable(coord.values.tolist()),
                "attributes": make_json_serializable(dict(coord.attrs))
            }
        
        # Add warnings if there are any.
        if validation.get("warnings"):
            result["warnings"] = validation["warnings"]

        # Add coordinate adjustment info for transparency
        if coordinate_adjustments:
            result["coordinate_adjustments"] = coordinate_adjustments
            result["note"] = "Coordinates were snapped to nearest grid points. See 'coordinate_adjustments' for details."

        return result
        
    except Exception as e:
        return {"error": f"Query failed: {str(e)}"}

def get_data_summary_statistics(
    variable: str = "tas",  # Default values, the LLM will replace the values when fuction calling.
    start_date: str = None,
    end_date: str = None,
    lat_range: List[float] = None,
    lon_range: List[float] = None,
    scenario: str = "scenarioSSP5-85",
    ensemble_member: str = "r15i1p1f1",
    frequency: str = "Amon",
    grid: str = "gr3",
    version: str = "v20210201"
) -> Dict[str, Any]:
    """
    Get summary statistics for data without returning full arrays. **This tool is still being developed and doesnt yet return 'true statistics'.**
    """
    try:
        # Validate parameters.
        validation = validate_query_parameters(start_date, end_date, lat_range, lon_range, variable,
                                             scenario, ensemble_member, frequency, grid, version)
        # Check if validation returned an error (e.g., file not found)
        if "error" in validation:
            return validation
        if not validation["valid"]:
            return {"error": "Invalid parameters", "details": validation["errors"]}

        ds = load_dataset_if_needed(scenario, ensemble_member, frequency, variable, grid, version,
                                     start_date=start_date, end_date=end_date)
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
        
        # Calculate statistics (work in progress!)
        stats = {
            "variable": variable,
            "file_info": {
                "scenario": scenario,
                "ensemble_member": ensemble_member,
                "frequency": frequency,
                "grid": grid,
                "version": version
            },
            "query_parameters": {
                "start_date": start_date,
                "end_date": end_date,
                "lat_range": lat_range,
                "lon_range": lon_range
            },
            "shape": list(data_var.shape),
            "data_size_info": {
                "total_values": int(np.prod(data_var.shape)),
                "estimated_full_size_mb": estimate_response_size(data_var.shape) / 1024 / 1024,
                "would_need_chunking": calculate_chunk_size(data_var.shape)["needs_chunking"]
            }
        }
        
        return stats # More to come!
        
    except Exception as e:
        return {"error": f"Statistics calculation failed: {str(e)}"}

# For Claude we need the dattime to be in iso format for plotting. Its very possible this will be changed with other LLM implementation.
def convert_cftime_to_string(obj):
    """Convert cftime objects to ISO format strings"""
    if isinstance(obj, (cftime._cftime.DatetimeJulian, cftime._cftime.DatetimeGregorian, 
                       cftime._cftime.DatetimeNoLeap, cftime._cftime.Datetime360Day)):
        return obj.isoformat()
    return obj

def make_json_serializable(obj):
    """Recursively convert objects to JSON-serializable format. This is meant to be an as-needed helper function."""
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
    elif isinstance(obj, (cftime._cftime.DatetimeJulian, cftime._cftime.DatetimeGregorian,
                         cftime._cftime.DatetimeNoLeap, cftime._cftime.Datetime360Day)):
        return obj.isoformat()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif not isinstance(obj, (str, int, float, bool, type(None))):
        return str(obj)
    return obj

def test_spear_connection():
    """Test basic S3 connection to SPEAR bucket. Useful for development. Can modify in the future for other tests!"""
    try:
        fs = s3fs.S3FileSystem(anon=True)
        bucket_path = "noaa-gfdl-spear-large-ensembles-pds"
        files = fs.ls(bucket_path, detail=False)[:5]
        return {
            "status": "success",
            "message": "Successfully connected to SPEAR S3 bucket",
            "sample_files": files
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}