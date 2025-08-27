"""MCP tool script for working with the public SPEAR NetCDF output on AWS server."""
# Copied from other tools.py. Will clean up later!
import asyncio
import calendar
from typing import Annotated, List, Dict, Optional, Tuple, Union, Any
from urllib.parse import quote, urlparse, urljoin
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
from bs4 import BeautifulSoup
import re
import logging
import tempfile
import os
import json
import s3fs
import cftime
import datetime
from datetime import datetime as dt
import sys

# Global variables to cache loaded dataset (necessary).
_cached_dataset = None
_cached_file_path = None

def get_s3_file_metadata_only(
    scenario: str = "scenarioSSP5-85",   # Default values, the LLM will replace the values when fuction calling.
    ensemble_member: str = "r15i1p1f1", 
    frequency: str = "Amon",
    variable: str = "tas",
    grid: str = "gr3",
    version: str = "v20210201"
) -> Dict[str, Any]:
    """
    Get ONLY the metadata of a SPEAR NetCDF file without loading data arrays.
    
    Args:
        scenario: "historical" or "scenarioSSP5-85"
        ensemble_member: e.g., "r1i1p1f1", "r15i1p1f1"
        frequency: e.g., "Amon" (monthly), "day" (daily)
        variable: e.g., "tas", "pr", "ua"
        grid: e.g., "gr3"
        version: e.g., "v20210201"
    
    Returns:
        Dictionary containing only file metadata, dimensions, coordinates info.
    """
    try:
        # Construct the file path based on the queried scenario.
        if scenario == "historical":
            date_range = "192101-201412"  # Historical span 1921-2014.
        elif scenario == "scenarioSSP5-85":
            date_range = "201501-210012"  # SSP5-8.5 runs span 2015-2100.
        else:
            return {"error": f"Unknown scenario: {scenario}. Use 'historical' or 'scenarioSSP5-85' or add new scenario and date range to options."}
        
        filename = f"{variable}_{frequency}_GFDL-SPEAR-MED_{scenario}_{ensemble_member}_{grid}_{date_range}.nc"
        
        s3_file_path = f"noaa-gfdl-spear-large-ensembles-pds/SPEAR/GFDL-LARGE-ENSEMBLES/CMIP/NOAA-GFDL/GFDL-SPEAR-MED/{scenario}/{ensemble_member}/{frequency}/{variable}/{grid}/{version}/{filename}"
        
        fs = s3fs.S3FileSystem(anon=True)
        
        # Check if queried file exists.
        if not fs.exists(s3_file_path):
            return {"error": f"File not found: {s3_file_path}"}
        
        # Open dataset but DO NOT load data arrays!
        with fs.open(s3_file_path, mode="rb") as f:
            ds = xr.open_dataset(f, engine="h5netcdf", decode_cf=True)
            
            # Get basic file info.
            metadata = {
                "file_path": s3_file_path,
                "filename": filename,
                "file_size_info": {
                    "dimensions": dict(ds.dims),
                    "total_variables": len(ds.data_vars),
                    "total_coordinates": len(ds.coords)
                },
                "dimensions": dict(ds.dims),
                "coordinates": {},
                "variables": {},
                "global_attributes": make_json_serializable(dict(ds.attrs))
            }
            
            # Get coordinate information (dimensions, shape, etc.). 
            for coord_name, coord in ds.coords.items():
                coord_info = {
                    "dimensions": list(coord.dims),
                    "shape": list(coord.shape),
                    "dtype": str(coord.dtype),
                    "attributes": make_json_serializable(dict(coord.attrs))
                }
                
                # For small coordinate arrays, include values.
                if coord.size <= 1000:  # Only load small coordinate arrays (the MCP server will only accept JSON responses < or = 1mb)
                    coord_info["values"] = make_json_serializable(coord.values.tolist())
                else:
                    # Will need to handle min/max values safely for different data types.
                    min_val = coord.min().values
                    max_val = coord.max().values
                    
                    # Convert to JSON-serializable format (cftime conversions are included).
                    min_serialized = make_json_serializable(min_val)
                    max_serialized = make_json_serializable(max_val)
                    
                    coord_info["values_info"] = {
                        "size": int(coord.size),
                        "min": min_serialized,
                        "max": max_serialized,
                        "first_few": make_json_serializable(coord.values[:5].tolist()),
                        "last_few": make_json_serializable(coord.values[-5:].tolist())
                    }
                
                metadata["coordinates"][coord_name] = coord_info
            
            # Get all variable information (metadata ONLY, NO data with this tool).
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
    version: str = "v20210201"
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
    
    Returns:
        Dictionary containing file info plus data ranges for validation.
    """
    try:
        # Construct the file path based on scenario provided in the query.
        if scenario == "historical":
            date_range = "192101-201412"  # Historical runs span 1921-2014.
        elif scenario == "scenarioSSP5-85":
            date_range = "201501-210012"  # SSP5-8.5 runs span 2015-2100.
        else:
            return {"error": f"Unknown scenario: {scenario}. Use 'historical' or 'scenarioSSP5-85'"}
        
        filename = f"{variable}_{frequency}_GFDL-SPEAR-MED_{scenario}_{ensemble_member}_{grid}_{date_range}.nc"
        
        s3_file_path = f"noaa-gfdl-spear-large-ensembles-pds/SPEAR/GFDL-LARGE-ENSEMBLES/CMIP/NOAA-GFDL/GFDL-SPEAR-MED/{scenario}/{ensemble_member}/{frequency}/{variable}/{grid}/{version}/{filename}"
        
        fs = s3fs.S3FileSystem(anon=True)
        
        # Check if the nc file exists.
        if not fs.exists(s3_file_path):
            return {"error": f"File not found: {s3_file_path}"}
        
        with fs.open(s3_file_path, mode="rb") as f:
            ds = xr.open_dataset(f, engine="h5netcdf", decode_cf=True)
            
            # Extract the time range.
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
                    "total_steps": len(time_values)
                },
                "variables": variables,
                "global_attributes": make_json_serializable(dict(ds.attrs)),
                "note": "Use this info to validate query parameters"
            }
            
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
        
        # Validate spatial range.
        if "spatial_range" in file_info:
            if "latitude" in file_info["spatial_range"] and lat_range:
                file_lat = file_info["spatial_range"]["latitude"]
                if lat_range[0] < file_lat[0] or lat_range[1] > file_lat[1]:
                    warnings.append(f"Requested lat range {lat_range} extends beyond file range {file_lat}")
            
            if "longitude" in file_info["spatial_range"] and lon_range:
                file_lon = file_info["spatial_range"]["longitude"]
                if lon_range[0] < file_lon[0] or lon_range[1] > file_lon[1]:
                    warnings.append(f"Requested lon range {lon_range} extends beyond file range {file_lon}")
        
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

def load_dataset_if_needed(
    scenario: str = "scenarioSSP5-85",  # Default values, the LLM will replace the values when fuction calling.
    ensemble_member: str = "r15i1p1f1",
    frequency: str = "Amon",
    variable: str = "tas",
    grid: str = "gr3", 
    version: str = "v20210201"
):
    """Load dataset into cache if not already loaded"""
    global _cached_dataset, _cached_file_path
    
    # Construct file path.
    if scenario == "historical":
        date_range = "192101-201412"
    elif scenario == "scenarioSSP5-85":
        date_range = "201501-210012"
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    filename = f"{variable}_{frequency}_GFDL-SPEAR-MED_{scenario}_{ensemble_member}_{grid}_{date_range}.nc"
    s3_file_path = f"noaa-gfdl-spear-large-ensembles-pds/SPEAR/GFDL-LARGE-ENSEMBLES/CMIP/NOAA-GFDL/GFDL-SPEAR-MED/{scenario}/{ensemble_member}/{frequency}/{variable}/{grid}/{version}/{filename}"
    
    if _cached_dataset is None or _cached_file_path != s3_file_path:
        fs = s3fs.S3FileSystem(anon=True)
        with fs.open(s3_file_path, mode="rb") as f:
            _cached_dataset = xr.open_dataset(f, engine="h5netcdf").load()
            _cached_file_path = s3_file_path
    
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
        if not validation["valid"]:
            return {
                "error": "Invalid query parameters",
                "details": validation["errors"],
                "warnings": validation.get("warnings", [])
            }
        
        # Load dataset.
        ds = load_dataset_if_needed(scenario, ensemble_member, frequency, variable, grid, version)
        
        # Select variable.
        if variable not in ds.data_vars:
            return {"error": f"Variable '{variable}' not found in dataset"}
        
        data_var = ds[variable]
        
        # Extract spatial selection.
        if lat_range and 'lat' in ds.coords:
            data_var = data_var.sel(lat=slice(lat_range[0], lat_range[1]))
        if lon_range and 'lon' in ds.coords:
            data_var = data_var.sel(lon=slice(lon_range[0], lon_range[1]))
        
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
        if not validation["valid"]:
            return {"error": "Invalid parameters", "details": validation["errors"]}
        
        ds = load_dataset_if_needed(scenario, ensemble_member, frequency, variable, grid, version)
        data_var = ds[variable]
        
        # Apply selections.
        if lat_range and 'lat' in ds.coords:
            data_var = data_var.sel(lat=slice(lat_range[0], lat_range[1]))
        if lon_range and 'lon' in ds.coords:
            data_var = data_var.sel(lon=slice(lon_range[0], lon_range[1]))
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