"""MCP server tools for dynamic SPEAR public AWS portal navigation - NC File Reading is handled in 'tools_nc.py'"""

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

# Root mount path (read-only)
MOUNT_ROOT = "/data/2/GFDL-LARGE-ENSEMBLES/TFTEST"


def _clean_local_path(subpath: str) -> str:
    """Sanitize and ensure subpath stays inside the mount root."""
    clean = os.path.normpath(subpath).lstrip("./")
    full_path = os.path.join(MOUNT_ROOT, clean)
    abs_path = os.path.abspath(full_path)
    if not abs_path.startswith(MOUNT_ROOT):
        raise ValueError("Path traversal outside allowed root is not permitted.")
    return abs_path


async def list_local_directory(subpath: str = "") -> Dict[str, Any]:
    """List directories and .nc files within a subpath from the root."""
    full_path = _clean_local_path(subpath)
    logger.info(f"Browsing local directory: {full_path}")

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Path does not exist: {full_path}")

    directories = []
    netcdf_files = []

    for item in os.listdir(full_path):
        item_path = os.path.join(full_path, item)
        if os.path.isdir(item_path):
            directories.append(item)
        elif os.path.isfile(item_path) and item.endswith(".nc"):
            netcdf_files.append(item)

    parent = os.path.relpath(os.path.dirname(full_path), MOUNT_ROOT)
    if parent == ".":
        parent = None

    return {
        "current_path": os.path.relpath(full_path, MOUNT_ROOT),
        "directories": sorted(directories),
        "netcdf_files": sorted(netcdf_files),
        "parent_path": parent
    }


@alru_cache(maxsize=8, ttl=3600)
async def load_netcdf_metadata(subpath: str) -> Dict[str, Any]:
    """Load only the metadata of a NetCDF file."""
    full_path = _clean_local_path(subpath)

    if not os.path.isfile(full_path) or not full_path.endswith(".nc"):
        raise ValueError("File is not a valid .nc file.")

    logger.info(f"Loading metadata from: {full_path}")

    ds = await asyncio.to_thread(xr.open_dataset, full_path, decode_cf=True)

    return {
        "filename": os.path.basename(full_path),
        "dimensions": dict(ds.dims),
        "variables": list(ds.data_vars.keys()),
        "coords": list(ds.coords.keys()),
        "attrs": dict(ds.attrs),
    }


@alru_cache(maxsize=8, ttl=1800)
async def load_netcdf_variable(
    subpath: str,
    variable: str,
    time_start: Optional[int] = None,
    time_end: Optional[int] = None,
    lat_slice: Optional[List[float]] = None,
    lon_slice: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Load a subset of a variable from a NetCDF file using nearest-match slicing."""
    full_path = _clean_local_path(subpath)

    if not os.path.isfile(full_path) or not full_path.endswith(".nc"):
        raise ValueError("File is not a valid .nc file.")

    logger.info(f"Loading data: {variable} from {full_path}")
    ds = await asyncio.to_thread(xr.open_dataset, full_path, decode_cf=True)

    if variable not in ds.data_vars:
        raise ValueError(f"Variable '{variable}' not found in file.")

    var_data = ds[variable]

    # Apply slicing
    if time_start is not None and time_end is not None and 'time' in var_data.dims:
        var_data = var_data.isel(time=slice(time_start, time_end))

    if lat_slice and 'lat' in var_data.coords:
        lat_vals = ds['lat'].values
        lat_index_start = int(pd.Series(lat_vals).sub(lat_slice[0]).abs().idxmin())
        lat_index_end = int(pd.Series(lat_vals).sub(lat_slice[1]).abs().idxmin())
        var_data = var_data.isel(lat=slice(min(lat_index_start, lat_index_end), max(lat_index_start, lat_index_end) + 1))

    if lon_slice and 'lon' in var_data.coords:
        lon_vals = ds['lon'].values
        lon_index_start = int(pd.Series(lon_vals).sub(lon_slice[0]).abs().idxmin())
        lon_index_end = int(pd.Series(lon_vals).sub(lon_slice[1]).abs().idxmin())
        var_data = var_data.isel(lon=slice(min(lon_index_start, lon_index_end), max(lon_index_start, lon_index_end) + 1))

    return {
        "variable": variable,
        "shape": list(var_data.shape),
        "dims": list(var_data.dims),
        "attrs": dict(var_data.attrs),
        "data_sample": var_data.values.tolist()[:1]  # Only 1st slice for safety
    }
