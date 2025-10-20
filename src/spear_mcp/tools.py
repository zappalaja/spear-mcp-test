"""
MCP Tools for navigating the SPEAR STAC browser and extracting NetCDF data.
JSON-based STAC navigation with a high-level query_stac() interface for natural language queries.
"""

import asyncio
import aiohttp
import json
import re
import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional, Dict, Any, List
from async_lru import alru_cache
from loguru import logger
import cftime
from urllib.parse import urljoin

STAC_BASE = "http://pp009.princeton.rdhpcs.noaa.gov:11622"
COLLECTION_ID = "SPEAR-FLP"
COLLECTION_URL = f"{STAC_BASE}/collections/{COLLECTION_ID}"
ITEMS_URL = f"{COLLECTION_URL}/items"

def safe_serialize(val):
    """Convert NumPy, datetime, cftime, and complex types to JSON-safe values."""
    if isinstance(val, (np.generic, np.bool_)):
        return val.item()
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, (pd.Timestamp, np.datetime64)):
        return str(val)
    elif isinstance(val, (cftime.DatetimeNoLeap, cftime.datetime)):
        return str(val)
    elif isinstance(val, (list, tuple)):
        return [safe_serialize(v) for v in val]
    elif isinstance(val, dict):
        return {str(k): safe_serialize(v) for k, v in val.items()}
    return val


async def fetch_json(url: str) -> Dict[str, Any]:
    """Fetch JSON from a STAC endpoint."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.json()

async def query_stac(
    collection: str = COLLECTION_ID,
    scenario: Optional[str] = None,
    variable: Optional[str] = None,
    member: Optional[int] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Search and summarize STAC items from the SPEAR collection.

    Examples:
      query_stac(scenario="SSP585", variable="precip", member=10)
    """

    # 1. Build the base STAC items URL
    url = f"{STAC_BASE}/collections/{collection}/items?limit={limit}"
    logger.info(f"Querying STAC: {url}")

    results = []
    while url:
        data = await fetch_json(url)
        items = data.get("features", [])
        for item in items:
            item_id = item.get("id", "")
            props = item.get("properties", {})
            matched = True

            # 2. Apply filters if provided
            if scenario and scenario.lower() not in item_id.lower():
                matched = False
            if variable and variable.lower() not in item_id.lower():
                matched = False
            if member is not None:
                # Look for patterns like _K10 or _K010 in item_id
                if not re.search(rf"K0*{member}\b", item_id):
                    matched = False

            if matched:
                assets = item.get("assets", {})
                netcdf_assets = {
                    k: v.get("href") for k, v in assets.items() if v.get("type", "").endswith("netcdf")
                }
                results.append(
                    {
                        "id": item_id,
                        "start_time": props.get("start_datetime"),
                        "end_time": props.get("end_datetime"),
                        "assets": netcdf_assets,
                    }
                )

        # 3. Handle pagination if there’s a “next” link
        next_links = [l["href"] for l in data.get("links", []) if l.get("rel") == "next"]
        url = next_links[0] if next_links else None

        if len(results) >= limit:
            break

    return {"count": len(results), "matches": results}


# Helper Tools 

# async def list_stac_items(collection: str = COLLECTION_ID, limit: int = 20) -> Dict[str, Any]:
#     """List all items within a STAC collection."""
#     url = f"{STAC_BASE}/collections/{collection}/items?limit={limit}"
#     data = await fetch_json(url)
#     items = [
#         {"id": f["id"], "start": f["properties"].get("start_datetime"), "end": f["properties"].get("end_datetime")}
#         for f in data.get("features", [])
#     ]
#     return {"collection": collection, "count": len(items), "items": items}


# async def get_stac_item_assets(collection: str, item_id: str) -> Dict[str, Any]:
#     """Return the asset dictionary for a specific STAC item."""
#     url = f"{STAC_BASE}/collections/{collection}/items/{item_id}"
#     data = await fetch_json(url)
#     assets = data.get("assets", {})
#     return {
#         "item": item_id,
#         "asset_count": len(assets),
#         "assets": {k: v.get("href") for k, v in assets.items()},
#     }

@alru_cache(maxsize=8, ttl=3600)
async def load_netcdf_metadata(file_url: str) -> Dict[str, Any]:
    """Load NetCDF metadata via xarray from an HTTP URL."""
    logger.info(f"Loading NetCDF metadata: {file_url}")
    ds = await asyncio.to_thread(xr.open_dataset, file_url, engine="netcdf4", decode_cf=True)
    return {
        "url": file_url,
        "dimensions": {k: int(v) for k, v in ds.dims.items()},
        "variables": list(ds.data_vars.keys()),
        "coords": list(ds.coords.keys()),
        "attrs": safe_serialize(ds.attrs),
    }


@alru_cache(maxsize=8, ttl=1800)
async def load_netcdf_variable(
    file_url: str,
    variable: str,
    time_start: Optional[int] = None,
    time_end: Optional[int] = None,
    lat_slice: Optional[List[float]] = None,
    lon_slice: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Subset a variable from a remote NetCDF file."""
    logger.info(f"Loading variable '{variable}' from {file_url}")
    ds = await asyncio.to_thread(xr.open_dataset, file_url, engine="netcdf4", decode_cf=True)

    if variable not in ds.data_vars:
        raise ValueError(f"Variable '{variable}' not found.")

    var_data = ds[variable]

    # Time slicing
    if time_start is not None and time_end is not None and "time" in var_data.dims:
        var_data = var_data.isel(time=slice(time_start, time_end))

    # Lat/lon slicing
    if lat_slice and "lat" in var_data.coords:
        lat_vals = ds["lat"].values
        i0 = int(pd.Series(lat_vals).sub(lat_slice[0]).abs().idxmin())
        i1 = int(pd.Series(lat_vals).sub(lat_slice[1]).abs().idxmin())
        var_data = var_data.isel(lat=slice(min(i0, i1), max(i0, i1) + 1))

    if lon_slice and "lon" in var_data.coords:
        lon_vals = ds["lon"].values
        j0 = int(pd.Series(lon_vals).sub(lon_slice[0]).abs().idxmin())
        j1 = int(pd.Series(lon_vals).sub(lon_slice[1]).abs().idxmin())
        var_data = var_data.isel(lon=slice(min(j0, j1), max(j0, j1) + 1))

    return {
        "url": file_url,
        "variable": variable,
        "shape": list(var_data.shape),
        "dims": list(var_data.dims),
        "attrs": safe_serialize(dict(var_data.attrs)),
        "data_sample": safe_serialize(var_data.values.tolist()[:1]),
    }

