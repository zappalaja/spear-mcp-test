"""MCP server tools for dynamic web-based SPEAR portal navigation and NetCDF inspection."""

import asyncio
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from async_lru import alru_cache
from loguru import logger
import cftime
from typing import Optional, List, Dict, Any
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin

warnings.filterwarnings("ignore")

# Base URL for the web portal (localhost or remote)
# BASE_URL = "http://localhost:8000" #place holder
# BASE_URL = "pp009.princeton.rdhpcs.noaa.gov:11624/collections/SPEAR-FLP"
STARTING_URL = "pp009.princeton.rdhpcs.noaa.gov:11624/collections/SPEAR-FLP?.language=en&.itemFilterOpen=1" # where the datasets are shown
BASE_URL = "pp009.princeton.rdhpcs.noaa.gov:11624/collections/SPEAR-FLP/items" # Where the selected files info gets added to the path.
BASE_URL_AND_ASSET = "pp009.princeton.rdhpcs.noaa.gov:11624/collections/SPEAR-FLP/items" + ITEM_NAME + "?.asset=asset-" + ASSET_NAME #how to access the dropdown off a selected asset.
# next we need to copy the url of the asset and send it back to the user, as a test to see if navigation is working. then we can try to pull the data.
# BASE_URL = "140.208.147.13:11624/collections/SPEAR-FLP" #url second option


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
    else:
        return val


# ---------------------------------------------------------------------------
# Web Navigation Tools
# ---------------------------------------------------------------------------

async def fetch_page(session, url: str) -> str:
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.text()


async def navigate_web_portal(subpath: str = "") -> Dict[str, Any]:
    """
    Browse a webpage under BASE_URL.
    Returns page title, visible links, .nc file links, and short text preview.
    """
    url = urljoin(BASE_URL + "/", subpath)
    if not url.startswith(BASE_URL):
        raise ValueError("Navigation outside the base URL is not permitted.")

    logger.info(f"Navigating web portal: {url}")

    async with aiohttp.ClientSession() as session:
        html = await fetch_page(session, url)
        soup = BeautifulSoup(html, "html.parser")

        links = []
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])
            if href.startswith(BASE_URL):
                links.append(href)

        title = soup.title.string if soup.title else None
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        preview = " ".join(paragraphs[:3])[:500]

        nc_links = [link for link in links if link.endswith(".nc")]

        return {
            "url": url,
            "title": title,
            "links": sorted(set(links)),
            "netcdf_links": sorted(set(nc_links)),
            "text_preview": preview,
        }


async def fetch_link_content(link: str) -> Dict[str, Any]:
    """
    Fetch and summarize the content from a given link within the allowed base URL.
    If the link is a .nc file, basic header info is returned.
    """
    url = urljoin(BASE_URL + "/", link)
    if not url.startswith(BASE_URL):
        raise ValueError("Access outside the base URL is not permitted.")

    logger.info(f"Fetching link content: {url}")

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                title = soup.title.string if soup.title else None
                text = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))[:1000]
                return {
                    "url": url,
                    "type": "html",
                    "title": title,
                    "text_excerpt": text
                }

            elif url.endswith(".nc"):
                return {
                    "url": url,
                    "type": "netcdf",
                    "message": "NetCDF file link detected. Use NetCDF tools to load metadata or variables."
                }

            else:
                size = response.headers.get("Content-Length")
                return {
                    "url": url,
                    "type": "binary",
                    "size_bytes": int(size) if size else None,
                    "message": "Binary or non-HTML file available for download."
                }


# ---------------------------------------------------------------------------
# NetCDF Metadata & Variable Loading (from HTTP)
# ---------------------------------------------------------------------------

@alru_cache(maxsize=8, ttl=3600)
async def load_netcdf_metadata(file_url: str) -> Dict[str, Any]:
    """
    Load only the metadata of a NetCDF file from the web portal via HTTP.
    """
    url = urljoin(BASE_URL + "/", file_url)
    if not url.startswith(BASE_URL):
        raise ValueError("Access outside the base URL is not permitted.")

    logger.info(f"Loading NetCDF metadata from: {url}")
    ds = await asyncio.to_thread(xr.open_dataset, url, engine="netcdf4", decode_cf=True)

    return {
        "url": url,
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
    lon_slice: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Load a subset of a variable from a NetCDF file served via HTTP.
    """
    url = urljoin(BASE_URL + "/", file_url)
    if not url.startswith(BASE_URL):
        raise ValueError("Access outside the base URL is not permitted.")

    logger.info(f"Loading variable '{variable}' from remote dataset: {url}")
    ds = await asyncio.to_thread(xr.open_dataset, url, engine="netcdf4", decode_cf=True)

    if variable not in ds.data_vars:
        raise ValueError(f"Variable '{variable}' not found in dataset.")

    var_data = ds[variable]

    # Apply slicing
    if time_start is not None and time_end is not None and 'time' in var_data.dims:
        var_data = var_data.isel(time=slice(time_start, time_end))

    if lat_slice and 'lat' in var_data.coords:
        lat_vals = ds['lat'].values
        lat_index_start = int(pd.Series(lat_vals).sub(lat_slice[0]).abs().idxmin())
        lat_index_end = int(pd.Series(lat_vals).sub(lat_slice[1]).abs().idxmin())
        var_data = var_data.isel(lat=slice(min(lat_index_start, lat_index_end),
                                           max(lat_index_start, lat_index_end) + 1))

    if lon_slice and 'lon' in var_data.coords:
        lon_vals = ds['lon'].values
        lon_index_start = int(pd.Series(lon_vals).sub(lon_slice[0]).abs().idxmin())
        lon_index_end = int(pd.Series(lon_vals).sub(lon_slice[1]).abs().idxmin())
        var_data = var_data.isel(lon=slice(min(lon_index_start, lon_index_end),
                                           max(lon_index_start, lon_index_end) + 1))

    return {
        "url": url,
        "variable": variable,
        "shape": list(var_data.shape),
        "dims": list(var_data.dims),
        "attrs": safe_serialize(dict(var_data.attrs)),
        "data_sample": safe_serialize(var_data.values.tolist()[:1]),
    }
