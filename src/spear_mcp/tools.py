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

# Define base SPEAR scenarios
SPEAR_SCENARIOS = ["historical", "scenarioSSP5-85"]   # We will likely want to move away from this type of structure.
SPEAR_BASE_URL = "https://noaa-gfdl-spear-large-ensembles-pds.s3.amazonaws.com"
SPEAR_BASE_PATH = "SPEAR/GFDL-LARGE-ENSEMBLES/CMIP/NOAA-GFDL/GFDL-SPEAR-MED"

logger = logging.getLogger(__name__)

class SPEARNavigationResult:
    """Result from SPEAR directory navigation."""
    def __init__(self, current_path: str, directories: List[str], files: List[str], 
                 parent_path: Optional[str] = None):
        self.current_path = current_path
        self.directories = directories
        self.files = files
        self.parent_path = parent_path
        
    def to_dict(self) -> Dict:
        return {
            "current_path": self.current_path,
            "directories": self.directories,
            "files": self.files,
            "parent_path": self.parent_path,
            "full_url": f"{SPEAR_BASE_URL}/{SPEAR_BASE_PATH}/{self.current_path}" if self.current_path else f"{SPEAR_BASE_URL}/{SPEAR_BASE_PATH}"
        }

# Fixed datetime handling functions
def _safe_datetime_conversion(time_coord):
    """Safely convert various time coordinate formats to pandas datetime."""
    try:
        # Handle cftime objects
        if hasattr(time_coord.values[0], 'datetime'):
            # cftime objects - convert to pandas datetime
            times = []
            for t in time_coord.values:
                if hasattr(t, 'year'):
                    times.append(pd.Timestamp(t.year, t.month, t.day))
                else:
                    times.append(pd.to_datetime(str(t)))
            return pd.DatetimeIndex(times)
        else:
            # Standard numpy datetime64 or already pandas
            return pd.to_datetime(time_coord.values)
    except Exception as e:
        logger.warning(f"Could not convert time coordinate: {e}")
        # Fallback: create a simple index # I think this should be removed.
        return pd.date_range('1900-01-01', periods=len(time_coord), freq='M')

async def browse_spear_directory(path: str = "") -> Dict:
    """
    Dynamically browse SPEAR directory structure step by step.
    
    This tool allows the LLM to navigate through the SPEAR data portal by building
    paths incrementally, starting from the root and exploring available subdirectories
    and files at each level.
    
    Args:
        path: Relative path from SPEAR base (e.g., "historical" or "historical/r1i1p1f1/Amon")
        
    Returns:
        Dict containing:
        - current_path: The current directory path
        - directories: List of subdirectories available at this level
        - files: List of files available at this level  
        - parent_path: Path to parent directory (if any)
        - full_url: Complete URL to current directory
        
    Raises:
        ValueError: If path is invalid or directory cannot be accessed.
    """
    # Clean and validate the path
    clean_path = _clean_path(path)
    
    # Build the full path for browsing
    if clean_path:
        full_browse_path = f"{SPEAR_BASE_PATH}/{clean_path}"
    else:
        full_browse_path = SPEAR_BASE_PATH
    
    try:
        async with aiohttp.ClientSession() as session:
            # Try S3 listing API first
            listing_url = f"{SPEAR_BASE_URL}/?list-type=2&prefix={full_browse_path}/&delimiter=/"
            logger.info(f"Browsing SPEAR directory: {listing_url}")
            
            async with session.get(listing_url, timeout=30) as response:
                if response.status == 200:
                    xml_content = await response.text()
                    result = _parse_s3_directory_listing(xml_content, full_browse_path)
                    if result:
                        return result.to_dict()
                
                logger.info("S3 API listing failed, trying direct access")
            
            # Fallback: try direct directory access
            direct_url = f"{SPEAR_BASE_URL}/{full_browse_path}/"
            async with session.get(direct_url, timeout=30) as response:
                if response.status == 200:
                    content = await response.text()
                    result = _parse_directory_content(content, clean_path)
                    if result:
                        return result.to_dict()
                
                logger.warning(f"Direct access failed: HTTP {response.status}")
        
        # If all else fails, provide known structure for common paths
        result = _get_known_structure(clean_path)
        return result.to_dict()
        
    except asyncio.TimeoutError as e:
        raise ValueError(f"Timeout accessing directory after 30 seconds") from e
    except Exception as e:
        logger.error(f"Error browsing SPEAR directory '{path}': {e}")
        raise ValueError(f"Failed to browse directory: {e}") from e

async def navigate_spear_path(path_components: List[str]) -> Dict:
    """
    Build and navigate to a specific SPEAR path by combining path components.
    
    This tool helps the LLM build complete paths step by step by providing
    a list of path components that get joined together.
    
    Args:
        path_components: List of path parts (e.g., ["historical", "r1i1p1f1", "Amon", "tas"])
        
    Returns:
        Dict with navigation result including available subdirectories and files
        
    Example:
        navigate_spear_path(["historical", "r1i1p1f1"]) 
        -> browses to historical/r1i1p1f1/ directory
    """
    # Join path components with forward slashes
    full_path = "/".join(str(component).strip("/") for component in path_components if component)
    
    return await browse_spear_directory(full_path)

async def search_spear_variables(scenario: str, variable_pattern: str = "", 
                                frequency: str = "") -> List[Dict]:
    """
    Search for variables across SPEAR datasets matching given criteria.
    
    Args:
        scenario: "historical" or "scenarioSSP5-85"
        variable_pattern: Pattern to match variable names (e.g., "tas", "pr", "temp")
        frequency: Data frequency ("Amon" for monthly, "day" for daily, etc.)
        
    Returns:
        List of dictionaries containing variable information and paths.
    """
    if scenario not in SPEAR_SCENARIOS:
        raise ValueError(f"Invalid scenario. Must be one of: {SPEAR_SCENARIOS}")
    
    results = []
    
    try:
        # Browse the options in the SPEAR scenario directory.
        scenario_content = await browse_spear_directory(scenario)
        
        # Look through available ensemble member runs.
        for run in scenario_content.get("directories", []):
            run_path = f"{scenario}/{run}"
            
            # Browse each run directory.
            run_content = await browse_spear_directory(run_path)
            
            # Look through output frequency directories (daily, monthly, etc.).
            freq_dirs = run_content.get("directories", [])
            if frequency:
                freq_dirs = [d for d in freq_dirs if frequency.lower() in d.lower()]
            
            for freq_dir in freq_dirs:
                freq_path = f"{run_path}/{freq_dir}"
                freq_content = await browse_spear_directory(freq_path)
                
                # Look through variable directories.
                var_dirs = freq_content.get("directories", [])
                if variable_pattern:
                    var_dirs = [v for v in var_dirs if variable_pattern.lower() in v.lower()]
                
                for var_dir in var_dirs:
                    results.append({
                        "scenario": scenario,
                        "run": run,
                        "frequency": freq_dir,
                        "variable": var_dir,
                        "path": f"{freq_path}/{var_dir}"
                    })
    
    except Exception as e:
        logger.error(f"Error searching variables: {e}")
        raise ValueError(f"Search failed: {e}") from e
    
    return results

# I think this tool can be removed now since I have a version of it in 'tools_nc.py' now. Will take action shortly!
def _convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj

def _clean_path(path: str) -> str:
    """Clean and normalize a path string."""
    if not path:
        return ""
    
    # Remove leading/trailing slashes and normalize.
    clean = path.strip("/").replace("\\", "/")
    
    # Remove any double slashes.
    while "//" in clean:
        clean = clean.replace("//", "/")
    
    return clean

def _parse_s3_directory_listing(xml_content: str, current_full_path: str) -> Optional[SPEARNavigationResult]:
    """Parse S3 XML directory listing to extract directories and files."""
    try:
        from xml.etree import ElementTree as ET
        root = ET.fromstring(xml_content)
        
        directories = []
        files = []
        prefix_base = f"{current_full_path}/"
        
        # Extract directories from 'CommonPrefixes'.
        for common_prefix in root.findall('.//{http://s3.amazonaws.com/doc/2006-03-01/}CommonPrefixes'):
            prefix_elem = common_prefix.find('{http://s3.amazonaws.com/doc/2006-03-01/}Prefix')
            if prefix_elem is not None:
                full_prefix = prefix_elem.text
                if full_prefix.startswith(prefix_base):
                    dir_name = full_prefix[len(prefix_base):].rstrip('/')
                    if dir_name:
                        directories.append(dir_name)
        
        # Extract files from 'Contents'.
        for content in root.findall('.//{http://s3.amazonaws.com/doc/2006-03-01/}Contents'):
            key_elem = content.find('{http://s3.amazonaws.com/doc/2006-03-01/}Key')
            if key_elem is not None:
                full_key = key_elem.text
                if full_key.startswith(prefix_base) and full_key != prefix_base:
                    file_name = full_key[len(prefix_base):]
                    if '/' not in file_name:  # Only direct files, not subdirectory files
                        files.append(file_name)
        
        # Create relative path and parent path.
        relative_path = current_full_path.replace(f"{SPEAR_BASE_PATH}/", "").replace(SPEAR_BASE_PATH, "")
        parent_path = "/".join(relative_path.split("/")[:-1]) if "/" in relative_path else None
        
        return SPEARNavigationResult(
            current_path=relative_path,
            directories=sorted(directories),
            files=sorted(files),
            parent_path=parent_path
        )
        
    except Exception as e:
        logger.error(f"Failed to parse S3 XML listing: {e}")
        return None

def _parse_directory_content(content: str, current_path: str) -> Optional[SPEARNavigationResult]:
    """Parse directory content (HTML or XML) to extract directories and files."""
    try:
        directories = []
        files = []
        
        # Parse as HTML first.
        soup = BeautifulSoup(content, 'html.parser')
        
        for link in soup.find_all('a'):
            href = link.get('href', '')
            text = link.get_text().strip()
            
            # Skip parent directory links.
            if href in ['../', '../', '..']:
                continue
            
            # Check if it's a directory (ends with /).
            if href.endswith('/'):
                dir_name = href.rstrip('/')
                if dir_name and dir_name not in directories:
                    directories.append(dir_name)
            else:
                # It's a file.
                if href and href not in files:
                    files.append(href)
        
        # Create parent path.
        parent_path = "/".join(current_path.split("/")[:-1]) if "/" in current_path else None
        
        return SPEARNavigationResult(
            current_path=current_path,
            directories=sorted(directories),
            files=sorted(files),
            parent_path=parent_path
        )
        
    except Exception as e:
        logger.error(f"Failed to parse directory content: {e}")
        return None

def _get_known_structure(path: str) -> SPEARNavigationResult:
    """Provide known directory structure for common SPEAR paths."""
    path_parts = path.split("/") if path else []
    
    # Root level, displays scenarios.
    if not path:
        return SPEARNavigationResult("", SPEAR_SCENARIOS, [], None)
    
    # Scenario level, displays the different ensemble member runs.
    if len(path_parts) == 1 and path_parts[0] in SPEAR_SCENARIOS:
        runs = [f"r{i}i1p1f1" for i in range(1, 31)]
        return SPEARNavigationResult(path, runs, [], "")
    
    # Run level, displays the available output frequency directories
    if len(path_parts) == 2 and path_parts[0] in SPEAR_SCENARIOS:
        freq_dirs = ["Amon", "day", "fx", "Ofx"]
        return SPEARNavigationResult(path, freq_dirs, [], path_parts[0])
    
    # Frequency level, displays available variables.
    if len(path_parts) == 3:
        freq = path_parts[2]
        if freq == "Amon":
            variables = ["tas", "pr", "ps", "hur", "hus", "ua", "va"]
        elif freq == "day":
            variables = ["tas", "pr", "psl"]
        else:
            variables = ["areacella", "areacello", "orog"]
        
        parent = "/".join(path_parts[:2])
        return SPEARNavigationResult(path, variables, [], parent)
    
    # Default value fallback to the root.
    parent = "/".join(path_parts[:-1]) if len(path_parts) > 1 else ""
    return SPEARNavigationResult(path, [], [], parent)

def _extract_time_range(dataset: xr.Dataset) -> Optional[Dict]:
    """Extract time range information with robust datetime handling."""
    try:
        if 'time' not in dataset.coords:
            return None
        
        time_coord = dataset.time
        
        # Datetime conversion.
        time_index = _safe_datetime_conversion(time_coord)
        
        return {
            "start": time_index[0].strftime("%Y-%m-%d %H:%M:%S"),
            "end": time_index[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "length": len(time_coord),
            "frequency": _infer_frequency(time_index)
        }
    except Exception as e:
        logger.warning(f"Time range extraction failed: {e}")
        return {
            "start": "unknown",
            "end": "unknown", 
            "length": len(dataset.time) if 'time' in dataset.coords else 0,
            "frequency": "unknown"
        }

def _infer_frequency(time_index) -> str:
    """Infer frequency from pandas datetime index."""
    try:
        if len(time_index) < 2:
            return "unknown"
        
        # Calculate differences in the output frequencies.
        diffs = time_index[1:5] - time_index[0:4]
        avg_diff = diffs[0]
        
        if avg_diff.days >= 28 and avg_diff.days <= 31:
            return "monthly"
        elif avg_diff.days == 1:
            return "daily"
        elif avg_diff.days >= 365:
            return "yearly"
        elif avg_diff.total_seconds() <= 3600:
            return "hourly"
        else:
            return f"{avg_diff.days}days"
    except Exception:
        return "unknown"

@alru_cache(maxsize=5, ttl=3600)
async def _cached_open_dataset(spear_url: str) -> xr.Dataset:
    """Cached dataset opening without timeout to avoid cache key issues."""
    logger.info('Opening dataset (cache miss): {url}', url=spear_url)
    return await asyncio.to_thread(xr.open_dataset, spear_url)

async def open_dataset(spear_url: str, timeout: int = 30) -> xr.Dataset:
    """Get dataset from cache or open it if not cached. Cache expires after 1 hour."""
    validate_spear_url(spear_url)

    try:
        return await asyncio.wait_for(_cached_open_dataset(spear_url), timeout=timeout)
    except TimeoutError as e:
        raise ValueError(
            f'Dataset access timed out after {timeout} seconds. This dataset may be '
            'very large or the server is slow. Try a different dataset or retry later.'
        ) from e

def validate_spear_url(url: str) -> bool:
    """
    Validate that the URL is from the allowed SPEAR domain and path.
    
    Args:
        url: The URL to validate
    Returns:
        True if the URL is allowed, False otherwise
    Raises:
        ValueError: If the URL is not allowed with explanation
    """
    if not url.startswith(SPEAR_BASE_URL):
        raise ValueError(f"URL must start with {SPEAR_BASE_URL}")
    
    # Check if URL contains valid scenario path
    valid_path_found = False
    for scenario in SPEAR_SCENARIOS:
        if f"/{scenario}/" in url or url.endswith(f"/{scenario}"):
            valid_path_found = True
            break
    
    if not valid_path_found:
        raise ValueError(f"URL must contain a valid scenario path: {SPEAR_SCENARIOS}")
    
    return True