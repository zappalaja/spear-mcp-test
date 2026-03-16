"""Server creation with FastMCP and calling all of the tools defined in 'tools.py' and 'tools_nc.py'."""

import argparse
import asyncio

from fastmcp import FastMCP
from loguru import logger
from starlette.requests import Request
from starlette.responses import PlainTextResponse, JSONResponse

from . import tools, tools_nc, tools_zarr

##############################################################################################
##############################################################################################
# Add or remove tools as needed.
async def create_server() -> FastMCP:
    """Create and configure the MCP server and register tools"""
    mcp = FastMCP('Test server for SPEAR NetCDF Public data.')

    mcp.tool()(tools.validate_spear_url) #######################################
    """
    Check that the SPEAR url is still live and reachable.
    """

    # NEW Tools - For the AWS server specifically.
    mcp.tool()(tools.browse_spear_directory) #Works well
    """
    Dynamically browse SPEAR directory structure step by step.
    Starts with 'empty' path and then navigates deeper by providing path components.
    Example: browse_spear_directory("historical/r1i1p1f1/Amon")
    """
    
    mcp.tool()(tools.navigate_spear_path) #Works well
    """
    Build and navigate to a specific SPEAR path by combining path components.
    Useful for building complete paths step by step.
    Example: navigate_spear_path(["historical", "r1i1p1f1", "Amon"])
    """
    
    mcp.tool()(tools.search_spear_variables) #Works well
    """
    Search for variables across SPEAR datasets matching given criteria.
    Useful for finding specific climate variables across runs and frequencies.
    Example: search_spear_variables("historical", "tas", "Amon")
    """

    # mcp.tool()(tools_nc.truncate_array_values) # Was used for development and testing system RAM capacity.

    
    mcp.tool()(tools_nc.make_json_serializable)
    """
    Recursively convert objects to JSON-serializable format. Handles numpy arrays, 
    cftime objects, and nested data structures. Essential helper for returning 
    complex scientific data through the MCP protocol.
    """

    mcp.tool()(tools_nc.convert_cftime_to_string)
    """
    Convert cftime datetime objects to ISO format strings for JSON compatibility.
    Handles various cftime calendar types (Julian, Gregorian, NoLeap, 360Day).
    """

    mcp.tool()(tools_nc.test_spear_connection)
    """
    Test basic S3 connection to SPEAR bucket and return sample file listings.
    Useful for development and debugging S3 connectivity issues.
    """

    mcp.tool()(tools_nc.get_file_info_and_validation)
    """
    Get comprehensive file information including metadata, dimensions, time ranges,
    and spatial coverage. Returns validation data for verifying query parameters
    against actual file contents.
    Example: get_file_info_and_validation("historical", "r1i1p1f1", "Amon", "tas")
    """

    mcp.tool()(tools_nc.validate_query_parameters)
    """
    Validate query parameters (dates, spatial ranges, variables) against actual
    file data ranges. Returns validation status, errors, and warnings before
    attempting data queries.
    """

    mcp.tool()(tools_nc.estimate_response_size)
    """
    Estimate response size in bytes for given data shape and dtype. Used to
    determine if data needs chunking to stay within MCP response limits (~1MB).
    """

    mcp.tool()(tools_nc.calculate_chunk_size)
    """
    Calculate optimal chunk dimensions to keep responses under size limits.
    Returns chunking strategy and estimated chunk count for large datasets.
    Prioritizes time-dimension chunking for 3D climate data.
    """

    mcp.tool()(tools_nc.load_dataset_if_needed)
    """
    Load NetCDF dataset into memory cache if not already loaded. Maintains
    global cache to avoid repeated S3 reads for the same file. Returns
    cached xarray Dataset object.
    """

    mcp.tool()(tools_nc.query_netcdf_data)
    """
    Query NetCDF data with spatial/temporal subsetting and automatic chunking.
    Main data extraction tool - handles parameter validation, spatial/temporal
    slicing, chunking for large responses, and JSON serialization.
    Example: query_netcdf_data("tas", "2020-01", "2021-12", [30, 50], [-120, -80])
    """

    mcp.tool()(tools_nc.get_data_summary_statistics)
    """
    Get summary statistics for data selections without returning full arrays.
    Currently returns basic shape and size information. Statistical calculations
    are still in development.
    """

    mcp.tool()(tools_nc.get_s3_file_metadata_only)
    """
    Extract only file metadata without loading data arrays. Returns dimensions,
    coordinates, variable information, and attributes. Efficient for exploring
    file structure without memory overhead.
    Example: get_s3_file_metadata_only("scenarioSSP5-85", "r15i1p1f1", "Amon", "pr")
    """

    # ========== ZARR TOOLS (CMIP6) ==========
    mcp.tool()(tools_zarr.test_cmip6_connection)
    """
    Test basic S3 connection to CMIP6 Zarr store and return store information.
    Useful for verifying access to Zarr data on AWS S3.
    Example: test_cmip6_connection("s3://cmip6-pds/CMIP6/CMIP/NOAA-GFDL/GFDL-CM4/historical/r1i1p1f1/Amon/tas/gr1/v20180701/")
    """

    mcp.tool()(tools_zarr.get_zarr_store_info)
    """
    Get metadata from a CMIP6 Zarr store without loading data arrays.
    Returns dimensions, coordinates, variables, and attributes.
    Set include_full_details=True for complete coordinate/variable information.
    Example: get_zarr_store_info(include_full_details=True)
    """

    mcp.tool()(tools_zarr.load_zarr_dataset)
    """
    Load Zarr dataset with lazy loading - data only downloaded when accessed.
    Maintains global cache to avoid repeated S3 reads.
    Example: load_zarr_dataset("s3://cmip6-pds/CMIP6/CMIP/...")
    """

    mcp.tool()(tools_zarr.query_zarr_data)
    """
    Query Zarr data with spatial/temporal subsetting.
    Main data extraction tool for Zarr stores - handles parameter validation,
    spatial/temporal slicing, and JSON serialization.
    Example: query_zarr_data("tas", "1850-01", "1860-12", [30, 50], [-120, -80])
    """

    mcp.tool()(tools_zarr.get_zarr_summary_statistics)
    """
    Get summary statistics (min, max, mean, std) for Zarr data selections
    without returning full arrays. More efficient than loading all data.
    Example: get_zarr_summary_statistics("tas", "1850-01", "1860-12", [30, 50], [-120, -80])
    """

    # Future Tools! Coming soon!
    # mcp.tool()(tools_nc.get_catalog_file_metadata_only)


##############################################################################################
##############################################################################################
# Residual functions. Will explore more in depth.

    # Add health check endpoint, mainly for Docker purposes.
    @mcp.custom_route('/health', methods=['GET'])
    async def health_check(request: Request) -> PlainTextResponse:
        return PlainTextResponse('OK')

    # Expose registered tools as a REST endpoint for the Streamlit UI.
    @mcp.custom_route('/tools', methods=['GET'])
    async def list_tools(request: Request) -> JSONResponse:
        tool_list = []
        tools = await mcp._tool_manager.list_tools()
        for t in tools:
            tool_list.append({
                "name": t.name,
                "description": t.description or "",
                "parameters": t.parameters,
            })
        return JSONResponse(tool_list)

    return mcp

async def async_main(transport: str, host: str, port: int):
    # Disable logging for stdio transport to avoid interfering with MCP protocol.
    if transport == 'stdio':
        logger.remove()
        logger.add(lambda _: None)

    server = await create_server()
    logger.info('Server created with enhanced SPEAR navigation tools')
    if transport == 'stdio':
        await server.run_async(transport='stdio')
    elif transport in ['http', 'sse']:
        # Configure uvicorn with extended timeouts for large S3 data transfers
        uvicorn_config = {
            "timeout_keep_alive": 1800,  # 30 minutes keep-alive timeout
        }
        await server.run_async(transport=transport, host=host, port=port, uvicorn_config=uvicorn_config)


def main():
    parser = argparse.ArgumentParser(
        description='Test server for SPEAR NetCDF Public data with dynamic navigation.'
    )
    parser.add_argument(
        '--transport',
        choices=['stdio', 'http', 'sse'],
        default='sse',
        help='Transport protocol to use (default: sse for HTTP mode)',
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to for http/sse transport (default: 0.0.0.0 for container access)',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind to for http/sse transport (default: 8000)',
    )

    args = parser.parse_args()

    # Limit what host can be
    allowed_hosts = ['127.0.0.1', 'localhost', '0.0.0.0']
    if args.host not in allowed_hosts:
        raise ValueError(f"Host '{args.host}' not allowed. Use one of: {allowed_hosts}")

    # A separate sync main function is needed because it is the entry point
    asyncio.run(async_main(args.transport, args.host, args.port))
