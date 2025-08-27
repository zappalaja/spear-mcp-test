"""Server creation with FastMCP and calling all of the tools defined in 'tools.py' and 'tools_nc.py'."""

import argparse
import asyncio

from fastmcp import FastMCP
from loguru import logger
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from . import tools, tools_nc

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
    # Future Tools! Coming soon!
    # mcp.tool()(tools_nc.get_PPan_file_metadata_only)


##############################################################################################
##############################################################################################
# Residual functions. Will explore more in depth.

    # Add health check endpoint, mainly for Docker purposes.
    @mcp.custom_route('/health', methods=['GET'])
    async def health_check(request: Request) -> PlainTextResponse:
        return PlainTextResponse('OK')

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
        await server.run_async(transport=transport, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(
        description='Test server for SPEAR NetCDF Public data with dynamic navigation.'
    )
    parser.add_argument(
        '--transport',
        choices=['stdio', 'http', 'sse'],
        default='stdio',
        help='Transport protocol to use (default: stdio)',
    )
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind to for http/sse transport (default: 127.0.0.1)',
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