"""Server creation with FastMCP"""

import argparse
import asyncio

from fastmcp import FastMCP
from loguru import logger
from starlette.requests import Request
from starlette.responses import PlainTextResponse

# from . import catalog, tools ########### Not sure we need the catalog script since ours is pre-cataloged (?).
from . import tools

##############################################################################################
##############################################################################################

async def create_server() -> FastMCP:
    """Create and configure the MCP server and register tools"""
    mcp = FastMCP('Test server for SPEAR NetCDF Public data.')

    # Register tools here instead of with a decorator

    #These are where you call up you tools you defined in catalog.py and tools.py.
    #Edit as needed to add or remove tools.

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
    
    mcp.tool()(tools.get_spear_file_info) # NOT WORKING!!!
    """
    Get detailed information about a specific SPEAR NetCDF file including
    metadata, dimensions, variables, and access URL.
    Example: get_spear_file_info("historical/r1i1p1f1/Amon/tas/gr3/v20210201/tas_file.nc")
    """
    
    mcp.tool()(tools.search_spear_variables) #Works well
    """
    Search for variables across SPEAR datasets matching given criteria.
    Useful for finding specific climate variables across runs and frequencies.
    Example: search_spear_variables("historical", "tas", "Amon")
    """
##############################################################################################
##############################################################################################

    # Add health check endpoint, mainly for Docker purposes
    @mcp.custom_route('/health', methods=['GET'])
    async def health_check(request: Request) -> PlainTextResponse:
        return PlainTextResponse('OK')

    return mcp

async def async_main(transport: str, host: str, port: int):
    # Disable logging for stdio transport to avoid interfering with MCP protocol
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