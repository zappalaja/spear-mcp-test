"""Server creation with FastMCP and calling all of the tools defined in 'tools.py' and 'tools_nc.py'."""

import argparse
import asyncio

from fastmcp import FastMCP
from loguru import logger
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from . import tools

##############################################################################################
##############################################################################################
# Add or remove tools as needed.
async def create_server() -> FastMCP:
    """Create and configure the MCP server and register tools"""
    mcp = FastMCP('Test server for SPEAR NetCDF Public data.')

    # Register local-only tools
    mcp.tool()(tools.list_local_directory)
    mcp.tool()(tools.load_netcdf_metadata)
    mcp.tool()(tools.load_netcdf_variable)

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
