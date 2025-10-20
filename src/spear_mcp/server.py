import argparse
import asyncio

from fastmcp import FastMCP
from loguru import logger
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from . import tools  # New STAC + NetCDF tools


async def create_server() -> FastMCP:
    """Create and configure the MCP server and register STAC-aware tools."""
    mcp = FastMCP("SPEAR STAC MCP Server")

    # Register primary conversational tools
    mcp.tool()(tools.query_stac)
    mcp.tool()(tools.load_netcdf_metadata.__wrapped__)
    mcp.tool()(tools.load_netcdf_variable.__wrapped__)

    # Optional helpers for manual browsing or debugging
    # mcp.tool()(tools.list_stac_items)
    # mcp.tool()(tools.get_stac_item_assets)

    # Health endpoint for container / pod monitoring
    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> PlainTextResponse:
        return PlainTextResponse("OK")

    return mcp


async def async_main(transport: str, host: str, port: int):
    """Launch the MCP server using the specified transport."""
    if transport == "stdio":
        # Disable logging if using stdio to avoid interfering with MCP comms
        logger.remove()
        logger.add(lambda _: None)

    server = await create_server()
    logger.info("SPEAR STAC MCP Server launched")

    if transport == "http":
        await server.run_http_async(host=host, port=port)
    elif transport == "stdio":
        await server.run_stdio_async()
    else:
        raise ValueError(f"Unsupported transport: {transport}")


def main():
    """Command-line entrypoint for running the MCP server."""
    parser = argparse.ArgumentParser(description="Run SPEAR STAC MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http", "sse"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    allowed_hosts = ["127.0.0.1", "localhost", "0.0.0.0"]
    if args.host not in allowed_hosts:
        raise ValueError(f"Invalid host: {args.host}")

    asyncio.run(async_main(args.transport, args.host, args.port))
