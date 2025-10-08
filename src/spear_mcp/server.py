import argparse
import asyncio

from fastmcp import FastMCP
from loguru import logger
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from . import tools_local  # NEW: Use local-only tools


async def create_server() -> FastMCP:
    """Create and configure the MCP server and register local tools."""
    mcp = FastMCP("MCP Server for Local NetCDF Directory (Read-Only)")

    # Register local-only tools
    mcp.tool()(tools_local.list_local_directory)
    mcp.tool()(tools_local.load_netcdf_metadata)
    mcp.tool()(tools_local.load_netcdf_variable)

    # Health endpoint for container / pod monitoring
    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> PlainTextResponse:
        return PlainTextResponse("OK")

    return mcp


async def async_main(transport: str, host: str, port: int):
    # Disable logging if using stdio to avoid interfering with MCP comms
    if transport == "stdio":
        logger.remove()
        logger.add(lambda _: None)

    server = await create_server()
    logger.info("MCP Server launched with local tools")
    await server.run_async(transport=transport, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="Run MCP Server for Local NetCDF Directory")
    parser.add_argument("--transport", choices=["stdio", "http", "sse"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    allowed_hosts = ["127.0.0.1", "localhost", "0.0.0.0"]
    if args.host not in allowed_hosts:
        raise ValueError(f"Invalid host: {args.host}")

    asyncio.run(async_main(args.transport, args.host, args.port))
