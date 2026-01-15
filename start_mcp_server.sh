#!/usr/bin/env bash
# Run SPEAR MCP server locally (no container)

#set -e

# --------------------------------------------------
# Ensure uv is available
# --------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' is not installed or not on PATH"
    echo "Install it from: https://github.com/astral-sh/uv"
    exit 1
fi

# --------------------------------------------------
# Optional: load environment variables
# --------------------------------------------------
#if [ -f .env ]; then
#    set -a
#    source .env
#    set +a
#fi

# --------------------------------------------------
# Run MCP server (as defined by the repo)
# --------------------------------------------------
echo "Starting SPEAR MCP server..."
#exec uv run spear-mcp-test/src/spear_mcp
cd ~/Bot_Test/spear-mcp-test
uv run python -m spear_mcp
