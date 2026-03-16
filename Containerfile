# Containerfile for SPEAR MCP Server
# Provides MCP tools for accessing SPEAR climate model data from AWS S3

FROM python:3.13-slim

WORKDIR /app

# Install system dependencies (required for cartopy, netcdf4, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/

# Install the package and dependencies
RUN pip install --no-cache-dir .

# Environment variables (can be overridden at runtime)
ENV PYTHONUNBUFFERED=1

# Expose MCP server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run MCP server with SSE transport (HTTP mode for container networking)
CMD ["spear-mcp", "--transport", "sse", "--host", "0.0.0.0", "--port", "8000"]
