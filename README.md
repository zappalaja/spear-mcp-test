# Model Context Protocol (MCP) Server for Accessing NOAA-SPEAR Climate Model Output
An MCP is a standarized form of two-way communication for connecting AI applications to external services, devices and databases.
This MCP server is designed for accessing NOAA SPEAR output. It has been created and testing using Claude Desktop.

To download Claude Desktop, follow the instructions from this link:

https://claude.ai/download

Once Claude Desktop is installed, go to the settings and proceed to the 'Developer' section. Click 'Edit config' and insert the correct form of the 'Local Install' into the 'claude_desktop_config.json'.

# Local Install

Configuration in LLM JSON file:
```python
{
  "mcpServers": {
    "spear_mcp": {
      "command": "uv", // May need a full path, like /opt/homebrew/bin/uv
      "args": [
        "--directory",
        "/Absolute/path/to/spear-mcp",
        "run",
        "spear-mcp"
      ]
    }
  },
}
```
