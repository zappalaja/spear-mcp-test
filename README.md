# spear-mcp-test
MCP serve for accessing SPEAR climate output.

How to configure in LLM JSON file:
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
