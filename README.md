#  Model Context Protocol (MCP) Test Server for Accessing SPEAR Model Output
Model Context Protocol is a standarized form of two-way communication for connecting AI applications to external services, devices and databases.
This test MCP server is designed for accessing SPEAR output. There are 3 different 'flavors' of SPEAR MCP server labeled as different branches in this repository. Each branch is catered to a specified SPEAR output location which include AWS hosted, STAC API hosted and local mounted directory hosted files.

The SPEAR MCP server can be utilized in multiple ways. Currently, the most supported method is using the MCP server in parrallel with our SPEAR Climate Chatbot model. You can find this chatbot here:

https://github.com/zappalaja/spear-climate-chatbot

The MCP servers also work well with and have been tested using Claude Desktop and Claude Code.

This is not intended for operational purposes. Contact through GitHub issues for questions. 

# Running the MCP server in terminal

UV will need to be installed in the environment to run the MCP server.

You can download UV from here: https://github.com/astral-sh/uv
```bash
# On macOS and Linux. Run this in your chosen MCP env
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To run the MCP server edit **`start_mcp_server.sh`** as needed, make it executable and run it in terminal:
```bash
chmod +x start_mcp_server.sh
./start_mcp_server.sh
```

# Claude Desktop Install

To download Claude Desktop, follow the instructions from this link:

https://claude.ai/download

Once Claude Desktop is installed, go to the settings and proceed to the 'Developer' section. Click 'Edit config' and insert the correct form of the 'Local Install' into the 'claude_desktop_config.json'.

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
  }
}
```
# Claude Code Install

Claude Code Installation Page:

https://code.claude.com/docs/en/overview

One downloaded in terminal, run this command to connect to the SPEAR MCP server:

```python
claude mcp add spear_mcp -- /bin/bash -c "cd /path/to/spear-mcp-test && uv run spear-mcp"
```

To check if you have successfully connected to the server, check with the "/mcp" command.
