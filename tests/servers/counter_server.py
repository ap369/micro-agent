"""Stateful MCP test server: `increment` returns how many times it was called in this session."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("counter")
count = 0


@mcp.tool()
def increment() -> str:
    """Increment the counter and return its value."""
    global count
    count += 1
    return str(count)


mcp.run()
