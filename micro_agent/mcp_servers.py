"""MCP servers from a Claude-Desktop style `mcp.json`, exposed as LangChain tools."""

import json
import os
import re
from contextlib import AsyncExitStack
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.stdio import get_default_environment

from micro_agent.config import ConfigError
from micro_agent.console import log


class McpServers:
    """Connects to every enabled server for the duration of an `async with` block.

    Each server keeps one session for the whole run, so stateful servers (browsers, databases)
    keep their state between tool calls and stdio servers are started only once.

        async with McpServers(path) as tools:
            ...
    """

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> list[BaseTool]:
        connections = self._connections()
        client = MultiServerMCPClient(connections)
        tools: list[BaseTool] = []
        for name in connections:
            server = AsyncExitStack()  # so a server that fails half-way is fully closed
            try:
                session = await server.enter_async_context(client.session(name))
                server_tools = await load_mcp_tools(session, server_name=name)
            except Exception as e:
                await _close_quietly(server, name)
                log(f"[mcp] WARNING: failed to load server '{name}': {e!r}")
                continue
            self._stack.push_async_exit(server)
            tools.extend(server_tools)
            log(f"[mcp] {name}: {len(server_tools)} tool(s) — {', '.join(t.name for t in server_tools)}")
        return tools

    async def __aexit__(self, *exc_info) -> None:
        await self._stack.__aexit__(*exc_info)

    def _connections(self) -> dict[str, dict]:
        if not self.config_path.is_file():
            log(f"[mcp] no config at {self.config_path}, running without MCP tools")
            return {}
        try:
            servers = json.loads(self.config_path.read_text(encoding="utf-8")).get("mcpServers", {})
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in {self.config_path}: {e}") from e
        connections = {
            name: _to_connection(name, cfg)
            for name, cfg in _expand_env(servers).items()
            if not cfg.get("disabled")
        }
        if not connections:
            log("[mcp] no servers configured")
        return connections


def _expand_env(value):
    """Recursively replace ${VAR} placeholders with environment values."""
    if isinstance(value, str):

        def lookup(m: re.Match) -> str:
            if m.group(1) not in os.environ:
                raise ConfigError(f"mcp.json references ${{{m.group(1)}}} but it is not set (add it to .env)")
            return os.environ[m.group(1)]

        return re.sub(r"\$\{(\w+)\}", lookup, value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def _to_connection(name: str, cfg: dict) -> dict:
    if "command" in cfg:
        conn = {"transport": "stdio", "command": cfg["command"], "args": cfg.get("args", [])}
        if cfg.get("env"):
            # only safe defaults (PATH, HOME, ...) — never leak the agent's secrets
            conn["env"] = {**get_default_environment(), **cfg["env"]}
        if cfg.get("cwd"):
            conn["cwd"] = cfg["cwd"]
        return conn
    if "url" in cfg:
        transport = cfg.get("transport", "streamable_http").replace("-", "_")
        if transport == "http":
            transport = "streamable_http"
        conn = {"transport": transport, "url": cfg["url"]}
        if cfg.get("headers"):
            conn["headers"] = cfg["headers"]
        return conn
    raise ConfigError(f"MCP server '{name}' needs either 'command' or 'url'")


async def _close_quietly(stack: AsyncExitStack, name: str) -> None:
    try:
        await stack.aclose()
    except Exception as e:  # the server already failed; don't hide that error behind a cleanup one
        log(f"[mcp] WARNING: error while closing server '{name}': {e!r}")
