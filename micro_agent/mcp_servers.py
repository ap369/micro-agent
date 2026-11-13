"""MCP servers from a Claude-Desktop style `mcp.json`, exposed as LangChain tools."""

import json
import os
import re
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.client.stdio import get_default_environment

from micro_agent.config import ConfigError
from micro_agent.console import log


class McpToolLoader:
    def __init__(self, config_path: Path):
        self.config_path = config_path

    async def load(self) -> list[BaseTool]:
        """Connect to every enabled server; a server that fails is logged and skipped."""
        connections = self._connections()
        if not connections:
            return []
        client = MultiServerMCPClient(connections)
        tools: list[BaseTool] = []
        for name in connections:
            try:
                server_tools = await client.get_tools(server_name=name)
                tools.extend(server_tools)
                log(f"[mcp] {name}: {len(server_tools)} tool(s) — {', '.join(t.name for t in server_tools)}")
            except Exception as e:
                log(f"[mcp] WARNING: failed to load server '{name}': {e!r}")
        return tools

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
