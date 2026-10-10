"""One-shot LangChain agent.

Loads a goal + prompt, a markdown knowledge base, MCP servers from a JSON config
and an OpenAI-compatible LLM from .env, then runs until the goal is done.
File access is sandboxed to WORKSPACE_DIR.

Usage: uv run agent.py [--goal goal.md] [--prompt prompt.md] [--env .env] [--mcp mcp.json]
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.client.stdio import get_default_environment
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError

BASE_DIR = Path(__file__).resolve().parent


class ConfigError(Exception):
    pass


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def truncate(text: str, limit: int = 300) -> str:
    text = str(text).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (BASE_DIR / p).resolve()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigError(f"Missing required env var {name} (see .env.example)")
    return value


def env_number(name: str, default: str, cast):
    raw = os.getenv(name, default)
    try:
        return cast(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from None


def read_text(path: Path, what: str) -> str:
    if not path.is_file():
        raise ConfigError(f"{what} file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


# --- Knowledge -----------------------------------------------------------------

def load_knowledge(knowledge_dir: Path) -> str:
    if not knowledge_dir.is_dir():
        log(f"[knowledge] directory not found, skipping: {knowledge_dir}")
        return ""
    files = sorted(knowledge_dir.rglob("*.md"))
    parts = [
        f"## Knowledge: {f.relative_to(knowledge_dir)}\n\n{f.read_text(encoding='utf-8').strip()}"
        for f in files
    ]
    text = "\n\n".join(parts)
    log(f"[knowledge] loaded {len(files)} file(s), {len(text)} chars")
    return text


# --- Sandboxed file tools ------------------------------------------------------

def make_file_tools(workspace: Path) -> list:
    workspace = workspace.resolve()

    def safe_path(path: str) -> Path:
        target = (workspace / path).resolve()
        if target != workspace and workspace not in target.parents:
            raise ValueError(f"Access denied: '{path}' is outside the workspace")
        return target

    @tool
    def list_files(subdir: str = ".") -> str:
        """List all files (recursively) under a directory of the workspace."""
        try:
            root = safe_path(subdir)
            if not root.is_dir():
                return f"Not a directory: {subdir}"
            files = [str(p.relative_to(workspace)) for p in sorted(root.rglob("*")) if p.is_file()]
            return "\n".join(files) or "(empty)"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def read_file(path: str) -> str:
        """Read a text file from the workspace. `path` is relative to the workspace."""
        try:
            return safe_path(path).read_text(encoding="utf-8")
        except Exception as e:
            return f"Error: {e}"

    @tool
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a text file in the workspace. Parent folders are created."""
        try:
            target = safe_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} chars to {path}"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def append_file(path: str, content: str) -> str:
        """Append text to a file in the workspace (creates it if missing)."""
        try:
            target = safe_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as f:
                f.write(content)
            return f"Appended {len(content)} chars to {path}"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def delete_file(path: str) -> str:
        """Delete a file from the workspace."""
        try:
            target = safe_path(path)
            if not target.is_file():
                return f"Not a file: {path}"
            target.unlink()
            return f"Deleted {path}"
        except Exception as e:
            return f"Error: {e}"

    return [list_files, read_file, write_file, append_file, delete_file]


# --- MCP -----------------------------------------------------------------------

def expand_env(value):
    """Recursively replace ${VAR} placeholders with environment values."""
    if isinstance(value, str):
        def lookup(m: re.Match) -> str:
            if m.group(1) not in os.environ:
                raise ConfigError(f"mcp.json references ${{{m.group(1)}}} but it is not set (add it to .env)")
            return os.environ[m.group(1)]

        return re.sub(r"\$\{(\w+)\}", lookup, value)
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


def to_connection(name: str, cfg: dict) -> dict:
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


async def load_mcp_tools(config_path: Path) -> list:
    if not config_path.is_file():
        log(f"[mcp] no config at {config_path}, running without MCP tools")
        return []
    try:
        servers = json.loads(config_path.read_text(encoding="utf-8")).get("mcpServers", {})
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {config_path}: {e}") from e

    connections = {}
    for name, cfg in expand_env(servers).items():
        if cfg.get("disabled"):
            continue
        connections[name] = to_connection(name, cfg)
    if not connections:
        log("[mcp] no servers configured")
        return []

    client = MultiServerMCPClient(connections)
    tools = []
    for name in connections:
        try:
            server_tools = await client.get_tools(server_name=name)
            tools.extend(server_tools)
            log(f"[mcp] {name}: {len(server_tools)} tool(s) — {', '.join(t.name for t in server_tools)}")
        except Exception as e:
            log(f"[mcp] WARNING: failed to load server '{name}': {e!r}")
    return tools


# --- Main ----------------------------------------------------------------------

def build_system_prompt(prompt: str, knowledge: str, workspace: Path) -> str:
    sections = [
        prompt,
        "# Workspace rules\n\n"
        f"You can read and write files only inside the workspace ({workspace}). "
        "Always use paths relative to the workspace root with the file tools.",
    ]
    if knowledge:
        sections.append(
            "# Knowledge base\n\n"
            "The knowledge files below are already loaded in full. They are NOT in the "
            "workspace — do not try to read them with the file tools.\n\n" + knowledge
        )
    return "\n\n".join(sections)


async def run(args) -> int:
    load_dotenv(resolve(args.env) if args.env else BASE_DIR / ".env")

    goal = read_text(resolve(args.goal or os.getenv("GOAL_FILE", "goal.md")), "Goal")
    prompt = read_text(resolve(args.prompt or os.getenv("PROMPT_FILE", "prompt.md")), "Prompt")
    knowledge = load_knowledge(resolve(os.getenv("KNOWLEDGE_DIR", "knowledge")))
    workspace = resolve(os.getenv("WORKSPACE_DIR", "workspace"))
    workspace.mkdir(parents=True, exist_ok=True)
    max_steps = env_number("MAX_STEPS", "50", int)

    model = ChatOpenAI(
        base_url=require_env("OPENAI_BASE_URL"),
        api_key=require_env("OPENAI_API_KEY"),
        model=require_env("LLM_MODEL"),
        temperature=env_number("LLM_TEMPERATURE", "0", float),
    )

    tools = make_file_tools(workspace) + await load_mcp_tools(
        resolve(args.mcp or os.getenv("MCP_CONFIG", "mcp.json"))
    )
    log(f"[agent] {len(tools)} tool(s) available, workspace: {workspace}")

    agent = create_agent(
        model,
        tools=tools,
        system_prompt=build_system_prompt(prompt, knowledge, workspace),
    )

    final = None
    async for update in agent.astream(
        {"messages": [HumanMessage(goal)]},
        # each tool round = model step + tools step
        {"recursion_limit": 2 * max_steps + 1},
        stream_mode="updates",
    ):
        for node in update.values():
            for msg in (node or {}).get("messages", []):
                if isinstance(msg, AIMessage):
                    for call in msg.tool_calls:
                        log(f"[tool →] {call['name']}({truncate(json.dumps(call['args'], ensure_ascii=False))})")
                    if not msg.tool_calls:
                        final = msg
                elif isinstance(msg, ToolMessage):
                    log(f"[tool ←] {msg.name}: {truncate(msg.content)}")

    result = final.text if final else ""
    (workspace / "_result.md").write_text(result, encoding="utf-8")
    print(result)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot LangChain + MCP agent")
    parser.add_argument("--goal", help="goal file (default: $GOAL_FILE or goal.md)")
    parser.add_argument("--prompt", help="prompt file (default: $PROMPT_FILE or prompt.md)")
    parser.add_argument("--env", help="env file (default: .env)")
    parser.add_argument("--mcp", help="MCP config (default: $MCP_CONFIG or mcp.json)")
    args = parser.parse_args()

    try:
        sys.exit(asyncio.run(run(args)))
    except ConfigError as e:
        log(f"[config] {e}")
        sys.exit(2)
    except GraphRecursionError:
        log("[agent] stopped: MAX_STEPS reached before the goal was completed")
        sys.exit(3)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        log(f"[agent] error: {e!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
