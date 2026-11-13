"""One-shot deep agent built on LangChain `deepagents`.

Loads a goal + prompt, skills (Agent Skills format, loaded on demand), sub-agents
(called through the `task` tool), MCP servers from a JSON config and an
OpenAI-compatible LLM from .env, then runs until the goal is done.
File access is sandboxed to WORKSPACE_DIR; skills are mounted read-only at /skills/.

Usage: uv run agent.py [--goal goal.md] [--prompt prompt.md] [--env .env] [--mcp mcp.json]
                       [--skills skills] [--agents agents]
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import yaml
from deepagents import FilesystemPermission, SubAgent, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend
from dotenv import load_dotenv
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from mcp.client.stdio import get_default_environment

BASE_DIR = Path(__file__).resolve().parent
SKILLS_ROUTE = "/skills/"
# skills are mounted into the agent's filesystem but must never be modified
SKILLS_READ_ONLY = FilesystemPermission(operations=["write"], paths=[f"{SKILLS_ROUTE}**"], mode="deny")


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


def parse_markdown(path: Path) -> tuple[dict, str]:
    """Split a markdown file into (YAML frontmatter, body)."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text.strip()
    try:
        _, front, body = text.split("---", 2)
        meta = yaml.safe_load(front) or {}
    except (ValueError, yaml.YAMLError) as e:
        raise ConfigError(f"Invalid frontmatter in {path}: {e}") from None
    if not isinstance(meta, dict):
        raise ConfigError(f"Frontmatter in {path} must be a mapping")
    return meta, body.strip()


def make_model(settings: dict, model_name: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings["base_url"],
        api_key=settings["api_key"],
        model=model_name or settings["model"],
        temperature=settings["temperature"],
    )


# --- Skills ----------------------------------------------------------------------

def check_skills(skills_dir: Path) -> bool:
    """Validate skills up front so mistakes are config errors, not silently skipped skills."""
    if not skills_dir.is_dir():
        log(f"[skills] directory not found, skipping: {skills_dir}")
        return False
    names = []
    for path in sorted(skills_dir.glob("*/SKILL.md")):
        meta, _ = parse_markdown(path)
        for field in ("name", "description"):
            if not str(meta.get(field) or "").strip():
                raise ConfigError(f"{path} needs a '{field}' in its frontmatter")
        if meta["name"] in names:
            raise ConfigError(f"Duplicate skill name '{meta['name']}' ({path})")
        names.append(meta["name"])
    log(f"[skills] found {len(names)}: {', '.join(names) or '-'}")
    return bool(names)


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


# --- Agents ----------------------------------------------------------------------

def load_subagents(
    agents_dir: Path, mcp_tools: list, has_skills: bool, settings: dict
) -> list[SubAgent]:
    """Turn agents/*.md into deepagents SubAgent specs (frontmatter = config, body = prompt)."""
    if not agents_dir.is_dir():
        log(f"[agents] directory not found, skipping: {agents_dir}")
        return []
    mcp_by_name = {t.name: t for t in mcp_tools}
    specs: list[SubAgent] = []
    for path in sorted(agents_dir.glob("*.md")):
        meta, body = parse_markdown(path)
        name = str(meta.get("name") or path.stem)
        description = str(meta.get("description") or "").strip()
        if not description:
            raise ConfigError(f"{path} needs a 'description' in its frontmatter")
        if not body:
            raise ConfigError(f"{path} has no body (the sub-agent's system prompt)")
        if any(s["name"] == name for s in specs):
            raise ConfigError(f"Duplicate agent name '{name}' ({path})")

        spec: SubAgent = {"name": name, "description": description, "system_prompt": body}

        # `tools` = MCP tools to give this agent (file tools are always built in)
        tools = meta.get("tools")
        if tools is not None:
            if not (isinstance(tools, list) and all(isinstance(t, str) for t in tools)):
                raise ConfigError(f"{path}: 'tools' must be a list of MCP tool names")
            unknown = [t for t in tools if t not in mcp_by_name]
            if unknown:
                raise ConfigError(
                    f"Agent '{name}' lists unknown MCP tool(s): {', '.join(unknown)}. "
                    f"Available: {', '.join(mcp_by_name) or '(none)'}"
                )
            spec["tools"] = [mcp_by_name[t] for t in tools]

        if meta.get("model"):
            spec["model"] = make_model(settings, str(meta["model"]))

        if has_skills and meta.get("skills", True):
            spec["skills"] = [SKILLS_ROUTE]

        # sub-agent rules replace the parent's, so always keep skills read-only
        permissions = [SKILLS_READ_ONLY]
        if meta.get("read_only"):
            permissions.append(FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"))
        spec["permissions"] = permissions

        max_steps = meta.get("max_steps", settings["max_steps"])
        if not isinstance(max_steps, int) or max_steps < 1:
            raise ConfigError(f"{path}: 'max_steps' must be a positive integer")
        # 'end' = stop gracefully and hand a notice back to the main agent
        spec["middleware"] = [ModelCallLimitMiddleware(run_limit=max_steps, exit_behavior="end")]

        specs.append(spec)
    log(f"[agents] loaded {len(specs)}: {', '.join(s['name'] for s in specs) or '-'}")
    return specs


# --- Main ----------------------------------------------------------------------

WORKSPACE_PROMPT = """# Filesystem

Your filesystem root `/` is the workspace: save every deliverable there (e.g. `/report.md`).
`/skills/` contains the skills and is read-only."""


async def run_agent(agent, goal: str, max_steps: int) -> str:
    """Stream the run (including sub-agents), log tool calls, and return the final answer."""
    final = None
    pending: list[str] = []  # sub-agents requested via `task`, in call order
    labels: dict[str, str] = {}  # subgraph namespace -> sub-agent name
    async for namespace, update in agent.astream(
        {"messages": [HumanMessage(goal)]},
        # backstop only; MAX_STEPS is enforced by ModelCallLimitMiddleware
        {"recursion_limit": 10 * max_steps + 100},
        stream_mode="updates",
        subgraphs=True,
    ):
        if namespace and namespace[0] not in labels:
            labels[namespace[0]] = pending.pop(0) if pending else "subagent"
        label = labels[namespace[0]] if namespace else "main"
        for node in update.values():
            if not isinstance(node, dict):
                continue
            for msg in node.get("messages") or []:
                if isinstance(msg, AIMessage):
                    for call in msg.tool_calls:
                        if call["name"] == "task" and not namespace:
                            pending.append(call["args"].get("subagent_type", "subagent"))
                        log(f"[{label}] tool → {call['name']}({truncate(json.dumps(call['args'], ensure_ascii=False))})")
                    if not msg.tool_calls and not namespace:
                        final = msg
                elif isinstance(msg, ToolMessage):
                    log(f"[{label}] tool ← {msg.name}: {truncate(msg.content)}")
    return final.text if final else ""


async def run(args) -> int:
    load_dotenv(resolve(args.env) if args.env else BASE_DIR / ".env")

    goal = read_text(resolve(args.goal or os.getenv("GOAL_FILE", "goal.md")), "Goal")
    prompt = read_text(resolve(args.prompt or os.getenv("PROMPT_FILE", "prompt.md")), "Prompt")
    skills_dir = resolve(args.skills or os.getenv("SKILLS_DIR", "skills"))
    agents_dir = resolve(args.agents or os.getenv("AGENTS_DIR", "agents"))
    workspace = resolve(os.getenv("WORKSPACE_DIR", "workspace"))
    workspace.mkdir(parents=True, exist_ok=True)

    settings = {
        "base_url": require_env("OPENAI_BASE_URL"),
        "api_key": require_env("OPENAI_API_KEY"),
        "model": require_env("LLM_MODEL"),
        "temperature": env_number("LLM_TEMPERATURE", "0", float),
        "max_steps": env_number("MAX_STEPS", "50", int),
    }

    has_skills = check_skills(skills_dir)
    mcp_tools = await load_mcp_tools(resolve(args.mcp or os.getenv("MCP_CONFIG", "mcp.json")))
    subagents = load_subagents(agents_dir, mcp_tools, has_skills, settings)

    routes = {SKILLS_ROUTE: FilesystemBackend(root_dir=skills_dir, virtual_mode=True)} if has_skills else {}
    agent = create_deep_agent(
        model=make_model(settings),
        tools=mcp_tools,
        system_prompt=f"{prompt}\n\n{WORKSPACE_PROMPT}",
        subagents=subagents,
        skills=[SKILLS_ROUTE] if has_skills else None,
        backend=CompositeBackend(
            default=FilesystemBackend(root_dir=workspace, virtual_mode=True), routes=routes
        ),
        permissions=[SKILLS_READ_ONLY],
        # 'error' = the run fails with exit code 3 when the main agent runs out of steps
        middleware=[ModelCallLimitMiddleware(run_limit=settings["max_steps"], exit_behavior="error")],
    )
    log(f"[main] workspace: {workspace}")

    result = await run_agent(agent, goal, settings["max_steps"])
    (workspace / "_result.md").write_text(result, encoding="utf-8")
    print(result)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot deep agent (LangChain deepagents + MCP)")
    parser.add_argument("--goal", help="goal file (default: $GOAL_FILE or goal.md)")
    parser.add_argument("--prompt", help="prompt file (default: $PROMPT_FILE or prompt.md)")
    parser.add_argument("--env", help="env file (default: .env)")
    parser.add_argument("--mcp", help="MCP config (default: $MCP_CONFIG or mcp.json)")
    parser.add_argument("--skills", help="skills dir (default: $SKILLS_DIR or skills)")
    parser.add_argument("--agents", help="agents dir (default: $AGENTS_DIR or agents)")
    args = parser.parse_args()

    try:
        sys.exit(asyncio.run(run(args)))
    except ConfigError as e:
        log(f"[config] {e}")
        sys.exit(2)
    except (ModelCallLimitExceededError, GraphRecursionError):
        log("[main] stopped: MAX_STEPS reached before the goal was completed")
        sys.exit(3)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        log(f"[agent] error: {e!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
