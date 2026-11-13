"""One-shot LangChain agent.

Loads a goal + prompt, skills (loaded on demand), sub-agents (called via a
`delegate` tool), MCP servers from a JSON config and an OpenAI-compatible LLM
from .env, then runs until the goal is done. File access is sandboxed to WORKSPACE_DIR.

Usage: uv run agent.py [--goal goal.md] [--prompt prompt.md] [--env .env] [--mcp mcp.json]
                       [--skills skills] [--agents agents]
"""

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
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


def safe_join(root: Path, path: str) -> Path:
    """Resolve `path` under `root`, refusing anything that escapes it (.., absolute, symlinks)."""
    target = (root / path).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Access denied: '{path}' is outside {root.name}/")
    return target


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


def require_description(meta: dict, path: Path) -> str:
    description = str(meta.get("description") or "").strip()
    if not description:
        raise ConfigError(f"{path} needs a 'description' in its frontmatter")
    return description


# --- Skills ----------------------------------------------------------------------

@dataclass
class Skill:
    name: str
    description: str
    body: str
    dir: Path


def load_skills(skills_dir: Path) -> dict[str, Skill]:
    if not skills_dir.is_dir():
        log(f"[skills] directory not found, skipping: {skills_dir}")
        return {}
    skills: dict[str, Skill] = {}
    for path in sorted(skills_dir.glob("*/SKILL.md")):
        meta, body = parse_markdown(path)
        name = str(meta.get("name") or path.parent.name)
        if name in skills:
            raise ConfigError(f"Duplicate skill name '{name}' ({path})")
        skills[name] = Skill(name, require_description(meta, path), body, path.parent.resolve())
    log(f"[skills] loaded {len(skills)}: {', '.join(skills) or '-'}")
    return skills


def make_skill_tools(skills: dict[str, Skill]) -> list:
    if not skills:
        return []

    def get(name: str) -> Skill:
        if name not in skills:
            raise ValueError(f"Unknown skill '{name}'. Available: {', '.join(skills)}")
        return skills[name]

    @tool
    def load_skill(name: str) -> str:
        """Load the full instructions of a skill by name. Call this before doing a task the skill covers."""
        try:
            skill = get(name)
            files = [
                str(p.relative_to(skill.dir)) for p in sorted(skill.dir.rglob("*"))
                if p.is_file() and p.name != "SKILL.md"
            ]
            extra = "\n\nSupporting files (read with read_skill_file):\n" + "\n".join(f"- {f}" for f in files) if files else ""
            return skill.body + extra
        except Exception as e:
            return f"Error: {e}"

    @tool
    def read_skill_file(name: str, path: str) -> str:
        """Read a supporting file (template, reference, example) from a skill's folder."""
        try:
            return safe_join(get(name).dir, path).read_text(encoding="utf-8")
        except Exception as e:
            return f"Error: {e}"

    return [load_skill, read_skill_file]


# --- Sandboxed file tools ------------------------------------------------------

def make_file_tools(workspace: Path) -> list:
    workspace = workspace.resolve()

    def safe_path(path: str) -> Path:
        return safe_join(workspace, path)

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


# --- Agents ----------------------------------------------------------------------

@dataclass
class AgentSpec:
    name: str
    description: str
    prompt: str
    tools: list[str] | None
    model: str | None
    max_steps: int | None


def load_agents(agents_dir: Path) -> dict[str, AgentSpec]:
    if not agents_dir.is_dir():
        log(f"[agents] directory not found, skipping: {agents_dir}")
        return {}
    agents: dict[str, AgentSpec] = {}
    for path in sorted(agents_dir.glob("*.md")):
        meta, body = parse_markdown(path)
        name = str(meta.get("name") or path.stem)
        if name in agents:
            raise ConfigError(f"Duplicate agent name '{name}' ({path})")
        tools = meta.get("tools")
        if tools is not None and not (isinstance(tools, list) and all(isinstance(t, str) for t in tools)):
            raise ConfigError(f"{path}: 'tools' must be a list of tool names")
        max_steps = meta.get("max_steps")
        if max_steps is not None and not isinstance(max_steps, int):
            raise ConfigError(f"{path}: 'max_steps' must be an integer")
        if not body:
            raise ConfigError(f"{path} has no body (the sub-agent's system prompt)")
        agents[name] = AgentSpec(
            name, require_description(meta, path), body, tools, meta.get("model"), max_steps
        )
    log(f"[agents] loaded {len(agents)}: {', '.join(agents) or '-'}")
    return agents


def build_system_prompt(
    base: str, workspace: Path, skills: dict[str, Skill], agents: dict[str, AgentSpec] | None = None
) -> str:
    sections = [
        base,
        "# Workspace rules\n\n"
        f"You can read and write files only inside the workspace ({workspace}). "
        "Always use paths relative to the workspace root with the file tools.",
    ]
    if skills:
        sections.append(
            "# Skills\n\n"
            "Skills hold specialised instructions. When a task matches a skill, call `load_skill` "
            "with its name first and follow what it says. Skill files are NOT in the workspace; "
            "use `read_skill_file` for their supporting files.\n\n"
            + "\n".join(f"- **{s.name}**: {s.description}" for s in skills.values())
        )
    if agents:
        sections.append(
            "# Agents\n\n"
            "You can hand a task to a specialised sub-agent with the `delegate` tool. Sub-agents do "
            "NOT see this conversation: put everything they need in the task (goal, file paths, "
            "constraints). They share the workspace, so pass work to each other through files.\n\n"
            + "\n".join(f"- **{a.name}**: {a.description}" for a in agents.values())
        )
    return "\n\n".join(sections)


async def run_agent(agent, task: str, max_steps: int, label: str) -> str:
    """Stream an agent run, logging tool calls, and return its final answer."""
    final = None
    async for update in agent.astream(
        {"messages": [HumanMessage(task)]},
        # each tool round = model step + tools step
        {"recursion_limit": 2 * max_steps + 1},
        stream_mode="updates",
    ):
        for node in update.values():
            for msg in (node or {}).get("messages", []):
                if isinstance(msg, AIMessage):
                    for call in msg.tool_calls:
                        log(f"[{label}] tool → {call['name']}({truncate(json.dumps(call['args'], ensure_ascii=False))})")
                    if not msg.tool_calls:
                        final = msg
                elif isinstance(msg, ToolMessage):
                    log(f"[{label}] tool ← {msg.name}: {truncate(msg.content)}")
    return final.text if final else ""


def make_delegate_tool(
    specs: dict[str, AgentSpec], tool_pool: list, skills: dict[str, Skill], settings: dict
):
    """Build every sub-agent up front (so config errors surface at startup) and a `delegate` tool."""
    if not specs:
        return None
    by_name = {t.name: t for t in tool_pool}
    built = {}
    for spec in specs.values():
        if spec.tools is None:
            tools = list(tool_pool)
        else:
            unknown = [t for t in spec.tools if t not in by_name]
            if unknown:
                raise ConfigError(
                    f"Agent '{spec.name}' lists unknown tool(s): {', '.join(unknown)}. "
                    f"Available: {', '.join(by_name)}"
                )
            tools = [by_name[t] for t in spec.tools]
        agent_skills = skills if any(t.name == "load_skill" for t in tools) else {}
        model = make_model(settings, spec.model)
        built[spec.name] = (
            create_agent(
                model,
                tools=tools,
                system_prompt=build_system_prompt(spec.prompt, settings["workspace"], agent_skills),
            ),
            spec.max_steps or settings["max_steps"],
        )

    @tool
    async def delegate(agent: str, task: str) -> str:
        """Hand a self-contained task to a sub-agent and get its final answer back."""
        if agent not in built:
            return f"Error: unknown agent '{agent}'. Available: {', '.join(built)}"
        sub, steps = built[agent]
        log(f"[{agent}] started")
        try:
            result = await run_agent(sub, task, steps, agent)
        except GraphRecursionError:
            return f"Error: agent '{agent}' hit its step limit ({steps}) before finishing"
        except Exception as e:
            return f"Error: agent '{agent}' failed: {e!r}"
        log(f"[{agent}] finished")
        return result or "(the agent returned no answer)"

    return delegate


# --- Main ----------------------------------------------------------------------

def make_model(settings: dict, model_name: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings["base_url"],
        api_key=settings["api_key"],
        model=model_name or settings["model"],
        temperature=settings["temperature"],
    )


async def run(args) -> int:
    load_dotenv(resolve(args.env) if args.env else BASE_DIR / ".env")

    goal = read_text(resolve(args.goal or os.getenv("GOAL_FILE", "goal.md")), "Goal")
    prompt = read_text(resolve(args.prompt or os.getenv("PROMPT_FILE", "prompt.md")), "Prompt")
    skills = load_skills(resolve(args.skills or os.getenv("SKILLS_DIR", "skills")))
    agent_specs = load_agents(resolve(args.agents or os.getenv("AGENTS_DIR", "agents")))
    workspace = resolve(os.getenv("WORKSPACE_DIR", "workspace"))
    workspace.mkdir(parents=True, exist_ok=True)

    settings = {
        "base_url": require_env("OPENAI_BASE_URL"),
        "api_key": require_env("OPENAI_API_KEY"),
        "model": require_env("LLM_MODEL"),
        "temperature": env_number("LLM_TEMPERATURE", "0", float),
        "max_steps": env_number("MAX_STEPS", "50", int),
        "workspace": workspace,
    }

    tool_pool = (
        make_file_tools(workspace)
        + make_skill_tools(skills)
        + await load_mcp_tools(resolve(args.mcp or os.getenv("MCP_CONFIG", "mcp.json")))
    )
    delegate = make_delegate_tool(agent_specs, tool_pool, skills, settings)
    tools = tool_pool + ([delegate] if delegate else [])
    log(f"[main] {len(tools)} tool(s) available, workspace: {workspace}")

    agent = create_agent(
        make_model(settings),
        tools=tools,
        system_prompt=build_system_prompt(prompt, workspace, skills, agent_specs),
    )

    result = await run_agent(agent, goal, settings["max_steps"], "main")
    (workspace / "_result.md").write_text(result, encoding="utf-8")
    print(result)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot LangChain + MCP agent")
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
