"""Settings from .env, environment variables and CLI overrides."""

import os
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    """Invalid or missing configuration (exit code 2)."""


@dataclass(frozen=True)
class Settings:
    # LLM (OpenAI-compatible endpoint)
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_steps: int
    # inputs
    goal: str
    prompt: str
    mcp_config: Path
    skills_dir: Path
    agents_dir: Path
    workspace: Path

    @classmethod
    def load(cls, args: Namespace) -> "Settings":
        """Load .env, then build settings; CLI flags win over env vars, env vars over defaults."""
        load_dotenv(_resolve(args.env) if args.env else PROJECT_DIR / ".env")
        return cls(
            base_url=_require_env("OPENAI_BASE_URL"),
            api_key=_require_env("OPENAI_API_KEY"),
            model=_require_env("LLM_MODEL"),
            temperature=_env_number("LLM_TEMPERATURE", "0", float),
            max_steps=_env_number("MAX_STEPS", "50", int),
            goal=_read_text(_path(args.goal, "GOAL_FILE", "goal.md"), "Goal"),
            prompt=_read_text(_path(args.prompt, "PROMPT_FILE", "prompt.md"), "Prompt"),
            mcp_config=_path(args.mcp, "MCP_CONFIG", "mcp.json"),
            skills_dir=_path(args.skills, "SKILLS_DIR", "skills"),
            agents_dir=_path(args.agents, "AGENTS_DIR", "agents"),
            workspace=_path(None, "WORKSPACE_DIR", "workspace"),
        )


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_DIR / p).resolve()


def _path(cli_value: str | None, env_name: str, default: str) -> Path:
    return _resolve(cli_value or os.getenv(env_name, default))


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigError(f"Missing required env var {name} (see .env.example)")
    return value


def _env_number(name: str, default: str, cast):
    raw = os.getenv(name, default)
    try:
        return cast(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from None


def _read_text(path: Path, what: str) -> str:
    if not path.is_file():
        raise ConfigError(f"{what} file not found: {path}")
    return path.read_text(encoding="utf-8").strip()
