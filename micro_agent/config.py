"""Settings from environment variables, with optional overrides.

Loading a `.env` file is the caller's job (run.py does it), so this module never mutates `os.environ`.
"""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    """Invalid or missing configuration (exit code 2)."""


@dataclass(frozen=True)
class Settings:
    # LLM — any provider supported by LangChain's `init_chat_model`
    provider: str
    model: str
    base_url: str | None
    api_key: str | None
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
    def load(
        cls,
        *,
        goal: str | Path | None = None,
        prompt: str | Path | None = None,
        mcp_config: str | Path | None = None,
        skills_dir: str | Path | None = None,
        agents_dir: str | Path | None = None,
        workspace: str | Path | None = None,
    ) -> "Settings":
        """Build settings from env vars; arguments override the matching env var / default.

        Relative paths are resolved against the project folder.
        """
        return cls(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            model=_require_env("LLM_MODEL"),
            # OPENAI_* names are still accepted for backwards compatibility
            base_url=os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None,
            api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or None,
            temperature=_env_number("LLM_TEMPERATURE", "0", float),
            max_steps=_env_number("MAX_STEPS", "50", int),
            goal=_read_text(_path(goal, "GOAL_FILE", "goal.md"), "Goal"),
            prompt=_read_text(_path(prompt, "PROMPT_FILE", "prompt.md"), "Prompt"),
            mcp_config=_path(mcp_config, "MCP_CONFIG", "mcp.json"),
            skills_dir=_path(skills_dir, "SKILLS_DIR", "skills"),
            agents_dir=_path(agents_dir, "AGENTS_DIR", "agents"),
            workspace=_path(workspace, "WORKSPACE_DIR", "workspace"),
        )


def resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_DIR / p).resolve()


def _path(override: str | Path | None, env_name: str, default: str) -> Path:
    return resolve_path(override or os.getenv(env_name, default))


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
