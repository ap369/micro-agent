"""Shared fixtures: an isolated project folder and scripted fake models (no API key needed)."""

import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run  # noqa: E402
from micro_agent import llm  # noqa: E402

ENV_VARS = [
    "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "LLM_TEMPERATURE",
    "OPENAI_BASE_URL", "OPENAI_API_KEY", "MAX_STEPS", "GOAL_FILE", "PROMPT_FILE",
    "MCP_CONFIG", "SKILLS_DIR", "AGENTS_DIR", "WORKSPACE_DIR",
]  # fmt: skip


def call(name: str, args: dict | None = None, call_id: str | None = None) -> AIMessage:
    """An AI message with one tool call."""
    return AIMessage("", tool_calls=[{"name": name, "args": args or {}, "id": call_id or f"call_{name}"}])


class ScriptedModel(FakeMessagesListChatModel):
    """Replays scripted AI messages; tools are accepted but ignored."""

    def bind_tools(self, tools, **kwargs):
        return self


@dataclass
class Project:
    """A throw-away copy of the project inputs; `run()` executes run.py against it."""

    dir: Path
    mcp_servers: dict = field(default_factory=dict)

    @property
    def workspace(self) -> Path:
        return self.dir / "workspace"

    def write(self, relative: str, text: str) -> Path:
        path = self.dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def run(self, *extra: str) -> int:
        import json

        (self.dir / "mcp.json").write_text(json.dumps({"mcpServers": self.mcp_servers}))
        args = ["--env", str(self.dir / ".env"), "--goal", str(self.dir / "goal.md")]
        args += ["--prompt", str(self.dir / "prompt.md"), "--mcp", str(self.dir / "mcp.json")]
        args += ["--skills", str(self.dir / "skills"), "--agents", str(self.dir / "agents"), *extra]
        return run.main(args)


@pytest.fixture
def project(tmp_path, monkeypatch) -> Project:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_MODEL", "fake-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")  # never reached by scripted models
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "workspace"))

    shutil.copytree(ROOT / "skills", tmp_path / "skills")
    shutil.copytree(ROOT / "agents", tmp_path / "agents")
    p = Project(tmp_path)
    p.write(".env", "")
    p.write("goal.md", "Do the thing.")
    p.write("prompt.md", "You are a test agent.")
    return p


@pytest.fixture
def script(monkeypatch):
    """`script([...messages])` makes every model (main and sub-agents) replay the messages in order."""

    def use(messages: list[AIMessage]) -> None:
        model = ScriptedModel(responses=messages)
        monkeypatch.setattr(llm.ModelFactory, "create", lambda self, model_name=None: model)

    return use
