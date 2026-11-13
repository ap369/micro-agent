"""Configuration errors exit with code 2; Settings can be built without the CLI."""

import pytest

from micro_agent import ConfigError, Settings
from run import ExitCode


@pytest.mark.parametrize(
    ("path", "content", "message"),
    [
        ("skills/bad/SKILL.md", "---\nname: bad\n---\nbody", "needs a 'description'"),
        ("skills/nameless/SKILL.md", "---\ndescription: d\n---\nbody", "needs a 'name'"),
        ("agents/x.md", "---\ndescription: d\ntools: [missing_tool]\n---\nbody", "unknown MCP tool"),
        ("agents/y.md", "---\ndescription: d\nmax_steps: lots\n---\nbody", "'max_steps' must be"),
        ("agents/z.md", "---\ndescription: d\n---\n", "has no body"),
        ("agents/dup.md", "---\nname: reviewer\ndescription: d\n---\nbody", "Duplicate agent name"),
        ("agents/broken.md", "---\n: [\n---\nbody", "Invalid frontmatter"),
    ],
)
def test_invalid_skill_or_agent(project, capsys, path, content, message):
    project.write(path, content)
    assert project.run() == ExitCode.CONFIG
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("env", "value", "message"),
    [
        ("MAX_STEPS", "fifty", "MAX_STEPS must be a number"),
        ("LLM_TEMPERATURE", "hot", "LLM_TEMPERATURE must be a number"),
        ("LLM_MODEL", "", "Missing required env var LLM_MODEL"),
        ("LLM_PROVIDER", "no-such-provider", "Cannot create model"),
    ],
)
def test_invalid_env(project, monkeypatch, capsys, env, value, message):
    monkeypatch.setenv(env, value)
    assert project.run() == ExitCode.CONFIG
    assert message in capsys.readouterr().err


def test_missing_goal_file(project, capsys):
    assert project.run("--goal", str(project.dir / "nope.md")) == ExitCode.CONFIG
    assert "Goal file not found" in capsys.readouterr().err


def test_settings_load_with_keywords(project):
    settings = Settings.load(goal=project.dir / "goal.md", prompt=project.dir / "prompt.md")
    assert settings.goal == "Do the thing."
    assert settings.provider == "openai"
    assert settings.workspace == project.workspace


def test_settings_accept_legacy_openai_env_names(project, monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL")
    monkeypatch.delenv("LLM_API_KEY")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    settings = Settings.load(goal=project.dir / "goal.md", prompt=project.dir / "prompt.md")
    assert (settings.base_url, settings.api_key) == ("https://legacy/v1", "legacy-key")


def test_settings_load_reports_missing_model(project, monkeypatch):
    monkeypatch.delenv("LLM_MODEL")
    with pytest.raises(ConfigError, match="LLM_MODEL"):
        Settings.load(goal=project.dir / "goal.md", prompt=project.dir / "prompt.md")
