"""Rejected tool calls (400 tool_use_failed), tested through the real ChatOpenAI client and a fake server."""

import pytest
from fake_openai import FakeOpenAI, completion, error

from run import ExitCode

REJECTED = error(
    "tool_use_failed",
    "Tool call validation failed: missing properties: 'file_path'",
    failed_generation='{"name": "write_file", "arguments": {"path": "/a.md"}}',
)


@pytest.fixture
def server(project, monkeypatch):
    fake = FakeOpenAI([])
    monkeypatch.setenv("LLM_BASE_URL", fake.base_url)
    yield fake
    fake.close()


def last_user_message(request: dict) -> str:
    return next(m["content"] for m in reversed(request["messages"]) if m["role"] == "user")


def test_rejected_call_is_shown_to_the_model_and_fixed(project, server):
    server.responses = [
        REJECTED,
        completion(tool_calls=[("write_file", {"file_path": "/a.md", "content": "hi"})]),
        completion("done"),
    ]

    assert project.run() == ExitCode.OK

    assert (project.workspace / "a.md").read_text() == "hi"
    hint = last_user_message(server.requests[1])
    assert "rejected by the API" in hint
    assert '"path": "/a.md"' in hint  # the model sees its rejected call


def test_main_agent_gives_up_after_repeated_rejections(project, server, capsys):
    server.responses = [REJECTED] * 3

    assert project.run() == ExitCode.ERROR
    assert len(server.requests) == 3  # first try + 2 repairs
    assert "tool_use_failed" in capsys.readouterr().err


def test_other_bad_requests_are_not_retried(project, server):
    server.responses = [error("context_length_exceeded", "too long, mentions tool_use_failed in text")]

    assert project.run() == ExitCode.ERROR
    assert len(server.requests) == 1


def test_subagent_rejections_are_handed_back_to_main_agent(project, server, capsys):
    server.responses = [
        completion(tool_calls=[("task", {"subagent_type": "general-purpose", "description": "x"})]),
        REJECTED,  # general-purpose sub-agent
        REJECTED,
        REJECTED,
        completion("handled"),  # main agent continues
    ]

    assert project.run() == ExitCode.OK

    err = capsys.readouterr().err
    assert "[main] tool ← task: Error: the model kept producing invalid tool calls" in err
