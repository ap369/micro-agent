"""MCP servers: one persistent session per server, broken servers skipped, ${VAR} expansion."""

import sys
from pathlib import Path

from conftest import call
from langchain_core.messages import AIMessage

from run import ExitCode

COUNTER = {"command": sys.executable, "args": [str(Path(__file__).parent / "servers" / "counter_server.py")]}


def test_session_is_kept_for_the_whole_run(project, script, capsys):
    project.mcp_servers = {"counter": COUNTER}
    script([call("increment", {}, "c1"), call("increment", {}, "c2"), AIMessage("done")])

    assert project.run() == ExitCode.OK

    results = [line for line in capsys.readouterr().err.splitlines() if "tool ← increment" in line]
    # a new session per call would restart the server and return '1' twice
    assert "'text': '1'" in results[0]
    assert "'text': '2'" in results[1]


def test_subagents_get_mcp_tools(project, script, capsys):
    project.mcp_servers = {"counter": COUNTER}
    script([
        call("task", {"subagent_type": "reviewer", "description": "count"}),
        call("increment"),  # reviewer
        AIMessage("counted"),
        AIMessage("done"),
    ])  # fmt: skip

    assert project.run() == ExitCode.OK
    assert "[reviewer] tool ← increment" in capsys.readouterr().err


def test_broken_server_is_skipped(project, script, capsys):
    project.mcp_servers = {"broken": {"command": "definitely-not-a-command-xyz"}, "counter": COUNTER}
    script([call("increment"), AIMessage("done")])

    assert project.run() == ExitCode.OK

    err = capsys.readouterr().err
    assert "failed to load server 'broken'" in err
    assert "[mcp] counter: 1 tool(s)" in err


def test_unset_variable_is_a_config_error(project, capsys):
    project.mcp_servers = {"remote": {"url": "https://x/mcp", "headers": {"Authorization": "Bearer ${NOPE}"}}}
    assert project.run() == ExitCode.CONFIG
    assert "${NOPE} but it is not set" in capsys.readouterr().err
