"""End-to-end runs with scripted models: skills, sub-agents, sandbox, limits and logging."""

from conftest import ROOT, call
from langchain_core.messages import AIMessage

from run import ExitCode


def test_skill_subagent_flow(project, script, capsys):
    script([
        call("read_file", {"file_path": "/skills/report-writing/SKILL.md"}),
        call("read_file", {"file_path": "/skills/report-writing/template.md"}),
        call("write_file", {"file_path": "/report.md", "content": "# Report"}),
        call("task", {"subagent_type": "reviewer", "description": "Review /report.md into /review.md"}),
        call("read_file", {"file_path": "/report.md"}),  # reviewer
        call("write_file", {"file_path": "/review.md", "content": "LGTM"}),  # reviewer
        AIMessage("approved"),  # reviewer's answer
        AIMessage("All done."),  # main agent's answer
    ])  # fmt: skip

    assert project.run() == ExitCode.OK

    out = capsys.readouterr()
    assert out.out.strip() == "All done."
    assert (project.workspace / "report.md").read_text() == "# Report"
    assert (project.workspace / "review.md").read_text() == "LGTM"
    assert not (project.workspace / "_result.md").exists()  # nothing extra leaks into the workspace
    assert "[reviewer] tool → write_file" in out.err
    assert "[main] tool ← task: approved" in out.err


def test_skills_are_read_only_and_workspace_is_sandboxed(project, script, capsys):
    script([
        call("write_file", {"file_path": "/skills/report-writing/evil.md", "content": "x"}),
        call("read_file", {"file_path": "/../.env"}),
        AIMessage("done"),
    ])  # fmt: skip

    assert project.run() == ExitCode.OK

    err = capsys.readouterr().err
    assert "permission denied" in err
    assert "Path traversal not allowed" in err
    assert not (project.dir / "skills/report-writing/evil.md").exists()


def test_read_only_agent_cannot_write(project, script, capsys):
    project.write("agents/auditor.md", "---\ndescription: audits\nread_only: true\n---\nAudit things.")
    script([
        call("task", {"subagent_type": "auditor", "description": "write /x.md"}),
        call("write_file", {"file_path": "/x.md", "content": "x"}),  # auditor
        AIMessage("could not write"),  # auditor
        AIMessage("done"),
    ])  # fmt: skip

    assert project.run() == ExitCode.OK
    assert "permission denied" in capsys.readouterr().err
    assert not (project.workspace / "x.md").exists()


def test_unknown_subagent_is_reported_to_main_agent(project, script, capsys):
    script([call("task", {"subagent_type": "nobody", "description": "x"}), AIMessage("done")])

    assert project.run() == ExitCode.OK
    assert "We cannot invoke subagent nobody" in capsys.readouterr().err


def test_general_purpose_subagent_is_available(project, script, capsys):
    script([
        call("task", {"subagent_type": "general-purpose", "description": "list files"}),
        call("ls", {"path": "/"}),  # general-purpose
        AIMessage("nothing here"),  # general-purpose
        AIMessage("done"),
    ])  # fmt: skip

    assert project.run() == ExitCode.OK
    assert "[general-purpose] tool → ls" in capsys.readouterr().err


def test_parallel_subagents_are_labelled_by_name(project, script, capsys):
    project.write("agents/alpha.md", "---\ndescription: a\n---\nAlpha.")
    project.write("agents/beta.md", "---\ndescription: b\n---\nBeta.")
    both = AIMessage("", tool_calls=[
        {"name": "task", "args": {"subagent_type": "alpha", "description": "x"}, "id": "t1"},
        {"name": "task", "args": {"subagent_type": "beta", "description": "y"}, "id": "t2"},
    ])  # fmt: skip
    script([both, call("ls", {"path": "/"}, "l1"), AIMessage("1"), call("ls", {"path": "/"}, "l2"),
            AIMessage("2"), AIMessage("done")])  # fmt: skip

    assert project.run() == ExitCode.OK

    err = capsys.readouterr().err
    assert err.count("tool → ls") == 2
    assert "[subagent]" not in err
    # each `ls` is logged under the agent that actually made it
    assert {line.split("]")[0] + "]" for line in err.splitlines() if "tool → ls" in line} <= {
        "[alpha]",
        "[beta]",
    }


def test_main_agent_step_limit_exits_3(project, script, monkeypatch, capsys):
    monkeypatch.setenv("MAX_STEPS", "3")
    script([call("ls", {"path": "/"}, f"c{i}") for i in range(10)])

    assert project.run() == ExitCode.STEP_LIMIT
    assert "MAX_STEPS reached" in capsys.readouterr().err


def test_subagent_step_limit_returns_notice_to_main(project, script, monkeypatch, capsys):
    monkeypatch.setenv("MAX_STEPS", "3")
    script([
        call("task", {"subagent_type": "reviewer", "description": "loop"}),
        *[call("ls", {"path": "/"}, f"c{i}") for i in range(3)],  # reviewer loops
        AIMessage("done"),
    ])  # fmt: skip

    assert project.run() == ExitCode.OK
    assert "[main] tool ← task: Model call limits exceeded" in capsys.readouterr().err


def test_output_flag_saves_the_answer(project, script):
    script([AIMessage("The answer.")])

    assert project.run("--output", str(project.dir / "out/answer.md")) == ExitCode.OK
    assert (project.dir / "out/answer.md").read_text() == "The answer."


def test_filesystem_prompt_only_mentions_skills_when_there_are_some(tmp_path):
    from micro_agent.filesystem import filesystem_prompt
    from micro_agent.skills import SkillCatalog

    assert "/skills/" not in filesystem_prompt(SkillCatalog(tmp_path / "missing"))
    assert "/skills/" in filesystem_prompt(SkillCatalog(ROOT / "skills"))


def test_subagents_get_the_filesystem_prompt(project):
    from micro_agent import Settings
    from micro_agent.llm import ModelFactory
    from micro_agent.skills import SkillCatalog
    from micro_agent.subagents import SubAgentLoader

    settings = Settings.load(goal=project.dir / "goal.md", prompt=project.dir / "prompt.md")
    skills = SkillCatalog(project.dir / "skills")
    specs = SubAgentLoader(project.dir / "agents", [], skills, ModelFactory(settings), 5).load()

    assert {s["name"] for s in specs} == {"reviewer", "general-purpose"}
    assert all("# Filesystem" in s["system_prompt"] for s in specs)
