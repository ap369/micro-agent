# micro-agent

A one-shot agent built on LangChain's [`deepagents`](https://docs.langchain.com/oss/python/deepagents/overview) harness. You give it a **goal**, a **prompt**, **skills**, **sub-agents**, and a list of **MCP servers**. When you run it, it connects to everything, does the job, and exits.

- Works with any LLM provider LangChain supports (`init_chat_model`), including any **OpenAI-compatible** endpoint such as Groq
- **Skills** ([Agent Skills](https://agentskills.io) format): only names and descriptions go in the prompt; the agent reads a skill's full instructions when it needs them
- **Agents**: specialised sub-agents the main agent can hand tasks to (via `deepagents`' `task` tool), each with its own prompt, tools and model
- Connects to MCP servers (stdio, streamable HTTP, SSE) from a `mcp.json` file
- Can read and write files, but **only inside `workspace/`**

## How it works

```
goal.md ─────────────┐
prompt.md ───────────┤                      ┌──► file tools ──► /          = workspace/
skills/*/SKILL.md ───┼──► main agent ──────┤                  /skills/   = skills/ (read-only)
agents/*.md ─────────┤   (create_deep_agent)├──► MCP servers
.env (LLM config) ───┤                      └──► task ──► sub-agent (own prompt, tools, model)
mcp.json ────────────┘
                              final answer ──► stdout + workspace/_result.md
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/). It installs the right Python (3.12) for you:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- An LLM whose model supports **tool calling** (OpenAI-compatible endpoint, Anthropic, Ollama, ...)
- Node.js / `uvx` only if the MCP servers you configure need them

## Quick start

```bash
git clone https://github.com/ap369/micro-agent.git
cd micro-agent

# 1. Install dependencies
uv sync

# 2. Configure the LLM
cp .env.example .env
#    then edit .env:
#    LLM_BASE_URL=https://your-endpoint/v1
#    LLM_API_KEY=sk-...
#    LLM_MODEL=your-model-name

# 3. Describe the task
#    edit goal.md     -> what to accomplish
#    edit prompt.md   -> how the agent should behave
#    add skills to skills/<name>/SKILL.md
#    add sub-agents to agents/<name>.md

# 4. (optional) add MCP servers to mcp.json

# 5. Run
uv run run.py
```

The sample `goal.md` has the agent write `workspace/report.md` using the `report-writing` skill, then hand it to the `reviewer` sub-agent, which writes `workspace/review.md`. You can run it straight after setting up `.env`.

## Project files

| File / folder   | Purpose |
|-----------------|---------|
| `run.py`        | Entry point: command-line options and exit codes; calls `micro_agent` |
| `goal.md`       | The task, sent as the user message |
| `prompt.md`     | System instructions / persona |
| `skills/`       | One folder per skill, each with a `SKILL.md` and optional supporting files |
| `agents/`       | One `.md` file per sub-agent |
| `mcp.json`      | MCP server config |
| `.env`          | LLM endpoint and settings (copy from `.env.example`, never commit it) |
| `workspace/`    | The only folder the agent can read and write |

## Code layout

```
run.py              # entry point: CLI options, exit codes -> MicroAgent
micro_agent/
├── config.py       # Settings (env vars + keyword overrides), ConfigError
├── agent.py        # MicroAgent: builds the deep agent and runs it
├── llm.py          # ModelFactory (init_chat_model), ToolCallRepairMiddleware
├── skills.py       # SkillCatalog: validates skills/, mounted read-only at /skills/
├── subagents.py    # SubAgentLoader: agents/*.md -> deepagents SubAgent specs
├── mcp_servers.py  # McpServers: mcp.json -> LangChain tools, one session per server
├── streaming.py    # RunPrinter: streams the run, logs tool calls per agent
├── markdown.py     # YAML frontmatter parsing
└── console.py      # stderr logging helpers
tests/              # pytest suite (uv run pytest)
```

You can also use it from Python:

```python
import asyncio
from dotenv import load_dotenv
from micro_agent import MicroAgent, Settings

load_dotenv()  # Settings reads the environment; loading .env is up to you
settings = Settings.load(goal="goals/report.md")  # any argument left out uses its env var / default
result = asyncio.run(MicroAgent(settings).run())
```

## Configuration (`.env`)

| Variable          | Required | Default      | Description |
|-------------------|----------|--------------|-------------|
| `LLM_PROVIDER`    | no       | `openai`     | Any `init_chat_model` provider: `openai` (also for OpenAI-compatible endpoints), `anthropic`, `ollama`, ... Providers other than `openai` need their package, e.g. `uv add langchain-ollama` |
| `LLM_BASE_URL`    | no       |              | Custom endpoint, e.g. `https://api.groq.com/openai/v1` (`OPENAI_BASE_URL` also works) |
| `LLM_API_KEY`     | no       |              | API key; if unset, the provider's own variable is used, e.g. `ANTHROPIC_API_KEY` (`OPENAI_API_KEY` also works) |
| `LLM_MODEL`       | yes      |              | Model name, e.g. `openai/gpt-oss-120b` |
| `LLM_TEMPERATURE` | no       | `0`          | Sampling temperature |
| `GOAL_FILE`       | no       | `goal.md`    | Goal file |
| `PROMPT_FILE`     | no       | `prompt.md`  | Prompt file |
| `MCP_CONFIG`      | no       | `mcp.json`   | MCP config file |
| `SKILLS_DIR`      | no       | `skills`     | Skills folder |
| `AGENTS_DIR`      | no       | `agents`     | Sub-agents folder |
| `WORKSPACE_DIR`   | no       | `workspace`  | Sandboxed read/write folder |
| `MAX_STEPS`       | no       | `50`         | Max model calls for the main agent (and default for sub-agents) before it gives up |

Relative paths are resolved from the project folder.

## MCP servers (`mcp.json`)

Same format as Claude Desktop / Cursor:

```json
{
  "mcpServers": {
    "fetch":  { "command": "uvx", "args": ["mcp-server-fetch"] },
    "files":  { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/some/dir"] },
    "remote": { "url": "https://host/mcp", "headers": { "Authorization": "Bearer ${MY_TOKEN}" } },
    "legacy": { "url": "https://host/sse", "transport": "sse" },
    "off":    { "command": "foo", "disabled": true }
  }
}
```

- `command` (+ `args`, `env`, `cwd`) starts a local **stdio** server
- Each server is connected **once per run** and the session is kept open, so stateful servers (browser, database) keep their state between tool calls
- `url` (+ `headers`) connects over **streamable HTTP**; add `"transport": "sse"` for SSE servers
- `${VAR}` placeholders are filled from your environment / `.env`, so secrets stay out of `mcp.json`. An unset variable is a config error (exit code 2)
- stdio servers only get a minimal environment (`PATH`, `HOME`, ...) plus their own `env`; your `.env` secrets are never passed to them
- `"disabled": true` skips a server
- A server that fails to start prints a warning and is skipped. The run continues.

## Skills (`skills/`)

A skill is a folder with a `SKILL.md` file and, optionally, supporting files (templates, references, examples):

```
skills/
  report-writing/
    SKILL.md
    template.md
```

```markdown
---
name: report-writing          # required
description: Write a short, structured markdown report on any topic.   # required
---

# Report writing
1. Read `template.md` (next to this file) with `read_file` and follow its structure.
2. ...
```

- Skills are handled by `deepagents`' `SkillsMiddleware`. Only each skill's `name` and `description` (plus its path) go into the system prompt, so having many skills costs few tokens.
- When a task matches a skill, the agent reads `/skills/<name>/SKILL.md` and any supporting files with the normal `read_file` tool.
- `skills/` is mounted read-only at `/skills/`: writes there are denied.
- A skill with no `name` or `description`, or two skills with the same name, is a config error (exit code 2).

## Agents (`agents/`)

Each `.md` file is a sub-agent. Its frontmatter is the configuration and its body is the sub-agent's system prompt:

```markdown
---
name: reviewer                       # optional, defaults to the file name
description: Reviews a document in the workspace and writes feedback.   # required
model: openai/gpt-oss-20b            # optional, defaults to LLM_MODEL (same endpoint/key)
max_steps: 20                        # optional, defaults to MAX_STEPS
tools: [fetch]                       # optional, MCP tools to give it; defaults to all MCP tools
skills: true                         # optional, set false to hide skills from this agent
read_only: false                     # optional, true = it can read but not write files
---

You are a strict but fair editor. ...
```

- Each file becomes a `deepagents` `SubAgent`. The main agent sees every agent's name and description and calls `task(subagent_type, description)` to hand over a task. It gets the sub-agent's final answer back.
- A built-in **`general-purpose`** sub-agent (from `deepagents`) is always available, with the same tools as the main agent, for isolating big sub-tasks. Add `agents/general-purpose.md` to replace it.
- Sub-agents **don't see the main conversation**, so the task text must be self-contained. They share the workspace, so files are how agents pass work to each other.
- Sub-agents can't call `task` themselves, so they can't call each other in a loop.
- File tools are always available to sub-agents. `tools` only chooses which **MCP** tools a sub-agent gets; an unknown name is a config error.
- If a sub-agent hits its `max_steps`, it stops and returns a notice to the main agent. The run continues.

## Built-in file tools

These come from `deepagents`' `FilesystemMiddleware` on top of a `FilesystemBackend`. Paths are absolute inside a virtual filesystem: `/` is `workspace/`, and `/skills/` is the skills folder (read-only). Path traversal (`..`) and anything outside these folders is refused.

| Tool         | What it does |
|--------------|--------------|
| `ls`         | List a folder |
| `read_file`  | Read a file (supports line offset/limit) |
| `write_file` | Create or overwrite a file |
| `edit_file`  | Replace text inside a file |
| `glob`       | Find files by pattern |
| `grep`       | Search file contents |
| `delete`     | Delete a file |
| `task`       | Hand a task to a sub-agent (main agent only) |

## Running several setups

Override any input from the command line:

```bash
uv run run.py --goal goals/report.md --prompt prompts/analyst.md --mcp mcp.prod.json --env .env.prod \
    --skills skills-prod --agents agents-prod
```

## Output and exit codes

- Progress (MCP connections, every tool call and result) is printed to **stderr**, tagged `[main]` or with the sub-agent's name, e.g. `[reviewer]`
- The final answer is printed to **stdout** and saved to `workspace/_result.md`

So `uv run run.py > answer.md` captures only the answer.

| Code | Meaning |
|------|---------|
| `0`  | Success |
| `1`  | Runtime / LLM error |
| `2`  | Configuration error (missing file or env var, bad JSON, invalid skill/agent frontmatter, unknown tool name) |
| `3`  | `MAX_STEPS` reached before finishing |

## Troubleshooting

- **The agent never calls tools**: your model or endpoint probably doesn't support OpenAI-style tool calling. Try another model.
- **`[mcp] WARNING: failed to load server ...`**: check that the command is installed and on your `PATH`, or that the URL and headers are right.
- **Stops with exit code 3**: raise `MAX_STEPS` or make the goal more specific.
- **`tool_use_failed` (400)**: the model produced a malformed tool call and the provider (e.g. Groq) rejected it. The model is shown its rejected call and asked to fix it, up to 2 times. If a sub-agent still fails, the error goes back to the main agent; if the main agent still fails, the run exits with code 1. Rate limits, timeouts and 5xx errors are retried by the provider's SDK.
- **`Rate limit reached ... tokens per day` (429)**: your provider's daily quota is used up. Wait until it resets, use a smaller model, or upgrade your plan.

## Development

```bash
uv run pytest        # tests: scripted fake models and a fake OpenAI server, no API key or network needed
uvx ruff check .     # lint
uvx ruff format .    # format
```

`deepagents` is pinned below `0.8` because it is pre-1.0 and its API still changes; bump it deliberately and re-run the tests.
