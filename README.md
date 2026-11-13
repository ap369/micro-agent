# micro-agent

A one-shot LangChain agent. You give it a **goal**, a **prompt**, **skills**, **sub-agents**, and a list of **MCP servers**. When you run it, it connects to everything, does the job, and exits.

- Works with any **OpenAI-compatible** LLM endpoint (base URL + API key + model name)
- **Skills** (Claude-style): only names and descriptions go in the prompt; the agent loads a skill's full instructions when it needs them
- **Agents**: specialised sub-agents the main agent can hand tasks to, each with its own prompt, tools and model
- Connects to MCP servers (stdio, streamable HTTP, SSE) from a `mcp.json` file
- Can read and write files, but **only inside `workspace/`**

## How it works

```
goal.md ─────────────┐
prompt.md ───────────┤                      ┌──► file tools ──► workspace/
skills/*/SKILL.md ───┼──► main agent ──────┼──► load_skill / read_skill_file ──► skills/
agents/*.md ─────────┤                      ├──► MCP servers
.env (LLM config) ───┤                      └──► delegate ──► sub-agent (own prompt, tools, model)
mcp.json ────────────┘
                              final answer ──► stdout + workspace/_result.md
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/). It installs the right Python (3.12) for you:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- An OpenAI-compatible endpoint whose model supports **tool calling**
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
#    OPENAI_BASE_URL=https://your-endpoint/v1
#    OPENAI_API_KEY=sk-...
#    LLM_MODEL=your-model-name

# 3. Describe the task
#    edit goal.md     -> what to accomplish
#    edit prompt.md   -> how the agent should behave
#    add skills to skills/<name>/SKILL.md
#    add sub-agents to agents/<name>.md

# 4. (optional) add MCP servers to mcp.json

# 5. Run
uv run agent.py
```

The sample `goal.md` has the agent write `workspace/report.md` using the `report-writing` skill, then hand it to the `reviewer` sub-agent, which writes `workspace/review.md`. You can run it straight after setting up `.env`.

## Project files

| File / folder   | Purpose |
|-----------------|---------|
| `agent.py`      | The script |
| `goal.md`       | The task, sent as the user message |
| `prompt.md`     | System instructions / persona |
| `skills/`       | One folder per skill, each with a `SKILL.md` and optional supporting files |
| `agents/`       | One `.md` file per sub-agent |
| `mcp.json`      | MCP server config |
| `.env`          | LLM endpoint and settings (copy from `.env.example`, never commit it) |
| `workspace/`    | The only folder the agent can read and write |

## Configuration (`.env`)

| Variable          | Required | Default      | Description |
|-------------------|----------|--------------|-------------|
| `OPENAI_BASE_URL` | yes      |              | OpenAI-compatible endpoint, e.g. `https://host/v1` |
| `OPENAI_API_KEY`  | yes      |              | API key |
| `LLM_MODEL`       | yes      |              | Model name |
| `LLM_TEMPERATURE` | no       | `0`          | Sampling temperature |
| `GOAL_FILE`       | no       | `goal.md`    | Goal file |
| `PROMPT_FILE`     | no       | `prompt.md`  | Prompt file |
| `MCP_CONFIG`      | no       | `mcp.json`   | MCP config file |
| `SKILLS_DIR`      | no       | `skills`     | Skills folder |
| `AGENTS_DIR`      | no       | `agents`     | Sub-agents folder |
| `WORKSPACE_DIR`   | no       | `workspace`  | Sandboxed read/write folder |
| `MAX_STEPS`       | no       | `50`         | Max tool rounds (model call + tool call) before it gives up |

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
name: report-writing          # optional, defaults to the folder name
description: Write a short, structured markdown report on any topic.   # required
---

# Report writing
1. Read `template.md` with `read_skill_file` and follow its structure.
2. ...
```

- Only the `name` and `description` of each skill go into the system prompt, so having many skills costs few tokens.
- When a task matches a skill, the agent calls `load_skill(name)` to get the full body and the list of supporting files, then `read_skill_file(name, path)` to read those files.
- Skill folders are read-only for the agent, and `read_skill_file` can't leave the skill's folder.

## Agents (`agents/`)

Each `.md` file is a sub-agent. Its frontmatter is the configuration and its body is the sub-agent's system prompt:

```markdown
---
name: reviewer                       # optional, defaults to the file name
description: Reviews a document in the workspace and writes feedback.   # required
tools: [read_file, write_file]       # optional, defaults to all file, skill and MCP tools
model: openai/gpt-oss-20b            # optional, defaults to LLM_MODEL (same endpoint/key)
max_steps: 20                        # optional, defaults to MAX_STEPS
---

You are a strict but fair editor. ...
```

- The main agent sees each agent's name and description and calls `delegate(agent, task)` to hand over a task. It gets the sub-agent's final answer back.
- Sub-agents **don't see the main conversation**, so the task text must be self-contained. They share `workspace/`, so files are how agents pass work to each other.
- Sub-agents can't delegate further, so they can't call each other in a loop.
- `tools` can list any built-in tool (`list_files`, `read_file`, `write_file`, `append_file`, `delete_file`, `load_skill`, `read_skill_file`) or MCP tool by name. An unknown name is a config error. A sub-agent gets skills only if it has `load_skill`.
- If a sub-agent fails or hits its step limit, `delegate` returns the error to the main agent and the run continues.

## Built-in file tools

All file paths are relative to `workspace/`. Anything outside it (`../`, absolute paths, symlink escapes) is refused.

| Tool          | What it does |
|---------------|--------------|
| `list_files`  | Recursive listing of a workspace folder |
| `read_file`   | Read a text file |
| `write_file`  | Create or overwrite a file (creates parent folders) |
| `append_file` | Append to a file |
| `delete_file` | Delete a file |

## Running several setups

Override any input from the command line:

```bash
uv run agent.py --goal goals/report.md --prompt prompts/analyst.md --mcp mcp.prod.json --env .env.prod \
    --skills skills-prod --agents agents-prod
```

## Output and exit codes

- Progress (MCP connections, every tool call and result) is printed to **stderr**, tagged `[main]` or with the sub-agent's name, e.g. `[reviewer]`
- The final answer is printed to **stdout** and saved to `workspace/_result.md`

So `uv run agent.py > answer.md` captures only the answer.

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
