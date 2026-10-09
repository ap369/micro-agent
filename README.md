# micro-agent

A one-shot LangChain agent. You give it a **goal**, a **prompt**, a folder of **knowledge** (markdown), and a list of **MCP servers**. When you run it, it connects to everything, does the job, and exits.

- Works with any **OpenAI-compatible** LLM endpoint (base URL + API key + model name)
- Loads every `knowledge/**/*.md` into its instructions at startup
- Connects to MCP servers (stdio, streamable HTTP, SSE) from a `mcp.json` file
- Can read and write files, but **only inside `workspace/`**

## How it works

```
goal.md ─────────────┐
prompt.md ───────────┤
knowledge/*.md ──────┼──► agent (LangChain) ──► tools ──► workspace/ (files)
.env (LLM config) ───┤                            └──► MCP servers
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
#    add .md files to knowledge/

# 4. (optional) add MCP servers to mcp.json

# 5. Run
uv run agent.py
```

The sample `goal.md` asks the agent to summarise the knowledge base into `workspace/summary.md`, so you can run it straight after setting up `.env`.

## Project files

| File / folder   | Purpose |
|-----------------|---------|
| `agent.py`      | The script |
| `goal.md`       | The task, sent as the user message |
| `prompt.md`     | System instructions / persona |
| `knowledge/`    | Markdown files loaded at startup (recursive) |
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
| `KNOWLEDGE_DIR`   | no       | `knowledge`  | Knowledge folder |
| `WORKSPACE_DIR`   | no       | `workspace`  | Sandboxed read/write folder |
| `MAX_STEPS`       | no       | `50`         | Max agent steps before it gives up |

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
- `${VAR}` placeholders are filled from your environment / `.env`, so secrets stay out of `mcp.json`
- `"disabled": true` skips a server
- A server that fails to start prints a warning and is skipped. The run continues.

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
uv run agent.py --goal goals/report.md --prompt prompts/analyst.md --mcp mcp.prod.json --env .env.prod
```

## Output and exit codes

- Progress (MCP connections, every tool call and result) is printed to **stderr**
- The final answer is printed to **stdout** and saved to `workspace/_result.md`

So `uv run agent.py > answer.md` captures only the answer.

| Code | Meaning |
|------|---------|
| `0`  | Success |
| `1`  | Runtime / LLM error |
| `2`  | Configuration error (missing file or env var, bad JSON) |
| `3`  | `MAX_STEPS` reached before finishing |

## Troubleshooting

- **The agent never calls tools**: your model or endpoint probably doesn't support OpenAI-style tool calling. Try another model.
- **`[mcp] WARNING: failed to load server ...`**: check that the command is installed and on your `PATH`, or that the URL and headers are right.
- **Stops with exit code 3**: raise `MAX_STEPS` or make the goal more specific.
