"""Run the agent once: `uv run run.py [--goal ...] [--prompt ...] [--env ...] [--mcp ...]`.

The final answer goes to stdout, progress and errors to stderr. See `--help` for all options.
"""

import argparse
import asyncio
import sys
from enum import IntEnum

from dotenv import load_dotenv
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langgraph.errors import GraphRecursionError

from micro_agent import ConfigError, MicroAgent, Settings
from micro_agent.config import resolve_path
from micro_agent.console import log


class ExitCode(IntEnum):
    OK = 0
    ERROR = 1  # runtime / LLM error
    CONFIG = 2  # invalid or missing configuration
    STEP_LIMIT = 3  # MAX_STEPS reached before the goal was completed
    INTERRUPTED = 130


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="micro-agent", description="One-shot deep agent (LangChain deepagents + MCP)"
    )
    parser.add_argument("--goal", help="goal file (default: $GOAL_FILE or goal.md)")
    parser.add_argument("--prompt", help="prompt file (default: $PROMPT_FILE or prompt.md)")
    parser.add_argument("--env", help="env file (default: .env)")
    parser.add_argument("--mcp", help="MCP config (default: $MCP_CONFIG or mcp.json)")
    parser.add_argument("--skills", help="skills dir (default: $SKILLS_DIR or skills)")
    parser.add_argument("--agents", help="agents dir (default: $AGENTS_DIR or agents)")
    parser.add_argument("--output", help="also save the final answer to this file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> ExitCode:
    args = parse_args(argv)
    # variables already set in the environment win over the .env file
    load_dotenv(resolve_path(args.env or ".env"))
    try:
        settings = Settings.load(
            goal=args.goal,
            prompt=args.prompt,
            mcp_config=args.mcp,
            skills_dir=args.skills,
            agents_dir=args.agents,
        )
        result = asyncio.run(MicroAgent(settings).run())
        if args.output:
            output = resolve_path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(result, encoding="utf-8")
        print(result)
    except ConfigError as e:
        log(f"[config] {e}")
        return ExitCode.CONFIG
    except (ModelCallLimitExceededError, GraphRecursionError):
        log("[main] stopped: MAX_STEPS reached before the goal was completed")
        return ExitCode.STEP_LIMIT
    except KeyboardInterrupt:
        return ExitCode.INTERRUPTED
    except Exception as e:
        log(f"[agent] error: {e!r}")
        return ExitCode.ERROR
    return ExitCode.OK


if __name__ == "__main__":
    sys.exit(main())
