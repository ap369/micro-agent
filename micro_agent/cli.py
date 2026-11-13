"""Command line entry point and exit codes."""

import argparse
import asyncio
import sys
from enum import IntEnum

from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langgraph.errors import GraphRecursionError

from micro_agent.agent import MicroAgent
from micro_agent.config import ConfigError, Settings
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
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> ExitCode:
    """Run the agent once; the final answer goes to stdout, progress and errors to stderr."""
    args = parse_args(argv)
    try:
        print(asyncio.run(MicroAgent(Settings.load(args)).run()))
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


def main(argv: list[str] | None = None) -> None:
    sys.exit(run(argv))
