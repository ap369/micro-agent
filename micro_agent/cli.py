"""Command line entry point and exit codes."""

import argparse
import asyncio
import sys

from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langgraph.errors import GraphRecursionError

from micro_agent.agent import MicroAgent
from micro_agent.config import ConfigError, Settings
from micro_agent.console import log

EXIT_OK, EXIT_ERROR, EXIT_CONFIG, EXIT_STEP_LIMIT, EXIT_INTERRUPTED = 0, 1, 2, 3, 130


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="micro-agent", description="One-shot deep agent (LangChain deepagents + MCP)")
    parser.add_argument("--goal", help="goal file (default: $GOAL_FILE or goal.md)")
    parser.add_argument("--prompt", help="prompt file (default: $PROMPT_FILE or prompt.md)")
    parser.add_argument("--env", help="env file (default: .env)")
    parser.add_argument("--mcp", help="MCP config (default: $MCP_CONFIG or mcp.json)")
    parser.add_argument("--skills", help="skills dir (default: $SKILLS_DIR or skills)")
    parser.add_argument("--agents", help="agents dir (default: $AGENTS_DIR or agents)")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> None:
    result = await MicroAgent(Settings.load(args)).run()
    print(result)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        asyncio.run(_run(args))
    except ConfigError as e:
        log(f"[config] {e}")
        sys.exit(EXIT_CONFIG)
    except (ModelCallLimitExceededError, GraphRecursionError):
        log("[main] stopped: MAX_STEPS reached before the goal was completed")
        sys.exit(EXIT_STEP_LIMIT)
    except KeyboardInterrupt:
        sys.exit(EXIT_INTERRUPTED)
    except Exception as e:
        log(f"[agent] error: {e!r}")
        sys.exit(EXIT_ERROR)
    sys.exit(EXIT_OK)
