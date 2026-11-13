"""Entry point: `uv run agent.py [--goal ...] [--prompt ...] [--env ...] [--mcp ...] [--skills ...] [--agents ...]`.

The implementation lives in the `micro_agent` package.
"""

from micro_agent.cli import main

if __name__ == "__main__":
    main()
