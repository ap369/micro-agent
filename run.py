"""Entry point: `uv run run.py [--goal ...] [--prompt ...] [--env ...] [--mcp ...]`.

See `uv run run.py --help` for all options. The implementation lives in `micro_agent/`.
"""

from micro_agent.cli import main

if __name__ == "__main__":
    main()
