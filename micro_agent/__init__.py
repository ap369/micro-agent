"""One-shot deep agent built on LangChain `deepagents`.

Loads a goal + prompt, skills (Agent Skills format, loaded on demand), sub-agents
(called through the `task` tool), MCP servers from a JSON config and an
OpenAI-compatible LLM from .env, then runs until the goal is done.
File access is sandboxed to WORKSPACE_DIR; skills are mounted read-only at /skills/.
"""

from micro_agent.agent import MicroAgent
from micro_agent.config import ConfigError, Settings

__all__ = ["ConfigError", "MicroAgent", "Settings"]
