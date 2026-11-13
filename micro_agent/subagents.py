"""Sub-agents from `agents/*.md` (frontmatter = config, body = system prompt)."""

from pathlib import Path

from deepagents import FilesystemPermission, SubAgent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.tools import BaseTool

from micro_agent.config import ConfigError
from micro_agent.console import log
from micro_agent.llm import ModelFactory, retry_middleware
from micro_agent.markdown import parse_frontmatter
from micro_agent.skills import SKILLS_READ_ONLY, SkillCatalog

# `read_only: true` agents can read everything but write nothing
DENY_ALL_WRITES = FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")


class SubAgentLoader:
    """Turns agent markdown files into deepagents `SubAgent` specs."""

    def __init__(
        self,
        agents_dir: Path,
        mcp_tools: list[BaseTool],
        skills: SkillCatalog,
        models: ModelFactory,
        default_max_steps: int,
    ):
        self.dir = agents_dir
        self._mcp_tools = {t.name: t for t in mcp_tools}
        self._skills = skills
        self._models = models
        self._default_max_steps = default_max_steps

    def load(self) -> list[SubAgent]:
        if not self.dir.is_dir():
            log(f"[agents] directory not found, skipping: {self.dir}")
            return []
        specs: list[SubAgent] = []
        for path in sorted(self.dir.glob("*.md")):
            spec = self._build(path)
            if any(s["name"] == spec["name"] for s in specs):
                raise ConfigError(f"Duplicate agent name '{spec['name']}' ({path})")
            specs.append(spec)
        log(f"[agents] loaded {len(specs)}: {', '.join(s['name'] for s in specs) or '-'}")
        return specs

    def _build(self, path: Path) -> SubAgent:
        meta, body = parse_frontmatter(path)
        name = str(meta.get("name") or path.stem)
        description = str(meta.get("description") or "").strip()
        if not description:
            raise ConfigError(f"{path} needs a 'description' in its frontmatter")
        if not body:
            raise ConfigError(f"{path} has no body (the sub-agent's system prompt)")

        spec: SubAgent = {
            "name": name,
            "description": description,
            "system_prompt": body,
            "permissions": self._permissions(meta),
            "middleware": [self._step_limit(meta, path), retry_middleware()],
        }
        if (tools := self._tools(meta, name, path)) is not None:
            spec["tools"] = tools
        if meta.get("model"):
            spec["model"] = self._models.create(str(meta["model"]))
        if self._skills.enabled and meta.get("skills", True):
            spec["skills"] = self._skills.sources
        return spec

    def _tools(self, meta: dict, name: str, path: Path) -> list[BaseTool] | None:
        """`tools` = MCP tools to give this agent (file tools are always built in). None = inherit all."""
        tools = meta.get("tools")
        if tools is None:
            return None
        if not (isinstance(tools, list) and all(isinstance(t, str) for t in tools)):
            raise ConfigError(f"{path}: 'tools' must be a list of MCP tool names")
        unknown = [t for t in tools if t not in self._mcp_tools]
        if unknown:
            raise ConfigError(
                f"Agent '{name}' lists unknown MCP tool(s): {', '.join(unknown)}. "
                f"Available: {', '.join(self._mcp_tools) or '(none)'}"
            )
        return [self._mcp_tools[t] for t in tools]

    @staticmethod
    def _permissions(meta: dict) -> list[FilesystemPermission]:
        # sub-agent rules replace the parent's, so always keep skills read-only
        return [SKILLS_READ_ONLY, DENY_ALL_WRITES] if meta.get("read_only") else [SKILLS_READ_ONLY]

    def _step_limit(self, meta: dict, path: Path) -> ModelCallLimitMiddleware:
        max_steps = meta.get("max_steps", self._default_max_steps)
        if not isinstance(max_steps, int) or max_steps < 1:
            raise ConfigError(f"{path}: 'max_steps' must be a positive integer")
        # 'end' = stop gracefully and hand a notice back to the main agent
        return ModelCallLimitMiddleware(run_limit=max_steps, exit_behavior="end")
