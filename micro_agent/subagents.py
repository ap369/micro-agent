"""Sub-agents from `agents/*.md` (frontmatter = config, body = system prompt)."""

from pathlib import Path

from deepagents import FilesystemPermission, SubAgent
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.tools import BaseTool

from micro_agent.config import ConfigError
from micro_agent.console import log
from micro_agent.filesystem import filesystem_prompt
from micro_agent.llm import ModelFactory, ToolCallRepairMiddleware
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
        specs: list[SubAgent] = []
        if self.dir.is_dir():
            for path in sorted(self.dir.glob("*.md")):
                spec = self._from_file(path)
                if any(s["name"] == spec["name"] for s in specs):
                    raise ConfigError(f"Duplicate agent name '{spec['name']}' ({path})")
                specs.append(spec)
        else:
            log(f"[agents] directory not found, skipping: {self.dir}")
        log(f"[agents] loaded {len(specs)}: {', '.join(s['name'] for s in specs) or '-'}")
        if not any(s["name"] == GENERAL_PURPOSE_SUBAGENT["name"] for s in specs):
            specs.append(self._general_purpose())
        return specs

    def _general_purpose(self) -> SubAgent:
        """deepagents' built-in helper, declared explicitly so it gets the same middleware as ours."""
        gp = GENERAL_PURPOSE_SUBAGENT
        return self._spec(
            gp["name"], gp["description"], gp["system_prompt"], meta={}, source="general-purpose"
        )

    def _from_file(self, path: Path) -> SubAgent:
        meta, body = parse_frontmatter(path)
        description = str(meta.get("description") or "").strip()
        if not description:
            raise ConfigError(f"{path} needs a 'description' in its frontmatter")
        if not body:
            raise ConfigError(f"{path} has no body (the sub-agent's system prompt)")
        return self._spec(str(meta.get("name") or path.stem), description, body, meta, source=str(path))

    def _spec(self, name: str, description: str, prompt: str, meta: dict, source: str) -> SubAgent:
        spec: SubAgent = {
            "name": name,
            "description": description,
            "system_prompt": f"{prompt}\n\n{filesystem_prompt(self._skills)}",
            "permissions": self._permissions(meta),
            "middleware": [
                self._step_limit(meta, source),
                # on failure, hand an error back to the main agent instead of ending the run
                ToolCallRepairMiddleware(on_failure="continue"),
            ],
        }
        if (tools := self._tools(meta, name, source)) is not None:
            spec["tools"] = tools
        if meta.get("model"):
            spec["model"] = self._models.create(str(meta["model"]))
        if self._skills.enabled and meta.get("skills", True):
            spec["skills"] = self._skills.sources
        return spec

    def _tools(self, meta: dict, name: str, source: str) -> list[BaseTool] | None:
        """`tools` = MCP tools to give this agent (file tools are always built in). None = inherit all."""
        tools = meta.get("tools")
        if tools is None:
            return None
        if not (isinstance(tools, list) and all(isinstance(t, str) for t in tools)):
            raise ConfigError(f"{source}: 'tools' must be a list of MCP tool names")
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

    def _step_limit(self, meta: dict, source: str) -> ModelCallLimitMiddleware:
        max_steps = meta.get("max_steps", self._default_max_steps)
        if not isinstance(max_steps, int) or max_steps < 1:
            raise ConfigError(f"{source}: 'max_steps' must be a positive integer")
        # 'end' = stop gracefully and hand a notice back to the main agent
        return ModelCallLimitMiddleware(run_limit=max_steps, exit_behavior="end")
