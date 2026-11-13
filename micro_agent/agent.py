"""The main agent: wires settings, skills, MCP tools and sub-agents into `create_deep_agent`."""

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from micro_agent.config import Settings
from micro_agent.console import log
from micro_agent.llm import ModelFactory, ToolCallRepairMiddleware
from micro_agent.mcp_servers import McpServers
from micro_agent.skills import SKILLS_READ_ONLY, SKILLS_ROUTE, SkillCatalog
from micro_agent.streaming import RunPrinter
from micro_agent.subagents import SubAgentLoader

WORKSPACE_PROMPT = """# Filesystem

Your filesystem root `/` is the workspace: save every deliverable there (e.g. `/report.md`).
`/skills/` contains the skills and is read-only."""

RESULT_FILE = "_result.md"


class MicroAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.models = ModelFactory(settings)

    async def run(self) -> str:
        """Build the agent, pursue the goal, save and return the final answer."""
        s = self.settings
        # MCP sessions stay open for the whole run
        async with McpServers(s.mcp_config) as mcp_tools:
            agent = self._build(mcp_tools)
            # backstop only; MAX_STEPS is enforced by ModelCallLimitMiddleware
            result = await RunPrinter().run(agent, s.goal, recursion_limit=10 * s.max_steps + 100)
        (s.workspace / RESULT_FILE).write_text(result, encoding="utf-8")
        return result

    def _build(self, mcp_tools: list[BaseTool]) -> CompiledStateGraph:
        s = self.settings
        s.workspace.mkdir(parents=True, exist_ok=True)

        skills = SkillCatalog(s.skills_dir)
        subagents = SubAgentLoader(s.agents_dir, mcp_tools, skills, self.models, s.max_steps).load()

        agent = create_deep_agent(
            model=self.models.create(),
            tools=mcp_tools,
            system_prompt=f"{s.prompt}\n\n{WORKSPACE_PROMPT}",
            subagents=subagents,
            skills=skills.sources,
            backend=self._backend(skills),
            permissions=[SKILLS_READ_ONLY],
            # 'error' = the run fails with exit code 3 when the main agent runs out of steps
            middleware=[
                ModelCallLimitMiddleware(run_limit=s.max_steps, exit_behavior="error"),
                ToolCallRepairMiddleware(on_failure="error"),
            ],
        )
        log(f"[main] workspace: {s.workspace}")
        return agent

    def _backend(self, skills: SkillCatalog) -> CompositeBackend:
        """`/` = workspace, `/skills/` = skills folder."""
        routes = {}
        if skills.enabled:
            routes[SKILLS_ROUTE] = FilesystemBackend(root_dir=skills.dir, virtual_mode=True)
        return CompositeBackend(
            default=FilesystemBackend(root_dir=self.settings.workspace, virtual_mode=True),
            routes=routes,
        )
