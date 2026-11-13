"""Streams a deep agent run, logging every tool call of the main agent and its sub-agents."""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from micro_agent.console import log, truncate


class RunPrinter:
    """Runs the agent with `subgraphs=True` and tags each log line with the agent that produced it."""

    def __init__(self):
        self._labels: dict[str, str] = {}  # subgraph namespace -> sub-agent name
        self._final: AIMessage | None = None

    async def run(self, agent, goal: str, recursion_limit: int) -> str:
        """Return the main agent's final answer."""
        async for namespace, update in agent.astream(
            {"messages": [HumanMessage(goal)]},
            {"recursion_limit": recursion_limit},
            stream_mode="updates",
            subgraphs=True,
        ):
            self._handle(namespace, update)
        return self._final.text if self._final else ""

    def _handle(self, namespace: tuple, update: dict) -> None:
        is_main = not namespace
        for node in update.values():
            if not isinstance(node, dict):
                continue
            for msg in node.get("messages") or []:
                if isinstance(msg, AIMessage):
                    # sub-agent replies carry the sub-agent's name; its tool results don't,
                    # so remember the name per subgraph
                    if not is_main and msg.name:
                        self._labels[namespace[0]] = msg.name
                    label = self._label(namespace)
                    for call in msg.tool_calls:
                        args = truncate(json.dumps(call["args"], ensure_ascii=False))
                        log(f"[{label}] tool → {call['name']}({args})")
                    if is_main and not msg.tool_calls:
                        self._final = msg
                elif isinstance(msg, ToolMessage):
                    log(f"[{self._label(namespace)}] tool ← {msg.name}: {truncate(msg.content)}")

    def _label(self, namespace: tuple) -> str:
        return self._labels.get(namespace[0], "subagent") if namespace else "main"
