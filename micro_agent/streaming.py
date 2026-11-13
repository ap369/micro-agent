"""Streams a deep agent run, logging every tool call of the main agent and its sub-agents."""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from micro_agent.console import log, truncate


class RunPrinter:
    """Runs the agent with `subgraphs=True` and tags each log line with the agent that produced it."""

    def __init__(self):
        self._pending: list[str] = []  # sub-agents requested via `task`, in call order
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
        label = "main" if is_main else self._label_for(namespace[0])
        for node in update.values():
            if not isinstance(node, dict):
                continue
            for msg in node.get("messages") or []:
                if isinstance(msg, AIMessage):
                    for call in msg.tool_calls:
                        if is_main and call["name"] == "task":
                            self._pending.append(call["args"].get("subagent_type", "subagent"))
                        log(f"[{label}] tool → {call['name']}({truncate(json.dumps(call['args'], ensure_ascii=False))})")
                    if is_main and not msg.tool_calls:
                        self._final = msg
                elif isinstance(msg, ToolMessage):
                    log(f"[{label}] tool ← {msg.name}: {truncate(msg.content)}")

    def _label_for(self, namespace: str) -> str:
        # deepagents doesn't expose which `task` call a subgraph belongs to,
        # so new subgraphs are matched to pending `task` calls in order
        if namespace not in self._labels:
            self._labels[namespace] = self._pending.pop(0) if self._pending else "subagent"
        return self._labels[namespace]
