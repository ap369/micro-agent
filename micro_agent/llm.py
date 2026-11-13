"""Chat models (any LangChain provider) and recovery from rejected tool calls."""

from collections.abc import Awaitable, Callable
from typing import Literal

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from micro_agent.config import ConfigError, Settings
from micro_agent.console import log, truncate


class ModelFactory:
    """Creates chat models for the configured provider; sub-agents may pick another model name.

    Transient errors (rate limits, timeouts, 5xx) are retried by the provider SDK itself.
    """

    def __init__(self, settings: Settings):
        self._settings = settings

    def create(self, model_name: str | None = None) -> BaseChatModel:
        s = self._settings
        kwargs = {"temperature": s.temperature}
        if s.base_url:
            kwargs["base_url"] = s.base_url
        if s.api_key:
            kwargs["api_key"] = s.api_key
        try:
            return init_chat_model(model_name or s.model, model_provider=s.provider, **kwargs)
        except (ImportError, ValueError) as e:
            raise ConfigError(f"Cannot create model for provider {s.provider!r}: {e}") from None


class ToolCallRepairMiddleware(AgentMiddleware):
    """Recover when the provider rejects a malformed tool call (HTTP 400, code `tool_use_failed`).

    Some providers (e.g. Groq) validate tool calls server-side. Resending the same request often
    reproduces the same mistake, so the model is shown its rejected call and asked to fix it.
    """

    def __init__(self, max_repairs: int = 2, on_failure: Literal["error", "continue"] = "error"):
        super().__init__()
        self.max_repairs = max_repairs
        # 'continue' = give up with an error message (sub-agents hand it back to the main agent)
        self.on_failure = on_failure

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        for attempt in range(self.max_repairs + 1):
            try:
                return await handler(request)
            except Exception as exc:
                if getattr(exc, "code", None) != "tool_use_failed":
                    raise
                if attempt == self.max_repairs:
                    if self.on_failure == "continue":
                        return AIMessage(f"Error: the model kept producing invalid tool calls: {exc}")
                    raise
                log(f"[retry] tool call rejected, asking the model to fix it ({attempt + 1})")
                request = request.override(messages=[*request.messages, HumanMessage(_repair_hint(exc))])
        raise AssertionError("unreachable")


def _repair_hint(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    rejected = body.get("failed_generation") if isinstance(body, dict) else None
    error = getattr(exc, "message", None) or str(exc)
    hint = f"Your previous tool call was rejected by the API: {truncate(error, 500)}"
    if rejected:
        hint += f"\nRejected call: {truncate(rejected, 1000)}"
    return hint + "\nCall the tool again with arguments that exactly match its schema (parameter names too)."
