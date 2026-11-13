"""Chat models for the OpenAI-compatible endpoint configured in .env."""

from langchain.agents.middleware import ModelRetryMiddleware
from langchain.agents.middleware.model_retry import default_retry_on
from langchain_core.exceptions import ModelInvalidRequestError
from langchain_openai import ChatOpenAI

from micro_agent.config import Settings


class ModelFactory:
    """Creates chat models that share one endpoint and key; sub-agents may pick another model name."""

    def __init__(self, settings: Settings):
        self._settings = settings

    def create(self, model_name: str | None = None) -> ChatOpenAI:
        return ChatOpenAI(
            base_url=self._settings.base_url,
            api_key=self._settings.api_key,
            model=model_name or self._settings.model,
            temperature=self._settings.temperature,
        )


def _should_retry(exc: BaseException) -> bool:
    # Some providers (e.g. Groq) validate tool calls server-side and reject a malformed one
    # with a 400 `tool_use_failed`; the model usually gets it right on the next attempt.
    if isinstance(exc, ModelInvalidRequestError) and "tool_use_failed" in str(exc):
        return True
    return default_retry_on(exc)  # rate limits, timeouts, 5xx, ...


def retry_middleware() -> ModelRetryMiddleware:
    """Retry failed model calls a few times; give up with the original error afterwards."""
    return ModelRetryMiddleware(max_retries=3, retry_on=_should_retry, on_failure="error")
