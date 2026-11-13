"""Chat models for the OpenAI-compatible endpoint configured in .env."""

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
