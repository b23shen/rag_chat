"""Chat models for agents. The only place model clients and their retries are configured."""

from langchain_google_genai import ChatGoogleGenerativeAI

from promo_rag.agents.config import AgentConfigError, ModelRole
from promo_rag.config import get_settings

ROLE_SETTINGS: dict[ModelRole, str] = {
    "chat": "chat_model",
    "helper": "helper_model",
    "judge": "judge_model",
}
# First call + up to 2 retries (on rate limits, server errors, timeouts). LangChain's default
# is 6; add no retry middleware on top, or the retries multiply.
MAX_RETRIES = 2


def model_for_role(role: ModelRole) -> str:
    setting = ROLE_SETTINGS[role]
    name = getattr(get_settings(), setting)
    if not name:
        # Fail loudly: a role silently falling back to some default model is how a "helper"
        # ends up calling an expensive or unapproved model.
        raise AgentConfigError(f"{setting.upper()} is not set in .env")
    return name


def chat_model(name: str, temperature: float, timeout_s: float) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=name,
        google_api_key=get_settings().gemini_api_key.get_secret_value(),
        temperature=temperature,
        timeout=timeout_s,
        max_retries=MAX_RETRIES,
    )
