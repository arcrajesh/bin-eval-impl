"""OpenRouter LLM backend with ADK integration.

Provides an OpenAI-compatible client pointing at OpenRouter and a LiteLlm wrapper
for use with Google ADK agents.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_api_key() -> str:
    """Return the OpenRouter API key from environment."""
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not set. Add it to .env or export it."
        )
    return key


def get_model_name() -> str:
    """Return the configured model name (e.g. 'google/gemini-2.0-flash-001')."""
    return os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")


def get_temperature() -> float:
    """Return configured temperature (default 0 for deterministic evaluation)."""
    return float(os.getenv("OPENROUTER_TEMPERATURE", "0"))


def get_openai_client() -> OpenAI:
    """Create an OpenAI-compatible client targeting OpenRouter."""
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=get_api_key(),
    )


def get_adk_model() -> LiteLlm:
    """Create an ADK-compatible LiteLlm model wrapping OpenRouter.

    LiteLlm uses the 'openrouter/<model>' prefix to route through OpenRouter.
    Temperature is set to 0 for deterministic evaluation per the paper.
    """
    model_name = get_model_name()
    # LiteLlm expects the openrouter/ prefix for routing
    litellm_model_id = f"openrouter/{model_name}"
    return LiteLlm(model=litellm_model_id)


def call_llm_sync(
    prompt: str,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Synchronous LLM call via OpenAI-compatible client.

    Used for non-ADK direct calls (e.g., standalone evaluation).
    """
    client = get_openai_client()
    temp = temperature if temperature is not None else get_temperature()
    max_tok = max_tokens or int(os.getenv("OPENROUTER_MAX_TOKENS", "4096"))

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=get_model_name(),
        messages=messages,
        temperature=temp,
        max_tokens=max_tok,
    )
    return response.choices[0].message.content or ""
