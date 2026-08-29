"""
OpenRouter LLM plugin — wraps OpenRouter's OpenAI-compatible API via the
livekit-plugins-openai LLM adapter.

Model: deepseek/deepseek-chat-v3-0324 (DeepSeek-V3, 163k context). Plain
chat model with no hidden reasoning tokens, so no reasoning_effort is passed;
the adapter's auto-injection heuristic also returns False for this model id
(verified against livekit-plugins-openai 1.4.4). The request body therefore
stays plain OpenAI chat: model / messages / stream / temperature (+ the
adapter's stream_options.include_usage, which OpenRouter honours) + the
OpenRouter "provider" routing object.

OpenRouter specifics (https://openrouter.ai/docs):
  base_url  https://openrouter.ai/api/v1
  auth      Authorization: Bearer $OPENROUTER_API_KEY
  headers   HTTP-Referer / X-Title — optional app attribution
  provider  OPENROUTER_PROVIDER_PREFS (config.py): pinned upstream order by
            measured TTFT (crusoe bf16 first), fallbacks allowed, slow
            upstreams ignored. Voice TTFT matters more than a fraction of a
            cent, so this replaces OpenRouter's default price-weighted
            load balancing.
"""

from livekit.plugins.openai import LLM

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    OPENROUTER_HEADERS,
    OPENROUTER_PROVIDER_PREFS,
)


def create_openrouter_llm() -> LLM:
    """Create an OpenRouter-backed LLM instance using the OpenAI-compatible adapter."""
    return LLM(
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.85,
        extra_headers=OPENROUTER_HEADERS,
        extra_body={"provider": OPENROUTER_PROVIDER_PREFS},
    )
