"""
Voice LLM plugin — the conversation model, via the livekit-plugins-openai LLM
adapter (any OpenAI-compatible API).

Provider is chosen by VOICE_LLM_PROVIDER in config.py:
  parasail    (default) Parasail's own API, https://api.parasail.io/v1,
              model parasail-deepseek-v41-flash (DeepSeek V4.1 Flash).
              Thinking off via chat_template_kwargs (PARASAIL_EXTRA_BODY).
  openrouter  rollback: https://openrouter.ai/api/v1, deepseek/deepseek-v4.1-flash,
              provider routing (OPENROUTER_PROVIDER_PREFS) + reasoning off,
              plus OpenRouter's optional attribution headers.

The adapter's own reasoning_effort auto-injection only fires for gpt-5* ids
(checked against livekit-plugins-openai 1.4.4), so the request body stays plain
OpenAI chat: model / messages / stream / temperature (+ the adapter's
stream_options.include_usage) + the provider's extra_body.
"""

from livekit.plugins.openai import LLM

from config import LLM_API_KEY, LLM_BASE_URL, LLM_EXTRA_BODY, LLM_HEADERS, LLM_MODEL


def create_voice_llm() -> LLM:
    """Create the conversation LLM for the configured provider."""
    return LLM(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0.85,
        extra_body=LLM_EXTRA_BODY,
        **({"extra_headers": LLM_HEADERS} if LLM_HEADERS else {}),
    )
