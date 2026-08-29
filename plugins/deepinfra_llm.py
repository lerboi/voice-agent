"""
DeepInfra LLM plugin — wraps DeepInfra's OpenAI-compatible API via the
livekit-plugins-openai LLM adapter.

Model: deepseek-ai/DeepSeek-V3 (plain chat model, 163k context). There are
no hidden reasoning tokens, so no reasoning_effort is passed. The adapter's
own reasoning_effort auto-injection heuristic returns False for this model
id (verified against livekit-plugins-openai 1.4.4), so the request body
stays plain OpenAI chat: model / messages / stream / temperature.

Endpoint / auth per https://deepinfra.com/deepseek-ai/DeepSeek-V3/api:
  POST https://api.deepinfra.com/v1/openai/chat/completions
  Authorization: Bearer $DEEPINFRA_API_KEY
"""

from livekit.plugins.openai import LLM

from config import DEEPINFRA_API_KEY, DEEPINFRA_BASE_URL, DEEPINFRA_MODEL


def create_deepinfra_llm() -> LLM:
    """Create a DeepInfra-backed LLM instance using the OpenAI-compatible adapter."""
    return LLM(
        model=DEEPINFRA_MODEL,
        api_key=DEEPINFRA_API_KEY,
        base_url=DEEPINFRA_BASE_URL,
        temperature=0.85,
    )
