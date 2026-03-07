"""
Groq LLM plugin — wraps Groq's OpenAI-compatible API via the
livekit-plugins-openai LLM adapter. Targets < 500ms TTFT with
Llama 3.3 70B.
"""

from livekit.plugins.openai import LLM

from config import GROQ_API_KEY, GROQ_MODEL


def create_groq_llm() -> LLM:
    """Create a Groq-backed LLM instance using OpenAI-compatible adapter."""
    return LLM(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.85,
    )
