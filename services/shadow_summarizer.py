"""
Shadow Summarizer — rolling summarization every N turns during a voice call.

Runs in background (never blocks the Fast Loop). Condenses the transcript
buffer so the LLM's active context stays at ~2300 tokens regardless of
call length.

See Voice_Implementation.md Section 3.1.
"""

import logging

from openai import AsyncOpenAI

from config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger("anione-voice-agent")


def _create_groq_client() -> AsyncOpenAI:
    """Create a fresh AsyncOpenAI client for Groq.

    Created per-call rather than module-level to ensure the httpx client
    binds to the current event loop (matches transcript_archiver.py pattern).
    """
    return AsyncOpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )

SUMMARIZATION_PROMPT = """Condense the following conversation into a 200-word summary.
Summarize in the same language the conversation is primarily conducted in.
Preserve: emotional beats, relationship developments, key facts learned,
promises made, and any narrative progression.

Previous summary:
{existing_summary}

New turns:
{recent_turns}

Output ONLY the condensed summary, no explanations."""


async def generate_shadow_summary(
    existing_summary: str,
    recent_turns: list[dict],
    character_name: str = "Character",
) -> str | None:
    """
    Generate a rolling shadow summary by condensing recent turns
    with the existing summary.

    Args:
        existing_summary: Current shadow summary (empty string if first run).
        recent_turns: List of dicts with 'role' and 'content' keys.
        character_name: Name of the character for formatting.

    Returns:
        New condensed summary string, or None on failure.
    """
    turns_text = "\n".join(
        f"{'User' if t['role'] == 'user' else character_name}: {t['content']}"
        for t in recent_turns
    )

    prompt = SUMMARIZATION_PROMPT.format(
        existing_summary=existing_summary or "(No prior summary — this is the start of the call.)",
        recent_turns=turns_text,
    )

    try:
        client = _create_groq_client()
        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise conversation summarizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )

        summary = response.choices[0].message.content.strip()
        logger.info("Shadow summary generated: %d chars", len(summary))
        return summary

    except Exception:
        logger.exception("Shadow summarization failed")
        return None
