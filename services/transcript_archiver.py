"""
Transcript Archiver — post-call archival of voice turns into the shared
messages table and long-term memory system.

4-step process (order is CRITICAL — see Voice_Implementation.md Section 3.2):
  1. Lock lastMemorySync on UserCharacter (prevent summarizer race condition)
  2. Batch-embed via Voyage AI + bulk-insert turns into messages table
  3. Generate conversation_summary with metadata.source='voice_call'
  4. Re-extract user_fact entries from the transcript

Voice turns go into the SAME messages table with SAME memory_space_id
as text chat. Voice summaries use memory_type='conversation_summary'.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone

from openai import AsyncOpenAI
from supabase import create_client
import voyageai

from config import (
    DEEPINFRA_API_KEY,
    DEEPINFRA_BASE_URL,
    DEEPINFRA_MODEL,
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
    VOYAGE_API_KEY,
    VOYAGE_MODEL,
)
from services.xp_tracker import get_level_from_xp, get_stage_from_level

logger = logging.getLogger("anione-voice-agent")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)


def _create_llm_client() -> AsyncOpenAI:
    """Create a fresh AsyncOpenAI client for DeepInfra (OpenAI-compatible).

    Created per-call rather than module-level to ensure the httpx client binds
    to the current event loop (archival runs after session.start() returns).
    """
    return AsyncOpenAI(
        api_key=DEEPINFRA_API_KEY,
        base_url=DEEPINFRA_BASE_URL,
    )


async def _voyage_embed_with_retry(texts: list[str], max_retries: int = 2) -> list | None:
    """Embed texts via Voyage AI with retry logic. Returns embeddings list or None."""
    for attempt in range(max_retries + 1):
        try:
            result = await asyncio.to_thread(
                voyage_client.embed,
                texts=texts,
                model=VOYAGE_MODEL,
                input_type="document",
            )
            return result.embeddings
        except Exception:
            if attempt < max_retries:
                wait = (attempt + 1) * 1.5  # 1.5s, 3s
                logger.warning(
                    "Voyage embedding attempt %d/%d failed — retrying in %.1fs",
                    attempt + 1, max_retries + 1, wait,
                )
                await asyncio.sleep(wait)
            else:
                logger.warning(
                    "Voyage embedding failed after %d attempts", max_retries + 1,
                )
    return None


async def _mark_archival_failed(call_id: str, error_msg: str):
    """Mark a voice_calls record with archival failure metadata for retry."""
    try:
        # Merge into existing metadata rather than overwriting
        vc = await asyncio.to_thread(
            lambda: supabase.table("voice_calls")
            .select("metadata")
            .eq("id", call_id)
            .single()
            .execute()
        )
        existing_meta = (vc.data or {}).get("metadata") or {}
        existing_meta["archival_status"] = "failed"
        existing_meta["archival_error"] = str(error_msg)[:500]
        existing_meta["archival_failed_at"] = datetime.now(timezone.utc).isoformat()

        await asyncio.to_thread(
            lambda: supabase.table("voice_calls")
            .update({"metadata": existing_meta})
            .eq("id", call_id)
            .execute()
        )
        logger.info("Marked call %s archival as failed", call_id)
    except Exception:
        logger.exception("Failed to mark archival failure on call %s", call_id)


async def archive_call(
    call_id: str,
    user_id: str,
    character_id: int | None,
    custom_character_id: str | None,
    memory_space_id: str,
    transcript: list,
    shadow_summary: str,
    character_name: str = "Character",
    cumulative_xp: int = 0,
    final_mood: str = "Neutral",
    call_duration_seconds: int = 0,
):
    """
    Archive a completed voice call into the shared memory system.

    Args:
        call_id: UUID of the voice_calls record.
        user_id: User ID.
        character_id: Official character ID (or None).
        custom_character_id: Custom character UUID (or None).
        memory_space_id: Shared namespace (same as text chat).
        transcript: List of TranscriptTurn objects from CallState.
        shadow_summary: Final shadow summary from the call.
        character_name: Character name for prompt formatting.
        cumulative_xp: Total XP earned during the call.
        final_mood: Mood at end of call.
        call_duration_seconds: Total call duration.
    """
    if not transcript:
        logger.info("No transcript to archive for call %s", call_id)
        return

    logger.info(
        "Archiving call %s: %d turns, %ds duration",
        call_id, len(transcript), call_duration_seconds,
    )

    embeddings_failed = False  # Track across all steps for metadata marking
    critical_failure = False   # Track if a critical step (turns insert or summary) failed

    # ---------------------------------------------------------------
    # STEP 1: Lock lastMemorySync FIRST (prevent summarizer race)
    # ---------------------------------------------------------------
    try:
        filter_field = "customCharacterId" if custom_character_id else "characterId"
        filter_value = custom_character_id if custom_character_id else character_id

        await asyncio.to_thread(
            lambda: supabase.table("UserCharacter").update({
                "lastMemorySync": datetime.now(timezone.utc).isoformat(),
            }).eq("userId", user_id).eq(
                filter_field, filter_value
            ).eq("version", 1).execute()
        )

        logger.info("Step 1: lastMemorySync locked")
    except Exception:
        logger.exception("Failed to lock lastMemorySync — continuing anyway")

    # ---------------------------------------------------------------
    # STEP 2: Batch-embed + bulk-insert turns into messages table
    # ---------------------------------------------------------------
    try:
        contents = [t.content for t in transcript]

        # Batch embed via Voyage AI with retry (1-2 API calls for entire batch)
        embeddings = await _voyage_embed_with_retry(contents)
        if embeddings is None:
            embeddings_failed = True

        records = []
        for i, turn in enumerate(transcript):
            record = {
                "memory_space_id": memory_space_id,
                "user_id": user_id,
                "character_id": character_id,
                "custom_character_id": custom_character_id,
                "role": turn.role,
                "content": turn.content,
                "created_at": datetime.fromtimestamp(
                    turn.timestamp, tz=timezone.utc
                ).isoformat(),
                "metadata": {
                    "source": "voice_call",
                    "call_id": call_id,
                    "turn_index": i,
                    "was_interrupted": turn.was_interrupted,
                },
            }
            if embeddings and i < len(embeddings):
                record["embedding"] = embeddings[i]
            records.append(record)

        await asyncio.to_thread(
            lambda: supabase.table("messages").insert(records).execute()
        )
        logger.info("Step 2: %d turns inserted into messages table", len(records))

    except Exception:
        logger.exception("Failed to insert voice turns into messages")
        critical_failure = True

    # ---------------------------------------------------------------
    # STEP 2b: Voice call summary card
    # Skipped — inserted by endCallAPI on the Next.js side to
    # guarantee persistence regardless of agent lifecycle.
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # STEP 3: Generate voice call summary -> conversation_summary
    # ---------------------------------------------------------------
    try:
        final_summary = await _generate_call_summary(
            transcript, shadow_summary, character_name,
            call_duration_seconds, cumulative_xp,
        )

        if final_summary:
            importance = _calculate_importance(
                call_duration_seconds, len(transcript), cumulative_xp,
            )

            # Embed the summary with retry
            summary_embeddings = await _voyage_embed_with_retry([final_summary])
            summary_embedding = summary_embeddings[0] if summary_embeddings else None
            if summary_embedding is None:
                embeddings_failed = True

            memory_record = {
                "user_id": user_id,
                "character_id": character_id,
                "custom_character_id": custom_character_id,
                "memory_space_id": memory_space_id,
                "content": final_summary,
                "memory_type": "conversation_summary",
                "importance": importance,
                "metadata": {
                    "source": "voice_call",
                    "call_id": call_id,
                    "duration_seconds": call_duration_seconds,
                    "total_turns": len(transcript),
                    "started_at": datetime.fromtimestamp(
                        transcript[0].timestamp, tz=timezone.utc
                    ).isoformat() if transcript else None,
                },
            }
            if summary_embedding:
                memory_record["embedding"] = summary_embedding

            await asyncio.to_thread(
                lambda: supabase.table("memories").insert(memory_record).execute()
            )
            logger.info("Step 3: conversation_summary stored (importance=%d)", importance)

    except Exception:
        logger.exception("Failed to generate/store call summary")
        critical_failure = True

    # ---------------------------------------------------------------
    # STEP 4: Extract user_fact entries from transcript (additive)
    # Skip for short calls — not enough content to extract meaningful facts,
    # and we must NOT delete existing facts accumulated from text chat.
    # ---------------------------------------------------------------
    user_turn_count = sum(1 for t in transcript if t.role == "user")
    if user_turn_count >= 5:
        try:
            await _extract_user_facts(
                transcript, memory_space_id, user_id,
                character_id, custom_character_id,
            )
            logger.info("Step 4: user_fact extraction complete")
        except Exception:
            logger.warning("User fact extraction failed (non-blocking)")
    else:
        logger.info("Step 4: skipped user_fact extraction (%d user turns < 5)", user_turn_count)

    # ---------------------------------------------------------------
    # Update UserCharacter relationship state
    # ---------------------------------------------------------------
    try:
        filter_field = "customCharacterId" if custom_character_id else "characterId"
        filter_value = custom_character_id if custom_character_id else character_id

        uc = await asyncio.to_thread(
            lambda: supabase.table("UserCharacter").select(
                "total_call_duration, relationshipXP, relationshipLevel"
            ).eq("userId", user_id).eq(
                filter_field, filter_value,
            ).eq("version", 1).single().execute()
        )

        update_data = {
            "currentMood": final_mood,
        }

        if uc.data:
            current_duration = uc.data.get("total_call_duration", 0) or 0
            update_data["total_call_duration"] = current_duration + call_duration_seconds
            update_data["last_call_at"] = datetime.now(timezone.utc).isoformat()

            # Persist XP earned during the call
            old_xp = uc.data.get("relationshipXP", 0) or 0
            new_xp = old_xp + cumulative_xp
            new_level = get_level_from_xp(new_xp)
            new_stage = get_stage_from_level(new_level)
            update_data["relationshipXP"] = new_xp
            update_data["relationshipLevel"] = new_level
            update_data["relationship_stage"] = new_stage

        await asyncio.to_thread(
            lambda: supabase.table("UserCharacter").update(update_data).eq(
                "userId", user_id
            ).eq(
                filter_field, filter_value,
            ).eq("version", 1).execute()
        )

    except Exception:
        logger.exception("Failed to update UserCharacter post-call")

    # Mark critical failure on the voice_calls record for retry
    if critical_failure:
        await _mark_archival_failed(call_id, "Critical archival step failed (turns insert or summary)")

    # Mark embeddings_failed on the voice_calls record so they can be backfilled
    if embeddings_failed and not critical_failure:
        try:
            vc = await asyncio.to_thread(
                lambda: supabase.table("voice_calls")
                .select("metadata")
                .eq("id", call_id)
                .single()
                .execute()
            )
            meta = (vc.data or {}).get("metadata") or {}
            meta["embeddings_failed"] = True
            await asyncio.to_thread(
                lambda: supabase.table("voice_calls")
                .update({"metadata": meta})
                .eq("id", call_id)
                .execute()
            )
        except Exception:
            logger.warning("Failed to mark embeddings_failed on call %s", call_id)

    logger.info("Archival complete for call %s", call_id)


def _calculate_importance(duration_seconds: int, total_turns: int, xp_gained: int) -> int:
    """Calculate memory importance score (0-10). See Section 3.2."""
    score = 3  # base
    if duration_seconds > 300:   # > 5 min
        score += 1
    if duration_seconds > 900:   # > 15 min
        score += 1
    if xp_gained > 40:
        score += 1
    if total_turns > 20:
        score += 1
    return min(score, 10)


async def _generate_call_summary(
    transcript: list,
    shadow_summary: str,
    character_name: str,
    duration_seconds: int,
    xp_gained: int,
) -> str | None:
    """Generate a final call summary using DeepInfra (DeepSeek-V3)."""
    turns_text = "\n".join(
        f"{'User' if t.role == 'user' else character_name}: {t.content}"
        for t in transcript[-20:]  # last 20 turns for context if long call
    )

    prompt = f"""Summarize this voice call into a concise memory log (under 150 words, single paragraph, past tense, third person).

Call duration: {duration_seconds // 60}m {duration_seconds % 60}s
XP gained: {xp_gained}

Rolling summary from during the call:
{shadow_summary or '(none)'}

Recent transcript:
{turns_text}

Focus on: emotional beats, relationship developments, key facts learned, promises made, narrative progression.
Output ONLY the summary paragraph."""

    try:
        client = _create_llm_client()
        response = await client.chat.completions.create(
            model=DEEPINFRA_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise conversation summarizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        logger.exception("Call summary generation failed")
        return shadow_summary  # fall back to shadow summary


async def _extract_user_facts(
    transcript: list,
    memory_space_id: str,
    user_id: str,
    character_id: int | None,
    custom_character_id: str | None,
):
    """
    Extract user facts from transcript and store them.
    Mirrors extractAndStoreUserFacts() in summarizer.js:
    - Delete all existing user_fact for this space
    - Re-extract from transcript
    - Store each as a separate memory
    """
    user_turns = [t for t in transcript if t.role == "user" and t.content.strip()]
    if not user_turns:
        return

    turns_text = "\n".join(f"User: {t.content}" for t in user_turns)

    prompt = f"""Analyze this voice conversation to extract concrete facts the user revealed about themselves.

USER MESSAGES:
{turns_text}

Extract ONLY facts the USER explicitly stated. Focus on:
- Name, age, job, location, relationships, hobbies, fears, goals, significant events

RULES:
- Only include what was explicitly said, not inferred
- Ignore casual filler
- If no clear facts, return empty array

OUTPUT (JSON only):
{{"facts": [{{"key": "user_name", "value": "Jake", "sentence": "The user's name is Jake."}}]}}"""

    try:
        client = _create_llm_client()
        response = await client.chat.completions.create(
            model=DEEPINFRA_MODEL,
            messages=[
                {"role": "system", "content": "Extract user facts. Output ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
        )

        import json
        parsed = json.loads(response.choices[0].message.content)
        facts = parsed.get("facts", [])

        if not facts:
            logger.info("No user facts extracted from voice transcript")
            return

        # Voice archival is ADDITIVE — we never delete text-chat-derived facts.
        # Text chat's summarizer.js handles its own delete-and-reinsert cycle.
        # Fetch existing fact keys so we only add NEW facts (no duplicates).
        existing_keys = set()
        try:
            existing = await asyncio.to_thread(
                lambda: supabase.table("memories").select("metadata").eq(
                    "memory_space_id", memory_space_id
                ).eq("memory_type", "user_fact").execute()
            )
            for row in (existing.data or []):
                key = (row.get("metadata") or {}).get("fact_key")
                if key:
                    existing_keys.add(key)
        except Exception:
            logger.warning("Failed to fetch existing fact keys — will insert all")

        # Store only new facts (skip keys that already exist)
        fact_records = []
        for fact in facts:
            if not fact.get("key") or not fact.get("sentence"):
                continue
            if fact["key"] in existing_keys:
                continue

            # Embed the fact with retry
            fact_embeddings = await _voyage_embed_with_retry([fact["sentence"]])
            embedding = fact_embeddings[0] if fact_embeddings else None

            record = {
                "user_id": user_id,
                "character_id": character_id,
                "custom_character_id": custom_character_id,
                "memory_space_id": memory_space_id,
                "content": fact["sentence"],
                "memory_type": "user_fact",
                "importance": 9,
                "metadata": {"fact_key": fact["key"], "fact_value": fact.get("value", "")},
            }
            if embedding:
                record["embedding"] = embedding
            fact_records.append(record)

        if fact_records:
            await asyncio.to_thread(
                lambda: supabase.table("memories").insert(fact_records).execute()
            )
            logger.info("Stored %d user facts from voice transcript", len(fact_records))

    except Exception:
        logger.warning("User fact extraction failed (non-blocking)")
