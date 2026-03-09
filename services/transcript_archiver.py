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
from datetime import datetime, timezone

from openai import AsyncOpenAI
from supabase import create_client
import voyageai

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
    VOYAGE_API_KEY,
    VOYAGE_MODEL,
)
from services.xp_tracker import get_level_from_xp, get_stage_from_level

logger = logging.getLogger("anione-voice-agent")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
groq_client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


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

    # ---------------------------------------------------------------
    # STEP 1: Lock lastMemorySync FIRST (prevent summarizer race)
    # ---------------------------------------------------------------
    try:
        filter_field = "customCharacterId" if custom_character_id else "characterId"
        filter_value = custom_character_id if custom_character_id else character_id

        supabase.table("UserCharacter").update({
            "lastMemorySync": datetime.now(timezone.utc).isoformat(),
        }).eq("userId", user_id).eq(
            filter_field, filter_value
        ).eq("version", 1).execute()

        logger.info("Step 1: lastMemorySync locked")
    except Exception:
        logger.exception("Failed to lock lastMemorySync — continuing anyway")

    # ---------------------------------------------------------------
    # STEP 2: Batch-embed + bulk-insert turns into messages table
    # ---------------------------------------------------------------
    try:
        contents = [t.content for t in transcript]

        # Batch embed via Voyage AI (1-2 API calls for entire batch)
        # Runs in thread pool to avoid blocking the event loop
        embeddings = None
        try:
            result = await asyncio.to_thread(
                voyage_client.embed,
                texts=contents,
                model=VOYAGE_MODEL,
                input_type="document",
            )
            embeddings = result.embeddings
        except Exception:
            logger.warning("Voyage batch embedding failed — inserting without embeddings")

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

        supabase.table("messages").insert(records).execute()
        logger.info("Step 2: %d turns inserted into messages table", len(records))

    except Exception:
        logger.exception("Failed to insert voice turns into messages")

    # ---------------------------------------------------------------
    # STEP 2b: Insert voice call summary card into messages table
    # This is a role='system' message that ChatWindow renders as a
    # VoiceCallSummaryCard. It persists across page refreshes.
    # ---------------------------------------------------------------
    try:
        mins = call_duration_seconds // 60
        secs = call_duration_seconds % 60
        duration_display = f"{mins:02d}:{secs:02d}"

        # Fetch token_cost from the voice_calls record
        token_cost = 0
        try:
            vc = supabase.table("voice_calls").select("token_cost").eq(
                "id", call_id
            ).single().execute()
            if vc.data:
                token_cost = vc.data.get("token_cost", 0) or 0
        except Exception:
            logger.warning("Could not fetch token_cost for summary card")

        supabase.table("messages").insert({
            "memory_space_id": memory_space_id,
            "user_id": user_id,
            "character_id": character_id,
            "custom_character_id": custom_character_id,
            "role": "system",
            "content": "",
            "metadata": {
                "type": "voice_call_summary",
                "source": "voice_call",
                "call_id": call_id,
                "duration": call_duration_seconds,
                "durationDisplay": duration_display,
                "tokenCost": token_cost,
            },
        }).execute()
        logger.info("Step 2b: voice call summary card inserted")

    except Exception:
        logger.exception("Failed to insert voice call summary card (non-blocking)")

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

            # Embed the summary (thread pool to avoid blocking event loop)
            summary_embedding = None
            try:
                result = await asyncio.to_thread(
                    voyage_client.embed,
                    texts=[final_summary],
                    model=VOYAGE_MODEL,
                    input_type="document",
                )
                summary_embedding = result.embeddings[0]
            except Exception:
                logger.warning("Summary embedding failed")

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

            supabase.table("memories").insert(memory_record).execute()
            logger.info("Step 3: conversation_summary stored (importance=%d)", importance)

    except Exception:
        logger.exception("Failed to generate/store call summary")

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

        uc = supabase.table("UserCharacter").select(
            "total_call_duration, relationshipXP, relationshipLevel"
        ).eq("userId", user_id).eq(
            filter_field, filter_value,
        ).eq("version", 1).single().execute()

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

        supabase.table("UserCharacter").update(update_data).eq(
            "userId", user_id
        ).eq(
            filter_field, filter_value,
        ).eq("version", 1).execute()

    except Exception:
        logger.exception("Failed to update UserCharacter post-call")

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
    """Generate a final call summary using Groq."""
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
        response = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
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
        response = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
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
            existing = supabase.table("memories").select("metadata").eq(
                "memory_space_id", memory_space_id
            ).eq("memory_type", "user_fact").execute()
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

            # Embed the fact (thread pool to avoid blocking event loop)
            embedding = None
            try:
                result = await asyncio.to_thread(
                    voyage_client.embed,
                    texts=[fact["sentence"]],
                    model=VOYAGE_MODEL,
                    input_type="document",
                )
                embedding = result.embeddings[0]
            except Exception:
                pass

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
            supabase.table("memories").insert(fact_records).execute()
            logger.info("Stored %d user facts from voice transcript", len(fact_records))

    except Exception:
        logger.warning("User fact extraction failed (non-blocking)")
