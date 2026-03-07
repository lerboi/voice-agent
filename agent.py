"""
Anione Voice Agent — LiveKit Agents SDK entry point.

Connects to LiveKit Cloud, auto-joins rooms created by createRoomAPI,
and runs the real-time voice pipeline:
  Deepgram STT -> Groq LLM (Llama 3.3 70B) -> Fish Speech TTS

Phases 3-6: character context, shadow summarization, post-call archival,
token billing, XP tracking, relationship stage changes.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import httpx
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    WorkerType,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, silero

from config import (
    BILLING_INTERVAL_SECONDS,
    NEXTJS_API_URL,
    SHADOW_SUMMARY_INTERVAL,
    VOICE_AGENT_API_KEY,
)
from plugins.groq_llm import create_groq_llm
from plugins.fish_tts import create_fish_tts, get_archetype_voice_id
from services.context_loader import (
    load_character_context,
    build_voice_system_prompt,
    get_behavior_tier,
)
from services.shadow_summarizer import generate_shadow_summary
from services.transcript_archiver import archive_call
from services.xp_tracker import (
    calculate_xp_delta,
    check_stage_change,
    get_level_from_xp,
    get_stage_from_level,
)

logger = logging.getLogger("anione-voice-agent")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Fallback test prompt (used only when metadata is missing)
# ---------------------------------------------------------------------------
TEST_SYSTEM_PROMPT = (
    "You are a friendly anime character having a voice conversation. "
    "Keep responses to 1-3 sentences unless asked for more detail. "
    "Use natural speech patterns: contractions, filler words (um, well, hmm), "
    "and emotional interjections. Never use asterisks, markdown, or text "
    "formatting — this is spoken dialogue only. "
    "React to interruptions naturally. Match the user's energy and pacing. "
    "Be warm, expressive, and stay in character at all times."
)


# ---------------------------------------------------------------------------
# Transcript buffer — stores turns in memory during the call
# ---------------------------------------------------------------------------
@dataclass
class TranscriptTurn:
    role: str  # "user" or "assistant"
    content: str
    timestamp: float
    duration_ms: int | None = None
    was_interrupted: bool = False


@dataclass
class CallState:
    call_id: str = ""
    user_id: str = ""
    memory_space_id: str = ""
    character_name: str = ""
    metadata: dict = field(default_factory=dict)
    transcript: list[TranscriptTurn] = field(default_factory=list)
    turn_index: int = 0
    billing_task: asyncio.Task | None = None
    call_start_time: float = 0.0
    shadow_summary: str = ""
    shadow_summary_count: int = 0
    turns_since_shadow: int = 0
    # Phase 6: XP tracking
    cumulative_xp: int = 0
    current_relationship_xp: int = 0  # loaded from metadata
    current_relationship_level: int = 1
    current_relationship_stage: str = "Stranger"
    current_mood: str = "Neutral"


# ---------------------------------------------------------------------------
# Send data message to frontend via LiveKit
# ---------------------------------------------------------------------------
async def send_data_message(room: rtc.Room, msg_type: str, data: dict):
    """Send a JSON data message to all participants in the room."""
    payload = json.dumps({"type": msg_type, **data}).encode("utf-8")
    try:
        await room.local_participant.publish_data(payload, reliable=True)
    except Exception:
        logger.warning("Failed to send data message: %s", msg_type)


# ---------------------------------------------------------------------------
# Billing loop — calls useVoiceTokensAPI every 60 seconds
# ---------------------------------------------------------------------------
async def billing_loop(state: CallState, ctx: JobContext, room: rtc.Room):
    """Periodically deduct voiceMinutes via the Next.js API."""
    await asyncio.sleep(BILLING_INTERVAL_SECONDS)

    async with httpx.AsyncClient(timeout=15) as client:
        while True:
            try:
                resp = await client.post(
                    f"{NEXTJS_API_URL}/api/Voice/useVoiceTokensAPI",
                    json={"callId": state.call_id, "minutesUsed": 1},
                    headers={"Authorization": f"Bearer {VOICE_AGENT_API_KEY}"},
                )
                data = resp.json()

                if resp.status_code == 400:
                    # Minutes exhausted — send notification and shut down
                    logger.warning(
                        "Voice minutes exhausted for call %s — shutting down",
                        state.call_id,
                    )
                    await send_data_message(room, "minutes_exhausted", {
                        "remainingMinutes": 0,
                    })
                    await ctx.shutdown()
                    return

                if resp.status_code != 200:
                    logger.error("Billing API error %d: %s", resp.status_code, data)
                else:
                    remaining = data.get("remainingMinutes", 0)
                    logger.info(
                        "Billing tick for call %s: remaining=%d",
                        state.call_id,
                        remaining,
                    )

                    # Warn at 1 minute remaining
                    if remaining == 1:
                        await send_data_message(room, "minutes_warning", {
                            "remainingMinutes": 1,
                        })

            except Exception:
                logger.exception("Billing request failed for call %s", state.call_id)

            await asyncio.sleep(BILLING_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Shadow summarization — runs in background every N turns
# ---------------------------------------------------------------------------
async def maybe_shadow_summarize(state: CallState):
    """Check if shadow summarization should trigger and run it."""
    if state.turns_since_shadow < SHADOW_SUMMARY_INTERVAL:
        return

    start_idx = len(state.transcript) - state.turns_since_shadow
    recent_turns = [
        {"role": t.role, "content": t.content}
        for t in state.transcript[start_idx:]
    ]

    if not recent_turns:
        return

    state.turns_since_shadow = 0

    new_summary = await generate_shadow_summary(
        existing_summary=state.shadow_summary,
        recent_turns=recent_turns,
        character_name=state.character_name,
    )

    if new_summary:
        state.shadow_summary = new_summary
        state.shadow_summary_count += 1
        logger.info(
            "Shadow summary #%d generated for call %s",
            state.shadow_summary_count,
            state.call_id,
        )


# ---------------------------------------------------------------------------
# Prewarm — preload heavy models at worker startup (not per-room)
# ---------------------------------------------------------------------------
def prewarm(proc: JobProcess):
    """Load Silero VAD model into RAM on server start for fast cold-starts."""
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarm complete: Silero VAD loaded")


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------
async def _entrypoint(ctx: JobContext):
    """Called when the agent joins a room."""

    state = CallState(call_start_time=time.time())
    room = ctx.room

    # Parse room metadata set by createRoomAPI
    meta = {}
    raw_metadata = room.metadata
    if raw_metadata:
        try:
            meta = json.loads(raw_metadata)
            state.call_id = meta.get("callId", "")
            state.user_id = meta.get("userId", "")
            state.memory_space_id = meta.get("memorySpaceId", "")
            state.character_name = meta.get("characterName", "")
            state.current_relationship_xp = meta.get("relationshipXP", 0)
            state.current_relationship_level = meta.get("relationshipLevel", 1)
            state.current_relationship_stage = meta.get("relationshipStage", "Stranger")
            state.current_mood = meta.get("currentMood", "Neutral")
            state.metadata = meta
            logger.info(
                "Joined room for character=%s, call=%s, level=%d, stage=%s",
                state.character_name,
                state.call_id,
                state.current_relationship_level,
                state.current_relationship_stage,
            )
        except json.JSONDecodeError:
            logger.warning("Failed to parse room metadata: %s", raw_metadata[:200])

    # --- Load character context from Supabase ---
    system_prompt = TEST_SYSTEM_PROMPT
    char_ctx = None
    if meta:
        try:
            char_ctx = await load_character_context(meta)
            nsfw_enabled = meta.get("nsfwEnabled", False)
            system_prompt = build_voice_system_prompt(char_ctx, nsfw_enabled)
            state.character_name = char_ctx.character_name or state.character_name
            # Load starting XP from the UserCharacter (fetched via context_loader)
            # For now we compute from level; Phase 5+ context_loader can return exact XP
            logger.info(
                "Voice prompt built: %d chars for %s",
                len(system_prompt),
                state.character_name,
            )
        except Exception:
            logger.exception("Failed to load character context — using fallback prompt")

    # Build initial chat context
    initial_ctx = llm.ChatContext()
    initial_ctx.append(role="system", text=system_prompt)

    # Resolve voice profile for TTS
    voice_id = ""
    if char_ctx:
        # Priority: explicit voice_profile_id > reference URL > archetype fallback
        voice_id = char_ctx.voice_profile_id
        if not voice_id and char_ctx.voice_reference_url:
            # Custom reference audio — use the URL as the voice ID
            # Fish Speech accepts reference audio URLs as voice identifiers
            voice_id = char_ctx.voice_reference_url
        if not voice_id:
            # Fallback to archetype detection from persona gender/personality
            voice_id = get_archetype_voice_id(
                char_ctx.persona_gender,
                char_ctx.persona_personality,
            )
        if voice_id:
            logger.info("Voice profile resolved: %s", voice_id[:50])

    # Build voice pipeline with prewarmed VAD
    voice_assistant = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=create_groq_llm(),
        tts=create_fish_tts(voice_id=voice_id or None),
        chat_ctx=initial_ctx,
    )

    # --- Transcript tracking + XP via pipeline events ---

    @voice_assistant.on("user_speech_committed")
    def on_user_speech(msg: llm.ChatMessage):
        content = msg.content or ""
        if not content.strip():
            return

        state.transcript.append(
            TranscriptTurn(role="user", content=content, timestamp=time.time())
        )
        state.turn_index += 1
        state.turns_since_shadow += 1
        logger.info("[Turn %d] User: %s", state.turn_index, content[:100])

        # XP tracking — fast pattern classification, no API call
        intent, xp_delta = calculate_xp_delta(content)
        old_total_xp = state.current_relationship_xp + state.cumulative_xp
        state.cumulative_xp += xp_delta
        new_total_xp = state.current_relationship_xp + state.cumulative_xp

        logger.info(
            "[XP] intent=%s, delta=+%d, cumulative=%d, total=%d",
            intent, xp_delta, state.cumulative_xp, new_total_xp,
        )

        # Check for relationship stage change
        stage_change = check_stage_change(old_total_xp, new_total_xp)
        if stage_change:
            state.current_relationship_level = stage_change["new_level"]
            state.current_relationship_stage = stage_change["new_stage"]

            logger.info(
                "[STAGE CHANGE] %s -> %s (level %d -> %d)",
                stage_change["old_stage"],
                stage_change["new_stage"],
                stage_change["old_level"],
                stage_change["new_level"],
            )

            # Send data message to frontend for toast notification
            asyncio.create_task(send_data_message(room, "level_up", {
                "newLevel": stage_change["new_level"],
                "newStage": stage_change["new_stage"],
                "oldStage": stage_change["old_stage"],
                "totalXP": new_total_xp,
            }))

        # Shadow summarization check (non-blocking)
        asyncio.create_task(maybe_shadow_summarize(state))

    @voice_assistant.on("agent_speech_committed")
    def on_agent_speech(msg: llm.ChatMessage):
        content = msg.content or ""
        if content.strip():
            state.transcript.append(
                TranscriptTurn(role="assistant", content=content, timestamp=time.time())
            )
            state.turns_since_shadow += 1
            logger.info("[Turn %d] Agent: %s", state.turn_index, content[:100])

    @voice_assistant.on("agent_speech_interrupted")
    def on_agent_interrupted(msg: llm.ChatMessage):
        if state.transcript and state.transcript[-1].role == "assistant":
            state.transcript[-1].was_interrupted = True
            logger.info("[Turn %d] Agent interrupted", state.turn_index)

    # Register shutdown callback for cleanup and post-call archival
    async def on_shutdown():
        # Cleanup billing
        if state.billing_task and not state.billing_task.done():
            state.billing_task.cancel()

        call_duration = int(time.time() - state.call_start_time)
        logger.info(
            "Call ended: call=%s, turns=%d, duration=%ds, xp_earned=%d",
            state.call_id,
            len(state.transcript),
            call_duration,
            state.cumulative_xp,
        )

        # Post-call archival
        if state.call_id and state.transcript:
            try:
                await archive_call(
                    call_id=state.call_id,
                    user_id=state.user_id,
                    character_id=meta.get("characterId"),
                    custom_character_id=meta.get("customCharacterId"),
                    memory_space_id=state.memory_space_id,
                    transcript=state.transcript,
                    shadow_summary=state.shadow_summary,
                    character_name=state.character_name,
                    cumulative_xp=state.cumulative_xp,
                    final_mood=state.current_mood,
                    call_duration_seconds=call_duration,
                )
            except Exception:
                logger.exception("Post-call archival failed for call %s", state.call_id)

    ctx.add_shutdown_callback(on_shutdown)

    # Start billing loop
    if state.call_id:
        state.billing_task = asyncio.create_task(billing_loop(state, ctx, room))

    # Start the voice pipeline — non-blocking, framework keeps job alive
    # until all participants leave the room or ctx.shutdown() is called
    voice_assistant.start(ctx.room)


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=_entrypoint,
            prewarm_fnc=prewarm,
            worker_type=WorkerType.ROOM,
            auto_subscribe=AutoSubscribe.AUDIO_ONLY,
        ),
    )
