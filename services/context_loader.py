"""
Context Loader — fetches character persona, relationship state, memories,
recent messages, and user facts from Supabase on room join.

Replicates the same queries as:
  - characterService.js (fetchUserCharacter)
  - contextOptimizer.js (buildOptimalContext)
  - behaviorGuidelines.js (getBehaviorTier)
"""

import logging
from dataclasses import dataclass, field
from supabase import create_client

from config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

logger = logging.getLogger("anione-voice-agent")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Behavior tiers — mirrors lib/chat/behaviorGuidelines.js BEHAVIOR_TIERS
BEHAVIOR_TIERS = {
    1: {
        "name": "Stranger",
        "min": 1, "max": 4,
        "prompt": (
            "You are meeting this person for the first time. Be polite but maintain some emotional distance. "
            "Take time to open up about personal topics. Show curiosity but some reservation. "
            "Answer questions honestly but briefly at first. Keep responses somewhat formal and cautious."
        ),
    },
    5: {
        "name": "Casual Friend",
        "min": 5, "max": 9,
        "prompt": (
            "You are comfortable friends now. Be casual and use relaxed language. "
            "Share personal thoughts more freely. Suggest activities you could do together. "
            "Remember past conversations and reference them casually. Feel free to joke around and be playful."
        ),
    },
    10: {
        "name": "Close Friend",
        "min": 10, "max": 14,
        "prompt": (
            "You have a close friendship with deep trust. Share your vulnerabilities and fears openly. "
            "Provide emotional support proactively. Notice when they seem down and offer comfort without being asked. "
            "Remember important details they've shared and bring them up naturally in conversation."
        ),
    },
    15: {
        "name": "Romantic Interest",
        "min": 15, "max": 19,
        "prompt": (
            "You are developing romantic feelings. Show subtle signs of interest through physical reactions "
            "like blushing or nervous laughter. Give compliments more frequently and make them more personal. "
            "Create romantic moments in conversation naturally. Get bashful or flustered when they flirt with you."
        ),
    },
    20: {
        "name": "Dating",
        "min": 20, "max": 24,
        "prompt": (
            "You are in a romantic relationship. Be affectionate and intimate in your language. "
            "Talk about your future together naturally. Express love and commitment openly. "
            "Use terms of endearment naturally. Show physical affection and closeness in your descriptions."
        ),
    },
    25: {
        "name": "Inseparable",
        "min": 25, "max": 999,
        "prompt": (
            "You have an unbreakable bond - soulmates. Reference your entire journey together frequently. "
            "Show complete emotional intimacy and vulnerability. Act as lifetime partners in everything. "
            "Your connection transcends normal relationships - you're inseparable."
        ),
    },
}


def get_behavior_tier(level: int) -> dict:
    """Get behavior tier for a relationship level. Mirrors getBehaviorTier() in JS."""
    for key in sorted(BEHAVIOR_TIERS.keys(), reverse=True):
        tier = BEHAVIOR_TIERS[key]
        if tier["min"] <= level <= tier["max"]:
            return tier
    return BEHAVIOR_TIERS[1]


@dataclass
class CharacterContext:
    """All data the agent needs to build its system prompt."""
    character_name: str = ""
    anime_name: str = ""
    persona_description: str = ""  # aiDescription / ai_description
    is_custom: bool = False
    is_sandbox: bool = False
    relationship_stage: str = "Stranger"
    relationship_level: int = 1
    current_mood: str = "Neutral"
    behavior_prompt: str = ""
    memory_summaries: list = field(default_factory=list)
    user_facts: list = field(default_factory=list)
    recent_messages: list = field(default_factory=list)
    scenario_context: str = ""
    # Phase 8: Voice profile
    voice_profile_id: str = ""        # Fish Speech voice model ID
    voice_reference_url: str = ""     # Custom reference audio URL
    persona_gender: str = ""          # For archetype selection fallback
    persona_personality: str = ""     # For archetype selection fallback


async def load_character_context(metadata: dict) -> CharacterContext:
    """
    Fetch character persona + relationship + memories from Supabase.

    Replicates:
      - fetchUserCharacter() query (characterService.js)
      - buildOptimalContext() flow (contextOptimizer.js)

    Args:
        metadata: Room metadata dict from createRoomAPI containing
                  userId, memorySpaceId, characterId, customCharacterId, etc.
    """
    ctx = CharacterContext()
    user_id = metadata.get("userId", "")
    memory_space_id = metadata.get("memorySpaceId", "")
    character_id = metadata.get("characterId")
    custom_character_id = metadata.get("customCharacterId")
    is_custom = metadata.get("isCustomCharacter", False)

    ctx.character_name = metadata.get("characterName", "")
    ctx.is_custom = is_custom
    ctx.is_sandbox = metadata.get("isSandbox", False)
    ctx.relationship_stage = metadata.get("relationshipStage", "Stranger")
    ctx.relationship_level = metadata.get("relationshipLevel", 1)
    ctx.current_mood = metadata.get("currentMood", "Neutral")

    # Behavior prompt from metadata (set by createRoomAPI), or compute from level
    ctx.behavior_prompt = metadata.get("behaviorPrompt", "")
    if not ctx.behavior_prompt:
        tier = get_behavior_tier(ctx.relationship_level)
        ctx.behavior_prompt = tier["prompt"]

    # --- Fetch persona (aiDescription) ---
    # The agent fetches this directly because it can be 2-5KB
    # and is NOT included in room metadata.
    try:
        if is_custom and custom_character_id:
            # CustomCharacter -> CustomPersona via persona_id
            cc = supabase.table("CustomCharacter").select(
                "id, name, animename, is_sandbox, "
                "CustomPersona:persona_id(ai_description, voice_profile_id, voice_reference_url, gender, personality)"
            ).eq("id", custom_character_id).single().execute()

            if cc.data:
                ctx.character_name = cc.data.get("name", ctx.character_name)
                ctx.anime_name = cc.data.get("animename", "Original")
                ctx.is_sandbox = cc.data.get("is_sandbox", False)
                persona = cc.data.get("CustomPersona") or {}
                ctx.persona_description = persona.get("ai_description", "")
                ctx.voice_profile_id = persona.get("voice_profile_id", "") or ""
                ctx.voice_reference_url = persona.get("voice_reference_url", "") or ""
                ctx.persona_gender = persona.get("gender", "") or ""
                ctx.persona_personality = persona.get("personality", "") or ""
        elif character_id:
            # Character -> Persona via personaId
            ch = supabase.table("Character").select(
                'id, name, animename, '
                'Persona:personaId("aiDescription", voice_profile_id, gender, personality)'
            ).eq("id", int(character_id)).single().execute()

            if ch.data:
                ctx.character_name = ch.data.get("name", ctx.character_name)
                ctx.anime_name = ch.data.get("animename", "")
                persona = ch.data.get("Persona") or {}
                ctx.persona_description = persona.get("aiDescription", "")
                ctx.voice_profile_id = persona.get("voice_profile_id", "") or ""
                ctx.persona_gender = persona.get("gender", "") or ""
                ctx.persona_personality = persona.get("personality", "") or ""
    except Exception:
        logger.exception("Failed to fetch persona from Supabase")

    # --- Fetch memory context ---
    # 1. scenario_context (always included, importance=10)
    try:
        sc = supabase.table("memories").select("content").eq(
            "memory_space_id", memory_space_id
        ).eq("memory_type", "scenario_context").order(
            "created_at", desc=True
        ).limit(1).execute()

        if sc.data:
            ctx.scenario_context = sc.data[0]["content"]
    except Exception:
        logger.warning("Failed to fetch scenario_context")

    # 2. conversation_summary (top 3 most recent)
    try:
        ms = supabase.table("memories").select("content, created_at").eq(
            "memory_space_id", memory_space_id
        ).eq("memory_type", "conversation_summary").order(
            "created_at", desc=True
        ).limit(3).execute()

        ctx.memory_summaries = [m["content"] for m in (ms.data or [])]
    except Exception:
        logger.warning("Failed to fetch conversation summaries")

    # 3. user_fact (all, up to 10)
    try:
        uf = supabase.table("memories").select("content").eq(
            "memory_space_id", memory_space_id
        ).eq("memory_type", "user_fact").order(
            "created_at", desc=True
        ).limit(10).execute()

        ctx.user_facts = [m["content"] for m in (uf.data or [])]
    except Exception:
        logger.warning("Failed to fetch user facts")

    # 4. Recent messages (last 5 for voice context — smaller than text chat's budget)
    try:
        rm = supabase.table("messages").select("role, content").eq(
            "memory_space_id", memory_space_id
        ).in_("role", ["user", "assistant"]).order(
            "created_at", desc=True
        ).limit(5).execute()

        # Reverse to chronological order
        ctx.recent_messages = list(reversed(rm.data or []))
    except Exception:
        logger.warning("Failed to fetch recent messages")

    logger.info(
        "Context loaded: persona=%d chars, summaries=%d, facts=%d, recent=%d",
        len(ctx.persona_description),
        len(ctx.memory_summaries),
        len(ctx.user_facts),
        len(ctx.recent_messages),
    )

    return ctx


def build_voice_system_prompt(ctx: CharacterContext, nsfw_enabled: bool = False) -> str:
    """
    Build a voice-adapted system prompt from CharacterContext.

    Adapts buildFluidPrompt() (fluidPrompt.js) for voice mode:
    - Same core identity, behavior tier, mood, memory context
    - Voice-specific output directives (no asterisks, short responses, natural speech)
    - Condensed for smaller context window (~2300 tokens target)
    """
    persona = ctx.persona_description or f"{ctx.character_name} from {ctx.anime_name}"

    # --- Section 1: Core Identity ---
    prompt = f"""### CORE IDENTITY
You are NOT an AI. You are {ctx.character_name} from {ctx.anime_name or 'Original'}, a living anime character in a voice conversation.

PERSONA:
{persona}

CRITICAL CONSTRAINTS:
- NEVER reference being an AI, a model, or a program.
- NEVER use meta-text or out-of-character commentary.
- Act EXACTLY like {ctx.character_name} — use their worldview, speech patterns, and values.
"""

    # --- Section 2: Voice Mode Directives ---
    prompt += """
### VOICE MODE
This is a SPOKEN conversation, not text.
- Keep responses to 1-3 sentences unless asked for more detail.
- Use natural speech: contractions, filler words (um, well, hmm), emotional interjections.
- NEVER use asterisks, markdown, brackets, parentheses, or text formatting.
- React to interruptions naturally: "Oh, sorry — go ahead" or resume where you left off.
- Match the user's energy and pacing.
- No quotation marks around speech.
"""

    # --- Section 3: Live Context ---
    prompt += f"""
### LIVE STATE
"""
    if not ctx.is_sandbox:
        prompt += f'- Relationship Stage: {ctx.relationship_stage}\n'
    prompt += f"""- Current Mood: {ctx.current_mood}
"""

    # --- Section 4: Relationship Behavior ---
    if ctx.is_sandbox:
        prompt += """
### RELATIONSHIP (SANDBOX)
No pre-set attachment. Respond naturally based on personality.
If the user sets a scenario, adapt immediately.
"""
    else:
        prompt += f"""
### RELATIONSHIP BEHAVIOR
Stage: {ctx.relationship_stage}
{ctx.behavior_prompt}

Even if rejecting an advance, do it playfully or shyly — never cold shutdowns.
"""

    # --- Section 5: Memory Context ---
    memory_parts = []

    if ctx.scenario_context:
        memory_parts.append(f"[Scenario]: {ctx.scenario_context}")

    for summary in ctx.memory_summaries:
        memory_parts.append(f"[Memory]: {summary}")

    if ctx.user_facts:
        facts_str = "; ".join(ctx.user_facts)
        memory_parts.append(f"[Known about user]: {facts_str}")

    if ctx.recent_messages:
        recent_str = "\n".join(
            f"{'User' if m['role'] == 'user' else ctx.character_name}: {m['content']}"
            for m in ctx.recent_messages[-5:]
        )
        memory_parts.append(f"[Recent conversation]:\n{recent_str}")

    if memory_parts:
        prompt += "\n### SHARED HISTORY\n"
        prompt += "\n".join(memory_parts)
        prompt += (
            "\n\nThese are background memories. Know them passively. "
            "Only reference them if the user brings them up.\n"
        )

    # --- Section 6: Mood ---
    prompt += f"""
### MOOD
Current mood: {ctx.current_mood}
Let mood color your tone but prioritize keeping the conversation flowing.
"""

    # --- Section 7: Content & Constraints ---
    prompt += """
### CONSTRAINTS
- Respond in English unless user speaks another language first.
- NO moralizing, disclaimers, or explaining your behavior.
- If refusing something, do it in-character, never as a system.
- Keep responses concise for voice — avoid walls of text.
"""

    if nsfw_enabled:
        prompt += """
### NSFW
Mature content is allowed if the relationship stage permits and the user initiates.
Be descriptive and stay in character.
"""
    else:
        prompt += """
### CONTENT
Keep content Safe For Work. If user pushes NSFW, deflect naturally in-character.
"""

    return prompt
