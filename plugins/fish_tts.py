"""
Fish Speech TTS plugin — wraps Fish Audio's OpenAI-compatible
TTS endpoint via the livekit-plugins-openai TTS adapter.
Streams audio chunk-by-chunk for low-latency voice output.

Supports dynamic voice profiles per character:
  - voice_profile_id: Fish Speech voice model ID (for pre-built archetypes)
  - voice_reference_url: Custom reference audio URL (for voice cloning)
"""

from livekit.plugins.openai import TTS

from config import FISH_SPEECH_API_KEY, FISH_SPEECH_API_URL, FISH_SPEECH_MODEL, FISH_SPEECH_VOICE_ID

# Pre-built voice archetypes — map personality archetype to Fish Speech voice IDs.
# Populate these with actual Fish Audio model IDs after uploading reference audio.
# The agent selects an archetype based on the character's persona keywords.
VOICE_ARCHETYPES = {
    "tsundere_female":  "",  # Fiery, sharp tone with occasional softness
    "kuudere_female":   "",  # Cool, composed, low-energy delivery
    "genki_female":     "",  # Energetic, bright, upbeat
    "dandere_female":   "",  # Soft, shy, hesitant delivery
    "onee_san_female":  "",  # Mature, warm, gentle
    "yandere_female":   "",  # Sweet surface, intense undertone
    "default_female":   "",  # Neutral anime female voice
    "shonen_male":      "",  # Energetic, determined male voice
    "cool_male":        "",  # Calm, deep, composed male voice
    "gentle_male":      "",  # Soft-spoken, kind male voice
    "default_male":     "",  # Neutral anime male voice
}


def get_archetype_voice_id(gender: str | None, personality: str | None) -> str:
    """
    Select a voice archetype based on character gender and personality keywords.
    Falls back to gender-default, then global default.
    """
    gender_lower = (gender or "").lower()
    personality_lower = (personality or "").lower()

    if "female" in gender_lower or gender_lower in ("f", "girl", "woman"):
        if any(kw in personality_lower for kw in ("tsundere", "fiery", "hot-headed", "stubborn")):
            return VOICE_ARCHETYPES.get("tsundere_female", "")
        if any(kw in personality_lower for kw in ("kuudere", "cool", "stoic", "emotionless")):
            return VOICE_ARCHETYPES.get("kuudere_female", "")
        if any(kw in personality_lower for kw in ("genki", "energetic", "cheerful", "bubbly")):
            return VOICE_ARCHETYPES.get("genki_female", "")
        if any(kw in personality_lower for kw in ("dandere", "shy", "quiet", "timid")):
            return VOICE_ARCHETYPES.get("dandere_female", "")
        if any(kw in personality_lower for kw in ("onee", "mature", "motherly", "caring", "big sister")):
            return VOICE_ARCHETYPES.get("onee_san_female", "")
        if any(kw in personality_lower for kw in ("yandere", "obsessive", "possessive")):
            return VOICE_ARCHETYPES.get("yandere_female", "")
        return VOICE_ARCHETYPES.get("default_female", "")

    if "male" in gender_lower or gender_lower in ("m", "boy", "man"):
        if any(kw in personality_lower for kw in ("energetic", "determined", "shonen", "hot-blooded")):
            return VOICE_ARCHETYPES.get("shonen_male", "")
        if any(kw in personality_lower for kw in ("cool", "calm", "stoic", "composed")):
            return VOICE_ARCHETYPES.get("cool_male", "")
        if any(kw in personality_lower for kw in ("gentle", "kind", "soft", "caring")):
            return VOICE_ARCHETYPES.get("gentle_male", "")
        return VOICE_ARCHETYPES.get("default_male", "")

    return ""


def create_fish_tts(voice_id: str | None = None) -> TTS:
    """
    Create a Fish Speech TTS instance.

    Args:
        voice_id: Fish Speech voice model ID to use. Falls back to
                  FISH_SPEECH_VOICE_ID env var, then "default".
    """
    effective_voice = voice_id or FISH_SPEECH_VOICE_ID or "default"

    return TTS(
        model=FISH_SPEECH_MODEL,
        api_key=FISH_SPEECH_API_KEY,
        base_url=f"{FISH_SPEECH_API_URL}/v1",
        voice=effective_voice,
    )
