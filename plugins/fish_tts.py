"""
Fish Speech TTS plugin — uses Fish Audio's native /v1/tts endpoint.

The livekit-plugins-openai TTS wrapper cannot be used because Fish Audio
does NOT have an OpenAI-compatible /v1/audio/speech endpoint. This plugin
implements livekit.agents.tts.TTS directly with httpx.

Supports dynamic voice profiles per character:
  - voice_profile_id: Fish Speech reference_id (for pre-built voices)
  - voice_reference_url: Custom reference audio URL (for voice cloning)
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, replace

import httpx
from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from config import FISH_SPEECH_API_KEY, FISH_SPEECH_API_URL, FISH_SPEECH_VOICE_ID

logger = logging.getLogger("anione-voice-agent")

SAMPLE_RATE = 44100
NUM_CHANNELS = 1
MAX_TTS_RETRIES = 2  # Total attempts = MAX_TTS_RETRIES + 1

# Fish Speech S1 recognized emotion/tone tags.
# Tags NOT in this set will be stripped before sending to TTS to prevent
# the model from reading them aloud (e.g. "(smirking)" → spoken as "smirking").
_VALID_FISH_TAGS = frozenset({
    # Emotions
    "happy", "sad", "angry", "excited", "calm", "nervous", "confident",
    "surprised", "satisfied", "delighted", "scared", "worried", "upset",
    "frustrated", "depressed", "empathetic", "embarrassed", "disgusted",
    "moved", "proud", "relaxed", "grateful", "curious", "sarcastic",
    "disdainful", "unhappy", "anxious", "hysterical", "indifferent",
    "uncertain", "doubtful", "confused", "disappointed", "regretful",
    "guilty", "ashamed", "jealous", "envious", "hopeful", "optimistic",
    "pessimistic", "nostalgic", "lonely", "bored", "contemptuous",
    "sympathetic", "compassionate", "determined", "resigned",
    # Tone/effects
    "in a hurry tone", "shouting", "screaming", "whispering", "soft tone",
    "laughing", "chuckling", "sobbing", "crying loudly", "sighing",
    "groaning", "panting", "gasping", "yawning", "snoring",
    # Pauses
    "break", "long-break",
})

# Regex to match (tag) patterns — captures the tag content
_TAG_RE = re.compile(r"\(([^)]+)\)")

# Regex to match *action* patterns (roleplay actions the LLM shouldn't output in voice mode)
_ASTERISK_ACTION_RE = re.compile(r"\*[^*]+\*")


def _sanitize_tts_input(text: str) -> str:
    """
    Clean text before sending to Fish Speech TTS.

    1. Strip *action* text (e.g. *smirks*, *blushes*)
    2. Strip (tag) where tag is NOT in Fish Speech's recognized set
       (e.g. (smirking), (blushing), (teasing) get removed)
    3. Valid Fish Speech tags like (happy), (laughing) are preserved
    """
    # Step 1: Remove *action* patterns
    text = _ASTERISK_ACTION_RE.sub("", text)

    # Step 2: Remove invalid emotion tags, keep valid ones
    def _replace_tag(match: re.Match) -> str:
        tag_content = match.group(1).strip().lower()
        if tag_content in _VALID_FISH_TAGS:
            return match.group(0)  # Keep valid tag as-is
        return ""  # Strip invalid tag

    text = _TAG_RE.sub(_replace_tag, text)

    # Clean up extra whitespace from removals
    text = re.sub(r"  +", " ", text).strip()
    return text


# Pre-built voice archetypes — map personality archetype to Fish Speech reference IDs.
VOICE_ARCHETYPES = {
    "tsundere_female":  "",
    "kuudere_female":   "",
    "genki_female":     "",
    "dandere_female":   "",
    "onee_san_female":  "",
    "yandere_female":   "",
    "default_female":   "",
    "shonen_male":      "",
    "cool_male":        "",
    "gentle_male":      "",
    "default_male":     "",
}


def get_archetype_voice_id(gender: str | None, personality: str | None) -> str:
    """Select a voice archetype based on character gender and personality keywords."""
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


@dataclass
class _FishTTSOptions:
    reference_id: str
    format: str


class FishTTS(tts.TTS):
    """Fish Audio TTS using native /v1/tts endpoint."""

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "",
        reference_id: str = "",
        response_format: str = "mp3",
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._api_key = api_key or FISH_SPEECH_API_KEY
        self._base_url = (base_url or FISH_SPEECH_API_URL).rstrip("/")
        self._opts = _FishTTSOptions(
            reference_id=reference_id or FISH_SPEECH_VOICE_ID or "",
            format=response_format,
        )

    def update_options(self, *, reference_id: str | None = None) -> None:
        if reference_id is not None:
            self._opts.reference_id = reference_id

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "FishChunkedStream":
        return FishChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        pass


class FishChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: FishTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: FishTTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        url = f"{self._tts._base_url}/v1/tts"
        headers = {
            "Authorization": f"Bearer {self._tts._api_key}",
            "Content-Type": "application/json",
        }

        # Sanitize text: strip invalid emotion tags and *actions* before TTS
        clean_text = _sanitize_tts_input(self.input_text)
        if not clean_text:
            # Nothing left after sanitization — skip TTS
            return

        body: dict = {
            "text": clean_text,
            "format": self._opts.format,
        }
        if self._opts.reference_id:
            body["reference_id"] = self._opts.reference_id

        last_error = None
        streaming_started = False
        for attempt in range(MAX_TTS_RETRIES + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(30, connect=self._conn_options.timeout),
                ) as client:
                    async with client.stream("POST", url, headers=headers, json=body) as resp:
                        if resp.status_code != 200:
                            error_body = await resp.aread()
                            raise Exception(
                                f"Fish Audio TTS error {resp.status_code}: {error_body.decode()}"
                            )

                        output_emitter.initialize(
                            request_id=resp.headers.get("x-request-id", ""),
                            sample_rate=SAMPLE_RATE,
                            num_channels=NUM_CHANNELS,
                            mime_type=f"audio/{self._opts.format}",
                        )
                        streaming_started = True

                        async for chunk in resp.aiter_bytes():
                            output_emitter.push(chunk)

                        output_emitter.flush()
                        return  # Success — exit retry loop

            except Exception as exc:
                last_error = exc
                # Don't retry if we already started streaming — emitter can't be re-initialized
                if streaming_started:
                    raise
                if attempt < MAX_TTS_RETRIES:
                    wait = (attempt + 1) * 1.0  # 1s, 2s
                    logger.warning(
                        "Fish TTS attempt %d/%d failed: %s — retrying in %.0fs",
                        attempt + 1, MAX_TTS_RETRIES + 1, str(exc)[:200], wait,
                    )
                    await asyncio.sleep(wait)

        # All retries exhausted — raise last error
        raise last_error


def create_fish_tts(voice_id: str | None = None) -> FishTTS:
    """
    Create a Fish Audio TTS instance.

    Args:
        voice_id: Fish Audio reference_id for the voice to use.
                  Falls back to FISH_SPEECH_VOICE_ID env var.
    """
    return FishTTS(
        api_key=FISH_SPEECH_API_KEY,
        base_url=FISH_SPEECH_API_URL,
        reference_id=voice_id or FISH_SPEECH_VOICE_ID or "",
    )
