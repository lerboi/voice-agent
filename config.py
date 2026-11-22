import json
import os
from dotenv import load_dotenv

load_dotenv()

# LiveKit
LIVEKIT_URL = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

_split = lambda s: [p.strip() for p in s.split(",") if p.strip()]

# --- Voice LLM provider -------------------------------------------------------
# "parasail" (default, 2026-11-23): Parasail's own OpenAI-compatible API, called
# directly. Parasail was the fastest DeepSeek V4.1 Flash upstream in the
# 2026-11-23 benchmark (0.31-0.43s to first token, 217-322 tok/s) and its terms
# don't ban adult content. "openrouter": the previous path, kept for rollback
# (set VOICE_LLM_PROVIDER=openrouter + a valid OPENROUTER_API_KEY, no code change).
VOICE_LLM_PROVIDER = os.getenv("VOICE_LLM_PROVIDER", "parasail").strip().lower()

# Parasail (https://docs.parasail.io). parasail-deepseek-v41-flash = DeepSeek
# V4.1 Flash, the same model text chat uses.
PARASAIL_BASE_URL = os.getenv("PARASAIL_BASE_URL", "https://api.parasail.io/v1")
PARASAIL_MODEL = os.getenv("PARASAIL_MODEL", "parasail-deepseek-v41-flash")
# Thinking must stay OFF for voice — every reply would otherwise wait for hidden
# reasoning tokens. Parasail serves DeepSeek through its chat template, so the
# switch goes in chat_template_kwargs; both key spellings are sent (templates
# ignore the one they don't use). Override with PARASAIL_EXTRA_BODY='{...}'.
PARASAIL_EXTRA_BODY = json.loads(os.getenv(
    "PARASAIL_EXTRA_BODY",
    '{"chat_template_kwargs": {"thinking": false, "enable_thinking": false}}',
))

# OpenRouter (OpenAI-compatible) — rollback provider.
# deepseek/deepseek-v4.1-flash = DeepSeek V4.1 Flash. Switched from
# deepseek-chat-v3-0324 on 2026-11-09: that id is down to one slow upstream
# (SiliconFlow, ~1.9s to first token), while V4.1 Flash has 30+ upstreams.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash")
# V4.1 Flash can "think" before answering. Thinking must stay OFF for voice:
# every reply would otherwise wait for hidden reasoning tokens. Sent on every
# OpenRouter call (voice turns, shadow summaries, archiver).
OPENROUTER_REASONING = {"enabled": False}
# Optional app-attribution headers recommended by OpenRouter (shown on their
# dashboard; no functional effect).
OPENROUTER_HEADERS = {
    "HTTP-Referer": os.getenv("OPENROUTER_APP_URL", "https://anione.me"),
    "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Anione Voice"),
}
# Upstream routing (https://openrouter.ai/docs/features/provider-routing):
# try providers in this order, fall back to any other if all are down, and
# never use the ignored ones. V4.1 Flash median time to first token,
# 2026-11-07 (reasoning off): alibaba 0.42s (cheapest fast one, $0.15/$0.60
# per 1M), together 0.37s, coreweave 0.50s, parasail 0.47s. Ignored: novita
# (ignores reasoning-off, returns empty replies), baseten (429 rate limits),
# siliconflow + gmicloud (~1.7s). Tune via env without a deploy.
# 2026-11-23: parasail first (fastest in that day's benchmark); together moved
# to ignore — its terms ban pornographic content.
OPENROUTER_PROVIDER_PREFS = {
    "order": _split(os.getenv("OPENROUTER_PROVIDER_ORDER", "parasail,alibaba,wafer")),
    "allow_fallbacks": os.getenv("OPENROUTER_ALLOW_FALLBACKS", "true").lower() != "false",
    "ignore": _split(os.getenv("OPENROUTER_PROVIDER_IGNORE", "together,novita,baseten,siliconflow,gmicloud")),
}

# --- Resolved settings every LLM call site uses --------------------------------
# LLM_EXTRA_BODY: normal calls. LLM_JSON_EXTRA_BODY: JSON-mode calls (OpenRouter
# additionally requires upstreams that support response_format).
if VOICE_LLM_PROVIDER == "openrouter":
    if not OPENROUTER_API_KEY:
        raise RuntimeError("VOICE_LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is not set")
    LLM_API_KEY = OPENROUTER_API_KEY
    LLM_BASE_URL = OPENROUTER_BASE_URL
    LLM_MODEL = OPENROUTER_MODEL
    LLM_HEADERS = OPENROUTER_HEADERS
    LLM_EXTRA_BODY = {"provider": OPENROUTER_PROVIDER_PREFS, "reasoning": OPENROUTER_REASONING}
    LLM_JSON_EXTRA_BODY = {
        "provider": {**OPENROUTER_PROVIDER_PREFS, "require_parameters": True},
        "reasoning": OPENROUTER_REASONING,
    }
elif VOICE_LLM_PROVIDER == "parasail":
    LLM_API_KEY = os.environ["PARASAIL_API_KEY"]
    LLM_BASE_URL = PARASAIL_BASE_URL
    LLM_MODEL = PARASAIL_MODEL
    LLM_HEADERS = {}
    LLM_EXTRA_BODY = PARASAIL_EXTRA_BODY
    LLM_JSON_EXTRA_BODY = PARASAIL_EXTRA_BODY
else:
    raise RuntimeError(f"Unknown VOICE_LLM_PROVIDER={VOICE_LLM_PROVIDER!r} (use parasail or openrouter)")

# Deepgram
DEEPGRAM_API_KEY = os.environ["DEEPGRAM_API_KEY"]

# Fish Speech TTS
FISH_SPEECH_API_KEY = os.environ.get("FISH_SPEECH_API_KEY", "")
FISH_SPEECH_API_URL = os.environ.get(
    "FISH_SPEECH_API_URL", "https://api.fish.audio"
)
FISH_SPEECH_VOICE_ID = os.getenv("FISH_SPEECH_VOICE_ID", "")
FISH_SPEECH_MODEL = os.getenv("FISH_SPEECH_MODEL", "s2-pro")

# Next.js API (for billing)
NEXTJS_API_URL = os.environ.get("NEXTJS_API_URL", "http://localhost:3000")
VOICE_AGENT_API_KEY = os.environ["VOICE_AGENT_API_KEY"]

# Supabase (direct DB access)
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

# Voyage AI (embeddings)
VOYAGE_API_KEY = os.environ["VOYAGE_API_KEY"]
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-4")

# Agent settings
SHADOW_SUMMARY_INTERVAL = int(os.getenv("SHADOW_SUMMARY_INTERVAL", "5"))
BILLING_INTERVAL_SECONDS = int(os.getenv("BILLING_INTERVAL_SECONDS", "60"))
TURN_CHECKPOINT_INTERVAL = int(os.getenv("TURN_CHECKPOINT_INTERVAL", "2"))
INACTIVITY_TIMEOUT_SECONDS = int(os.getenv("INACTIVITY_TIMEOUT_SECONDS", "300"))  # 5 minutes
