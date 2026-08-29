import os
from dotenv import load_dotenv

load_dotenv()

# LiveKit
LIVEKIT_URL = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

# OpenRouter (OpenAI-compatible) — the voice LLM provider.
# deepseek/deepseek-chat-v3-0324 = DeepSeek-V3 (0324 checkpoint; 163k context,
# plain chat model with no hidden reasoning tokens, so NO reasoning_effort is
# passed anywhere). Chosen over the original "deepseek/deepseek-chat" id for
# VOICE LATENCY: that id is served by only 3 upstreams (best ~1.15s TTFT on an
# fp4 quant with 86% uptime), while v3-0324 has a bf16 upstream (Crusoe)
# measured at ~0.95s TTFT. Set OPENROUTER_MODEL=deepseek/deepseek-chat to go
# back. v3.1/v3.2 ids expose a "reasoning" param — keep them off for voice.
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324")
# Optional app-attribution headers recommended by OpenRouter (shown on their
# dashboard; no functional effect).
OPENROUTER_HEADERS = {
    "HTTP-Referer": os.getenv("OPENROUTER_APP_URL", "https://anione.me"),
    "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Anione Voice"),
}
# Upstream routing (https://openrouter.ai/docs/features/provider-routing):
# try providers in this order, fall back to any other if all are down, and
# never use the ignored ones. Order = measured median TTFT 2026-08-29:
# crusoe (bf16) 0.95s, novita (fp8) 0.95s, deepinfra (fp4) 1.14s,
# siliconflow 1.4s; gmicloud 4-14s (ignored). Tune via env without a deploy.
_split = lambda s: [p.strip() for p in s.split(",") if p.strip()]
OPENROUTER_PROVIDER_PREFS = {
    "order": _split(os.getenv("OPENROUTER_PROVIDER_ORDER", "crusoe,novita,deepinfra")),
    "allow_fallbacks": os.getenv("OPENROUTER_ALLOW_FALLBACKS", "true").lower() != "false",
    "ignore": _split(os.getenv("OPENROUTER_PROVIDER_IGNORE", "gmicloud")),
}

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
