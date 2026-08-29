import os
from dotenv import load_dotenv

load_dotenv()

# LiveKit
LIVEKIT_URL = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

# DeepInfra (OpenAI-compatible) — replaces Groq as the voice LLM provider.
# DeepSeek-V3 is a plain chat model (no hidden reasoning tokens), so NO
# reasoning_effort is passed anywhere. Endpoint + model id per
# https://deepinfra.com/deepseek-ai/DeepSeek-V3/api (163k context).
DEEPINFRA_API_KEY = os.environ["DEEPINFRA_API_KEY"]
DEEPINFRA_BASE_URL = os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")
DEEPINFRA_MODEL = os.getenv("DEEPINFRA_MODEL", "deepseek-ai/DeepSeek-V3")

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
