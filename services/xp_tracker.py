"""
XP Tracker — lightweight in-call intent classification and XP tracking.

Simplified version of analyzer.js for real-time voice. Uses fast pattern-based
classification (no API call) to avoid adding latency to the voice loop.

XP formula: intimacy_impact * 4 (same x4 multiplier as text chat).

See Voice_Implementation.md Section 7.1.
"""

import re
import logging

logger = logging.getLogger("anione-voice-agent")

# Intent -> base intimacy impact mapping
# Text chat analyzer uses -10 to +20 range; voice uses a simpler subset
INTENT_MAP = {
    "Casual":    1,
    "Kind":      2,
    "Flirty":    3,
    "Emotional": 3,
    "Playful":   2,
    "Aggressive": -2,
}

# Keyword patterns for fast classification
_FLIRTY_PATTERNS = re.compile(
    r"\b(cute|beautiful|pretty|handsome|gorgeous|love you|miss you|kiss|hug|darling|babe|sweetheart|baby)\b",
    re.IGNORECASE,
)
_KIND_PATTERNS = re.compile(
    r"\b(thank|thanks|appreciate|care about|proud of|happy for|support|glad|amazing|wonderful)\b",
    re.IGNORECASE,
)
_EMOTIONAL_PATTERNS = re.compile(
    r"\b(sad|crying|hurt|scared|afraid|lonely|depressed|anxious|worried|lost|pain|struggle)\b",
    re.IGNORECASE,
)
_AGGRESSIVE_PATTERNS = re.compile(
    r"\b(hate|stupid|idiot|shut up|leave me|go away|annoying|worst|ugly|dumb)\b",
    re.IGNORECASE,
)
_PLAYFUL_PATTERNS = re.compile(
    r"\b(haha|lol|lmao|funny|joke|tease|silly|goofy|prank|bet you)\b",
    re.IGNORECASE,
)

# Relationship level definitions — mirrors lib/chat/relationship.js
LEVEL_TABLE = [
    (1, 0), (2, 50), (3, 120), (4, 200), (5, 300),
    (6, 420), (7, 560), (8, 720), (9, 900), (10, 1100),
    (11, 1320), (12, 1560), (13, 1820), (14, 2100), (15, 2400),
    (16, 2720), (17, 3060), (18, 3420), (19, 3800), (20, 4200),
    (21, 4620), (22, 5060), (23, 5520), (24, 6000), (25, 6500),
]

# Stage boundaries — mirrors getRelationshipStage() in relationship.js
STAGE_BOUNDARIES = [
    (25, "Soulmate"),
    (20, "Lover"),
    (15, "Crush"),
    (10, "Friend"),
    (5,  "Acquaintance"),
    (1,  "Stranger"),
]


def get_level_from_xp(xp: int) -> int:
    """Get relationship level from XP. Mirrors getLevelFromXP()."""
    level = 1
    for lvl, min_xp in LEVEL_TABLE:
        if xp >= min_xp:
            level = lvl
    return level


def get_stage_from_level(level: int) -> str:
    """Get relationship stage name from level. Mirrors getRelationshipStage()."""
    for min_level, stage in STAGE_BOUNDARIES:
        if level >= min_level:
            return stage
    return "Stranger"


def classify_intent(user_text: str) -> str:
    """
    Fast pattern-based intent classification.
    Returns one of: Casual, Kind, Flirty, Emotional, Playful, Aggressive.
    """
    if _AGGRESSIVE_PATTERNS.search(user_text):
        return "Aggressive"
    if _FLIRTY_PATTERNS.search(user_text):
        return "Flirty"
    if _EMOTIONAL_PATTERNS.search(user_text):
        return "Emotional"
    if _KIND_PATTERNS.search(user_text):
        return "Kind"
    if _PLAYFUL_PATTERNS.search(user_text):
        return "Playful"
    return "Casual"


def calculate_xp_delta(user_text: str) -> tuple[str, int]:
    """
    Classify intent and calculate XP delta for a single user turn.

    Returns:
        (intent, xp_delta) where xp_delta already has the x4 multiplier applied.
    """
    intent = classify_intent(user_text)
    base_impact = INTENT_MAP.get(intent, 1)
    xp_delta = base_impact * 4  # Same x4 multiplier as text chat
    return intent, xp_delta


def check_stage_change(old_xp: int, new_xp: int) -> dict | None:
    """
    Check if XP change crossed a relationship stage boundary.

    Returns:
        Dict with stage change info, or None if no change.
        { "old_stage", "new_stage", "old_level", "new_level" }
    """
    old_level = get_level_from_xp(old_xp)
    new_level = get_level_from_xp(new_xp)

    if new_level <= old_level:
        return None

    old_stage = get_stage_from_level(old_level)
    new_stage = get_stage_from_level(new_level)

    if old_stage == new_stage:
        return None

    return {
        "old_stage": old_stage,
        "new_stage": new_stage,
        "old_level": old_level,
        "new_level": new_level,
    }
