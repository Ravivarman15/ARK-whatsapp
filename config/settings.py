"""
config/settings.py
──────────────────
Centralised application configuration loaded from environment variables.

Uses Pydantic Settings for type-safe validation. All secrets and tunables
live here so no other module reads os.getenv() directly.
"""

from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings — automatically loaded from .env file
    and environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Supabase ──────────────────────────────────────────────────
    SUPABASE_URL: str
    SUPABASE_KEY: str

    # ── HuggingFace ───────────────────────────────────────────────
    HF_API_TOKEN: str

    # ── LLM Selection ─────────────────────────────────────────────
    # Switch models via env var without code changes.
    # Primary:     Qwen/Qwen2.5-72B-Instruct  (free tier, fast)
    # Alternative: meta-llama/Llama-3.3-70B-Instruct
    LLM_MODEL: str = "Qwen/Qwen2.5-72B-Instruct"

    # ── AiSensy (WhatsApp) ───────────────────────────────────────
    AISENSY_API_KEY: str = ""                  # AiSensy Project API key
    AISENSY_PROJECT_ID: str = ""               # AiSensy project ID (for session replies)
    AISENSY_CAMPAIGN_NAME: str = ""            # Campaign name for template messages (follow-ups / admin alerts)

    # ── Admin / Escalation ────────────────────────────────────────
    ADMIN_WHATSAPP_NUMBER: str = ""       # Admin phone for lead alerts
    ESCALATION_COOLDOWN: int = 600        # Seconds before re-notifying for same user

    # ── Zapier (Google Sheets integration) ────────────────────────
    ZAPIER_WEBHOOK_URL: str = ""          # Zapier Catch Hook URL for Google Sheets

    # ── Redis (optional — falls back to in-memory cache) ──────────
    REDIS_URL: str = ""

    # ── RAG Tuning ────────────────────────────────────────────────
    TOP_K: int = 3               # Number of chunks to retrieve
    CHUNK_SIZE: int = 500        # Characters per chunk
    CHUNK_OVERLAP: int = 80      # Overlap between chunks

    # ── Cache ─────────────────────────────────────────────────────
    CACHE_TTL: int = 3600        # Seconds before cached answer expires

    # ── Conversation Memory ───────────────────────────────────────
    MEMORY_MAX_TURNS: int = 3    # Number of past turns to keep per user

    # ── Performance ───────────────────────────────────────────────
    HF_TIMEOUT: int = 30         # Seconds before HuggingFace call times out
    MAX_NEW_TOKENS: int = 150    # WhatsApp-friendly short responses

    # ── Follow-Up Automation (within 24-hour WhatsApp window) ─────
    FOLLOWUP_STAGE1_DELAY: int = 1800        # Seconds before 1st follow-up (30 min)
    FOLLOWUP_STAGE2_DELAY: int = 14400       # Seconds before 2nd follow-up (4 hr)
    FOLLOWUP_STAGE3_DELAY: int = 57600       # Seconds before 3rd follow-up (16 hr)
    FOLLOWUP_CHECK_INTERVAL: int = 60        # Seconds between scheduler runs
    MAX_FOLLOWUP_STAGE: int = 3              # Max number of follow-ups per user

    # ── Intelligence Engine ───────────────────────────────────────
    ADA_INJECTION_FREQUENCY: int = 0         # Disabled — ADA only after admission intent
    TRIGGER_ROTATION_FREQUENCY: int = 0      # Disabled — triggers only on intent match
    CONFUSION_ESCALATION_THRESHOLD: int = 3  # Escalate after N unanswered queries


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
