"""
backend/config/settings.py
Central application settings.

API Key Rotation architecture:
  - Single fixed model: llama-3.3-70b-versatile
  - Multiple API keys rotate automatically on rate-limit errors
  - No model switching — only key switching

Groq free-tier limits per key (June 2026):
  llama-3.3-70b-versatile: TPM 12,000 | TPD 100,000 | RPD 1,000
  (Confirmed from live Groq 429 responses — "Limit 12000" — not the
   stale 6,000 figure previously assumed here.)
"""
from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):

    # ── App ──────────────────────────────────────────────────────
    app_name:    str  = "LawCaseIntelligence"
    app_version: str  = "1.0.0"
    debug:       bool = False
    log_level:   str  = "INFO"

    # ── Groq — Fixed Model ────────────────────────────────────────
    # Single model, never changes
    groq_primary_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Fixed Groq model. Do NOT change this.",
    )
    # Legacy alias kept so existing code using settings.groq_model still works
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Alias for groq_primary_model.",
    )

    # ── Groq — API Key Pool ───────────────────────────────────────
    # Primary key (legacy / single-key fallback)
    groq_api_key: str = Field(default="", description="Primary Groq API key (legacy)")
    # Numbered pool — loaded directly by APIKeyManager via os.environ
    # These are NOT declared as Settings fields to avoid pydantic validation issues
    # with an arbitrary number of keys. APIKeyManager reads them from os.environ.

    # ── Groq — Limits (per key) ───────────────────────────────────
    groq_max_tokens:     int   = 8192
    groq_temperature:    float = 0.1
    groq_tpm_limit:      int   = 12_000
    groq_tpd_limit:      int   = 100_000
    groq_rpd_limit:      int   = 1_000
    groq_rpm_limit:      int   = 30
    groq_context_window: int   = 128_000

    # ── API Key Manager — cooldowns ───────────────────────────────
    # Applied when a Groq rate-limit error doesn't include its own
    # retry_after value.
    key_cooldown_seconds:         int = 80
    key_max_consecutive_failures: int = 3
    # Minimum cooldown when a TPD/RPD error puts a key on "until midnight
    # UTC" cooldown — avoids a near-zero cooldown if the error happens at
    # 23:59:59 UTC.
    key_daily_cooldown_min_seconds: int = 60
    # Persisted daily-exhaustion / cooldown state (survives restarts)
    key_state_file: str = str(BASE_DIR / "data" / "system" / "key_state.json")

    # ── TokenTracker — sliding-window sizes ────────────────────────
    # TPM/RPM are defined by Groq over a rolling 60s window; TPD/RPD over
    # a rolling 24h window. These are protocol-defined, not provider-
    # tunable, but centralized here so no file declares its own copy.
    tpm_window_seconds: float = 60.0
    tpd_window_seconds: float = 86_400.0

    # ── TokenScheduler — score-based key selection ─────────────────
    #   score = remaining_tpm_pct * scheduler_weight_remaining_tpm
    #         + success_rate      * scheduler_weight_success_rate
    #         + idle_time_score   * scheduler_weight_idle_time
    scheduler_weight_remaining_tpm:     float = 0.70
    scheduler_weight_success_rate:      float = 0.20
    scheduler_weight_idle_time:         float = 0.10
    scheduler_safe_threshold_tpm:       int   = 2_000  # min remaining TPM to select a key
    scheduler_idle_full_credit_seconds: float = 30.0   # idle >= this -> full idle score
    # In-flight "reservation" of a call's estimated tokens against the key
    # TokenScheduler just selected, so concurrent callers see this key as
    # busier before its actual usage is recorded (prevents a "thundering
    # herd" where several concurrent agents all pick the same idle key).
    # Reservations are released when the call completes, but auto-expire
    # after this many seconds as a safety net if release is ever missed.
    scheduler_reservation_ttl_seconds:  float = 30.0

    # ── agents.utils.rate_guard — per-chunk pacing ──────────────────
    inter_chunk_delay_seconds:      float = 6.0    # pause between every chunk LLM call
    chunk_failure_backoff_seconds:  float = 12.0   # extra pause after a chunk failure
    cooldown_poll_interval_seconds: float = 5.0    # poll interval while waiting for a key
    max_cooldown_wait_seconds:      float = 120.0  # max time to wait for any key to recover

    # ── Universal Chunk Extractor (UCE) ─────────────────────────────
    uce_max_tokens:               int   = 1_500  # normal extraction call (small fixed JSON schema)
    uce_retry_max_tokens:         int   = 3_000  # retry budget if a prior attempt returned truncated JSON
    uce_max_chunk_attempts:       int   = 4      # in-line attempts per chunk before deferring
    uce_final_pass_attempts:      int   = 2      # extra attempts per chunk in the cleanup pass
    uce_key_wait_timeout_seconds: float = 90.0   # max seconds to wait for any key to free up
    uce_tpm_low_threshold:        int   = 1_000  # below this remaining TPM -> long throttle sleep
    uce_tpm_warn_threshold:       int   = 3_000  # below this remaining TPM -> short throttle sleep

    # ── Embeddings ────────────────────────────────────────────────
    bge_model_name:      str = "BAAI/bge-large-en-v1.5"
    embedding_dimension: int = 1024

    # ── ChromaDB ──────────────────────────────────────────────────
    chroma_persist_dir:       str = str(BASE_DIR / "data" / "chroma")
    chroma_collection_prefix: str = "legal_"

    # ── PDF Processing ────────────────────────────────────────────
    upload_dir:           str = str(BASE_DIR / "data" / "uploads")
    processed_dir:        str = str(BASE_DIR / "data" / "processed")
    reports_dir:          str = str(BASE_DIR / "data" / "reports")
    exports_dir:          str = str(BASE_DIR / "data" / "exports")
    max_upload_mb:        int = 100
    large_page_threshold: int = 100

    chunk_size:    int = 2_000
    chunk_overlap: int = 200

    # ── Database ──────────────────────────────────────────────────
    database_url: str = Field(
        default=f"sqlite:///{BASE_DIR / 'data' / 'lawcase.db'}"
    )

    # ── RAG ───────────────────────────────────────────────────────
    rag_top_k:                int   = 5
    rag_rerank_top_k:         int   = 3
    rag_similarity_threshold: float = 0.3

    # ── Rate Limiting & Retries ───────────────────────────────────
    inter_call_delay: float = 2.0
    max_retries:      int   = 3
    backoff_base:     float = 2.0
    # wait = backoff_base ** attempt * retry_backoff_multiplier_seconds
    retry_backoff_multiplier_seconds: float = 5.0
    # Extra buffer added to Groq's "try again in Xs" retry_after value
    retry_after_buffer_seconds: int = 1

    # ── LangGraph ─────────────────────────────────────────────────
    langgraph_max_retry: int = 2
    checkpoint_dir:      str = str(BASE_DIR / "data" / "checkpoints")
    pipeline_total_steps: int = 7

    # ── Flask ─────────────────────────────────────────────────────
    flask_secret: str = Field(default_factory=lambda: secrets.token_hex(32))
    port:         int = 5001

    # ── Chat System ──────────────────────────────────────────────────
    chat_max_recent_messages: int = 6
    chat_summarize_threshold:   int = 20
    chat_max_summary_tokens:    int = 300
    chat_legal_disclaimer:      str = "⚠️ This is general legal information and not legal advice. For advice specific to your situation, please consult a qualified lawyer."

    # ── Chat Token Budget ────────────────────────────────────────────
    chat_total_request_budget:  int = 7000
    chat_output_budget:         int = 2500
    chat_input_budget:          int = 4500
    chat_system_prompt_budget:   int = 500
    chat_user_query_budget:     int = 300
    chat_rag_context_budget:   int = 2500
    chat_history_budget:        int = 1000
    chat_summary_budget:        int = 300

    class Config:
        env_file          = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        case_sensitive    = False
        extra             = "ignore"

    def validate_provider(self) -> None:
        """Ensure at least one API key is configured."""
        import os
        has_numbered = any(
            os.environ.get(f"GROQ_API_KEY_{i}", "").strip()
            for i in range(1, 11)
        )
        if not has_numbered and not self.groq_api_key:
            raise EnvironmentError(
                "No Groq API keys found. "
                "Set GROQ_API_KEY_1 (and optionally _2, _3, _4) in your .env file. "
                "Or set the legacy GROQ_API_KEY for single-key operation."
            )

    def ensure_dirs(self) -> None:
        for d in [
            self.upload_dir, self.processed_dir, self.reports_dir,
            self.exports_dir, self.chroma_persist_dir, self.checkpoint_dir,
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)
        Path(self.key_state_file).parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
